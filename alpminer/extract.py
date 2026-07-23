"""LLM extraction stage, driven by the active domain profile.

Two-pass design to control cost:
  1. triage (cheap model, truncated text): is this paper relevant at all,
     per the profile's triage prompt? Reviews/theory papers stop here.
  2. extraction (strong model, full text): forced tool call returning
     structured records, validated against the profile's field specs.

Robustness:
  - every raw LLM response is cached to data/raw_llm/<id>.json BEFORE
    parsing, so a parsing bug never wastes paid tokens (fix + re-run
    reparses the cache; caches from alpminer 1.x parse unchanged);
  - each paper is committed independently; failures are recorded and skipped;
  - Ctrl-C exits cleanly between papers with all progress saved;
  - extract_workers > 1 in alpminer.toml parallelizes the LLM calls (the
    database stays single-threaded; per-paper commits are unchanged);
  - a database is locked to one profile: extracting with a different profile
    is refused with guidance, so fields never silently mix.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from . import db, profiles
from .config import Config
from .pdftext import ensure_text
from .providers import Backend, LLMError, QuotaExhausted, get_backend
from .schema import (ExtractionResult, TriageResult, apply_backfills,
                     apply_derivations)
from .utils import atomic_write_text, log

PROFILE_META_KEY = "profile"


class ExtractionFailed(LLMError):
    """Kept as a name for backward compatibility; identical to LLMError."""


def _paper_header(paper) -> str:
    return (f"TITLE: {paper['title']}\n"
            f"JOURNAL: {paper['journal']}  YEAR: {paper['year']}\n"
            f"DOI: {paper['doi']}\n")


def _cached_call(cache_path: Path, refresh: bool, do_call) -> dict:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = do_call()
    atomic_write_text(cache_path, json.dumps(result, ensure_ascii=False,
                                             indent=1))
    return result


def triage_paper(backend: Backend, cfg: Config, profile: profiles.Profile,
                 paper, text: str, refresh: bool = False) -> TriageResult:
    cache = cfg.raw_llm_dir / f"{paper['id']}.triage.json"
    raw = _cached_call(cache, refresh, lambda: backend.call_tool(
        cfg.active_triage_model, profile.triage_prompt,
        _paper_header(paper) + "\nTEXT (truncated):\n" + text[:cfg.triage_chars],
        profile.triage_tool(), max_tokens=300,
    ))
    return TriageResult.model_validate(raw)


def extract_paper(backend: Backend, cfg: Config, profile: profiles.Profile,
                  paper, text: str, refresh: bool = False) -> list[dict]:
    """Return the validated records for one paper (may be empty)."""
    cache = cfg.raw_llm_dir / f"{paper['id']}.json"
    body = text
    if len(body) > cfg.max_paper_chars:
        # keep the head (front matter + experimental usually) and the tail
        head = int(cfg.max_paper_chars * 0.8)
        tail = cfg.max_paper_chars - head
        body = body[:head] + "\n[...TRUNCATED...]\n" + body[-tail:]
    raw = _cached_call(cache, refresh, lambda: backend.call_tool(
        cfg.active_extraction_model, profile.extraction_prompt,
        _paper_header(paper) + "\nFULL TEXT:\n" + body,
        profile.extraction_tool(), max_tokens=cfg.max_output_tokens,
    ))
    result = ExtractionResult.model_validate(raw)
    records = []
    for i, rec in enumerate(result.records, 1):
        try:
            records.append(
                apply_derivations(apply_backfills(profile.validate_record(rec))))
        except profiles.ProfileError as exc:
            raise ExtractionFailed(f"record {i}: {exc}") from exc
    return records


def _check_profile_lock(conn, profile_name: str) -> None:
    locked = db.get_meta(conn, PROFILE_META_KEY)
    if locked is None:
        db.set_meta(conn, PROFILE_META_KEY, profile_name)
    elif locked != profile_name:
        raise ExtractionFailed(
            f"this database was built with profile {locked!r} but the "
            f"config now says {profile_name!r}. One project holds one "
            "profile's data. Either switch the profile back, or start a new "
            "project folder (new data_dir) for the new profile."
        )


def run_extract(conn, cfg: Config, limit: int | None = None,
                refresh: bool = False, only: str | None = None) -> dict:
    cfg.ensure_dirs()
    profile = profiles.load(cfg.profile, cfg.base_dir)
    _check_profile_lock(conn, profile.name)
    backend = get_backend(cfg)

    if only:
        papers = db.papers_where(conn, "id = ?", (only,))
        if not papers:
            raise ExtractionFailed(f"paper id {only!r} not found")
    else:
        papers = db.papers_where(
            conn,
            "download_status = 'downloaded' AND "
            "extract_status IN ('pending', 'failed')",
            limit=limit,
        )

    stats = {"done": 0, "no_recipes": 0, "triaged_out": 0, "failed": 0,
             "text_failed": 0, "ocr_deferred": 0, "recipes": 0}
    workers = max(1, int(getattr(cfg, "extract_workers", 1) or 1))
    ocr = ("defer" if getattr(cfg, "ocr_mode", "inline") == "deferred"
           else "inline")
    log.info("Extracting %s from %d paper(s) with profile=%s provider=%s "
             "model=%s (triage: %s%s)...", profile.record_noun, len(papers),
             profile.name, cfg.provider, cfg.active_extraction_model,
             cfg.active_triage_model if cfg.triage_enabled else "off",
             f", {workers} parallel calls" if workers > 1 else "")

    interrupted = False
    if workers > 1 and len(papers) > 1:
        interrupted = _extract_parallel(conn, cfg, profile, backend, papers,
                                        stats, refresh, workers, ocr)
    else:
        for i, paper in enumerate(papers, 1):
            try:
                _process_one(conn, cfg, profile, backend, paper, i,
                             len(papers), stats, refresh, ocr)
            except KeyboardInterrupt:
                log.warning("Interrupted -- all completed papers are saved. "
                            "Re-run `alpminer extract` to continue.")
                interrupted = True
                break
            except QuotaExhausted:
                log.error("Provider quota exhausted -- stopping this run "
                          "early instead of retrying every remaining paper. "
                          "Completed papers are saved; the rest stay pending.")
                interrupted = True
                break

    # Deferred OCR phase: every text-layer paper is done; now OCR the flagged
    # ones and run them through the same triage + extraction.
    if ocr == "defer" and not interrupted:
        flagged = db.papers_where(
            conn, "text_status = 'ocr_pending' AND "
                  "extract_status IN ('pending', 'failed')")
        if flagged:
            log.info("OCR phase: %d flagged paper(s) -- OCR is slow "
                     "(~1-3 s/page), Ctrl-C between papers is safe...",
                     len(flagged))
            for i, paper in enumerate(flagged, 1):
                try:
                    _process_one(conn, cfg, profile, backend, paper, i,
                                 len(flagged), stats, refresh, ocr="force")
                except KeyboardInterrupt:
                    log.warning("Interrupted during the OCR phase -- "
                                "completed papers are saved; the rest stay "
                                "flagged (text: ocr_pending).")
                    break
        # report what is STILL flagged (an uninterrupted phase clears it)
        stats["ocr_deferred"] = len(db.papers_where(
            conn, "text_status = 'ocr_pending'"))

    log.info("Extraction pass finished: %(done)d with records "
             "(%(recipes)d records), %(no_recipes)d without, "
             "%(triaged_out)d triaged out, %(failed)d failed, "
             "%(text_failed)d unreadable PDFs, %(ocr_deferred)d left "
             "flagged for OCR.", stats)
    _record_usage(conn, backend, stats)
    return stats


def _process_one(conn, cfg, profile, backend, paper, i, total, stats,
                 refresh, ocr) -> None:
    """Process one paper end to end (text -> triage -> extraction -> commit).
    Handles its own failures so the batch survives; KeyboardInterrupt
    propagates to the caller. Shared by the serial loop and the deferred
    OCR phase."""
    pid = paper["id"]
    try:
        text = ensure_text(conn, cfg, paper, ocr=ocr)
        if text is None:
            row = db.get_paper(conn, pid)
            if row and row["text_status"] == "ocr_pending":
                stats["ocr_deferred"] += 1
            else:
                stats["text_failed"] += 1
            return

        if cfg.triage_enabled:
            triage = triage_paper(backend, cfg, profile, paper, text, refresh)
            if not triage.relevant:
                db.set_fields(conn, pid, extract_status="triaged_out",
                              extract_error=None)
                stats["triaged_out"] += 1
                log.info("[%d/%d] %s triaged out (%s)", i, total, pid,
                         (triage.reason or "")[:80])
                return

        records = extract_paper(backend, cfg, profile, paper, text, refresh)
        db.replace_recipes(conn, pid, records)
        if records:
            db.set_fields(conn, pid, extract_status="done",
                          extract_error=None)
            stats["done"] += 1
            stats["recipes"] += len(records)
            log.info("[%d/%d] %s -> %d %s", i, total, pid,
                     len(records), profile.record_noun)
        else:
            db.set_fields(conn, pid, extract_status="no_recipes",
                          extract_error=None)
            stats["no_recipes"] += 1
            log.info("[%d/%d] %s -> no qualifying %s", i, total, pid,
                     profile.record_noun)

    except KeyboardInterrupt:
        raise
    except QuotaExhausted as exc:
        # record this paper, then let the caller abort the whole pass:
        # a daily cap cannot be waited out inside the run
        db.set_fields(conn, pid, extract_status="failed",
                      extract_error=str(exc)[:500])
        stats["failed"] += 1
        log.error("[%d/%d] %s failed: %s", i, total, pid, str(exc)[:200])
        raise
    except (LLMError, ValidationError) as exc:
        db.set_fields(conn, pid, extract_status="failed",
                      extract_error=str(exc)[:500])
        stats["failed"] += 1
        log.error("[%d/%d] %s failed: %s", i, total, pid, str(exc)[:200])
    except Exception as exc:  # noqa: BLE001 - keep the batch alive
        db.set_fields(conn, pid, extract_status="failed",
                      extract_error=f"{type(exc).__name__}: {exc}"[:500])
        stats["failed"] += 1
        log.error("[%d/%d] %s failed: %s: %s", i, total, pid,
                  type(exc).__name__, str(exc)[:200])


def _llm_work(backend, cfg, profile, paper, text, refresh):
    """The network-only portion of one paper: triage + extraction. Runs in a
    worker thread; touches no database connection (SQLite connections are
    bound to the thread that opened them). Returns ("triaged_out", reason)
    or ("done", records); exceptions propagate through the future."""
    if cfg.triage_enabled:
        triage = triage_paper(backend, cfg, profile, paper, text, refresh)
        if not triage.relevant:
            return ("triaged_out", triage.reason)
    return ("done", extract_paper(backend, cfg, profile, paper, text, refresh))


def _extract_parallel(conn, cfg, profile, backend, papers, stats,
                      refresh, workers, ocr="inline") -> bool:
    """Run the LLM calls for several papers concurrently. Returns True when
    interrupted (so the caller skips the deferred OCR phase).

    Only the provider calls parallelize; text preparation and every database
    write stay on this (main) thread, so per-paper commit semantics, failure
    isolation, and Ctrl-C behavior match the serial path. In-flight work is
    capped at workers*2 so texts are prepared lazily, not all up front.
    """
    import concurrent.futures as cf

    total = len(papers)
    queue = iter(enumerate(papers, 1))
    futures: dict = {}

    def submit_next(pool) -> bool:
        for i, paper in queue:
            text = ensure_text(conn, cfg, paper, ocr=ocr)
            if text is None:
                row = db.get_paper(conn, paper["id"])
                if row and row["text_status"] == "ocr_pending":
                    stats["ocr_deferred"] += 1
                else:
                    stats["text_failed"] += 1
                continue
            fut = pool.submit(_llm_work, backend, cfg, profile, paper,
                              text, refresh)
            futures[fut] = (i, paper)
            return True
        return False

    def apply_result(i, paper, fut) -> None:
        pid = paper["id"]
        try:
            kind, payload = fut.result()
            if kind == "triaged_out":
                db.set_fields(conn, pid, extract_status="triaged_out",
                              extract_error=None)
                stats["triaged_out"] += 1
                log.info("[%d/%d] %s triaged out (%s)", i, total, pid,
                         (payload or "")[:80])
            elif payload:
                db.replace_recipes(conn, pid, payload)
                db.set_fields(conn, pid, extract_status="done",
                              extract_error=None)
                stats["done"] += 1
                stats["recipes"] += len(payload)
                log.info("[%d/%d] %s -> %d %s", i, total, pid,
                         len(payload), profile.record_noun)
            else:
                db.replace_recipes(conn, pid, [])
                db.set_fields(conn, pid, extract_status="no_recipes",
                              extract_error=None)
                stats["no_recipes"] += 1
                log.info("[%d/%d] %s -> no qualifying %s", i, total, pid,
                         profile.record_noun)
        except QuotaExhausted as exc:
            # record this paper, then abort the whole pass (see caller)
            db.set_fields(conn, pid, extract_status="failed",
                          extract_error=str(exc)[:500])
            stats["failed"] += 1
            log.error("[%d/%d] %s failed: %s", i, total, pid, str(exc)[:200])
            raise
        except (LLMError, ValidationError) as exc:
            db.set_fields(conn, pid, extract_status="failed",
                          extract_error=str(exc)[:500])
            stats["failed"] += 1
            log.error("[%d/%d] %s failed: %s", i, total, pid,
                      str(exc)[:200])
        except Exception as exc:  # noqa: BLE001 - keep the batch alive
            db.set_fields(conn, pid, extract_status="failed",
                          extract_error=f"{type(exc).__name__}: {exc}"[:500])
            stats["failed"] += 1
            log.error("[%d/%d] %s failed: %s: %s", i, total, pid,
                      type(exc).__name__, str(exc)[:200])

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for _ in range(workers * 2):        # prime the window
                if not submit_next(pool):
                    break
            while futures:
                done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    i, paper = futures.pop(fut)
                    apply_result(i, paper, fut)
                    submit_next(pool)
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            log.warning("Interrupted -- all completed papers are saved; "
                        "in-flight calls were cancelled. Re-run "
                        "`alpminer extract` to continue.")
            return True
        except QuotaExhausted:
            pool.shutdown(wait=False, cancel_futures=True)
            log.error("Provider quota exhausted -- stopping this run early "
                      "instead of retrying every remaining paper. Completed "
                      "papers are saved; the rest stay pending.")
            return True
    return False


def _record_usage(conn, backend, stats: dict) -> None:
    """Log this run's real token spend (as reported by the provider) and add
    it to the cumulative per-project counters in the meta table. Cached
    responses never reach the backend, so cache hits cost -- and count --
    nothing."""
    usage = getattr(backend, "usage", None)
    if not usage or not usage.get("calls"):
        return
    stats["input_tokens"] = usage["input"]
    stats["output_tokens"] = usage["output"]
    stats["llm_calls"] = usage["calls"]
    total_in = db.add_meta_int(conn, "usage_input_tokens", usage["input"])
    total_out = db.add_meta_int(conn, "usage_output_tokens", usage["output"])
    db.add_meta_int(conn, "usage_llm_calls", usage["calls"])
    log.info("LLM spend this run: %s in / %s out tokens over %d call(s) "
             "(project total: %s in / %s out).",
             f"{usage['input']:,}", f"{usage['output']:,}", usage["calls"],
             f"{total_in:,}", f"{total_out:,}")
