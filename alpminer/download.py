"""Download legally available open-access PDFs.

Sources tried in order for each paper:
  1. the OpenAlex best_oa_location pdf_url captured at harvest time,
  2. Unpaywall's best OA location(s) for the DOI.

Anything without a working OA copy is marked `manual` and lands in the manual
queue: the user downloads it via institutional access and drops it into
data/manual_inbox/<paper_id>.pdf (see manual.py). The tool never attempts to
bypass paywalls.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import requests

from . import __version__, db
from .config import Config
from .utils import log, sha256_file, with_retries, RetryError

UNPAYWALL = "https://api.unpaywall.org/v2/{doi}"
CHUNK = 1 << 16


class NotAPdf(RuntimeError):
    pass


def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": f"alpminer/{__version__} (mailto:{cfg.email})",
        "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    })
    return s


def unpaywall_pdf_urls(session: requests.Session, doi: str, email: str,
                       timeout: float) -> list[str]:
    """Return candidate OA pdf URLs from Unpaywall for a DOI (may be empty)."""
    def _get():
        r = session.get(UNPAYWALL.format(doi=doi), params={"email": email},
                        timeout=(10, timeout))
        if r.status_code == 404:
            return None
        if r.status_code == 429 or r.status_code >= 500:
            raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
        r.raise_for_status()
        return r.json()

    try:
        payload = with_retries(_get, desc=f"Unpaywall lookup {doi}",
                               retry_on=(requests.RequestException,))
    except RetryError as exc:
        log.debug("Unpaywall lookup failed for %s: %s", doi, exc)
        return []
    if not payload:
        return []

    urls: list[str] = []
    best = payload.get("best_oa_location") or {}
    if best.get("url_for_pdf"):
        urls.append(best["url_for_pdf"])
    for loc in payload.get("oa_locations") or []:
        u = loc.get("url_for_pdf")
        if u and u not in urls:
            urls.append(u)
    return urls[:4]


def fetch_pdf(session: requests.Session, url: str, dest: Path, cfg: Config) -> None:
    """Stream url to dest atomically; raise NotAPdf if it is not a real PDF."""
    max_bytes = cfg.max_pdf_mb * (1 << 20)
    with session.get(url, stream=True, timeout=(10, cfg.download_timeout_s),
                     allow_redirects=True) as r:
        if r.status_code == 429 or r.status_code >= 500:
            raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
        r.raise_for_status()
        it = r.iter_content(chunk_size=CHUNK)
        first = next(it, b"")
        if b"%PDF" not in first[:1024]:
            raise NotAPdf(f"content at {url} is not a PDF "
                          f"(starts with {first[:16]!r})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=".dl_", suffix=".pdf")
        total = 0
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(first)
                total += len(first)
                for chunk in it:
                    total += len(chunk)
                    if total > max_bytes:
                        raise NotAPdf(f"file exceeds max_pdf_mb={cfg.max_pdf_mb}")
                    f.write(chunk)
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def candidate_urls(session: requests.Session, paper, cfg: Config) -> list[str]:
    urls: list[str] = []
    if paper["oa_pdf_url"]:
        urls.append(paper["oa_pdf_url"])
    if paper["doi"]:
        for u in unpaywall_pdf_urls(session, paper["doi"], cfg.email,
                                    cfg.download_timeout_s):
            if u not in urls:
                urls.append(u)
    return urls


def download_pending(conn, cfg: Config, limit: int | None = None) -> dict:
    """Try to auto-download every paper with download_status='pending'."""
    cfg.require_email()
    cfg.ensure_dirs()
    session = _session(cfg)
    papers = db.papers_where(conn, "download_status = 'pending'", limit=limit)
    ok = manual = 0
    log.info("Attempting auto-download for %d paper(s)...", len(papers))

    for i, paper in enumerate(papers, 1):
        try:
            urls = candidate_urls(session, paper, cfg)
        except Exception as exc:  # noqa: BLE001 - never kill the batch
            log.warning("[%s] OA lookup error: %s", paper["id"], exc)
            urls = [paper["oa_pdf_url"]] if paper["oa_pdf_url"] else []

        if not urls:
            db.set_fields(conn, paper["id"], download_status="manual",
                          download_error="no open-access copy found")
            manual += 1
        else:
            dest = cfg.pdf_dir / f"{paper['id']}.pdf"
            last_err = "unknown error"
            for url in urls:
                try:
                    with_retries(lambda u=url: fetch_pdf(session, u, dest, cfg),
                                 desc=f"download {paper['id']}",
                                 retry_on=(requests.RequestException,),
                                 give_up_on=(NotAPdf,), attempts=3)
                    db.set_fields(conn, paper["id"],
                                  download_status="downloaded",
                                  download_source=url,
                                  download_error=None,
                                  pdf_path=str(dest),
                                  pdf_sha256=sha256_file(dest))
                    ok += 1
                    log.info("[%d/%d] downloaded %s", i, len(papers), paper["id"])
                    break
                except (NotAPdf, RetryError, requests.RequestException) as exc:
                    last_err = str(exc)
                    log.debug("[%s] candidate failed: %s", paper["id"], exc)
            else:
                db.set_fields(conn, paper["id"], download_status="manual",
                              download_error=last_err[:500])
                manual += 1
                log.info("[%d/%d] %s -> manual queue (%s)",
                         i, len(papers), paper["id"], last_err[:80])
        time.sleep(cfg.request_delay_s)

    log.info("Download pass finished: %d downloaded, %d routed to the manual "
             "queue.", ok, manual)
    return {"downloaded": ok, "manual": manual, "attempted": len(papers)}
