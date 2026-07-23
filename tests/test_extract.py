import json

from alpminer import db, extract
from alpminer.providers import Backend, LLMError
from tests.conftest import make_paper

RECIPE_PAYLOAD = {
    "reports_own_ald_experiment": True,
    "recipes": [{
        "material": "Al2O3",
        "technique": "thermal ALD",
        "metal_precursor_abbrev": "TMA",
        "co_reactant": "water",
        "deposition_temperature_c": 200,
        "gpc_as_reported": "0.11 nm/cycle",
        "gpc_angstrom_per_cycle": None,
        "confidence": 0.9,
    }],
    "paper_notes": None,
}


class FakeBackend(Backend):
    """Implements the same call_tool(model, system, user_text, tool,
    max_tokens) interface real backends use, so extract.py is exercised
    exactly as it would be against Anthropic or Gemini."""

    name = "fake"

    def __init__(self, triage_answer=True, payload=RECIPE_PAYLOAD,
                fail_extraction_for=()):
        self.triage_answer = triage_answer
        self.payload = payload
        self.fail_extraction_for = fail_extraction_for
        self.calls = []

    def call_tool(self, model, system, user_text, tool, max_tokens):
        self.calls.append(tool["name"])
        self.record_usage(1000, 100)      # like a real provider response
        if tool["name"] == "triage_result":
            return {"reports_own_ald_experiment": self.triage_answer,
                    "reason": "fake"}
        assert tool["name"] == "record_findings"
        for pid in self.fail_extraction_for:
            if pid in user_text:
                raise RuntimeError(f"simulated API blowup for {pid}")
        return self.payload


def _ready_paper(cfg, conn, i=1):
    rec = make_paper(i, doi=f"10.1/e{i}")
    db.upsert_paper(conn, rec)
    text = f"Paper {rec['id']}: TMA and H2O ALD at 200 C. " * 40
    (cfg.text_dir / f"{rec['id']}.txt").write_text(text)
    db.set_fields(conn, rec["id"], download_status="downloaded",
                  pdf_path="/fake.pdf", text_status="ok",
                  text_path=str(cfg.text_dir / f"{rec['id']}.txt"))
    return rec


def _patch(monkeypatch, backend):
    monkeypatch.setattr(extract, "get_backend", lambda cfg: backend)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_happy_path_with_triage_and_gpc_backfill(monkeypatch, cfg, conn):
    rec = _ready_paper(cfg, conn, 1)
    backend = FakeBackend()
    _patch(monkeypatch, backend)

    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 1 and stats["recipes"] == 1
    assert backend.calls == ["triage_result", "record_findings"]

    row = db.get_paper(conn, rec["id"])
    assert row["extract_status"] == "done"
    stored = db.recipes_for(conn, rec["id"])[0]
    assert stored["material"] == "Al2O3"
    assert stored["gpc_angstrom_per_cycle"] == 1.1  # backfilled from nm/cycle
    # raw responses were cached to disk before parsing
    assert (cfg.raw_llm_dir / f"{rec['id']}.json").exists()

    # real token spend was recorded: 2 calls (triage + extract) x 1000/100
    assert stats["input_tokens"] == 2000 and stats["output_tokens"] == 200
    assert stats["llm_calls"] == 2
    assert db.get_meta(conn, "usage_input_tokens") == "2000"

    # a re-run hits the response cache: no new calls, no new spend
    backend2 = FakeBackend()
    _patch(monkeypatch, backend2)
    stats2 = extract.run_extract(conn, cfg, only=rec["id"])
    assert backend2.calls == []                      # cache, not the API
    assert "input_tokens" not in stats2              # nothing spent
    assert db.get_meta(conn, "usage_input_tokens") == "2000"  # unchanged
    assert (cfg.raw_llm_dir / f"{rec['id']}.triage.json").exists()


def test_cached_responses_avoid_repeat_spend(monkeypatch, cfg, conn):
    rec = _ready_paper(cfg, conn, 2)
    _patch(monkeypatch, FakeBackend())
    extract.run_extract(conn, cfg)

    fresh = FakeBackend()
    _patch(monkeypatch, fresh)
    extract.run_extract(conn, cfg, only=rec["id"])  # re-process explicitly
    assert fresh.calls == []  # everything served from the raw_llm cache
    assert db.get_paper(conn, rec["id"])["extract_status"] == "done"


def test_triaged_out_skips_extraction(monkeypatch, cfg, conn):
    rec = _ready_paper(cfg, conn, 3)
    backend = FakeBackend(triage_answer=False)
    _patch(monkeypatch, backend)
    stats = extract.run_extract(conn, cfg)
    assert stats["triaged_out"] == 1
    assert backend.calls == ["triage_result"]
    assert db.get_paper(conn, rec["id"])["extract_status"] == "triaged_out"


def test_failure_is_recorded_and_retried_on_next_run(monkeypatch, cfg, conn):
    ok = _ready_paper(cfg, conn, 4)
    bad = _ready_paper(cfg, conn, 5)
    _patch(monkeypatch, FakeBackend(fail_extraction_for=(bad["id"],)))
    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 1 and stats["failed"] == 1
    assert db.get_paper(conn, bad["id"])["extract_status"] == "failed"
    assert "simulated" in db.get_paper(conn, bad["id"])["extract_error"]

    # next run picks up only the failed paper and succeeds
    healed = FakeBackend()
    _patch(monkeypatch, healed)
    stats2 = extract.run_extract(conn, cfg)
    assert stats2["done"] == 1 and stats2["failed"] == 0
    assert db.get_paper(conn, bad["id"])["extract_status"] == "done"
    assert db.get_paper(conn, ok["id"])["extract_status"] == "done"


def test_invalid_llm_output_marks_failed(monkeypatch, cfg, conn):
    rec = _ready_paper(cfg, conn, 6)
    payload = {"reports_own_ald_experiment": True,
               "recipes": [{"technique": "no material given"}]}
    _patch(monkeypatch, FakeBackend(payload=payload))
    stats = extract.run_extract(conn, cfg)
    assert stats["failed"] == 1
    assert db.get_paper(conn, rec["id"])["extract_status"] == "failed"


def test_parallel_extraction_matches_serial_semantics(monkeypatch, cfg, conn):
    """extract_workers=3: everything completes, failures stay isolated, and
    token spend is fully accounted (record_usage is thread-safe)."""
    papers = [_ready_paper(cfg, conn, 20 + i) for i in range(6)]
    bad = papers[2]
    backend = FakeBackend(fail_extraction_for=(bad["id"],))
    _patch(monkeypatch, backend)
    cfg.extract_workers = 3

    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 5 and stats["failed"] == 1
    assert stats["recipes"] == 5
    for p in papers:
        expected = "failed" if p["id"] == bad["id"] else "done"
        assert db.get_paper(conn, p["id"])["extract_status"] == expected
    # 6 triage + 6 extraction attempts, 1000/100 tokens each, none lost
    assert backend.usage == {"input": 12000, "output": 1200, "calls": 12}
    assert stats["input_tokens"] == 12000
    # and results are re-runnable from cache exactly like the serial path
    healed = FakeBackend()
    _patch(monkeypatch, healed)
    stats2 = extract.run_extract(conn, cfg)
    assert stats2["done"] == 1 and healed.calls == ["record_findings"]


def test_deferred_ocr_phase_runs_after_text_papers(monkeypatch, cfg, conn):
    """ocr_mode='deferred': the scanned paper is flagged, every text-layer
    paper is extracted first, then the OCR phase handles the flagged one in
    the same run -- and the result carries text_ocr=1 for the exports."""
    import fitz

    from alpminer import pdftext

    normal = _ready_paper(cfg, conn, 40)
    scanned = make_paper(41, doi="10.1/e41")     # title: "... part 41"
    db.upsert_paper(conn, scanned)
    pdf = cfg.pdf_dir / f"{scanned['id']}.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "tiny")  # no useful text layer
    doc.save(pdf)
    doc.close()
    db.set_fields(conn, scanned["id"], download_status="downloaded",
                  pdf_path=str(pdf))

    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text",
                        lambda p, **kw: "OCR: TMA and water at 200 C. " * 200)

    class OrderBackend(FakeBackend):
        def __init__(self):
            super().__init__()
            self.seen = []

        def call_tool(self, model, system, user_text, tool, max_tokens):
            for tag in ("part 40", "part 41"):
                if tag in user_text:
                    self.seen.append(tag)
            return super().call_tool(model, system, user_text, tool,
                                     max_tokens)

    backend = OrderBackend()
    _patch(monkeypatch, backend)
    cfg.ocr_mode = "deferred"

    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 2
    assert stats["ocr_deferred"] == 0            # the phase cleared the flag
    row = db.get_paper(conn, scanned["id"])
    assert row["extract_status"] == "done"
    assert row["text_status"] == "ok" and row["text_ocr"] == 1
    assert db.get_paper(conn, normal["id"])["text_ocr"] == 0
    # every call for the text-layer paper happened BEFORE any OCR-paper call
    assert backend.seen == ["part 40", "part 40", "part 41", "part 41"]


def test_quota_exhaustion_aborts_the_pass(monkeypatch, cfg, conn):
    """A daily-quota QuotaExhausted must stop the whole run: the failing
    paper is recorded, the REST stay pending (not churned through futile
    retries), and the run ends cleanly."""
    from alpminer.providers import QuotaExhausted

    papers = [_ready_paper(cfg, conn, 50 + i) for i in range(4)]

    class QuotaBackend(FakeBackend):
        def call_tool(self, model, system, user_text, tool, max_tokens):
            if papers[1]["id"] in user_text:      # second paper hits the wall
                raise QuotaExhausted("Gemini daily request quota exhausted")
            return super().call_tool(model, system, user_text, tool,
                                     max_tokens)

    _patch(monkeypatch, QuotaBackend())
    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 1 and stats["failed"] == 1
    assert db.get_paper(conn, papers[0]["id"])["extract_status"] == "done"
    assert db.get_paper(conn, papers[1]["id"])["extract_status"] == "failed"
    assert "quota" in db.get_paper(conn, papers[1]["id"])["extract_error"]
    for p in papers[2:]:                          # untouched, ready for later
        assert db.get_paper(conn, p["id"])["extract_status"] == "pending"


def test_quota_exhaustion_aborts_parallel_pass_too(monkeypatch, cfg, conn):
    from alpminer.providers import QuotaExhausted

    papers = [_ready_paper(cfg, conn, 60 + i) for i in range(6)]

    class QuotaBackend(FakeBackend):
        def call_tool(self, model, system, user_text, tool, max_tokens):
            raise QuotaExhausted("daily quota exhausted")

    _patch(monkeypatch, QuotaBackend())
    cfg.extract_workers = 3
    stats = extract.run_extract(conn, cfg)        # must return, not hang
    assert stats["failed"] >= 1
    statuses = [db.get_paper(conn, p["id"])["extract_status"] for p in papers]
    assert "pending" in statuses                  # the pass was cut short


def test_no_recipes_status(monkeypatch, cfg, conn):
    rec = _ready_paper(cfg, conn, 7)
    payload = {"reports_own_ald_experiment": False, "recipes": []}
    _patch(monkeypatch, FakeBackend(triage_answer=True, payload=payload))
    stats = extract.run_extract(conn, cfg)
    assert stats["no_recipes"] == 1
    assert db.get_paper(conn, rec["id"])["extract_status"] == "no_recipes"
    assert json.loads(
        (cfg.raw_llm_dir / f"{rec['id']}.json").read_text()
    )["recipes"] == []


def test_missing_api_key_raises_llm_error_before_any_paper_runs(cfg, conn,
                                                                 monkeypatch):
    _ready_paper(cfg, conn, 8)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    try:
        extract.run_extract(conn, cfg)
        assert False, "expected LLMError"
    except LLMError as exc:
        assert "ANTHROPIC_API_KEY" in str(exc)


def test_gemini_provider_selected_and_missing_key_reports_gemini(cfg, conn,
                                                                  monkeypatch):
    cfg.provider = "gemini"
    _ready_paper(cfg, conn, 9)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    try:
        extract.run_extract(conn, cfg)
        assert False, "expected LLMError"
    except LLMError as exc:
        assert "GEMINI_API_KEY" in str(exc)


# ---- profile lock + non-ALD profile (v2) -----------------------------------------

def test_profile_lock_refuses_mixed_databases(monkeypatch, cfg, conn):
    _ready_paper(cfg, conn, 10)
    _patch(monkeypatch, FakeBackend())
    extract.run_extract(conn, cfg)                    # locks db to 'ald'
    cfg.profile = "ale"
    try:
        extract.run_extract(conn, cfg)
        assert False, "expected profile-lock error"
    except LLMError as exc:
        assert "'ald'" in str(exc) and "'ale'" in str(exc)


def test_extract_with_ale_profile_and_epc_backfill(monkeypatch, cfg, conn):
    cfg.profile = "ale"
    rec = _ready_paper(cfg, conn, 11)
    payload = {"relevant": True, "records": [{
        "material": "Al2O3", "ale_type": "thermal ALE",
        "modification_reactant": "HF", "removal_reactant": "TMA",
        "epc_as_reported": "0.61 A/cycle at 300 C",
        "epc_angstrom_per_cycle": None}]}

    class AleBackend(FakeBackend):
        def call_tool(self, model, system, user_text, tool, max_tokens):
            self.calls.append(tool["name"])
            if tool["name"] == "triage_result":
                assert "atomic layer etching" in system
                return {"relevant": True}
            assert "modification_reactant" in str(tool["input_schema"])
            return payload

    _patch(monkeypatch, AleBackend())
    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 1
    stored = db.recipes_for(conn, rec["id"])[0]
    assert stored["removal_reactant"] == "TMA"
    assert stored["epc_angstrom_per_cycle"] == 0.61
    assert "gpc_angstrom_per_cycle" not in stored     # ALD fields absent


def test_legacy_cached_response_reparses_under_v2(monkeypatch, cfg, conn):
    """A raw cache written by alpminer 1.x (old wrapper keys) must extract
    without a single new LLM call."""
    rec = _ready_paper(cfg, conn, 12)
    legacy = {"reports_own_ald_experiment": True,
              "recipes": [{"material": "SnO2", "co_reactant": "H2O2"}],
              "paper_notes": "from-1.x"}
    (cfg.raw_llm_dir / f"{rec['id']}.json").write_text(json.dumps(legacy))
    (cfg.raw_llm_dir / f"{rec['id']}.triage.json").write_text(
        json.dumps({"reports_own_ald_experiment": True, "reason": "old"}))
    silent = FakeBackend()
    _patch(monkeypatch, silent)
    stats = extract.run_extract(conn, cfg)
    assert stats["done"] == 1 and silent.calls == []
    assert db.recipes_for(conn, rec["id"])[0]["material"] == "SnO2"
