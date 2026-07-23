import fitz

from alpminer import db, pdftext
from alpminer.pdftext import ensure_text, pdf_to_text


def _write_pdf(path, text: str, repeats: int = 1):
    doc = fitz.open()
    for _ in range(repeats):
        page = doc.new_page()
        y = 72
        for line in text.split("\n"):
            for start in range(0, len(line), 80):
                page.insert_text((72, y), line[start:start + 80], fontsize=11)
                y += 14
    doc.save(path)
    doc.close()


def test_pdf_to_text_and_hyphen_join(tmp_path):
    pdf = tmp_path / "a.pdf"
    _write_pdf(pdf, "Atomic layer depo-\nsition of Al2O3 films.")
    text = pdf_to_text(pdf)
    assert "deposition" in text  # hyphen line break re-joined
    assert "Al2O3" in text


def test_ensure_text_success_and_cache(cfg, conn, paper):
    pdf = cfg.pdf_dir / f"{paper['id']}.pdf"
    body = ("Trimethylaluminum and water were pulsed alternately at 200 C. " * 30)
    _write_pdf(pdf, body, repeats=2)
    db.set_fields(conn, paper["id"], download_status="downloaded",
                  pdf_path=str(pdf))
    row = db.get_paper(conn, paper["id"])
    text = ensure_text(conn, cfg, row)
    assert text and "Trimethylaluminum" in text
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "ok"
    # cached path is reused without re-parsing
    assert ensure_text(conn, cfg, row) == text


def test_ensure_text_flags_scanned_pdf(cfg, conn, paper, monkeypatch):
    monkeypatch.setattr(pdftext, "ocr_available", lambda: False)  # no OCR here
    pdf = cfg.pdf_dir / f"{paper['id']}.pdf"
    _write_pdf(pdf, "tiny")  # far below MIN_TEXT_CHARS
    db.set_fields(conn, paper["id"], download_status="downloaded",
                  pdf_path=str(pdf))
    row = db.get_paper(conn, paper["id"])
    assert ensure_text(conn, cfg, row) is None
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "failed"
    assert "scanned" in row["text_error"]


def _scanned_paper(cfg, conn, paper):
    pdf = cfg.pdf_dir / f"{paper['id']}.pdf"
    _write_pdf(pdf, "tiny")                       # no useful text layer
    db.set_fields(conn, paper["id"], download_status="downloaded",
                  pdf_path=str(pdf))
    return db.get_paper(conn, paper["id"])


def test_scanned_pdf_recovered_via_ocr_when_available(cfg, conn, paper,
                                                      monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    body = "OCR recovered: TMA and water pulsed at 200 C. " * 30
    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text", lambda p, **kw: body)
    text = ensure_text(conn, cfg, row)
    assert text and "OCR recovered" in text
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "ok"
    # the OCR'd text is cached like any other
    assert (cfg.text_dir / f"{paper['id']}.txt").exists()


def test_ocr_text_is_flagged_and_text_layer_is_not(cfg, conn, paper,
                                                   monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text",
                        lambda p, **kw: "recovered text " * 100)
    assert ensure_text(conn, cfg, row)
    assert db.get_paper(conn, paper["id"])["text_ocr"] == 1

    # a normal text-layer paper stays unflagged
    pdf2 = cfg.pdf_dir / "W9999.pdf"
    _write_pdf(pdf2, "Trimethylaluminum and water at 200 C. " * 40)
    db.upsert_paper(conn, {"id": "W9999", "doi": "10.1/nn", "title": "n",
                           "year": 2024, "journal": "J", "authors": [],
                           "oa_pdf_url": None, "landing_url": None,
                           "is_oa": 0})
    db.set_fields(conn, "W9999", download_status="downloaded",
                  pdf_path=str(pdf2))
    assert ensure_text(conn, cfg, db.get_paper(conn, "W9999"))
    assert db.get_paper(conn, "W9999")["text_ocr"] == 0


def test_deferred_mode_flags_instead_of_ocring(cfg, conn, paper, monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    called = []
    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text",
                        lambda p, **kw: called.append(p) or "x " * 2000)
    # defer: no OCR yet, paper is flagged for the later phase
    assert ensure_text(conn, cfg, row, ocr="defer") is None
    assert called == []
    assert db.get_paper(conn, paper["id"])["text_status"] == "ocr_pending"
    # force: the deferred phase OCRs it for real
    row = db.get_paper(conn, paper["id"])
    assert ensure_text(conn, cfg, row, ocr="force")
    assert len(called) == 1
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "ok" and row["text_ocr"] == 1


def test_deferred_mode_without_ocr_installed_still_fails_clearly(cfg, conn,
                                                                 paper,
                                                                 monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    monkeypatch.setattr(pdftext, "ocr_available", lambda: False)
    assert ensure_text(conn, cfg, row, ocr="defer") is None
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "failed"        # not flagged: nothing to defer to
    assert "scanned" in row["text_error"]


def test_ocr_respects_config_switch(cfg, conn, paper, monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    cfg.ocr_enabled = False
    called = []
    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text",
                        lambda p, **kw: called.append(p) or "x" * 5000)
    assert ensure_text(conn, cfg, row) is None
    assert called == []                           # OCR never attempted
    assert "scanned" in db.get_paper(conn, paper["id"])["text_error"]


def test_ocr_failure_and_short_ocr_are_flagged(cfg, conn, paper, monkeypatch):
    row = _scanned_paper(cfg, conn, paper)
    monkeypatch.setattr(pdftext, "ocr_available", lambda: True)
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text",
                        lambda p, **kw: "still nothing")   # OCR found no text
    assert ensure_text(conn, cfg, row) is None
    assert db.get_paper(conn, paper["id"])["text_status"] == "failed"

    def boom(p, **kw):
        raise RuntimeError("tesseract exploded")
    monkeypatch.setattr(pdftext, "ocr_pdf_to_text", boom)
    db.set_fields(conn, paper["id"], text_status="pending")
    row = db.get_paper(conn, paper["id"])
    assert ensure_text(conn, cfg, row) is None
    assert "OCR failed" in db.get_paper(conn, paper["id"])["text_error"]


def test_stale_absolute_pdf_path_self_heals(cfg, conn, paper):
    """A moved/renamed project folder strands the absolute pdf_path stored in
    the database; ensure_text must fall back to the canonical location and
    repair the row (regression: the ALPminer folder rename flagged every
    downloaded paper as 'unreadable')."""
    pdf = cfg.pdf_dir / f"{paper['id']}.pdf"
    _write_pdf(pdf, "Trimethylaluminum and water pulsed at 200 C. " * 40)
    db.set_fields(conn, paper["id"], download_status="downloaded",
                  pdf_path=r"C:\some\old\project\path\W1001.pdf",  # stale
                  text_status="failed", text_error="pdf file missing on disk")
    row = db.get_paper(conn, paper["id"])
    text = ensure_text(conn, cfg, row)
    assert text and "Trimethylaluminum" in text
    row = db.get_paper(conn, paper["id"])
    assert row["text_status"] == "ok"
    assert row["pdf_path"] == str(pdf)            # row repaired permanently


def test_ensure_text_missing_file(cfg, conn, paper):
    db.set_fields(conn, paper["id"], download_status="downloaded",
                  pdf_path=str(cfg.pdf_dir / "missing.pdf"))
    row = db.get_paper(conn, paper["id"])
    assert ensure_text(conn, cfg, row) is None
    assert db.get_paper(conn, paper["id"])["text_status"] == "failed"
