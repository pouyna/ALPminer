from pathlib import Path

import pytest

from alpminer import db, manual
from tests.conftest import make_paper


def _make_manual_paper(conn, i=1, **over):
    rec = make_paper(i, **over)
    db.upsert_paper(conn, rec)
    db.set_fields(conn, rec["id"], download_status="manual",
                  download_error="no open-access copy found")
    return rec


def test_export_queue_writes_csv_and_html(cfg, conn):
    rec = _make_manual_paper(conn, 1)
    info = manual.export_queue(conn, cfg)
    assert info["count"] == 1
    csv_text = info["csv"].read_text()
    assert f"{rec['id']}.pdf" in csv_text
    assert "10.1000/test.1" in csv_text
    html_text = info["html"].read_text()
    assert "https://doi.org/10.1000/test.1" in html_text
    assert rec["title"] in html_text


def test_ingest_matches_case_insensitively_and_validates(cfg, conn):
    rec = _make_manual_paper(conn, 2)
    good = cfg.inbox_dir / f"{rec['id'].lower()}.PDF"  # wrong case on purpose
    good.write_bytes(b"%PDF-1.5 hand-downloaded article body")
    (cfg.inbox_dir / "not_matching.pdf").write_bytes(b"%PDF-1.5 orphan")
    (cfg.inbox_dir / f"{rec['id']}.txt").write_text("wrong extension")

    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == [rec["id"]]
    assert len(result["unmatched"]) == 2

    row = db.get_paper(conn, rec["id"])
    assert row["download_status"] == "downloaded"
    assert row["download_source"] == "manual"
    assert (cfg.pdf_dir / f"{rec['id']}.pdf").exists()
    assert not good.exists()  # moved out of the inbox


def test_ingest_rejects_fake_pdf(cfg, conn):
    rec = _make_manual_paper(conn, 3)
    bad = cfg.inbox_dir / f"{rec['id']}.pdf"
    bad.write_bytes(b"<html>this is a saved login page</html>")
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == []
    assert len(result["bad"]) == 1
    assert db.get_paper(conn, rec["id"])["download_status"] == "manual"
    assert bad.exists()  # left in place for the user to inspect


# ---- content-based matching (v1.1) -----------------------------------------

import fitz

from alpminer import manual as manual_mod


def _write_real_pdf(path, lines):
    doc = fitz.open()
    page = doc.new_page()
    y = 72
    for line in lines:
        for start in range(0, len(line), 80):
            page.insert_text((72, y), line[start:start + 80], fontsize=10)
            y += 13
    doc.save(path)
    doc.close()


def test_ingest_matches_by_doi_with_publisher_filename(cfg, conn):
    rec = _make_manual_paper(conn, 10)
    _write_real_pdf(cfg.inbox_dir / "1-s2.0-S0169433225-main.pdf", [
        "Journal of Testing Science",
        f"DOI: {rec['doi']}",
        "Atomic layer deposition of testium oxide films...",
    ])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == [rec["id"]]
    row = db.get_paper(conn, rec["id"])
    assert row["download_status"] == "downloaded"
    assert (cfg.pdf_dir / f"{rec['id']}.pdf").exists()


def test_ingest_matches_by_title_when_no_doi_printed(cfg, conn):
    rec = _make_manual_paper(conn, 11, title="A very long and distinctive "
                             "title about testium oxide growth kinetics")
    _write_real_pdf(cfg.inbox_dir / "download (3).pdf", [
        "A very long and distinctive title",
        "about testium oxide growth kinetics",
        "A. Author and B. Author",
    ])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == [rec["id"]]


def test_ingest_skips_browser_temp_files(cfg, conn):
    _make_manual_paper(conn, 12)
    (cfg.inbox_dir / "still-downloading.pdf.crdownload").write_bytes(b"x")
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == []
    assert result["unmatched"] == []           # not even reported as junk
    assert (cfg.inbox_dir / "still-downloading.pdf.crdownload").exists()


def test_ingest_reports_duplicates_and_leaves_file(cfg, conn):
    rec = _make_manual_paper(conn, 13)
    db.set_fields(conn, rec["id"], download_status="downloaded",
                  pdf_path="/already/there.pdf")
    dup = cfg.inbox_dir / "again.pdf"
    _write_real_pdf(dup, [f"DOI: {rec['doi']}", "Same paper again"])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == []
    assert len(result["duplicate"]) == 1
    assert dup.exists()


def test_open_queue_opens_tabs_in_order(cfg, conn, monkeypatch):
    recs = [_make_manual_paper(conn, 20 + i) for i in range(3)]
    opened = []
    monkeypatch.setattr(manual_mod.webbrowser, "open_new_tab",
                        lambda url: opened.append(url))
    monkeypatch.setattr(manual_mod.time, "sleep", lambda s: None)
    n = manual.open_queue(conn, cfg, n=2)
    assert n == 2
    assert opened == [f"https://doi.org/{recs[0]['doi']}",
                      f"https://doi.org/{recs[1]['doi']}"]


def test_scan_skip_fresh_defers_just_written_files(cfg, conn):
    rec = _make_manual_paper(conn, 30)
    f = cfg.inbox_dir / f"{rec['id']}.pdf"
    f.write_bytes(b"%PDF-1.5 fresh file body")
    fresh = manual_mod._scan(conn, cfg, cfg.inbox_dir, skip_fresh=True)
    assert fresh["matched"] == []              # too new; browser may be writing
    settled = manual_mod._scan(conn, cfg, cfg.inbox_dir, skip_fresh=False)
    assert [pid for pid, _ in settled["matched"]] == [rec["id"]]


# ---- matching freshly harvested (pending) papers -------------------------

def test_queue_includes_pending_papers_manual_listed_first(conn):
    # freshly harvested paper (default download_status='pending')
    db.upsert_paper(conn, make_paper(1))                     # W1001, pending
    m = make_paper(2)                                        # W1002, manual
    db.upsert_paper(conn, m)
    db.set_fields(conn, m["id"], download_status="manual",
                  download_error="no open-access copy found")
    ids = [r["id"] for r in manual.queue(conn)]
    assert set(ids) == {"W1001", "W1002"}                   # both awaiting a PDF
    assert ids[0] == "W1002"                                # manual before pending


def test_ingest_matches_a_pending_paper_by_doi(cfg, conn):
    rec = make_paper(50)
    db.upsert_paper(conn, rec)                               # stays 'pending'
    _write_real_pdf(cfg.inbox_dir / "sciencedirect_main.pdf", [
        "Journal of Testing Science",
        f"https://doi.org/{rec['doi']}",
        "The article body about testium oxide.",
    ])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == [rec["id"]]
    assert db.get_paper(conn, rec["id"])["download_status"] == "downloaded"


def test_queue_count_and_page(conn):
    for i in range(1, 8):                                    # 7 pending papers
        db.upsert_paper(conn, make_paper(i))
    m = make_paper(20)                                       # + 1 manual paper
    db.upsert_paper(conn, m)
    db.set_fields(conn, m["id"], download_status="manual")
    assert manual.queue_count(conn) == 8
    p1 = manual.queue_page(conn, offset=0, limit=3)
    assert len(p1) == 3
    assert p1[0]["id"] == m["id"]                            # manual listed first
    p2 = manual.queue_page(conn, offset=3, limit=3)
    assert len(p2) == 3
    assert {r["id"] for r in p1}.isdisjoint({r["id"] for r in p2})
    assert manual.queue_page(conn, offset=100, limit=3) == []   # past the end


def test_queue_search_filters_title_doi_journal_and_id(conn):
    db.upsert_paper(conn, make_paper(1, title="ALD of hafnium oxide"))
    db.upsert_paper(conn, make_paper(2, title="ALE of gallium nitride"))
    assert manual.queue_count(conn, "hafnium") == 1
    assert manual.queue_page(conn, q="hafnium")[0]["id"] == "W1001"
    assert manual.queue_count(conn, "10.1000/test.2") == 1   # DOI
    assert manual.queue_count(conn, "W1002") == 1            # id
    assert manual.queue_count(conn, "J. Test") == 2          # journal
    assert manual.queue_count(conn, "") == 2                 # blank = all
    assert manual.queue_count(conn, "zzz") == 0


def test_remove_and_restore_queue(conn):
    for i in range(1, 4):
        db.upsert_paper(conn, make_paper(i))                 # W1001-3, pending
    assert manual.queue_count(conn) == 3
    assert manual.remove_from_queue(conn, "W1002") is True
    assert manual.queue_count(conn) == 2                     # out of the queue
    assert manual.removed_count(conn) == 1
    assert [r["id"] for r in manual.removed_list(conn)] == ["W1002"]
    # a paper that already has a PDF can't be removed from the queue
    db.set_fields(conn, "W1001", download_status="downloaded")
    assert manual.remove_from_queue(conn, "W1001") is False
    # restore brings it back as pending, and only works once
    assert manual.restore_to_queue(conn, "W1002") is True
    assert db.get_paper(conn, "W1002")["download_status"] == "pending"
    assert manual.queue_count(conn) == 2
    assert manual.restore_to_queue(conn, "W1002") is False


def test_restore_all(conn):
    for i in range(1, 4):
        db.upsert_paper(conn, make_paper(i))
        manual.remove_from_queue(conn, f"W{1000 + i}")
    assert manual.removed_count(conn) == 3
    assert manual.restore_all(conn) == 3
    assert manual.removed_count(conn) == 0
    assert manual.queue_count(conn) == 3


def test_removed_paper_is_not_matched_by_a_dropped_pdf(cfg, conn):
    rec = make_paper(60, title="Removed distinctive title about testium oxide "
                     "growth kinetics")
    db.upsert_paper(conn, rec)
    assert manual.remove_from_queue(conn, rec["id"]) is True
    _write_real_pdf(cfg.inbox_dir / "drop.pdf", [
        f"DOI: {rec['doi']}",
        "Removed distinctive title about testium oxide growth kinetics"])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == []                          # not resurrected
    assert db.get_paper(conn, rec["id"])["download_status"] == "removed"


def test_ingest_matches_a_pending_paper_by_title_without_doi(cfg, conn):
    rec = make_paper(51, title="A very long and distinctive title about "
                     "testium oxide growth kinetics")
    db.upsert_paper(conn, rec)                               # 'pending', never downloaded
    _write_real_pdf(cfg.inbox_dir / "download (7).pdf", [
        "A very long and distinctive title",
        "about testium oxide growth kinetics",
        "A. Author and B. Author",
    ])
    result = manual.ingest_inbox(conn, cfg)
    assert result["matched"] == [rec["id"]]                 # matched pre-download
    assert db.get_paper(conn, rec["id"])["download_status"] == "downloaded"


# ---- copy-mode watching over shared folders (v2) --------------------------------

def test_scan_copies_from_external_folder_and_leaves_original(cfg, conn,
                                                              tmp_path):
    rec = _make_manual_paper(conn, 40)
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    src = downloads / "publisher_file (1).pdf"
    _write_real_pdf(src, [f"DOI: {rec['doi']}", "The article body."])
    (downloads / "unrelated.pdf").write_bytes(b"%PDF-1.4 someone's tax form")

    seen = set()
    res = manual_mod._scan(conn, cfg, downloads, seen=seen)
    assert [pid for pid, _ in res["matched"]] == [rec["id"]]
    assert src.exists()                              # copy, not move
    assert (cfg.pdf_dir / f"{rec['id']}.pdf").exists()
    assert res["unmatched"] == []                    # shared folder: silent
    assert db.get_paper(conn, rec["id"])["download_status"] == "downloaded"

    # second pass: nothing re-reported, nothing re-filed
    res2 = manual_mod._scan(conn, cfg, downloads, seen=seen)
    assert res2["matched"] == [] and res2["duplicate"] == []


def test_inbox_still_moves_and_reports_junk(cfg, conn):
    rec = _make_manual_paper(conn, 41)
    f = cfg.inbox_dir / f"{rec['id']}.pdf"
    f.write_bytes(b"%PDF-1.5 inbox file")
    (cfg.inbox_dir / "junk.txt").write_text("x")
    res = manual_mod._scan(conn, cfg, cfg.inbox_dir)
    assert not f.exists()                            # moved out of the inbox
    assert len(res["unmatched"]) == 1                # junk.txt reported


def test_watch_dirs_composition(cfg):
    import tempfile
    dirs = manual_mod.watch_dirs(cfg, ["/some/downloads"], include_temp=True)
    assert dirs[0] == cfg.inbox_dir
    assert dirs[1] == Path("/some/downloads")   # compare paths, not OS strings
    assert dirs[-1] == Path(tempfile.gettempdir())
    assert manual_mod.watch_dirs(cfg, None, False) == [cfg.inbox_dir]


def test_scan_missing_folder_is_reported_not_crashed(cfg, conn):
    res = manual_mod._scan(conn, cfg, cfg.root / "nope")
    assert res["unmatched"][0][1] == "folder does not exist"


# ---- external papers (not in OpenAlex) ------------------------------------------

def test_add_external_creates_downloaded_pending_paper(cfg, conn, tmp_path):
    pdf = tmp_path / "my_preprint.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"x" * 400)
    info = manual.add_external(conn, cfg, pdf, title="My Preprint")
    assert info["id"].startswith("ext-") and info["reused"] is False
    row = db.get_paper(conn, info["id"])
    assert row["download_status"] == "downloaded"
    assert row["download_source"] == "external"
    assert row["extract_status"] == "pending"
    assert row["title"] == "My Preprint"
    assert (cfg.pdf_dir / f"{info['id']}.pdf").exists()


def test_add_external_rejects_non_pdf(cfg, conn, tmp_path):
    bad = tmp_path / "notes.txt"
    bad.write_text("just text, not a pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        manual.add_external(conn, cfg, bad)


def test_add_external_missing_file_raises(cfg, conn, tmp_path):
    with pytest.raises(FileNotFoundError):
        manual.add_external(conn, cfg, tmp_path / "ghost.pdf")


def test_add_external_reuses_a_row_with_the_same_doi(cfg, conn, tmp_path):
    db.upsert_paper(conn, make_paper(1, doi="10.9/known"))
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.7\n" + b"y" * 300)
    info = manual.add_external(conn, cfg, pdf, doi="10.9/known")
    assert info["reused"] is True
    assert info["id"] == "W1001"                       # existing row, not ext-
    assert db.get_paper(conn, "W1001")["download_status"] == "downloaded"
