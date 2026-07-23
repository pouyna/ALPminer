import requests

from alpminer import db, download
from tests.conftest import make_paper

PDF_BYTES = b"%PDF-1.7 fake pdf body " + b"x" * 2000
HTML_BYTES = b"<!doctype html><html>login wall</html>"


class FakeResponse:
    def __init__(self, content=PDF_BYTES, status=200, json_data=None):
        self._content = content
        self.status_code = status
        self._json = json_data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_content(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class FakeSession:
    """Maps URL substrings to responses; records what was requested."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.headers = {}

    def get(self, url, **kw):
        self.calls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return FakeResponse(status=404)


def _run(monkeypatch, cfg, conn, routes):
    monkeypatch.setattr(download, "_session", lambda c: FakeSession(routes))
    monkeypatch.setattr("time.sleep", lambda s: None)
    return download.download_pending(conn, cfg)


def test_download_success(monkeypatch, cfg, conn, paper):
    stats = _run(monkeypatch, cfg, conn, {"oa.example.org": FakeResponse()})
    assert stats == {"downloaded": 1, "manual": 0, "attempted": 1}
    row = db.get_paper(conn, paper["id"])
    assert row["download_status"] == "downloaded"
    pdf = cfg.pdf_dir / f"{paper['id']}.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert row["pdf_sha256"]


def test_html_login_wall_goes_to_manual_queue(monkeypatch, cfg, conn, paper):
    routes = {"oa.example.org": FakeResponse(content=HTML_BYTES),
              "unpaywall": FakeResponse(status=404)}
    stats = _run(monkeypatch, cfg, conn, routes)
    assert stats["manual"] == 1
    row = db.get_paper(conn, paper["id"])
    assert row["download_status"] == "manual"
    assert "not a PDF" in row["download_error"]


def test_no_oa_url_falls_back_to_unpaywall(monkeypatch, cfg, conn):
    rec = make_paper(3, oa_pdf_url=None)
    db.upsert_paper(conn, rec)
    unpaywall_payload = {
        "best_oa_location": {"url_for_pdf": "https://repo.example.org/x.pdf"},
        "oa_locations": [],
    }
    routes = {
        "unpaywall": FakeResponse(json_data=unpaywall_payload),
        "repo.example.org": FakeResponse(),
    }
    stats = _run(monkeypatch, cfg, conn, routes)
    assert stats["downloaded"] == 1
    assert db.get_paper(conn, rec["id"])["download_source"].endswith("x.pdf")


def test_nothing_available_marks_manual(monkeypatch, cfg, conn):
    rec = make_paper(4, oa_pdf_url=None)
    db.upsert_paper(conn, rec)
    routes = {"unpaywall": FakeResponse(status=404)}
    stats = _run(monkeypatch, cfg, conn, routes)
    assert stats["manual"] == 1
    row = db.get_paper(conn, rec["id"])
    assert row["download_error"] == "no open-access copy found"


def test_resume_skips_completed_downloads(monkeypatch, cfg, conn, paper):
    _run(monkeypatch, cfg, conn, {"oa.example.org": FakeResponse()})
    stats = _run(monkeypatch, cfg, conn, {"oa.example.org": FakeResponse()})
    assert stats["attempted"] == 0  # nothing pending on the second run
