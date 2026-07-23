from alpminer import db, harvest


def test_empty_query_falls_back_to_profile_default(cfg):
    """An empty configured query must resolve to the active profile's
    default query when the OpenAlex filter is built."""
    from alpminer.harvest import build_filter

    cfg.query = ""
    cfg.profile = "ale"
    filt = build_filter(cfg)
    assert '"atomic layer etching" OR "atomic layer etch"' in filt

    cfg.profile = "ald"
    filt = build_filter(cfg)
    assert '"atomic layer deposition" OR "atomic layer epitaxy"' in filt

    cfg.query = '"my custom query"'
    assert '"my custom query"' in build_filter(cfg)


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def json(self):
        return self._payload

    def raise_for_status(self):
        pass


def _fake_openalex(pages):
    """A session that returns each page keyed by the request cursor."""
    class _Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            return _FakeResp(pages["*" if params["cursor"] == "*"
                                   else params["cursor"]])
    return _Session()


def test_max_ingests_whole_page_and_resumes_without_gaps(cfg, conn, monkeypatch):
    """Regression: --max must not checkpoint past a partially consumed page.

    Two pages of three works; with --max 2 the first run still ingests the
    whole first page and checkpoints to the second, so a resume run picks the
    second page up with nothing skipped. On the pre-fix code the first run
    stopped mid-page after checkpointing the next cursor, silently dropping the
    remainder of page one.
    """
    def works(page):
        return [{"id": f"https://openalex.org/W{page}{i}",
                 "doi": f"https://doi.org/10.1/{page}{i}",
                 "display_name": f"P{page}{i}", "publication_year": 2025}
                for i in range(3)]

    pages = {
        "*":  {"meta": {"count": 6, "next_cursor": "c2"}, "results": works(1)},
        "c2": {"meta": {"count": 6, "next_cursor": None}, "results": works(2)},
    }
    monkeypatch.setattr(harvest, "_session",
                        lambda c: _fake_openalex(pages))

    r1 = harvest.harvest(conn, cfg, max_records=2)
    assert r1["new"] == 3                      # whole page ingested, not just 2

    r2 = harvest.harvest(conn, cfg, max_records=2)
    assert r2["new"] == 3                      # resume picks up page two
    assert db.counts(conn)["papers"] == 6      # no paper skipped across the gap
