from alpminer import db
from tests.conftest import make_paper


def test_upsert_is_idempotent(conn):
    rec = make_paper(1)
    assert db.upsert_paper(conn, rec) is True
    assert db.upsert_paper(conn, rec) is False
    assert db.counts(conn)["papers"] == 1


def test_upsert_preserves_stage_progress(conn):
    rec = make_paper(2)
    db.upsert_paper(conn, rec)
    db.set_fields(conn, rec["id"], download_status="downloaded",
                  pdf_path="/x.pdf")
    db.upsert_paper(conn, rec)  # re-harvest must not reset progress
    row = db.get_paper(conn, rec["id"])
    assert row["download_status"] == "downloaded"
    assert row["pdf_path"] == "/x.pdf"


def test_papers_where_and_limit(conn):
    for i in range(5):
        db.upsert_paper(conn, make_paper(i, doi=f"10.1/{i}"))
    db.set_fields(conn, "W1000", download_status="downloaded")
    pending = db.papers_where(conn, "download_status = 'pending'")
    assert len(pending) == 4
    assert len(db.papers_where(conn, "download_status = 'pending'", limit=2)) == 2


def test_replace_recipes_is_atomic_and_idempotent(conn, paper):
    pid = paper["id"]
    db.replace_recipes(conn, pid, [{"material": "Al2O3"}, {"material": "TiO2"}])
    assert db.get_paper(conn, pid)["n_recipes"] == 2
    db.replace_recipes(conn, pid, [{"material": "HfO2"}])
    rows = db.recipes_for(conn, pid)
    assert [r["material"] for r in rows] == ["HfO2"]
    assert db.get_paper(conn, pid)["n_recipes"] == 1
    assert db.counts(conn)["recipes"] == 1


def test_meta_roundtrip(conn):
    assert db.get_meta(conn, "cursor") is None
    db.set_meta(conn, "cursor", "abc")
    db.set_meta(conn, "cursor", "def")
    assert db.get_meta(conn, "cursor") == "def"


def test_reset_clears_papers_recipes_and_meta(conn, paper):
    db.replace_recipes(conn, paper["id"], [{"material": "Al2O3"}])
    db.set_meta(conn, "profile", "ald")
    db.set_meta(conn, "harvest_cursor:abc", "somecursor")
    assert db.counts(conn)["papers"] == 1 and db.counts(conn)["recipes"] == 1

    db.reset(conn)

    c = db.counts(conn)
    assert c["papers"] == 0 and c["recipes"] == 0
    assert db.get_meta(conn, "profile") is None          # profile lock cleared
    assert db.get_meta(conn, "harvest_cursor:abc") is None  # checkpoint cleared
