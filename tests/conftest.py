import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpminer import db  # noqa: E402
from alpminer.config import Config  # noqa: E402


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config(email="test@example.edu", base_dir=tmp_path)
    c.ensure_dirs()
    return c


@pytest.fixture
def conn(cfg):
    connection = db.connect(cfg.db_path)
    yield connection
    connection.close()


def make_paper(i: int = 1, **over) -> dict:
    rec = {
        "id": f"W{1000 + i}",
        "doi": f"10.1000/test.{i}",
        "title": f"ALD of testium oxide, part {i}",
        "year": 2024,
        "journal": "J. Test. Sci.",
        "authors": ["A. Author", "B. Author"],
        "oa_pdf_url": f"https://oa.example.org/{i}.pdf",
        "landing_url": f"https://doi.org/10.1000/test.{i}",
        "is_oa": 1,
    }
    rec.update(over)
    return rec


@pytest.fixture
def paper(conn) -> dict:
    rec = make_paper(1)
    db.upsert_paper(conn, rec)
    return rec
