import csv
import json

from alpminer import db
from alpminer.export import build_export
from tests.conftest import make_paper


def _done_paper(conn, i, recipes):
    rec = make_paper(i, doi=f"10.1/x{i}")
    db.upsert_paper(conn, rec)
    db.replace_recipes(conn, rec["id"], recipes)
    db.set_fields(conn, rec["id"], download_status="downloaded",
                  download_source="https://oa.example.org/x.pdf",
                  extract_status="done")
    return rec


def test_export_structure_counts_and_units(cfg, conn):
    _done_paper(conn, 1, [
        {"material": "Al2O3", "deposition_temperature_c": 200.0,
         "additional_reactants": ["O3"]},
        {"material": "TiO2"},
    ])
    _done_paper(conn, 2, [{"material": "HfO2", "confidence": 0.8}])
    # a paper with no recipes must be excluded
    rec3 = make_paper(3, doi="10.1/x3")
    db.upsert_paper(conn, rec3)
    db.set_fields(conn, rec3["id"], extract_status="no_recipes")

    result = build_export(conn, cfg)
    assert result["papers"] == 2 and result["recipes"] == 3

    data = json.loads((cfg.export_dir / "ald_recipes.json").read_text())
    assert data["schema"] == "alpminer.records.v2"
    assert data["profile"] == "ald"
    assert data["n_papers"] == 2 and data["n_recipes"] == 3
    assert data["units"]["growth_per_cycle"] == "angstrom/cycle"
    first = data["papers"][0]
    assert first["doi"] and first["title"] and first["authors"]
    assert first["recipes"][0]["material"] == "Al2O3"

    flat = json.loads((cfg.export_dir / "recipes_flat.json").read_text())
    assert len(flat) == 3
    assert {"paper_id", "doi", "title", "material"} <= set(flat[0])

    with open(cfg.export_dir / "recipes_flat.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert rows[0]["material"] == "Al2O3"
    assert rows[0]["additional_reactants"] == '["O3"]'


def test_export_is_rerunnable(cfg, conn):
    _done_paper(conn, 9, [{"material": "ZnO"}])
    build_export(conn, cfg)
    result = build_export(conn, cfg)  # atomic overwrite, no error
    assert result["recipes"] == 1


def test_export_flags_ocr_sourced_papers(cfg, conn):
    normal = _done_paper(conn, 20, [{"material": "Al2O3"}])
    scanned = _done_paper(conn, 21, [{"material": "TiN"}])
    db.set_fields(conn, scanned["id"], text_ocr=1)

    build_export(conn, cfg)
    data = json.loads((cfg.export_dir / "ald_recipes.json").read_text())
    by_id = {p["paper_id"]: p for p in data["papers"]}
    assert by_id[normal["id"]]["ocr"] is False
    assert by_id[scanned["id"]]["ocr"] is True

    flat = json.loads((cfg.export_dir / "recipes_flat.json").read_text())
    assert {r["paper_id"]: r["ocr"] for r in flat} == {
        normal["id"]: False, scanned["id"]: True}

    with open(cfg.export_dir / "recipes_flat.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert "ocr" in rows[0]
    ocr_by_id = {r["paper_id"]: r["ocr"] for r in rows}
    assert ocr_by_id[scanned["id"]] == "True"
    assert ocr_by_id[normal["id"]] == "False"
