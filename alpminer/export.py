"""Build the final recipe database files from the SQLite store.

Outputs (all written atomically to data/exports/):
  ald_recipes.json   nested: one entry per paper with its recipe list
  recipes_flat.json  flat list: one entry per recipe with source fields inlined
  recipes_flat.csv   the same flat list as a spreadsheet-friendly CSV
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from . import __version__, db, profiles
from .config import Config
from .utils import atomic_write_text, log

# "ocr" is True when the paper's text came from OCR rather than the PDF's
# own text layer -- carried into every output so OCR-sourced records are
# always distinguishable (OCR can misread characters, e.g. O/0 in formulas).
_SOURCE_FIELDS = ("paper_id", "doi", "title", "year", "journal", "ocr")


def _paper_entry(paper, recipes: list[dict]) -> dict:
    authors = paper["authors"]
    try:
        authors = json.loads(authors) if authors else []
    except (TypeError, json.JSONDecodeError):
        authors = [authors] if authors else []
    return {
        "paper_id": paper["id"],
        "doi": paper["doi"],
        "title": paper["title"],
        "year": paper["year"],
        "journal": paper["journal"],
        "authors": authors,
        "pdf_source": paper["download_source"],
        "ocr": bool(paper["text_ocr"]),
        "n_recipes": len(recipes),
        "recipes": recipes,
    }


def build_export(conn, cfg: Config) -> dict:
    cfg.ensure_dirs()
    profile = profiles.load(cfg.profile, cfg.base_dir)
    locked = db.get_meta(conn, "profile")
    if locked and locked != profile.name:
        log.warning("database was extracted with profile %r; exporting with "
                    "%r field order/units. Switch profile back for a "
                    "faithful export.", locked, profile.name)
    papers = db.papers_where(
        conn, "extract_status = 'done' AND n_recipes > 0")
    entries = [_paper_entry(p, db.recipes_for(conn, p["id"])) for p in papers]
    n_recipes = sum(e["n_recipes"] for e in entries)

    database = {
        "schema": "alpminer.records.v2",
        "profile": profile.name,
        "record_noun": profile.record_noun,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": f"alpminer {__version__}",
        "query": cfg.query,
        "units": profile.units,
        "n_papers": len(entries),
        "n_recipes": n_recipes,
        "papers": entries,
    }
    main_path = cfg.export_dir / "ald_recipes.json"
    atomic_write_text(main_path,
                      json.dumps(database, ensure_ascii=False, indent=2))

    # Flat variants -------------------------------------------------------------
    flat = []
    for entry in entries:
        src = {k: entry[k] for k in _SOURCE_FIELDS}
        for recipe in entry["recipes"]:
            flat.append({**src, **recipe})
    flat_path = cfg.export_dir / "recipes_flat.json"
    atomic_write_text(flat_path,
                      json.dumps(flat, ensure_ascii=False, indent=2))

    csv_path = cfg.export_dir / "recipes_flat.csv"
    buf = io.StringIO()
    cols = list(_SOURCE_FIELDS) + profile.field_names()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in flat:
        flat_row = {k: (json.dumps(v, ensure_ascii=False)
                        if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()}
        writer.writerow(flat_row)
    atomic_write_text(csv_path, buf.getvalue())

    log.info("Exported %d recipes from %d papers:", n_recipes, len(entries))
    for p in (main_path, flat_path, csv_path):
        log.info("  %s", p)
    return {"papers": len(entries), "recipes": n_recipes,
            "paths": [main_path, flat_path, csv_path]}
