"""Build the final recipe database files from the SQLite store.

Outputs (all written atomically to data/exports/):
  ald_recipes.json   nested: one entry per paper with its recipe list
  recipes_flat.json  flat list: one entry per recipe with source fields inlined
  recipes_flat.csv   the same flat list as a spreadsheet-friendly CSV

The JSON carries a "units" block: for each quantity field its unit symbol
and QUDT unit and quantity-kind IRIs. Records extracted by alpminer 1.x are
upgraded on the way out (SI field names and values, reactant columns from
the built-in table), so old projects export in the current schema.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from . import __version__, chemistry, db, profiles
from .config import Config
from .schema import upgrade_legacy_fields
from .utils import atomic_write_text, log

SCHEMA_TAG = "alpminer.records.v3"


def offline_resolver(cfg: Config) -> chemistry.Resolver:
    """Built-in table plus already-cached PubChem answers; no network."""
    return chemistry.Resolver(cache_path=cfg.root / chemistry.CACHE_FILENAME,
                              online=False)


def stored_record(rec: dict, profile: profiles.Profile,
                  resolver: chemistry.Resolver | None = None) -> dict:
    """A record from the database in the current schema: 1.x quantity
    fields renamed and converted (and listed in unit_conversions), missing
    reactant columns filled."""
    conversions: list[str] = []
    rec = upgrade_legacy_fields(rec, profile.field_names(), conversions)
    if conversions:
        logged = rec.get(profiles.CONVERSIONS_FIELD)
        rec[profiles.CONVERSIONS_FIELD] = (
            (logged if isinstance(logged, list) else []) + conversions)
    if resolver is not None and any(
            f.chemical and f"{f.name}_name" not in rec
            for f in profile.fields):
        chemistry.normalize_record(rec, profile, resolver)
    return rec

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
    resolver = offline_resolver(cfg)
    entries = [_paper_entry(p, [stored_record(r, profile, resolver)
                                for r in db.recipes_for(conn, p["id"])])
               for p in papers]
    n_recipes = sum(e["n_recipes"] for e in entries)

    database = {
        "schema": SCHEMA_TAG,
        "profile": profile.name,
        "record_noun": profile.record_noun,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": f"alpminer {__version__}",
        "query": cfg.query,
        "units": profile.units_metadata(),
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
    cols = list(_SOURCE_FIELDS) + profile.export_field_names()
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
