"""SQLite state store. Single source of truth for pipeline progress.

Every stage reads "what is still pending" from here and commits each unit of
work as it completes, so the pipeline can be interrupted (Ctrl-C, crash,
network loss) and re-run without losing progress or repeating paid LLM calls.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# download_status: pending | downloaded | manual | removed
#   (removed = user took it out of the manual queue; kept so it can be
#    restored, but hidden from the queue, counts, and tab-opening)
# text_status:     pending | ok | failed | ocr_pending
#   (ocr_pending = scanned PDF flagged for the deferred OCR phase)
# text_ocr:        1 when the text came from OCR rather than the PDF's own
#                  text layer; exported with every record so OCR-sourced
#                  data is always distinguishable
# extract_status:  pending | done | no_recipes | triaged_out | failed
SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id              TEXT PRIMARY KEY,
    doi             TEXT UNIQUE,
    title           TEXT,
    year            INTEGER,
    journal         TEXT,
    authors         TEXT,
    oa_pdf_url      TEXT,
    landing_url     TEXT,
    is_oa           INTEGER DEFAULT 0,
    download_status TEXT NOT NULL DEFAULT 'pending',
    download_error  TEXT,
    download_source TEXT,
    pdf_path        TEXT,
    pdf_sha256      TEXT,
    text_status     TEXT NOT NULL DEFAULT 'pending',
    text_error      TEXT,
    text_path       TEXT,
    text_ocr        INTEGER NOT NULL DEFAULT 0,
    extract_status  TEXT NOT NULL DEFAULT 'pending',
    extract_error   TEXT,
    n_recipes       INTEGER DEFAULT 0,
    updated_at      TEXT
);

CREATE TABLE IF NOT EXISTS recipes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id   TEXT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    data       TEXT NOT NULL,
    created_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_recipes_paper ON recipes(paper_id);
CREATE INDEX IF NOT EXISTS idx_papers_download ON papers(download_status);
CREATE INDEX IF NOT EXISTS idx_papers_extract ON papers(extract_status);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""

PAPER_COLUMNS = (
    "id", "doi", "title", "year", "journal", "authors",
    "oa_pdf_url", "landing_url", "is_oa",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    conn.commit()
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was created (SQLite has no
    IF NOT EXISTS for columns; a duplicate-column error means it's present)."""
    for ddl in (
        "ALTER TABLE papers ADD COLUMN text_ocr INTEGER NOT NULL DEFAULT 0",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass


# ---- meta -------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT v FROM meta WHERE k = ?", (key,)).fetchone()
    return row["v"] if row else None


def add_meta_int(conn: sqlite3.Connection, key: str, delta: int) -> int:
    """Add delta to an integer meta counter (created at 0). Returns the new
    value. Used for the cumulative real-token-spend counters."""
    current = int(get_meta(conn, key) or 0) + int(delta)
    set_meta(conn, key, str(current))
    return current


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (k, v) VALUES (?, ?) "
        "ON CONFLICT(k) DO UPDATE SET v = excluded.v",
        (key, value),
    )
    conn.commit()


# ---- papers -----------------------------------------------------------------

def upsert_paper(conn: sqlite3.Connection, rec: dict) -> bool:
    """Insert a harvested paper. Existing rows are left untouched (their
    per-stage status is preserved). Returns True if a new row was inserted."""
    values = {k: rec.get(k) for k in PAPER_COLUMNS}
    if isinstance(values.get("authors"), (list, tuple)):
        values["authors"] = json.dumps(values["authors"], ensure_ascii=False)
    values["updated_at"] = _now()
    cols = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    cur = conn.execute(
        f"INSERT OR IGNORE INTO papers ({cols}) VALUES ({placeholders})",
        tuple(values.values()),
    )
    conn.commit()
    return cur.rowcount > 0


def set_fields(conn: sqlite3.Connection, paper_id: str, **fields) -> None:
    fields["updated_at"] = _now()
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(
        f"UPDATE papers SET {assignments} WHERE id = ?",
        (*fields.values(), paper_id),
    )
    conn.commit()


def get_paper(conn: sqlite3.Connection, paper_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()


def papers_where(conn: sqlite3.Connection, clause: str, params: tuple = (),
                 limit: int | None = None) -> list[sqlite3.Row]:
    sql = f"SELECT * FROM papers WHERE {clause} ORDER BY id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return conn.execute(sql, params).fetchall()


# ---- recipes ----------------------------------------------------------------

def replace_recipes(conn: sqlite3.Connection, paper_id: str,
                    recipes: list[dict]) -> int:
    """Atomically replace all recipes for a paper (idempotent re-extraction)."""
    with conn:  # single transaction
        conn.execute("DELETE FROM recipes WHERE paper_id = ?", (paper_id,))
        now = _now()
        conn.executemany(
            "INSERT INTO recipes (paper_id, data, created_at) VALUES (?, ?, ?)",
            [(paper_id, json.dumps(r, ensure_ascii=False), now) for r in recipes],
        )
        conn.execute(
            "UPDATE papers SET n_recipes = ?, updated_at = ? WHERE id = ?",
            (len(recipes), now, paper_id),
        )
    return len(recipes)


def recipes_for(conn: sqlite3.Connection, paper_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT data FROM recipes WHERE paper_id = ? ORDER BY id", (paper_id,)
    ).fetchall()
    return [json.loads(r["data"]) for r in rows]


# ---- reporting --------------------------------------------------------------

def counts(conn: sqlite3.Connection) -> dict:
    out: dict = {"papers": conn.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]}
    for col in ("download_status", "text_status", "extract_status"):
        rows = conn.execute(
            f"SELECT {col} AS s, COUNT(*) AS c FROM papers GROUP BY {col}"
        ).fetchall()
        out[col] = {r["s"]: r["c"] for r in rows}
    out["recipes"] = conn.execute("SELECT COUNT(*) c FROM recipes").fetchone()["c"]
    return out


# ---- reset ------------------------------------------------------------------

def reset(conn: sqlite3.Connection) -> None:
    """Clear all harvested state -- papers, recipes, and meta (the harvest
    checkpoint and the profile lock) -- so the project starts from empty.
    Files already on disk are left untouched; callers wipe those separately."""
    with conn:  # single transaction
        conn.execute("DELETE FROM recipes")
        conn.execute("DELETE FROM papers")
        conn.execute("DELETE FROM meta")
    try:
        conn.execute("VACUUM")   # reclaim the freed space; harmless if it can't
    except sqlite3.OperationalError:
        pass
