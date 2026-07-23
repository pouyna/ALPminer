"""Manual-download workflow for every paper still awaiting a PDF.

The queue holds all harvested papers without a file yet -- both the ones the
auto-downloader has not tried and the ones it gave up on -- so PDFs can be
dropped in right after harvest. The loop is designed to be as low-friction
as legitimately possible:

  alpminer manual open --n 10   -> opens the next 10 DOI links as browser
                                   tabs (you download each through your
                                   library access; ANY filename is fine)
  alpminer manual watch         -> watches the inbox and files each PDF the
                                   moment it lands, matching it to its paper
                                   by the DOI printed on page 1 (or by title)
  alpminer manual list          -> prints the queue and writes
                                   manual_queue.csv/.html to data/exports/
  alpminer manual ingest        -> one-shot version of watch

Renaming files to <paper_id>.pdf still works and is matched first, but is no
longer required: publisher default filenames are matched by content.

Queue entries can also be soft-deleted (download_status='removed'): removed
papers are hidden from the queue, the counts, tab-opening, and PDF matching,
but are kept in the database and can be restored at any time (GUI: the
"Removed papers" list on the Manual queue tab).
"""

from __future__ import annotations

import csv
import html
import io
import re
import shutil
import tempfile
import time
import webbrowser
from pathlib import Path

from . import db
from .config import Config
from .utils import atomic_write_text, log, sha256_file

BROWSER_TEMP_SUFFIXES = {".crdownload", ".part", ".tmp", ".download", ".partial"}
FRESH_FILE_SECONDS = 1.5   # watch mode: let the browser finish writing

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


# Papers that still need a PDF: freshly harvested ones (pending) and any the
# auto-downloader could not fetch (manual). Right after `harvest` every paper
# is here, so a copied-in PDF can be matched to its metadata immediately --
# no download pass required first. Auto-download and manually filed PDFs pull
# papers out of this set as their files arrive. Known-manual papers (auto-
# download already gave up) are listed before untried pending ones.
AWAITING_PDF_SQL = "download_status IN ('pending', 'manual')"


_QUEUE_ORDER = "ORDER BY (download_status <> 'manual'), id"


def _queue_filter(q: str | None) -> tuple[str, tuple]:
    """WHERE clause + params for an optional search term over the queue
    (matches title, DOI, journal, or paper id, case-insensitively)."""
    if not (q or "").strip():
        return AWAITING_PDF_SQL, ()
    like = f"%{q.strip()}%"
    return (f"{AWAITING_PDF_SQL} AND (title LIKE ? OR doi LIKE ? "
            "OR journal LIKE ? OR id LIKE ?)", (like, like, like, like))


def queue(conn) -> list:
    return conn.execute(
        f"SELECT * FROM papers WHERE {AWAITING_PDF_SQL} {_QUEUE_ORDER}"
    ).fetchall()


def queue_count(conn, q: str | None = None) -> int:
    """How many papers are awaiting a PDF (optionally matching a search)."""
    where, params = _queue_filter(q)
    return conn.execute(
        f"SELECT COUNT(*) AS c FROM papers WHERE {where}", params
    ).fetchone()["c"]


def queue_page(conn, offset: int = 0, limit: int = 50,
               q: str | None = None) -> list:
    """One page of the (optionally searched) queue, in the same order as
    queue(). Fetches only the rows for that page so the GUI can browse a
    large queue without loading it all at once."""
    where, params = _queue_filter(q)
    return conn.execute(
        f"SELECT * FROM papers WHERE {where} {_QUEUE_ORDER} "
        "LIMIT ? OFFSET ?", (*params, max(1, int(limit)), max(0, int(offset)))
    ).fetchall()


# ---- removing / restoring queue entries -----------------------------------------
# "Removed" is a soft-delete: the paper stays in the database (so it can be
# brought back), but download_status='removed' keeps it out of the queue, the
# awaiting-PDF count, tab-opening, and PDF matching until it is restored.

def remove_from_queue(conn, paper_id: str) -> bool:
    """Take a paper out of the manual queue. Only papers still awaiting a PDF
    can be removed; returns False if the id is unknown or already downloaded."""
    row = db.get_paper(conn, paper_id)
    if row is None or row["download_status"] not in ("pending", "manual"):
        return False
    db.set_fields(conn, paper_id, download_status="removed")
    return True


def restore_to_queue(conn, paper_id: str) -> bool:
    """Bring a removed paper back into the queue (as pending)."""
    row = db.get_paper(conn, paper_id)
    if row is None or row["download_status"] != "removed":
        return False
    db.set_fields(conn, paper_id, download_status="pending",
                  download_error=None)
    return True


def restore_all(conn) -> int:
    """Bring every removed paper back into the queue. Returns how many."""
    ids = [r["id"] for r in
           conn.execute("SELECT id FROM papers WHERE download_status = 'removed'")]
    for pid in ids:
        db.set_fields(conn, pid, download_status="pending", download_error=None)
    return len(ids)


def removed_count(conn) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM papers WHERE download_status = 'removed'"
    ).fetchone()["c"]


def removed_list(conn, limit: int = 200) -> list:
    return conn.execute(
        "SELECT * FROM papers WHERE download_status = 'removed' "
        "ORDER BY id LIMIT ?", (max(1, int(limit)),)
    ).fetchall()


# ---- queue export -------------------------------------------------------------

def export_queue(conn, cfg: Config) -> dict:
    """Write manual_queue.csv and manual_queue.html; return paths + count."""
    cfg.ensure_dirs()
    rows = queue(conn)

    def _why(p) -> str:
        if p["download_error"]:
            return p["download_error"]
        return ("not tried yet" if p["download_status"] == "pending" else "")

    csv_path = cfg.export_dir / "manual_queue.csv"
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["optional_filename", "title", "year", "journal", "doi_url",
                     "landing_url", "why_manual"])
    for p in rows:
        doi_url = f"https://doi.org/{p['doi']}" if p["doi"] else ""
        writer.writerow([f"{p['id']}.pdf", p["title"], p["year"], p["journal"],
                         doi_url, p["landing_url"] or "", _why(p)])
    atomic_write_text(csv_path, buf.getvalue())

    html_path = cfg.export_dir / "manual_queue.html"
    items = []
    for p in rows:
        doi_url = f"https://doi.org/{p['doi']}" if p["doi"] else (p["landing_url"] or "#")
        items.append(
            "<tr>"
            f"<td><a href='{html.escape(doi_url)}' target='_blank'>"
            f"{html.escape(p['title'] or '(untitled)')}</a></td>"
            f"<td>{p['year'] or ''}</td>"
            f"<td>{html.escape(p['journal'] or '')}</td>"
            f"<td><code>{html.escape(p['id'])}.pdf</code></td>"
            f"<td>{html.escape(_why(p)[:120])}</td>"
            "</tr>"
        )
    inbox = html.escape(str(cfg.inbox_dir))
    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>alpminer manual download queue ({len(rows)})</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px 8px;font-size:14px;text-align:left}}
th{{background:#f3f3f3}}code{{background:#eef}}</style></head><body>
<h2>Manual download queue &mdash; {len(rows)} article(s)</h2>
<p>Open each link (via your library/VPN if paywalled) and download the PDF
into <code>{inbox}</code>. <b>Any filename works</b> &mdash; run
<code>alpminer manual watch</code> and each file is matched to its paper by
the DOI on its first page. (Renaming to the filename in the fourth column
also works and is matched first.) Tip: <code>alpminer manual open</code>
opens these links as browser tabs for you.</p>
<table><tr><th>Title (link)</th><th>Year</th><th>Journal</th>
<th>Optional filename</th><th>Why manual</th></tr>{''.join(items)}</table>
</body></html>"""
    atomic_write_text(html_path, page)

    return {"count": len(rows), "csv": csv_path, "html": html_path}


# ---- browser assistance ---------------------------------------------------------

def open_queue(conn, cfg: Config, n: int = 10) -> int:
    """Open the next n queue entries as browser tabs. Returns tabs opened."""
    rows = queue(conn)[:max(0, n)]
    opened = 0
    for p in rows:
        url = (f"https://doi.org/{p['doi']}" if p["doi"]
               else (p["landing_url"] or ""))
        if not url:
            log.warning("[%s] has no DOI or landing URL to open", p["id"])
            continue
        webbrowser.open_new_tab(url)
        opened += 1
        time.sleep(0.35)  # keep tab order, don't hammer the browser
    return opened


# ---- content-based matching ------------------------------------------------------

def _clean_doi(doi: str) -> str:
    return doi.rstrip(".,;:)]}\"'").lower()


def _norm(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _pdf_front_text(path: Path, pages: int = 2) -> str:
    import fitz
    try:
        doc = fitz.open(path)
        try:
            k = min(pages, doc.page_count)
            return "\n".join(doc[i].get_text("text") for i in range(k))
        finally:
            doc.close()
    except Exception:  # noqa: BLE001 - unmatchable, caller reports
        return ""


def match_paper(conn, f: Path) -> tuple[str | None, str]:
    """Identify which paper a PDF file is. Returns (paper_id, how) on
    success or (None, reason) on failure.

    Priority: exact filename stem == paper id; then a DOI printed in the
    first two pages that matches a paper in the database; then an exact
    normalized-title match against any paper still awaiting a PDF (so a PDF
    dropped in right after harvest matches its harvested metadata, not only
    papers the auto-downloader already routed to manual).
    """
    row = conn.execute(
        "SELECT id FROM papers WHERE lower(id) = ? "
        "AND download_status != 'removed'", (f.stem.lower(),)
    ).fetchone()
    if row:
        return row["id"], "filename"

    text = _pdf_front_text(f)
    if not text.strip():
        return None, "no text layer to match; rename to <paper_id>.pdf"

    dois = {_clean_doi(m) for m in _DOI_RE.findall(text)}
    if dois:
        placeholders = ",".join("?" * len(dois))
        rows = conn.execute(
            f"SELECT id, doi, title FROM papers "
            f"WHERE lower(doi) IN ({placeholders}) "
            f"AND download_status != 'removed'", tuple(dois)
        ).fetchall()
        if len(rows) == 1:
            return rows[0]["id"], f"doi:{rows[0]['doi']}"
        if len(rows) > 1:
            norm_text = _norm(text)
            hits = [r for r in rows
                    if len(_norm(r["title"])) >= 25
                    and _norm(r["title"]) in norm_text]
            if len(hits) == 1:
                return hits[0]["id"], f"doi+title:{hits[0]['doi']}"
            return None, ("several known DOIs appear in this file; "
                          "rename to <paper_id>.pdf to disambiguate")

    norm_text = _norm(text)
    for r in conn.execute(
            f"SELECT id, title FROM papers WHERE {AWAITING_PDF_SQL}"):
        t = _norm(r["title"])
        if len(t) >= 25 and t in norm_text:
            return r["id"], "title"
    return None, "no matching DOI or title found; rename to <paper_id>.pdf"


# ---- external papers (not in OpenAlex) ------------------------------------------

def _pdf_title(path: Path) -> str | None:
    """The PDF's embedded document title, if it has a useful one."""
    import fitz
    try:
        doc = fitz.open(path)
        try:
            title = ((doc.metadata or {}).get("title") or "").strip()
        finally:
            doc.close()
        return title or None
    except Exception:  # noqa: BLE001
        return None


def add_external(conn, cfg: Config, pdf_path, title: str | None = None,
                 doi: str | None = None) -> dict:
    """Register a PDF that OpenAlex does not index so it flows through the
    extraction pipeline. Copies the file into the store, creates (or reuses,
    when the DOI already matches a row) a downloaded paper record, and marks it
    pending for text + extraction. Returns {id, title, doi, reused}."""
    cfg.ensure_dirs()
    src = Path(pdf_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"no such file: {src}")
    with open(src, "rb") as fh:
        if b"%PDF" not in fh.read(1024):
            raise ValueError(f"{src.name} is not a PDF (no %PDF header)")

    front = _pdf_front_text(src)
    if not doi:
        found = {_clean_doi(m) for m in _DOI_RE.findall(front)}
        doi = next(iter(found), None)
    title = (title or "").strip() or _pdf_title(src) or src.stem

    existing = None
    if doi:
        row = conn.execute("SELECT id FROM papers WHERE lower(doi) = ?",
                           (doi.lower(),)).fetchone()
        existing = row["id"] if row else None
    pid = existing or ("ext-" + sha256_file(src)[:10])

    dest = cfg.pdf_dir / f"{pid}.pdf"
    shutil.copy2(str(src), dest)
    if not existing:
        db.upsert_paper(conn, {"id": pid, "doi": doi, "title": title,
                               "year": None, "journal": None, "authors": [],
                               "oa_pdf_url": None, "landing_url": None,
                               "is_oa": 0})
    db.set_fields(conn, pid, download_status="downloaded",
                  download_source="external", download_error=None,
                  pdf_path=str(dest), pdf_sha256=sha256_file(dest),
                  text_status="pending", text_error=None,
                  extract_status="pending", extract_error=None)
    log.info("added external paper %s (%s) <- %s", pid, doi or "no DOI",
             src.name)
    return {"id": pid, "title": title, "doi": doi, "reused": bool(existing)}


# ---- ingest ---------------------------------------------------------------------

def _scan(conn, cfg: Config, directory: Path, skip_fresh: bool = False,
          move: bool | None = None, seen: set | None = None) -> dict:
    """One pass over `directory`: file every matchable PDF.

    move: True moves matched files into the store (right for the dedicated
    inbox); False copies and leaves the original untouched (right for temp
    folders, Downloads, or any folder the tool does not own). Defaults to
    move only for the project inbox.
    seen: optional set of already-processed source signatures, used by watch
    mode so copy-mode folders are not re-reported every poll.
    Returns lists of (id/name, detail) tuples under matched / bad /
    unmatched / duplicate."""
    matched, bad, unmatched, duplicate = [], [], [], []
    if move is None:
        move = directory == cfg.inbox_dir
    strict = directory == cfg.inbox_dir  # complain about junk only in the inbox
    if not directory.is_dir():
        return {"matched": matched, "bad": bad,
                "unmatched": [(str(directory), "folder does not exist")],
                "duplicate": duplicate}

    for f in sorted(directory.iterdir()):
        if not f.is_file() or f.name.startswith("."):
            continue
        if f.suffix.lower() in BROWSER_TEMP_SUFFIXES:
            continue  # browser still writing
        if skip_fresh:
            try:
                if time.time() - f.stat().st_mtime < FRESH_FILE_SECONDS:
                    continue
            except OSError:
                continue
        if f.suffix.lower() != ".pdf":
            if strict:
                unmatched.append((f.name, "not a .pdf filename"))
            continue
        try:
            stat = f.stat()
            sig = f"{f.resolve()}::{stat.st_size}::{int(stat.st_mtime)}"
            if seen is not None and sig in seen:
                continue
            with open(f, "rb") as fh:
                head = fh.read(1024)
        except OSError:
            continue  # locked by the browser; next pass
        if b"%PDF" not in head:
            bad.append((f.name, "file is not a valid PDF"))
            continue

        paper_id, how = match_paper(conn, f)
        if paper_id is None:
            if not strict:
                if seen is not None:
                    seen.add(sig)   # unrelated PDF in a shared folder
                continue            # only the inbox reports unmatched files
            unmatched.append((f.name, how))
            continue
        row = db.get_paper(conn, paper_id)
        if row["download_status"] == "downloaded":
            duplicate.append((f.name, f"already have {paper_id}"))
            if seen is not None:
                seen.add(sig)
            continue

        dest = cfg.pdf_dir / f"{paper_id}.pdf"
        if move:
            shutil.move(str(f), dest)
        else:
            shutil.copy2(str(f), dest)
            if seen is not None:
                seen.add(sig)
        db.set_fields(conn, paper_id,
                      download_status="downloaded",
                      download_source="manual",
                      download_error=None,
                      pdf_path=str(dest),
                      pdf_sha256=sha256_file(dest),
                      text_status="pending", text_error=None)
        matched.append((paper_id, how))
        log.info("ingested %s (matched by %s) <- %s", paper_id, how, f.name)

    return {"matched": matched, "bad": bad, "unmatched": unmatched,
            "duplicate": duplicate}


def ingest_inbox(conn, cfg: Config, directory: Path | str | None = None) -> dict:
    """One-shot ingest. Matches by filename, then DOI, then title.
    Files are MOVED out of the project inbox but only COPIED from any other
    folder (Downloads, temp, ...), leaving originals untouched."""
    cfg.ensure_dirs()
    directory = Path(directory) if directory else cfg.inbox_dir
    res = _scan(conn, cfg, directory)
    for name, why in res["unmatched"] + res["bad"]:
        log.warning("skipped: %s (%s)", name, why)
    for name, why in res["duplicate"]:
        log.info("left in place: %s (%s)", name, why)
    log.info("Manual ingest: %d filed, %d invalid, %d unmatched, "
             "%d duplicates.", len(res["matched"]), len(res["bad"]),
             len(res["unmatched"]), len(res["duplicate"]))
    # Backward-compatible shape: matched as a plain list of paper ids.
    res["matched"] = [pid for pid, _ in res["matched"]]
    return res


def watch_dirs(cfg: Config, directories=None, include_temp: bool = False
               ) -> list[Path]:
    """Resolve the folder list for watch mode: given folders (or the project
    inbox), plus the system temp folder when requested. The inbox is always
    included so the standard flow keeps working."""
    dirs = [Path(d) for d in (directories or []) if str(d).strip()]
    if cfg.inbox_dir not in dirs:
        dirs.insert(0, cfg.inbox_dir)
    if include_temp:
        tmp = Path(tempfile.gettempdir())
        if tmp not in dirs:
            dirs.append(tmp)
    return dirs


def watch_inbox(conn, cfg: Config, directories=None,
                include_temp: bool = False, interval: float = 2.0) -> dict:
    """Poll one or more folders and file PDFs as they arrive, until Ctrl-C.

    The project inbox is moved-from; every other folder (your Downloads, the
    system temp folder with include_temp, a browser folder) is copy-only:
    matched PDFs are copied into the store and the original is left exactly
    where it was. Unrelated PDFs in shared folders are ignored silently.
    Prints one line per filed paper and the live remaining-queue count.
    """
    cfg.ensure_dirs()
    dirs = watch_dirs(cfg, directories, include_temp)
    total_filed = 0
    reported: set[tuple[str, str]] = set()
    seen: set = set()
    remaining = len(queue(conn))
    log.info("Watching %s for PDFs (%d in queue). Ctrl-C to stop.",
             ", ".join(str(d) for d in dirs), remaining)
    try:
        while True:
            filed_this_pass = 0
            for directory in dirs:
                res = _scan(conn, cfg, directory, skip_fresh=True, seen=seen)
                filed_this_pass += len(res["matched"])
                for name, why in res["unmatched"] + res["bad"]:
                    key = (name, why)
                    if key not in reported:
                        reported.add(key)
                        log.warning("unfiled: %s (%s)", name, why)
            total_filed += filed_this_pass
            if filed_this_pass:
                remaining = len(queue(conn))
                log.info("-> %d remaining in the manual queue", remaining)
                if remaining == 0:
                    log.info("Manual queue is empty. Nice work.")
                    break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    log.info("Watch stopped: %d paper(s) filed this session.", total_filed)
    return {"filed": total_filed, "remaining": len(queue(conn))}
