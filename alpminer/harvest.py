"""Harvest ALD article metadata from OpenAlex (cursor-paginated, resumable).

OpenAlex is free, requires no key, covers essentially all DOI-registered
journal articles, and already includes each work's best legal open-access
location. The pagination cursor is checkpointed in the DB after every page,
so an interrupted harvest resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import time

import requests

from . import __version__, db
from .config import Config
from .utils import log, with_retries

OPENALEX_WORKS = "https://api.openalex.org/works"
PER_PAGE = 200
SELECT_FIELDS = ("id,doi,display_name,publication_year,primary_location,"
                 "best_oa_location,open_access,authorships")


def _session(cfg: Config) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = f"alpminer/{__version__} (mailto:{cfg.email})"
    return s


def build_filter(cfg: Config) -> str:
    query = cfg.query.replace(",", " ").strip()
    if not query:
        from . import profiles
        query = profiles.load(cfg.profile, cfg.base_dir).default_query
        log.info("No query configured; using the %r profile default: %s",
                 cfg.profile, query)
    parts = [f"title_and_abstract.search:{query}", "type:article"]
    if cfg.from_year:
        parts.append(f"from_publication_date:{cfg.from_year}-01-01")
    if cfg.to_year:
        parts.append(f"to_publication_date:{cfg.to_year}-12-31")
    return ",".join(parts)


def _short_id(url_or_id: str | None) -> str | None:
    if not url_or_id:
        return None
    return url_or_id.rstrip("/").rsplit("/", 1)[-1]


def _short_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")


def _work_to_record(work: dict) -> dict | None:
    paper_id = _short_id(work.get("id"))
    if not paper_id:
        return None
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    source = primary.get("source") or {}
    doi = _short_doi(work.get("doi"))
    authors = []
    for auth in (work.get("authorships") or [])[:15]:
        name = (auth.get("author") or {}).get("display_name")
        if name:
            authors.append(name)
    landing = (best_oa.get("landing_page_url")
               or primary.get("landing_page_url")
               or (f"https://doi.org/{doi}" if doi else None))
    return {
        "id": paper_id,
        "doi": doi,
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "authors": authors,
        "oa_pdf_url": best_oa.get("pdf_url"),
        "landing_url": landing,
        "is_oa": 1 if (work.get("open_access") or {}).get("is_oa") else 0,
    }


def harvest(conn, cfg: Config, max_records: int | None = None) -> dict:
    """Fetch works matching the query into the DB. Returns summary counts."""
    cfg.require_email()
    filt = build_filter(cfg)
    cursor_key = "harvest_cursor:" + hashlib.sha1(filt.encode()).hexdigest()[:12]
    cursor = db.get_meta(conn, cursor_key) or "*"
    if cursor == "DONE":
        log.info("Harvest for this query already completed; nothing to do. "
                 "(Delete meta key %s to force a re-harvest.)", cursor_key)
        return {"new": 0, "seen": 0, "total_available": None}

    session = _session(cfg)
    new = seen = 0
    total_available = None

    log.info("Harvesting OpenAlex works matching filter: %s", filt)
    while True:
        params = {
            "filter": filt,
            "per-page": PER_PAGE,
            "cursor": cursor,
            "select": SELECT_FIELDS,
            "mailto": cfg.email,
        }

        def _get():
            r = session.get(OPENALEX_WORKS, params=params,
                            timeout=(10, cfg.download_timeout_s))
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            r.raise_for_status()
            return r.json()

        # OpenAlex rate-limits (429) can be sticky; retry a few times honoring
        # any Retry-After the server sends, but keep the total bounded (~35s)
        # so a rate-limited job stays responsive to Stop and to re-running.
        payload = with_retries(_get, desc="OpenAlex page fetch",
                               retry_on=(requests.RequestException,),
                               attempts=4, base_delay=5.0, max_delay=30.0)
        meta = payload.get("meta") or {}
        total_available = meta.get("count", total_available)
        results = payload.get("results") or []

        for work in results:
            rec = _work_to_record(work)
            if rec is None:
                continue
            seen += 1
            if db.upsert_paper(conn, rec):
                new += 1

        next_cursor = meta.get("next_cursor")
        if next_cursor:
            db.set_meta(conn, cursor_key, next_cursor)  # checkpoint each page
            cursor = next_cursor
        if not results or not next_cursor:
            db.set_meta(conn, cursor_key, "DONE")
            break
        # --max is a per-run floor honored only at a page boundary: the whole
        # page above is ingested and its cursor checkpointed before we stop, so
        # the next run resumes at the following page with no papers skipped.
        # (Breaking mid-page would checkpoint past the page's unprocessed tail.)
        if max_records is not None and seen >= max_records:
            log.info("Stopping after %d records (>= --max %d); the next harvest "
                     "run continues from the checkpointed cursor.", seen,
                     max_records)
            break

        log.info("  harvested %d works so far (%s available for this query)...",
                 seen, f"{total_available:,}" if total_available else "?")
        time.sleep(cfg.request_delay_s)   # be polite between pages

    log.info("Harvest pass finished: %d fetched this run, %d new papers added.",
             seen, new)
    if total_available:
        log.info("OpenAlex reports %s total works matching this query.",
                 f"{total_available:,}")
    return {"new": new, "seen": seen, "total_available": total_available}
