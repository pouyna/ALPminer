"""alpminer command-line interface.

    alpminer init                    write alpminer.toml + create data dirs
    alpminer harvest [--max N]       fetch article metadata from OpenAlex
    alpminer download [--limit N]    auto-download open-access PDFs
    alpminer manual list             show/export the awaiting-a-PDF queue
    alpminer manual open [--n N]     open the next N DOI links as browser tabs
    alpminer manual watch            auto-file PDFs the moment they arrive
    alpminer manual ingest           one-shot: file PDFs from data/manual_inbox/
    alpminer manual remove <id...>   take paper(s) out of the queue (kept)
    alpminer manual restore [--all]  bring removed paper(s) back
    alpminer add <pdf>               add a PDF not in OpenAlex, queue for extract
    alpminer extract [--limit N]     LLM triage + recipe extraction
    alpminer export                  build ald_recipes.json (+ flat JSON/CSV)
    alpminer status                  progress dashboard + token estimate
    alpminer reset [--all]           clear harvested data and start fresh
    alpminer run                     all stages end to end (resumable)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config, db
from .utils import human_int, log, setup_logging


def _open(args):
    cfg = config.load(Path(args.config) if args.config else None)
    cfg.ensure_dirs()
    setup_logging(cfg.root, verbose=args.verbose)
    conn = db.connect(cfg.db_path)
    return cfg, conn


# ---- commands -----------------------------------------------------------------

def cmd_init(args) -> int:
    path = Path(args.config) if args.config else Path.cwd() / config.CONFIG_FILENAME
    if path.exists() and args.force:
        path.unlink()
    config.write_template(path, email=args.email or "you@university.edu")
    cfg = config.load(path)
    cfg.ensure_dirs()
    print(f"Wrote {path} and created {cfg.root}/.")
    print("Next steps:")
    print(f"  1. Edit {path.name}: set your email"
          + ("" if args.email else " (required)")
          + ", adjust query/year range if desired.")
    print("  2. Pick a profile (ald, ale, or your own via `alpminer "
          "profiles new`)")
    print("     and a provider in alpminer.toml, then set the matching key:")
    print("       ANTHROPIC_API_KEY   (console.anthropic.com)")
    print("       GEMINI_API_KEY      (aistudio.google.com/apikey, free "
          "tier)")
    print("       or openai_compatible / your own plugin -- see README")
    print("  3. alpminer run --limit 25    (recommended pilot before scaling up)")
    print("  Or skip the terminal entirely:  alpminer gui")
    return 0


def cmd_harvest(args) -> int:
    cfg, conn = _open(args)
    if args.query:
        cfg.query = args.query
    from . import harvest
    harvest.harvest(conn, cfg, max_records=args.max)
    return 0


def cmd_download(args) -> int:
    cfg, conn = _open(args)
    from . import download, manual
    download.download_pending(conn, cfg, limit=args.limit)
    _announce_manual_queue(conn, cfg, manual)
    return 0


def _announce_manual_queue(conn, cfg, manual_mod) -> None:
    n = len(manual_mod.queue(conn))
    if not n:
        return
    info = manual_mod.export_queue(conn, cfg)
    print()
    print(f"ACTION NEEDED: {n} article(s) could not be downloaded automatically.")
    print(f"  Clickable list: {info['html']}")
    print("  Fast loop: `alpminer manual open --n 10` opens the next ten links")
    print(f"  as browser tabs; download each PDF (any filename is fine) into")
    print(f"  {cfg.inbox_dir}")
    print("  while `alpminer manual watch` runs in a second terminal and files")
    print("  each one automatically by its DOI. The pipeline continues fine")
    print("  without them in the meantime.")


def cmd_manual(args) -> int:
    cfg, conn = _open(args)
    from . import manual
    if args.manual_cmd == "ingest":
        directory = args.directory[0] if args.directory else None
        result = manual.ingest_inbox(conn, cfg, directory=directory)
        print(f"Filed {len(result['matched'])} PDF(s); "
              f"{len(result['bad'])} invalid; "
              f"{len(result['unmatched'])} unmatched; "
              f"{len(result['duplicate'])} duplicates (see log).")
    elif args.manual_cmd == "open":
        opened = manual.open_queue(conn, cfg, n=args.n)
        if not opened:
            print("Manual queue is empty; nothing to open.")
        else:
            inbox = args.directory or cfg.inbox_dir
            print(f"Opened {opened} tab(s). Download each PDF (any filename) "
                  f"into {inbox},")
            print("then run `alpminer manual watch` (or `manual ingest`) to "
                  "file them automatically.")
    elif args.manual_cmd == "watch":
        result = manual.watch_inbox(conn, cfg, directories=args.directory,
                                    include_temp=args.include_temp)
        print(f"Filed {result['filed']} paper(s); "
              f"{result['remaining']} still in the queue.")
    elif args.manual_cmd == "remove":
        if not args.ids:
            print("usage: alpminer manual remove <paper_id> [...]",
                  file=sys.stderr)
            return 2
        removed = 0
        for pid in args.ids:
            if manual.remove_from_queue(conn, pid):
                removed += 1
            else:
                print(f"  skipped {pid}: not in the queue "
                      "(unknown id, already downloaded, or already removed)")
        print(f"Removed {removed} paper(s) from the queue. They are kept and "
              "can be restored with `alpminer manual restore`.")
    elif args.manual_cmd == "restore":
        if args.all:
            n = manual.restore_all(conn)
            print(f"Restored {n} removed paper(s) to the queue.")
        elif not args.ids:
            rows = manual.removed_list(conn)
            if not rows:
                print("No removed papers.")
                return 0
            print(f"{manual.removed_count(conn)} removed paper(s):")
            for p in rows[:20]:
                print(f"  {p['id']}  {(p['title'] or '(untitled)')[:80]}")
            print("Restore with `alpminer manual restore <id>` or `--all`.")
        else:
            restored = 0
            for pid in args.ids:
                if manual.restore_to_queue(conn, pid):
                    restored += 1
                else:
                    print(f"  skipped {pid}: not a removed paper")
            print(f"Restored {restored} paper(s) to the queue.")
    else:  # list
        rows = manual.queue(conn)
        if not rows:
            print("Manual queue is empty.")
            return 0
        info = manual.export_queue(conn, cfg)
        print(f"{info['count']} article(s) awaiting manual download. "
              f"First {min(10, len(rows))}:")
        for p in rows[:10]:
            doi = f"https://doi.org/{p['doi']}" if p["doi"] else (p["landing_url"] or "-")
            print(f"  {p['id']}  <-  {doi}")
            print(f"      {(p['title'] or '(untitled)')[:90]}")
        print(f"\nFull clickable list: {info['html']}")
        print("Fast loop: `alpminer manual open --n 10` opens the next ten "
              "links as tabs;")
        print(f"download each PDF (any filename) into {cfg.inbox_dir} while")
        print("`alpminer manual watch` runs in a second terminal and files "
              "them as they land.")
    return 0


def cmd_add(args) -> int:
    cfg, conn = _open(args)
    from . import manual
    try:
        info = manual.add_external(conn, cfg, args.pdf, title=args.title,
                                   doi=args.doi)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Added paper {info['id']}: {info['title']}"
          + (f"  (matched existing DOI {info['doi']})" if info["reused"] else ""))
    print("Now run `alpminer extract` (or `alpminer run --skip-harvest`) to "
          "process it.")
    return 0


def cmd_extract(args) -> int:
    cfg, conn = _open(args)
    from . import extract
    extract.run_extract(conn, cfg, limit=args.limit, refresh=args.refresh,
                        only=args.only)
    return 0


def cmd_export(args) -> int:
    cfg, conn = _open(args)
    from . import export
    export.build_export(conn, cfg)
    return 0


def cmd_profiles(args) -> int:
    from . import profiles
    base = (Path(args.config).resolve().parent if args.config
            else Path.cwd())
    if args.profiles_cmd == "new":
        dest = profiles.write_new_profile(base, args.name)
        print(f"Wrote {dest}")
        print(f"Edit it, then set  profile = \"{args.name}\"  in "
              "alpminer.toml.")
    else:  # list
        for info in profiles.list_profiles(base):
            mark = "*" if info["name"] == _current_profile(base) else " "
            extra = (f"{info.get('fields', '?')} fields, "
                     f"{info.get('record_noun', '')}"
                     if "error" not in info else f"INVALID: {info['error']}")
            print(f" {mark} {info['name']:<14} {info['origin']:<12} "
                  f"{info.get('label', '')}  ({extra})")
    return 0


def _current_profile(base: Path) -> str:
    try:
        return config.load(base / config.CONFIG_FILENAME).profile
    except config.ConfigError:
        return ""


def cmd_providers(args) -> int:
    from . import providers
    base = (Path(args.config).resolve().parent if args.config
            else Path.cwd())
    if args.providers_cmd == "new":
        dest = providers.write_new_plugin(base, args.name)
        print(f"Wrote {dest}")
        print(f"Implement call_tool(), then set  provider = "
              f"\"{args.name}\"  in alpminer.toml. Extra settings go under "
              "[provider_settings].")
    else:  # list
        print("built-in: anthropic, gemini, openai_compatible")
        plugin_dir = Path(base) / providers.PLUGIN_DIR_NAME
        plugins = sorted(p.stem for p in plugin_dir.glob("*.py"))             if plugin_dir.is_dir() else []
        print("plugins:  " + (", ".join(plugins) if plugins else "(none; "
              "add one with `alpminer providers new <name>`)"))
    return 0


def cmd_gui(args) -> int:
    from . import gui
    base = (Path(args.config).resolve().parent if args.config
            else Path.cwd())
    return gui.serve(base, port=args.port, open_browser=not args.no_browser)


def _estimate_pending_tokens(conn, cfg) -> int:
    rows = db.papers_where(
        conn, "download_status = 'downloaded' AND "
              "extract_status IN ('pending', 'failed')")
    total_chars = 0
    for p in rows:
        txt = cfg.text_dir / f"{p['id']}.txt"
        if txt.exists():
            total_chars += min(txt.stat().st_size, cfg.max_paper_chars)
        else:
            total_chars += 45_000  # typical article length heuristic
    return total_chars // 4


def _wipe_data_files(cfg) -> None:
    """Delete the contents of the download / text / cache / export / inbox
    folders, leaving the (now empty) folders themselves in place."""
    import shutil
    for d in (cfg.pdf_dir, cfg.text_dir, cfg.raw_llm_dir, cfg.export_dir,
              cfg.inbox_dir):
        if not d.exists():
            continue
        for f in d.iterdir():
            try:
                shutil.rmtree(f) if f.is_dir() else f.unlink()
            except OSError as exc:  # noqa: PERF203
                log.warning("could not remove %s: %s", f, exc)


def cmd_reset(args) -> int:
    cfg, conn = _open(args)
    c = db.counts(conn)
    if not args.yes:
        print(f"This clears the database for {cfg.root}:")
        print(f"  {human_int(c['papers'])} papers, {human_int(c['recipes'])} "
              "recipes, and the harvest checkpoint / profile lock.")
        if args.all:
            print("  --all also deletes downloaded PDFs, extracted texts, raw "
                  "LLM caches, and exports.")
        if input("Type 'yes' to confirm: ").strip().lower() != "yes":
            print("Aborted; nothing was changed.")
            return 1
    db.reset(conn)
    if args.all:
        _wipe_data_files(cfg)
    print("Project data reset."
          + (" Downloaded files removed." if args.all
             else " (Downloaded files kept; use --all to remove them too.)"))
    return 0


def cmd_status(args) -> int:
    cfg, conn = _open(args)
    c = db.counts(conn)
    dl, tx, ex = c["download_status"], c["text_status"], c["extract_status"]

    def line(label, mapping, order):
        parts = [f"{k} {human_int(mapping.get(k, 0))}"
                 for k in order if mapping.get(k)]
        print(f"  {label:<12} " + ("  |  ".join(parts) if parts else "-"))

    print(f"alpminer {__version__}  --  {cfg.db_path}")
    print(f"  profile      {cfg.profile}")
    print(f"  provider     {cfg.provider}  "
          f"(extraction={cfg.active_extraction_model}, "
          f"triage={cfg.active_triage_model if cfg.triage_enabled else 'off'})")
    print(f"  papers       {human_int(c['papers'])}")
    line("download:", dl, ("pending", "downloaded", "manual"))
    line("text:", tx, ("pending", "ok", "ocr_pending", "failed"))
    line("extract:", ex, ("pending", "done", "no_recipes", "triaged_out", "failed"))
    print(f"  recipes      {human_int(c['recipes'])}")

    pending_tokens = _estimate_pending_tokens(conn, cfg)
    if pending_tokens:
        print(f"  est. LLM input remaining: ~{human_int(pending_tokens)} tokens "
              f"({cfg.active_extraction_model}"
              + (", + triage" if cfg.triage_enabled else "") + ")")
    spent_in = int(db.get_meta(conn, "usage_input_tokens") or 0)
    spent_out = int(db.get_meta(conn, "usage_output_tokens") or 0)
    if spent_in or spent_out:
        calls = int(db.get_meta(conn, "usage_llm_calls") or 0)
        print(f"  LLM spend so far: {human_int(spent_in)} in / "
              f"{human_int(spent_out)} out tokens ({human_int(calls)} calls, "
              "as reported by the provider; cached re-runs cost nothing)")
    if dl.get("manual"):
        print(f"  -> {dl['manual']} article(s) in the manual queue: "
              "run `alpminer manual list`")
    if ex.get("failed"):
        print(f"  -> {ex['failed']} extraction failure(s): re-run "
              "`alpminer extract` to retry them")
    return 0


def cmd_run(args) -> int:
    cfg, conn = _open(args)
    import requests
    from . import download, export, extract, harvest, manual
    from .utils import RetryError
    if not args.skip_harvest:
        try:
            harvest.harvest(conn, cfg, max_records=args.limit)
        except (RetryError, requests.RequestException) as exc:
            log.warning("Harvest did not finish (%s). Continuing with the "
                        "papers already in the database; wait a bit and re-run "
                        "`alpminer run` to fetch more -- nothing is lost.", exc)
    download.download_pending(conn, cfg, limit=args.limit)
    ingest = manual.ingest_inbox(conn, cfg)
    if ingest["matched"]:
        print(f"Picked up {len(ingest['matched'])} manually downloaded PDF(s).")
    _announce_manual_queue(conn, cfg, manual)
    extract.run_extract(conn, cfg, limit=args.limit)
    export.build_export(conn, cfg)
    print("\nDone. See `alpminer status` for the full picture.")
    return 0


# ---- parser ---------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alpminer",
        description="Resumable ALD-recipe literature mining pipeline "
                    "(OpenAlex + open-access PDFs + Claude extraction).")
    p.add_argument("--config", help="path to alpminer.toml "
                                    "(default: ./alpminer.toml)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="debug logging on the console")
    p.add_argument("--version", action="version",
                   version=f"alpminer {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="create alpminer.toml and data dirs")
    sp.add_argument("--email", help="contact email for API polite pools")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("harvest", help="fetch article metadata from OpenAlex")
    sp.add_argument("--max", type=int, help="harvest at least N records this "
                    "run (stops at the next page boundary)")
    sp.add_argument("--query", help="override the config query for this run")
    sp.set_defaults(func=cmd_harvest)

    sp = sub.add_parser("download", help="auto-download open-access PDFs")
    sp.add_argument("--limit", type=int, help="process at most N papers")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("manual", help="manual-download queue")
    sp.add_argument("manual_cmd", choices=["list", "ingest", "open", "watch",
                                           "remove", "restore"])
    sp.add_argument("ids", nargs="*", metavar="paper_id",
                    help="paper id(s) for `remove` / `restore`")
    sp.add_argument("--all", action="store_true",
                    help="with `restore`: bring back every removed paper")
    sp.add_argument("--n", type=int, default=10,
                    help="tabs to open with `manual open` (default 10)")
    sp.add_argument("--dir", dest="directory", action="append",
                    help="extra folder to ingest/watch (repeatable), e.g. "
                         "your browser's Downloads folder; watched folders "
                         "outside the inbox are copy-only")
    sp.add_argument("--include-temp", action="store_true",
                    help="also watch the system temp folder (some PDF "
                         "viewers drop copies there)")
    sp.set_defaults(func=cmd_manual)

    sp = sub.add_parser("add", help="add a PDF not indexed by OpenAlex and "
                        "queue it for extraction")
    sp.add_argument("pdf", help="path to the PDF file")
    sp.add_argument("--title", help="paper title (else read from the PDF)")
    sp.add_argument("--doi", help="DOI (else read from the PDF's first page)")
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("extract", help="LLM triage + recipe extraction")
    sp.add_argument("--limit", type=int, help="process at most N papers")
    sp.add_argument("--refresh", action="store_true",
                    help="ignore cached LLM responses (re-spend tokens)")
    sp.add_argument("--only", help="process a single paper id (any status)")
    sp.set_defaults(func=cmd_extract)

    sp = sub.add_parser("export", help="build the JSON/CSV recipe database")
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("status", help="progress dashboard")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("reset", help="clear harvested data (papers, recipes, "
                        "manual queue) and start the project fresh")
    sp.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    sp.add_argument("--all", action="store_true",
                    help="also delete downloaded PDFs, texts, caches, exports")
    sp.set_defaults(func=cmd_reset)

    sp = sub.add_parser("profiles",
                        help="list extraction profiles or scaffold a new one")
    sp.add_argument("profiles_cmd", choices=["list", "new"])
    sp.add_argument("name", nargs="?", default="",
                    help="profile name (for `new`)")
    sp.set_defaults(func=cmd_profiles)

    sp = sub.add_parser("providers",
                        help="list LLM providers or scaffold a plugin")
    sp.add_argument("providers_cmd", choices=["list", "new"])
    sp.add_argument("name", nargs="?", default="",
                    help="provider name (for `new`)")
    sp.set_defaults(func=cmd_providers)

    sp = sub.add_parser("gui", help="open the web dashboard in your browser")
    sp.add_argument("--port", type=int, default=8642)
    sp.add_argument("--no-browser", action="store_true",
                    help="start the server without opening a browser tab")
    sp.set_defaults(func=cmd_gui)

    sp = sub.add_parser("run", help="all stages end to end (resumable)")
    sp.add_argument("--limit", type=int,
                    help="cap records per stage (recommended for a pilot)")
    sp.add_argument("--skip-harvest", action="store_true")
    sp.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    import requests
    from . import profiles as _p
    from .providers import LLMError as _L
    from .utils import RetryError

    args = build_parser().parse_args(argv)
    # Most specific first. A single ordered chain -- an earlier `except` that
    # re-raised would escape the whole try, so the fatal handler below must be
    # the last clause, not preceded by a broad `except Exception`.
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted -- progress is saved; re-run to continue.",
              file=sys.stderr)
        return 130
    except config.ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except (_p.ProfileError, _L) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (RetryError, requests.RequestException) as exc:
        print(f"network error: {exc}\nThis is usually a transient rate limit "
              "(HTTP 429) or a connectivity problem. Wait a few minutes and "
              "re-run; completed work is saved.", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort fatal handler
        log.exception("fatal error")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
