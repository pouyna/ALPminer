"""Local web dashboard: `alpminer gui`.

A zero-dependency (stdlib http.server) control panel bound to 127.0.0.1.
Pipeline stages run as subprocesses of the real CLI, so the GUI exercises
exactly the same code paths as the terminal, inherits all resume guarantees,
and "Stop" can safely terminate a job (every paper commits independently).
In-process pieces (inbox watcher, manual ingest, tab opening) log through the
same buffer, so everything appears in one console strip.

API keys entered in the GUI are held in this process's environment only;
they are never written to disk and never echoed back by any endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import webbrowser
from collections import Counter, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import __version__, config, db, manual, profiles
from .utils import log

JOB_PREFIX = [sys.executable, "-m", "alpminer.cli"]
DEFAULT_PORT = 8642

# API keys go into environment variables named like ANTHROPIC_API_KEY; accept
# any provider's key-var name that looks like one.
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def _kill_process_tree(proc: "subprocess.Popen") -> bool:
    """Forcefully end a job subprocess and any children, and VERIFY it died.
    On Windows a plain terminate() only kills the direct child, so a job
    blocked in a network call or a retry sleep can seem unstoppable;
    taskkill /F /T reaps the whole tree. Falls back to terminate()/kill()
    whenever taskkill fails or the process is somehow still alive."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
            proc.wait(timeout=5)
            return True
        except Exception:  # noqa: BLE001 - fall through to terminate()
            pass
    try:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return proc.poll() is not None
    except Exception:  # noqa: BLE001
        pass
    return False

JOB_SPECS = {
    "harvest": lambda p: ["harvest"] + _opt("--max", p.get("max")),
    "download": lambda p: ["download"] + _opt("--limit", p.get("limit")),
    "extract": lambda p: (["extract"] + _opt("--limit", p.get("limit"))
                          + (["--refresh"] if p.get("refresh") else [])
                          + _opt("--only", p.get("only"))),
    "export": lambda p: ["export"],
    "run": lambda p: (["run"] + _opt("--limit", p.get("limit"))
                      + (["--skip-harvest"] if p.get("skip_harvest") else [])),
}


def _opt(flag: str, value) -> list[str]:
    return [flag, str(value)] if value not in (None, "", False) else []


# ---- shared state -----------------------------------------------------------------

class LogBuffer:
    """Thread-safe ring buffer of console lines with monotonically
    increasing ids so the browser can poll incrementally."""

    def __init__(self, maxlen: int = 3000):
        self._lines: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._next = 1
        self._lock = threading.Lock()

    def add(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._lines.append((self._next, f"{stamp}  {text}"))
            self._next += 1

    def since(self, last_id: int) -> tuple[int, list[dict]]:
        with self._lock:
            lines = [{"i": i, "t": t} for i, t in self._lines if i > last_id]
            return self._next - 1, lines


class BufferHandler(logging.Handler):
    def __init__(self, buffer: LogBuffer):
        super().__init__(level=logging.INFO)
        self.buffer = buffer

    def emit(self, record):  # noqa: D102
        try:
            self.buffer.add(self.format(record))
        except Exception:  # noqa: BLE001 - logging must never crash the app
            pass


class JobBusy(RuntimeError):
    pass


class JobRunner:
    """Runs one CLI subprocess at a time, pumping its output to the buffer."""

    def __init__(self, buffer: LogBuffer, cwd: Path):
        self.buffer = buffer
        self.cwd = cwd
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self.info: dict | None = None
        self.last: dict | None = None

    def running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self, name: str, cli_args: list[str]) -> dict:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                raise JobBusy(f"a job is already running: {self.info['name']}")
            proc = subprocess.Popen(
                JOB_PREFIX + cli_args, cwd=self.cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, encoding="utf-8", errors="replace",
                env=os.environ.copy(),
            )
            self._proc = proc
            self.info = {"name": name, "args": cli_args,
                         "started": time.time(), "pid": proc.pid}
        self.buffer.add(f"=== {name} started ({' '.join(cli_args)})")
        threading.Thread(target=self._pump, args=(proc, name),
                         daemon=True).start()
        return dict(self.info)

    def _pump(self, proc: subprocess.Popen, name: str) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if line.rstrip():
                self.buffer.add(line.rstrip())
        code = proc.wait()
        self.buffer.add(f"=== {name} finished (exit {code})")
        with self._lock:
            self.last = {"name": name, "code": code, "finished": time.time()}
            self._proc = None
            self.info = None

    def stop(self) -> bool:
        with self._lock:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return False
        self.buffer.add("Stop requested; ending the current job. Completed "
                        "papers are already saved.")
        if _kill_process_tree(proc):
            return True
        self.buffer.add("WARNING: the job process could not be confirmed "
                        f"dead (pid {proc.pid}); end it from Task Manager if "
                        "it lingers.")
        return False


class WatchController:
    """Background watcher over one or more folders (see manual.watch_dirs).
    The project inbox is moved-from; every other folder is copy-only."""

    def __init__(self, buffer: LogBuffer, base_dir: Path):
        self.buffer = buffer
        self.base_dir = base_dir
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.dirs: list[str] = []
        self.include_temp: bool = False

    def on(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, dirs: list[str] | None, include_temp: bool) -> None:
        if self.on():
            return
        # A fresh Event per run: a quick off->on cycle must not hand the new
        # loop an Event the just-stopped loop is still watching (that would
        # leave both threads scanning). Each loop watches the Event it was
        # started with.
        self._stop = threading.Event()
        self.dirs = [d for d in (dirs or []) if str(d).strip()]
        self.include_temp = bool(include_temp)
        self._thread = threading.Thread(
            target=self._loop, args=(self._stop,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread = None
        self.buffer.add("Auto-file turned off.")

    def _loop(self, stop: threading.Event) -> None:
        cfg = config.load(self.base_dir / config.CONFIG_FILENAME)
        folders = manual.watch_dirs(cfg, self.dirs, self.include_temp)
        self.buffer.add("Auto-file on: watching "
                        + ", ".join(str(d) for d in folders))
        reported: set[tuple[str, str]] = set()
        seen: set = set()
        while not stop.is_set():
            try:
                conn = db.connect(cfg.db_path)
                try:
                    filed = 0
                    for directory in folders:
                        res = manual._scan(conn, cfg, directory,
                                           skip_fresh=True, seen=seen)
                        filed += len(res["matched"])
                        for name, why in res["unmatched"] + res["bad"]:
                            if (name, why) not in reported:
                                reported.add((name, why))
                                self.buffer.add(f"unfiled: {name} ({why})")
                    if filed:
                        remaining = len(manual.queue(conn))
                        self.buffer.add(
                            f"-> {remaining} remaining in the manual queue")
                finally:
                    conn.close()
            except Exception as exc:  # noqa: BLE001 - keep watching
                self.buffer.add(f"auto-file error: {exc}")
            stop.wait(2.0)


class App:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.buffer = LogBuffer()
        self.runner = JobRunner(self.buffer, base_dir)
        self.watch = WatchController(self.buffer, base_dir)
        self.log_handler = BufferHandler(self.buffer)
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(self.log_handler)
        log.setLevel(logging.DEBUG)

    def close(self) -> None:
        log.removeHandler(self.log_handler)

    def job_state(self) -> dict:
        info = self.runner.info
        if self.runner.running() and info:
            return {"running": True, **info}
        return {"running": False, "last": self.runner.last}

    # ---- helpers ----
    @property
    def config_path(self) -> Path:
        return config.default_path(self.base_dir)

    def load_cfg(self) -> config.Config:
        return config.load(self.config_path)

    def open_conn(self, cfg):
        return db.connect(cfg.db_path)


# ---- request handling ---------------------------------------------------------------

def _config_dict(cfg: config.Config) -> dict:
    keys = ("email", "profile", "query", "from_year", "to_year", "data_dir",
            "provider", "provider_settings", "models",
            "triage_enabled", "triage_chars",
            "max_paper_chars", "max_output_tokens", "extract_workers",
            "ocr_enabled", "ocr_mode",
            "request_delay_s", "download_timeout_s", "max_pdf_mb")
    out = {k: getattr(cfg, k) for k in keys}
    # ship effective pairs (defaults overlaid) so the GUI always shows the
    # models that would actually be used, for every provider equally
    effective = {p: cfg.models_for(p) for p in config.DEFAULT_MODELS}
    for p in cfg.models:
        effective[p] = cfg.models_for(p)
    out["models"] = effective
    return out


def make_handler(app: App):
    index_html = (resources.files("alpminer") / "static" / "index.html"
                  ).read_text(encoding="utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # silence per-request stderr noise
            pass

        # ---- plumbing ----
        def _json(self, payload, code: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n).decode("utf-8"))
            except json.JSONDecodeError:
                return {}

        def _query(self) -> dict:
            return {k: v[0] for k, v in
                    parse_qs(urlparse(self.path).query).items()}

        def _same_origin(self) -> bool:
            """Refuse cross-origin / DNS-rebinding POSTs. The server binds to
            127.0.0.1, so a request whose Host or Origin names any other host
            was not issued by a page this server served -- this blocks a
            malicious web page from POSTing to /api/keys or /api/job."""
            def _local(netloc: str) -> bool:
                host = netloc.rsplit(":", 1)[0].strip("[]").lower()
                return host in ("127.0.0.1", "localhost", "::1", "")
            if not _local(self.headers.get("Host", "")):
                return False
            origin = self.headers.get("Origin")
            if origin and not _local(urlparse(origin).netloc):
                return False
            return True

        # ---- GET ----
        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    body = index_html.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/state":
                    self._json(self.state())
                elif path == "/api/logs":
                    last = int(self._query().get("since", 0))
                    next_id, lines = app.buffer.since(last)
                    self._json({"next": next_id, "lines": lines})
                elif path == "/api/manual":
                    self._json(self.manual_list())
                elif path == "/api/manual/removed":
                    self._json(self.manual_removed())
                elif path == "/api/profiles":
                    self._json({"profiles": self.profiles_list()})
                elif path == "/api/providers":
                    self._json(self.providers_list())
                elif path == "/api/keys":
                    self._json(self.key_status())
                elif path == "/api/profile":
                    self._json(self.profile_detail())
                elif path == "/api/recipes":
                    self._json(self.recipes())
                elif path == "/api/recipes.csv":
                    self.recipes_csv()
                elif path == "/api/recipes/stats":
                    self._json(self.recipe_stats())
                else:
                    self._json({"error": "not found"}, 404)
            except config.ConfigError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        # ---- POST ----
        def do_POST(self):  # noqa: N802
            path = urlparse(self.path).path
            if not self._same_origin():
                self._json({"error": "cross-origin request refused"}, 403)
                return
            body = self._body()
            try:
                if path == "/api/init":
                    self.init_project(body)
                elif path == "/api/config":
                    self.save_config(body)
                elif path == "/api/keys":
                    self.set_key(body)
                elif path == "/api/job":
                    self.start_job(body)
                elif path == "/api/job/stop":
                    self._json({"stopped": app.runner.stop()})
                elif path == "/api/manual/open":
                    self.manual_open(body)
                elif path == "/api/manual/ingest":
                    self.manual_ingest(body)
                elif path == "/api/manual/add":
                    self.manual_add(body)
                elif path == "/api/manual/remove":
                    self.manual_remove(body)
                elif path == "/api/manual/restore":
                    self.manual_restore(body)
                elif path == "/api/watch":
                    self.set_watch(body)
                elif path == "/api/config/factory-reset":
                    self.factory_reset(body)
                elif path == "/api/profile":
                    self.save_profile_edits(body)
                elif path == "/api/profiles/create":
                    self.create_profile(body)
                elif path == "/api/profiles/delete":
                    self.delete_profile(body)
                elif path == "/api/reset":
                    self.reset_project(body)
                else:
                    self._json({"error": "not found"}, 404)
            except JobBusy as exc:
                self._json({"error": str(exc)}, 409)
            except config.ConfigError as exc:
                self._json({"error": str(exc)}, 400)
            except Exception as exc:  # noqa: BLE001
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        # ---- endpoint bodies ----
        def state(self) -> dict:
            out = {
                "version": __version__,
                "cwd": str(app.base_dir),
                "config_path": str(app.config_path),
                "initialized": app.config_path.exists(),
                "keys": {"anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
                         "openai": bool(os.environ.get("OPENAI_API_KEY")),
                         "gemini": bool(os.environ.get("GEMINI_API_KEY")
                                        or os.environ.get("GOOGLE_API_KEY"))},
                "job": app.job_state(),
                "watch": {"on": app.watch.on(), "dirs": app.watch.dirs,
                          "include_temp": app.watch.include_temp},
            }
            if not out["initialized"]:
                return out
            cfg = app.load_cfg()
            out["config"] = _config_dict(cfg)
            out["inbox"] = str(cfg.inbox_dir)
            out["exports_dir"] = str(cfg.export_dir)
            try:
                prof = profiles.load(cfg.profile, app.base_dir)
                out["profile"] = {"name": prof.name, "label": prof.label,
                                  "fields": len(prof.fields),
                                  "record_noun": prof.record_noun}
            except profiles.ProfileError as exc:
                out["profile"] = {"name": cfg.profile, "error": str(exc)}
            conn = app.open_conn(cfg)
            try:
                counts = db.counts(conn)
                from .cli import _estimate_pending_tokens
                counts["pending_tokens"] = _estimate_pending_tokens(conn, cfg)
                ds = counts["download_status"]
                counts["manual_count"] = (ds.get("pending", 0)
                                          + ds.get("manual", 0))
                # which profile this database's extractions are locked to
                # (extract.PROFILE_META_KEY); null until the first extraction
                counts["locked_profile"] = db.get_meta(conn, "profile")
                counts["tokens_spent"] = {
                    "input": int(db.get_meta(conn, "usage_input_tokens") or 0),
                    "output": int(db.get_meta(conn, "usage_output_tokens")
                                  or 0),
                    "calls": int(db.get_meta(conn, "usage_llm_calls") or 0),
                }
                out["status"] = counts
            finally:
                conn.close()
            return out

        def init_project(self, body: dict) -> None:
            email = (body.get("email") or "").strip()
            if "@" not in email:
                self._json({"error": "enter a valid contact email"}, 400)
                return
            config.write_template(app.config_path, email=email)
            app.load_cfg().ensure_dirs()
            app.buffer.add(f"Project initialized in {app.base_dir}")
            self._json({"ok": True})

        def save_config(self, body: dict) -> None:
            current = _config_dict(app.load_cfg())
            unknown = set(body) - set(current)
            if unknown:
                self._json({"error": f"unknown settings: {sorted(unknown)}"},
                           400)
                return
            current.update(body)
            cfg = config.save_config(app.config_path, current)
            cfg.ensure_dirs()
            app.buffer.add("Settings saved.")
            self._json({"ok": True, "config": _config_dict(cfg)})

        def set_key(self, body: dict) -> None:
            """Set (or clear, when key is blank) any provider's API key in the
            process environment. Accepts an explicit `env` var name, or a
            `provider` shorthand for the built-in providers. Session-only."""
            env_name = (body.get("env") or "").strip()
            if not env_name:
                env_name = {"anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY",
                            "gemini": "GEMINI_API_KEY"}.get(body.get("provider"),
                                                            "")
            if not _ENV_NAME_RE.match(env_name):
                self._json({"error": "invalid environment-variable name "
                                     "(expected e.g. OPENAI_API_KEY)"}, 400)
                return
            key = (body.get("key") or "").strip()
            if key:
                os.environ[env_name] = key
                app.buffer.add(f"{env_name} set for this session (not saved "
                               "to disk).")
            else:
                os.environ.pop(env_name, None)
                app.buffer.add(f"{env_name} cleared for this session.")
            self._json({"ok": True, "env": env_name, "set": bool(key)})

        def key_status(self) -> dict:
            """Whether a given key env var is set (boolean only; never the
            value). GEMINI_API_KEY also honours the GOOGLE_API_KEY fallback."""
            env = (self._query().get("env") or "").strip()
            if not _ENV_NAME_RE.match(env):
                return {"env": env, "set": False}
            names = [env] + (["GOOGLE_API_KEY"] if env == "GEMINI_API_KEY"
                             else [])
            return {"env": env, "set": any(os.environ.get(n) for n in names)}

        def start_job(self, body: dict) -> None:
            name = body.get("name")
            spec = JOB_SPECS.get(name)
            if spec is None:
                self._json({"error": f"unknown job {name!r}"}, 400)
                return
            if not app.config_path.exists():
                self._json({"error": "initialize the project first"}, 400)
                return
            info = app.runner.start(name, spec(body.get("params") or {}))
            self._json({"ok": True, "job": info})

        def manual_list(self) -> dict:
            q = self._query()
            try:
                limit = min(200, max(1, int(q.get("limit", 50))))
            except (TypeError, ValueError):
                limit = 50
            try:
                offset = max(0, int(q.get("offset", 0)))
            except (TypeError, ValueError):
                offset = 0
            term = (q.get("q") or "").strip()
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                total = manual.queue_count(conn, term)
                rows = manual.queue_page(conn, offset, limit, term)
                items = [{
                    "id": p["id"], "title": p["title"], "year": p["year"],
                    "journal": p["journal"],
                    "url": (f"https://doi.org/{p['doi']}" if p["doi"]
                            else (p["landing_url"] or "")),
                    "status": p["download_status"],
                    "why": (p["download_error"]
                            or ("not tried yet; drop a PDF or run download"
                                if p["download_status"] == "pending" else "")),
                } for p in rows]
                return {"count": total, "inbox": str(cfg.inbox_dir),
                        "items": items, "offset": offset, "limit": limit,
                        "removed_count": manual.removed_count(conn)}
            finally:
                conn.close()

        def manual_removed(self) -> dict:
            """The soft-deleted ('removed') papers, so the user can restore
            them. Capped at 200; the count is the true total."""
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                rows = manual.removed_list(conn, limit=200)
                items = [{
                    "id": p["id"], "title": p["title"], "year": p["year"],
                    "journal": p["journal"],
                    "url": (f"https://doi.org/{p['doi']}" if p["doi"]
                            else (p["landing_url"] or "")),
                } for p in rows]
                return {"count": manual.removed_count(conn), "items": items}
            finally:
                conn.close()

        def manual_open(self, body: dict) -> None:
            n = max(1, min(25, int(body.get("n") or 10)))
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                opened = manual.open_queue(conn, cfg, n=n)
            finally:
                conn.close()
            app.buffer.add(f"Opened {opened} browser tab(s) from the manual "
                           "queue.")
            self._json({"opened": opened})

        def manual_ingest(self, body: dict) -> None:
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                res = manual.ingest_inbox(conn, cfg,
                                          directory=body.get("dir") or None)
            finally:
                conn.close()
            self._json({"filed": len(res["matched"]),
                        "unmatched": len(res["unmatched"]),
                        "invalid": len(res["bad"]),
                        "duplicate": len(res["duplicate"])})

        def manual_add(self, body: dict) -> None:
            """Add a PDF that is not in OpenAlex (by local path) so it enters
            the pipeline at extraction."""
            path = (body.get("path") or "").strip()
            if not path:
                self._json({"error": "enter the path to a PDF file"}, 400)
                return
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                info = manual.add_external(
                    conn, cfg, path, title=(body.get("title") or None),
                    doi=(body.get("doi") or None))
            except (FileNotFoundError, ValueError) as exc:
                self._json({"error": str(exc)}, 400)
                return
            finally:
                conn.close()
            self._json({"ok": True, **info})

        def manual_remove(self, body: dict) -> None:
            """Take one paper ({"id": ...}) or several ({"ids": [...]}) out
            of the manual queue (soft-delete)."""
            ids = body.get("ids")
            if not isinstance(ids, list):
                ids = [body.get("id")]
            ids = [str(p).strip() for p in ids if p and str(p).strip()]
            if not ids:
                self._json({"error": "no paper id given"}, 400)
                return
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                removed = sum(manual.remove_from_queue(conn, pid)
                              for pid in ids)
            finally:
                conn.close()
            if not removed:
                self._json({"error": "none of the given papers are in the "
                                     "queue"}, 400)
                return
            app.buffer.add(f"Removed {removed} paper(s) from the queue "
                           "(restore them from Removed papers).")
            self._json({"ok": True, "removed": removed})

        def manual_restore(self, body: dict) -> None:
            """Bring a removed paper (or all of them) back into the queue."""
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                if body.get("all"):
                    n = manual.restore_all(conn)
                    app.buffer.add(f"Restored {n} paper(s) to the queue.")
                    self._json({"ok": True, "restored": n})
                    return
                pid = (body.get("id") or "").strip()
                ok = manual.restore_to_queue(conn, pid)
            finally:
                conn.close()
            if not ok:
                self._json({"error": f"{pid!r} is not a removed paper"}, 400)
                return
            app.buffer.add(f"Restored {pid} to the queue.")
            self._json({"ok": True, "restored": 1})

        def factory_reset(self, body: dict) -> None:
            """Restore alpminer.toml to the shipped defaults (keeping the
            contact email and data_dir so the project keeps working) and revert
            any GUI-edited built-in extraction rules to their defaults.
            Harvested papers, PDFs, recipes, and exports are NOT touched."""
            if not app.config_path.exists():
                self._json({"error": "initialize the project first"}, 400)
                return
            if app.runner.running():
                self._json({"error": "stop the running job before resetting "
                                     "settings"}, 409)
                return
            cfg = app.load_cfg()
            config.save_config(app.config_path,
                               {"email": cfg.email, "data_dir": cfg.data_dir})
            reverted = []
            pdir = profiles.project_profile_dir(app.base_dir)
            for name in profiles.builtin_names():
                f = pdir / f"{name}.toml"
                if f.exists():
                    f.unlink()
                    reverted.append(name)
            cfg = app.load_cfg()
            msg = "Settings restored to factory defaults"
            if reverted:
                msg += (f"; extraction rules for {', '.join(reverted)} "
                        "reverted too")
            app.buffer.add(msg + ". Your papers and data were not touched.")
            self._json({"ok": True, "config": _config_dict(cfg),
                        "reverted": reverted})

        def set_watch(self, body: dict) -> None:
            if body.get("on"):
                app.watch.start(body.get("dirs") or None,
                                bool(body.get("include_temp")))
            else:
                app.watch.stop()
            self._json({"on": app.watch.on()})

        @staticmethod
        def _recipe_where(term: str) -> tuple[str, list]:
            if not term:
                return "1=1", []
            like = f"%{term}%"
            return ("(r.data LIKE ? OR p.title LIKE ? "
                    "OR p.doi LIKE ? OR p.journal LIKE ?)",
                    [like, like, like, like])

        def recipes(self) -> dict:
            q = self._query()
            term = (q.get("q") or "").strip()
            limit = max(1, min(200, int(q.get("limit", 50))))
            offset = max(0, int(q.get("offset", 0)))
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                where, params = self._recipe_where(term)
                total = conn.execute(
                    f"SELECT COUNT(*) c FROM recipes r "
                    f"JOIN papers p ON p.id = r.paper_id WHERE {where}",
                    params).fetchone()["c"]
                rows = conn.execute(
                    f"SELECT r.id, r.data, p.id AS paper_id, p.doi, p.title, "
                    f"p.year, p.journal, p.text_ocr FROM recipes r "
                    f"JOIN papers p ON p.id = r.paper_id WHERE {where} "
                    f"ORDER BY r.id DESC LIMIT ? OFFSET ?",
                    params + [limit, offset]).fetchall()
                items = [{
                    "id": r["id"], "paper_id": r["paper_id"],
                    "doi": r["doi"], "title": r["title"],
                    "year": r["year"], "journal": r["journal"],
                    "ocr": bool(r["text_ocr"]),
                    "recipe": json.loads(r["data"]),
                } for r in rows]
                return {"total": total, "items": items,
                        "offset": offset, "limit": limit}
            finally:
                conn.close()

        def recipes_csv(self) -> None:
            """Download the CURRENT recipe search as CSV -- every match, not
            one page. Columns: paper metadata + the active profile's fields."""
            import csv
            import io
            term = (self._query().get("q") or "").strip()
            cfg = app.load_cfg()
            try:
                field_names = profiles.load(cfg.profile,
                                            app.base_dir).field_names()
            except profiles.ProfileError:
                field_names = []
            conn = app.open_conn(cfg)
            try:
                where, params = self._recipe_where(term)
                rows = conn.execute(
                    f"SELECT r.data, p.id AS paper_id, p.doi, p.title, "
                    f"p.year, p.journal, p.text_ocr FROM recipes r "
                    f"JOIN papers p ON p.id = r.paper_id WHERE {where} "
                    f"ORDER BY r.id DESC", params).fetchall()
            finally:
                conn.close()

            def cell(v):
                if v is None:
                    return ""
                if isinstance(v, list):
                    return "; ".join(str(x) for x in v)
                return v

            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["paper_id", "doi", "title", "year", "journal", "ocr",
                        *field_names])
            for r in rows:
                rec = json.loads(r["data"])
                w.writerow([r["paper_id"], r["doi"], r["title"], r["year"],
                            r["journal"], bool(r["text_ocr"]),
                            *[cell(rec.get(f)) for f in field_names]])
            # BOM so Excel opens the UTF-8 correctly on double-click
            body = buf.getvalue().encode("utf-8-sig")
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", term)[:40] or "all"
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition",
                             f'attachment; filename="recipes_{slug}.csv"')
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def recipe_stats(self) -> dict:
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                mats = Counter()
                for row in conn.execute(
                        "SELECT data FROM recipes LIMIT 20000"):
                    try:
                        mats[json.loads(row["data"]).get("material",
                                                          "?")] += 1
                    except json.JSONDecodeError:
                        pass
                return {"top_materials": mats.most_common(12)}
            finally:
                conn.close()

        # ---- providers & profile editing ----
        def providers_list(self) -> dict:
            from . import providers as _prov
            plugin_dir = app.base_dir / _prov.PLUGIN_DIR_NAME
            plugins = (sorted(p.stem for p in plugin_dir.glob("*.py"))
                       if plugin_dir.is_dir() else [])
            return {"builtin": ["anthropic", "openai", "gemini",
                                "openai_compatible"],
                    "plugins": plugins}

        def profile_detail(self) -> dict:
            """The active profile's editable query and prompts, for the
            Settings tab's extraction-rules editor."""
            cfg = app.load_cfg()
            try:
                prof = profiles.load(cfg.profile, app.base_dir)
            except profiles.ProfileError as exc:
                return {"name": cfg.profile, "error": str(exc)}
            proj = (profiles.project_profile_dir(app.base_dir)
                    / f"{prof.name}.toml")
            return {"name": prof.name, "label": prof.label,
                    "record_noun": prof.record_noun,
                    "default_query": prof.default_query,
                    "triage_prompt": prof.triage_prompt,
                    "extraction_prompt": prof.extraction_prompt,
                    "n_fields": len(prof.fields),
                    "field_names": prof.field_names(),
                    "fields": [{"name": f.name, "type": f.type,
                                "required": f.required,
                                "description": f.description,
                                "minimum": f.minimum, "maximum": f.maximum,
                                "max_len": f.max_len} for f in prof.fields],
                    "is_project_copy": proj.exists()}

        def profiles_list(self) -> list[dict]:
            """All profiles, each marked `deletable`: only custom (project)
            profiles whose name is not a built-in can be removed. A project
            copy shadowing a built-in name reverts via factory reset, not
            delete, so the list never loses the built-ins."""
            builtins = set(profiles.builtin_names())
            out = profiles.list_profiles(app.base_dir)
            for p in out:
                p["deletable"] = (p["origin"] != "built-in"
                                  and p["name"] not in builtins)
            return out

        def delete_profile(self, body: dict) -> None:
            """Delete a custom project profile. Built-in profiles (and
            project copies shadowing their names) are refused, as is the
            profile the project is currently using."""
            name = (body.get("name") or "").strip()
            if not profiles._NAME_RE.match(name):
                self._json({"error": f"invalid profile name {name!r}"}, 400)
                return
            if name in profiles.builtin_names():
                self._json({"error": f"{name!r} is a built-in profile and "
                                     "cannot be deleted. (To revert edits to "
                                     "it, use Reset to factory settings.)"},
                           400)
                return
            cfg = app.load_cfg()
            if cfg.profile == name:
                self._json({"error": f"{name!r} is the project's active "
                                     "profile; switch to another profile "
                                     "(and save) before deleting it"}, 400)
                return
            dest = profiles.project_profile_dir(app.base_dir) / f"{name}.toml"
            if not dest.exists():
                self._json({"error": f"no custom profile named {name!r}"},
                           404)
                return
            dest.unlink()
            app.buffer.add(f"Deleted custom extraction profile '{name}'.")
            self._json({"ok": True, "name": name})

        def create_profile(self, body: dict) -> None:
            """The Settings dropdown's "Create new profile" flow: make a new
            project profile named by the user -- either a commented starter
            template (from scratch) or a full copy of an existing profile --
            and switch the project to it. Built-ins are never modified."""
            name = (body.get("name") or "").strip()
            copy_from = (body.get("copy_from") or "").strip()
            if not profiles._NAME_RE.match(name):
                self._json({"error": f"invalid profile name {name!r} "
                                     "(use snake_case, e.g. my_ald)"}, 400)
                return
            if name in profiles.builtin_names():
                self._json({"error": f"{name!r} is a built-in profile name; "
                                     "pick a different one"}, 400)
                return
            dest = profiles.project_profile_dir(app.base_dir) / f"{name}.toml"
            if dest.exists():
                self._json({"error": f"a profile named {name!r} already "
                                     "exists in this project"}, 400)
                return

            try:
                if copy_from:
                    src = profiles.load(copy_from, app.base_dir)
                    src.name = name
                    profiles.write_profile(app.base_dir, src)
                    profiles.load(name, app.base_dir)   # validate round-trip
                else:
                    profiles.write_new_profile(app.base_dir, name)
            except profiles.ProfileError as exc:
                dest.unlink(missing_ok=True)
                self._json({"error": str(exc)}, 400)
                return

            cfg = app.load_cfg()
            current = _config_dict(cfg)
            current["profile"] = name
            config.save_config(app.config_path, current)
            origin = (f"copied from '{copy_from}'" if copy_from
                      else "from the starter template")
            app.buffer.add(f"New extraction profile '{name}' created "
                           f"({origin}) and selected. Edit its rules below.")
            self._json({"ok": True, "name": name,
                        "copied_from": copy_from or None})

        def save_profile_edits(self, body: dict) -> None:
            """Save an edited default query / triage / extraction prompt as a
            project profile that shadows the active one, or -- when the body
            carries `new_name` -- as a NEW project profile under that name
            (copy-from-built-in flow); the config then switches to it.
            Reverts on a bad round-trip so a typo never leaves an unloadable
            profile behind."""
            cfg = app.load_cfg()
            prof = profiles.load(cfg.profile, app.base_dir)

            new_name = (body.get("new_name") or "").strip()
            if new_name:
                if not profiles._NAME_RE.match(new_name):
                    self._json({"error": f"invalid profile name {new_name!r} "
                                         "(use snake_case, e.g. my_ald)"}, 400)
                    return
                if new_name in profiles.builtin_names():
                    self._json({"error": f"{new_name!r} is a built-in profile "
                                         "name; pick a different one"}, 400)
                    return
                if (profiles.project_profile_dir(app.base_dir)
                        / f"{new_name}.toml").exists():
                    self._json({"error": f"a profile named {new_name!r} "
                                         "already exists in this project"},
                               400)
                    return
                prof.name = new_name
            for key in ("default_query", "triage_prompt", "extraction_prompt"):
                if isinstance(body.get(key), str):
                    setattr(prof, key, body[key].strip())
            if isinstance(body.get("fields"), list):
                specs = []
                for f in body["fields"]:
                    if not isinstance(f, dict):
                        continue
                    name = str(f.get("name") or "").strip()
                    if not name:
                        continue
                    specs.append(profiles.FieldSpec(
                        name=name, type=str(f.get("type") or "string"),
                        description=str(f.get("description") or "").strip(),
                        required=bool(f.get("required")),
                        minimum=f.get("minimum"), maximum=f.get("maximum"),
                        max_len=f.get("max_len")))
                prof.fields = specs
            if not prof.triage_prompt or not prof.extraction_prompt:
                self._json({"error": "the triage and extraction prompts "
                                     "cannot be empty"}, 400)
                return
            if not prof.fields:
                self._json({"error": "at least one extracted field is "
                                     "required"}, 400)
                return
            dest = (profiles.project_profile_dir(app.base_dir)
                    / f"{prof.name}.toml")
            previous = (dest.read_text(encoding="utf-8")
                        if dest.exists() else None)
            profiles.write_profile(app.base_dir, prof)
            try:
                profiles.load(prof.name, app.base_dir)  # validate round-trip
            except profiles.ProfileError as exc:
                from .utils import atomic_write_text
                if previous is not None:
                    atomic_write_text(dest, previous)
                else:
                    dest.unlink(missing_ok=True)
                self._json({"error": f"profile did not validate: {exc}"}, 400)
                return
            if new_name:
                # switch the project to the freshly created profile
                current = _config_dict(cfg)
                current["profile"] = new_name
                config.save_config(app.config_path, current)
                app.buffer.add(f"New extraction profile '{new_name}' created "
                               "and selected. (If this project already holds "
                               "another profile's data, use the data-folder "
                               "switch shown in Settings.)")
                self._json({"ok": True, "name": new_name, "switched": True,
                            "is_project_copy": True})
                return
            app.buffer.add(f"Extraction profile '{prof.name}' saved to the "
                           "project (it now shadows the built-in).")
            self._json({"ok": True, "is_project_copy": True})

        def reset_project(self, body: dict) -> None:
            """Clear the database (papers, recipes, manual queue, harvest
            checkpoint, profile lock). With {"files": true}, also delete the
            downloaded PDFs, texts, caches, and exports on disk."""
            if app.runner.running():
                self._json({"error": "stop the running job before resetting"},
                           409)
                return
            cfg = app.load_cfg()
            conn = app.open_conn(cfg)
            try:
                db.reset(conn)
            finally:
                conn.close()
            if body.get("files"):
                from .cli import _wipe_data_files
                _wipe_data_files(cfg)
            app.buffer.add("Project data reset -- database cleared"
                           + (" and downloaded files removed."
                              if body.get("files") else "."))
            self._json({"ok": True})

    return Handler


# ---- entry points -------------------------------------------------------------------

class _ExclusiveServer(ThreadingHTTPServer):
    """On Windows, the default SO_REUSEADDR lets a second dashboard bind the
    SAME port and silently steal new connections from the first one: the
    browser then watches one server's job while Stop clicks land on the
    other, which has nothing to stop. Exclusive binding makes the second
    launch fail loudly instead, so exactly one dashboard owns the port."""
    allow_reuse_address = os.name != "nt"


def create_server(base_dir: Path, port: int = 0) -> tuple[ThreadingHTTPServer, App]:
    app = App(base_dir)
    server = _ExclusiveServer(("127.0.0.1", port), make_handler(app))
    return server, app


def serve(base_dir: Path, port: int = DEFAULT_PORT,
          open_browser: bool = True) -> int:
    try:
        server, app = create_server(base_dir, port)
    except OSError:
        url = f"http://127.0.0.1:{port}/"
        print(f"An ALPminer dashboard is already running at {url} -- "
              "opening it instead of starting a second one.\n"
              "(To run another instance, use `alpminer gui --port <other>`.)")
        if open_browser:
            webbrowser.open_new_tab(url)
        return 0
    actual_port = server.server_address[1]
    url = f"http://127.0.0.1:{actual_port}/"
    app.buffer.add(f"alpminer {__version__} dashboard at {url}")
    print(f"alpminer dashboard: {url}   (Ctrl-C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open_new_tab(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped. Pipeline progress is saved.")
    finally:
        server.shutdown()
    return 0
