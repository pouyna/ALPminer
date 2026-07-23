"""Real-browser smoke test: boots the dashboard and renders it in headless
Edge/Chrome, proving the page's JavaScript actually executes (fetches
/api/state, fills the gauges) -- the kind of frontend regression the Python
suite alone cannot catch. Skipped when no Chromium-based browser is present.
"""

import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from alpminer import __version__, config, db, gui
from tests.conftest import make_paper

BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "msedge", "chrome", "google-chrome", "chromium",
)


def _find_browser() -> str | None:
    for cand in BROWSER_CANDIDATES:
        if Path(cand).is_file():
            return cand
        found = shutil.which(cand)
        if found:
            return found
    return None


BROWSER = _find_browser()


@pytest.mark.skipif(BROWSER is None,
                    reason="no Chromium-based browser found for the smoke test")
def test_dashboard_renders_in_a_real_browser(tmp_path):
    cfg_path = tmp_path / config.CONFIG_FILENAME
    config.write_template(cfg_path, email="smoke@example.edu")
    cfg = config.load(cfg_path)
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    db.upsert_paper(conn, make_paper(1))
    db.upsert_paper(conn, make_paper(2))
    conn.close()

    httpd, app = gui.create_server(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    try:
        # a dedicated profile dir keeps this run independent of any Edge or
        # Chrome window the user has open (otherwise the launch delegates to
        # the running instance and returns an empty DOM)
        proc = subprocess.run(
            [BROWSER, "--headless=new", "--disable-gpu", "--dump-dom",
             "--no-first-run",
             f"--user-data-dir={tmp_path / 'browser-profile'}",
             "--virtual-time-budget=5000", url],
            capture_output=True, timeout=90)
        dom = proc.stdout.decode("utf-8", errors="replace")
    finally:
        httpd.shutdown()
        app.close()

    if not dom.strip():
        # headless Chromium yields rc=0 with no output when the desktop
        # browser is mid-update or otherwise refuses headless; that is an
        # environment problem, not a dashboard regression
        pytest.skip("headless browser produced no output; smoke test "
                    "cannot run in this environment right now")

    # JS executed: the version chip and gauges are filled by renderState()
    # from a live /api/state fetch; none of these exist in the raw HTML.
    assert f"v{__version__}" in dom
    assert '<b id="g-papers">2</b>' in dom          # both seeded papers
    assert '<b id="g-queue">2</b>' in dom           # awaiting a PDF
    assert "Manual queue" in dom and "About" in dom  # tabs rendered
