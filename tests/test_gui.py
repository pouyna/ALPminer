import json
import sys
import threading
import time
from pathlib import Path

import pytest
import requests

from alpminer import config, db, gui, manual
from tests.conftest import make_paper

TIMEOUT = 10


def _wait(predicate, timeout=TIMEOUT, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def server(tmp_path):
    """A live dashboard server over an initialized, seeded project."""
    cfg_path = tmp_path / config.CONFIG_FILENAME
    config.write_template(cfg_path, email="test@example.edu")
    cfg = config.load(cfg_path)
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    # one extracted paper with two recipes, one paper in the manual queue
    done = make_paper(1, doi="10.1/gui1")
    db.upsert_paper(conn, done)
    db.replace_recipes(conn, done["id"], [
        {"material": "TiO2", "metal_precursor_abbrev": "TDMAT",
         "deposition_temperature_c": 200, "confidence": 0.9},
        {"material": "Al2O3", "metal_precursor_abbrev": "TMA"},
    ])
    db.set_fields(conn, done["id"], download_status="downloaded",
                  extract_status="done")
    man = make_paper(2, doi="10.1/gui2")
    db.upsert_paper(conn, man)
    db.set_fields(conn, man["id"], download_status="manual",
                  download_error="no open-access copy found")
    conn.close()

    httpd, app = gui.create_server(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, app, tmp_path
    httpd.shutdown()
    app.close()


@pytest.fixture
def bare_server(tmp_path):
    """A live server over an EMPTY folder (no alpminer.toml)."""
    httpd, app = gui.create_server(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield url, app, tmp_path
    httpd.shutdown()
    app.close()


def test_serves_the_dashboard_page(server):
    url, _, _ = server
    r = requests.get(url + "/", timeout=TIMEOUT)
    assert r.status_code == 200
    assert "ALPminer console" in r.text


def test_state_reports_counts_config_and_manual_queue(server):
    url, _, _ = server
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["initialized"] is True
    assert s["status"]["papers"] == 2
    assert s["status"]["recipes"] == 2
    assert s["status"]["manual_count"] == 1
    assert s["config"]["provider"] == "anthropic"
    assert s["job"]["running"] is False


def test_uninitialized_state_and_init_flow(bare_server):
    url, _, tmp = bare_server
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["initialized"] is False
    r = requests.post(url + "/api/init", json={"email": "nope"},
                      timeout=TIMEOUT)
    assert r.status_code == 400
    r = requests.post(url + "/api/init", json={"email": "a@b.edu"},
                      timeout=TIMEOUT)
    assert r.status_code == 200
    assert (tmp / config.CONFIG_FILENAME).exists()
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["initialized"] is True and "status" in s


def test_config_save_roundtrip_preserves_quoted_query(server):
    url, _, tmp = server
    r = requests.post(url + "/api/config",
                      json={"provider": "gemini", "triage_chars": 12345,
                            "from_year": 2015,
                            "query": '"atomic layer etching" OR "atomic layer etch"'},
                      timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    cfg = config.load(tmp / config.CONFIG_FILENAME)
    assert cfg.provider == "gemini"
    assert cfg.triage_chars == 12345
    assert cfg.from_year == 2015
    assert cfg.query == '"atomic layer etching" OR "atomic layer etch"'
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["config"]["provider"] == "gemini"


def test_config_rejects_invalid_provider_and_restores_file(server):
    url, _, tmp = server
    before = (tmp / config.CONFIG_FILENAME).read_text()
    r = requests.post(url + "/api/config", json={"provider": "Not Valid!"},
                      timeout=TIMEOUT)
    assert r.status_code == 400
    assert config.load(tmp / config.CONFIG_FILENAME).provider == "anthropic"
    assert (tmp / config.CONFIG_FILENAME).read_text() == before


def test_keys_set_report_boolean_and_never_echo(server, monkeypatch):
    url, _, _ = server
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    secret = "AQ.test-secret-value-123"
    try:
        r = requests.post(url + "/api/keys",
                          json={"provider": "gemini", "key": secret},
                          timeout=TIMEOUT)
        assert r.status_code == 200
        raw = requests.get(url + "/api/state", timeout=TIMEOUT).text
        assert json.loads(raw)["keys"]["gemini"] is True
        assert secret not in raw
    finally:
        requests.post(url + "/api/keys",
                      json={"provider": "gemini", "key": ""}, timeout=TIMEOUT)
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["keys"]["gemini"] is False


def test_recipes_search_pagination_and_stats(server):
    url, _, _ = server
    j = requests.get(url + "/api/recipes", timeout=TIMEOUT).json()
    assert j["total"] == 2
    j = requests.get(url + "/api/recipes?q=TDMAT", timeout=TIMEOUT).json()
    assert j["total"] == 1
    assert j["items"][0]["recipe"]["material"] == "TiO2"
    st = requests.get(url + "/api/recipes/stats", timeout=TIMEOUT).json()
    assert dict(st["top_materials"])["TiO2"] == 1


def test_manual_list_and_open_tabs(server, monkeypatch):
    url, _, _ = server
    j = requests.get(url + "/api/manual", timeout=TIMEOUT).json()
    assert j["count"] == 1
    assert j["items"][0]["url"].startswith("https://doi.org/")

    opened = []
    monkeypatch.setattr(manual.webbrowser, "open_new_tab",
                        lambda u: opened.append(u))
    monkeypatch.setattr(manual.time, "sleep", lambda s: None)
    r = requests.post(url + "/api/manual/open", json={"n": 5},
                      timeout=TIMEOUT)
    assert r.json()["opened"] == 1  # only one paper in the queue
    assert len(opened) == 1


def test_manual_list_paginates(server):
    url, _, tmp = server
    # the seeded project already has 1 manual paper; add 120 harvested (pending)
    cfg = config.load(tmp / config.CONFIG_FILENAME)
    conn = db.connect(cfg.db_path)
    for i in range(100, 220):
        db.upsert_paper(conn, make_paper(i, doi=f"10.1/pg{i}"))
    conn.close()
    total = 121  # 120 pending + 1 manual

    j = requests.get(url + "/api/manual?offset=0&limit=50", timeout=TIMEOUT).json()
    assert j["count"] == total
    assert j["offset"] == 0 and j["limit"] == 50
    assert len(j["items"]) == 50
    assert j["items"][0]["status"] == "manual"           # manual sorts first

    j2 = requests.get(url + "/api/manual?offset=50&limit=50", timeout=TIMEOUT).json()
    assert len(j2["items"]) == 50
    assert j2["items"][0]["id"] != j["items"][0]["id"]   # a different page

    j3 = requests.get(url + "/api/manual?offset=100&limit=50", timeout=TIMEOUT).json()
    assert len(j3["items"]) == total - 100               # last, partial page

    j4 = requests.get(url + "/api/manual?limit=99999", timeout=TIMEOUT).json()
    assert j4["limit"] == 200                             # limit is capped


def test_second_dashboard_cannot_hijack_the_port(tmp_path):
    """Two dashboards silently sharing one port split the traffic: the
    browser watches one server's job while Stop lands on the other
    (regression: Stop Job appeared dead during a Gemini retry loop). The
    second bind must fail loudly instead."""
    httpd, app = gui.create_server(tmp_path, port=0)
    port = httpd.server_address[1]
    try:
        with pytest.raises(OSError):
            gui.create_server(tmp_path, port=port)
    finally:
        httpd.server_close()
        app.close()


def test_manual_remove_restore_and_removed_list(server):
    url, _, _ = server                              # seed has 1 manual paper W1002
    r = requests.post(url + "/api/manual/remove", json={"id": "W1002"},
                      timeout=TIMEOUT)
    assert r.json()["ok"] is True
    j = requests.get(url + "/api/manual", timeout=TIMEOUT).json()
    assert j["count"] == 0 and j["removed_count"] == 1
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["status"]["manual_count"] == 0        # no longer counted
    rem = requests.get(url + "/api/manual/removed", timeout=TIMEOUT).json()
    assert rem["count"] == 1 and rem["items"][0]["id"] == "W1002"

    r = requests.post(url + "/api/manual/restore", json={"id": "W1002"},
                      timeout=TIMEOUT)
    assert r.json()["restored"] == 1
    j = requests.get(url + "/api/manual", timeout=TIMEOUT).json()
    assert j["count"] == 1 and j["removed_count"] == 0
    # removing an unknown id is a 400
    assert requests.post(url + "/api/manual/remove", json={"id": "nope"},
                         timeout=TIMEOUT).status_code == 400


def test_state_reports_the_profile_lock(server):
    url, _, tmp = server
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["status"]["locked_profile"] is None      # nothing extracted yet
    cfg = config.load(tmp / config.CONFIG_FILENAME)
    conn = db.connect(cfg.db_path)
    db.set_meta(conn, "profile", "ald")               # extract.PROFILE_META_KEY
    conn.close()
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["status"]["locked_profile"] == "ald"


def test_recipes_csv_download_respects_search(server):
    url, _, _ = server
    r = requests.get(url + "/api/recipes.csv", timeout=TIMEOUT)
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/csv")
    assert 'filename="recipes_all.csv"' in r.headers["Content-Disposition"]
    text = r.content.decode("utf-8-sig")
    lines = text.strip().splitlines()
    assert lines[0].startswith("paper_id,doi,title,year,journal,ocr,material")
    assert len(lines) == 3                       # header + 2 seeded recipes
    assert "TiO2" in text and "TDMAT" in text

    r = requests.get(url + "/api/recipes.csv?q=TiO2", timeout=TIMEOUT)
    text = r.content.decode("utf-8-sig")
    assert len(text.strip().splitlines()) == 2   # header + the TiO2 recipe
    assert "TDMAT" in text and "TMA" not in text.splitlines()[1]
    assert 'filename="recipes_TiO2.csv"' in r.headers["Content-Disposition"]


def test_manual_search_and_bulk_remove(server):
    url, _, tmp = server
    cfg = config.load(tmp / config.CONFIG_FILENAME)
    conn = db.connect(cfg.db_path)
    db.upsert_paper(conn, make_paper(70, title="Searchable hafnium paper"))
    db.upsert_paper(conn, make_paper(71, title="Another queued paper"))
    conn.close()

    j = requests.get(url + "/api/manual?q=hafnium", timeout=TIMEOUT).json()
    assert j["count"] == 1 and j["items"][0]["id"] == "W1070"

    r = requests.post(url + "/api/manual/remove",
                      json={"ids": ["W1070", "W1071"]}, timeout=TIMEOUT)
    assert r.json()["removed"] == 2
    j = requests.get(url + "/api/manual", timeout=TIMEOUT).json()
    assert j["removed_count"] == 2
    requests.post(url + "/api/manual/restore", json={"all": True},
                  timeout=TIMEOUT)


def test_factory_reset_restores_defaults_and_keeps_data(server):
    url, _, _ = server
    requests.post(url + "/api/config",
                  json={"provider": "gemini", "query": "custom phrase",
                        "max_pdf_mb": 5}, timeout=TIMEOUT)
    before = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert before["config"]["provider"] == "gemini"

    r = requests.post(url + "/api/config/factory-reset", json={},
                      timeout=TIMEOUT)
    assert r.status_code == 200
    c = r.json()["config"]
    assert c["provider"] == "anthropic"            # back to shipped default
    assert c["query"] == "" and c["max_pdf_mb"] == 80
    assert c["email"] == "test@example.edu"        # email preserved

    after = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert after["status"]["papers"] == before["status"]["papers"]  # data kept
    assert after["status"]["recipes"] == 2


def test_jobs_run_conflict_stop_and_logs(server, monkeypatch):
    url, app, _ = server
    fake = [sys.executable, "-u", "-c",
            "import sys,time;print('hello-from-job');time.sleep(1.6);"
            "print('goodbye')"]
    monkeypatch.setattr(gui, "JOB_PREFIX", fake)

    r = requests.post(url + "/api/job",
                      json={"name": "export", "params": {}}, timeout=TIMEOUT)
    assert r.status_code == 200
    # starting a second job while one runs is refused
    r2 = requests.post(url + "/api/job",
                       json={"name": "harvest", "params": {}},
                       timeout=TIMEOUT)
    assert r2.status_code == 409

    def log_text():
        return json.dumps(
            requests.get(url + "/api/logs?since=0", timeout=TIMEOUT).json())

    assert _wait(lambda: "hello-from-job" in log_text())
    # stop terminates it and state settles back to idle
    requests.post(url + "/api/job/stop", timeout=TIMEOUT)
    assert _wait(lambda: not requests.get(
        url + "/api/state", timeout=TIMEOUT).json()["job"]["running"])
    assert "finished" in log_text()


def test_real_export_job_via_subprocess(server):
    url, _, tmp = server
    r = requests.post(url + "/api/job",
                      json={"name": "export", "params": {}}, timeout=TIMEOUT)
    assert r.status_code == 200

    def finished_ok():
        s = requests.get(url + "/api/state", timeout=TIMEOUT).json()["job"]
        return (not s["running"]) and s["last"] and s["last"]["code"] == 0

    # generous: the subprocess cold-imports the whole package, which can take
    # >30 s on a busy machine or a freshly rebuilt venv (observed flake)
    assert _wait(finished_ok, timeout=120)
    assert (tmp / "data" / "exports" / "ald_recipes.json").exists()


def test_watch_toggle(server):
    url, _, _ = server
    r = requests.post(url + "/api/watch", json={"on": True}, timeout=TIMEOUT)
    assert r.json()["on"] is True
    r = requests.post(url + "/api/watch", json={"on": False}, timeout=TIMEOUT)
    assert r.json()["on"] is False


def test_unknown_routes_and_jobs(server):
    url, _, _ = server
    assert requests.get(url + "/api/nope", timeout=TIMEOUT).status_code == 404
    r = requests.post(url + "/api/job", json={"name": "rm-rf", "params": {}},
                      timeout=TIMEOUT)
    assert r.status_code == 400


# ---- v2: profiles + provider_settings round-trip -----------------------------------

def test_profiles_endpoint_and_state_profile(server):
    url, _, _ = server
    j = requests.get(url + "/api/profiles", timeout=TIMEOUT).json()
    names = {p["name"] for p in j["profiles"]}
    assert {"ald", "ale"} <= names
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["profile"]["name"] == "ald"
    assert s["profile"]["fields"] > 20


def test_config_save_preserves_profile_and_provider_settings(server):
    url, _, tmp = server
    # simulate hand-edited provider_settings, then a GUI save of other fields
    cfg = config.load(tmp / config.CONFIG_FILENAME)
    values = {f: getattr(cfg, f) for f in
              ("email", "profile", "query", "from_year", "to_year",
               "data_dir", "provider", "provider_settings", "models",
               "triage_enabled", "triage_chars", "max_paper_chars",
               "max_output_tokens", "request_delay_s",
               "download_timeout_s", "max_pdf_mb")}
    values["provider_settings"] = {"base_url": "http://localhost:11434/v1"}
    values["profile"] = "ale"
    values["models"] = {"anthropic": {"extraction": "my-custom-model"}}
    config.save_config(tmp / config.CONFIG_FILENAME, values)

    r = requests.post(url + "/api/config", json={"triage_chars": 9999},
                      timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    reloaded = config.load(tmp / config.CONFIG_FILENAME)
    assert reloaded.triage_chars == 9999
    assert reloaded.profile == "ale"                       # preserved
    assert reloaded.provider_settings["base_url"].endswith("11434/v1")
    # per-provider model override survived the unrelated save
    assert reloaded.models_for("anthropic")["extraction"] == "my-custom-model"
    assert reloaded.models_for("anthropic")["triage"] == "claude-haiku-4-5"
    assert reloaded.models_for("gemini")["extraction"] == "gemini-2.5-flash"


# ---- provider dropdown + editable OpenAlex query / prompts --------------------

def test_providers_endpoint_lists_builtins(server):
    url, _, _ = server
    j = requests.get(url + "/api/providers", timeout=TIMEOUT).json()
    assert j["builtin"] == ["anthropic", "openai", "gemini", "openai_compatible"]
    assert isinstance(j["plugins"], list)


def test_profile_get_edit_and_shadow(server):
    url, _, tmp = server
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert j["name"] == "ald"
    assert j["n_fields"] == 30
    assert j["is_project_copy"] is False
    assert "atomic layer deposition" in j["default_query"]
    assert j["triage_prompt"] and j["extraction_prompt"]

    # edit the OpenAlex query and append an extraction rule
    new_query = '"atomic layer deposition" OR "ALD" OR "atomic layer epitaxy"'
    r = requests.post(url + "/api/profile", json={
        "default_query": new_query,
        "triage_prompt": j["triage_prompt"],
        "extraction_prompt": j["extraction_prompt"] + "\n8. Added via GUI."},
        timeout=TIMEOUT)
    assert r.status_code == 200, r.text

    # persisted as a project profile that shadows the built-in, and reloads
    assert (tmp / "profiles" / "ald.toml").exists()
    from alpminer import profiles
    prof = profiles.load("ald", tmp)
    assert prof.default_query == new_query
    assert prof.extraction_prompt.endswith("Added via GUI.")
    assert len(prof.fields) == 30                       # field set intact

    j2 = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert j2["is_project_copy"] is True
    assert j2["default_query"] == new_query


def test_profile_save_as_new_name_creates_and_switches(server):
    url, _, tmp = server
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    r = requests.post(url + "/api/profile", json={
        "new_name": "my_ald",
        "default_query": '"my special query"',
        "triage_prompt": j["triage_prompt"],
        "extraction_prompt": j["extraction_prompt"],
        "fields": j["fields"]}, timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "my_ald" and r.json()["switched"] is True

    # written as its own project profile; built-in ald untouched
    assert (tmp / "profiles" / "my_ald.toml").exists()
    assert not (tmp / "profiles" / "ald.toml").exists()
    from alpminer import profiles
    prof = profiles.load("my_ald", tmp)
    assert prof.default_query == '"my special query"'
    assert len(prof.fields) == 30                      # full copy of ald

    # config switched to the new profile; the editor now serves it
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["config"]["profile"] == "my_ald"
    j2 = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert j2["name"] == "my_ald" and j2["is_project_copy"] is True

    # listed alongside the built-ins
    names = [p["name"] for p in requests.get(
        url + "/api/profiles", timeout=TIMEOUT).json()["profiles"]]
    assert "my_ald" in names and "ald" in names


def test_create_profile_from_scratch_and_as_copy(server):
    url, _, tmp = server
    # from scratch: the commented starter template
    r = requests.post(url + "/api/profiles/create",
                      json={"name": "scratch_prof", "copy_from": ""},
                      timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "name": "scratch_prof",
                        "copied_from": None}
    assert (tmp / "profiles" / "scratch_prof.toml").exists()
    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()
    assert s["config"]["profile"] == "scratch_prof"      # switched to it
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert j["name"] == "scratch_prof"
    assert j["n_fields"] == 5                            # starter template

    # as a full copy of a built-in
    r = requests.post(url + "/api/profiles/create",
                      json={"name": "ale_copy", "copy_from": "ale"},
                      timeout=TIMEOUT)
    assert r.status_code == 200, r.text
    assert r.json()["copied_from"] == "ale"
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert j["name"] == "ale_copy" and j["n_fields"] == 27   # ale's fields
    # both appear in the dropdown list; built-ins untouched
    names = [p["name"] for p in requests.get(
        url + "/api/profiles", timeout=TIMEOUT).json()["profiles"]]
    assert {"scratch_prof", "ale_copy", "ald", "ale"} <= set(names)
    assert not (tmp / "profiles" / "ale.toml").exists()


def test_create_profile_rejects_bad_input(server):
    url, _, _ = server
    for body in ({"name": "Bad Name"},                    # not snake_case
                 {"name": "ald"},                         # built-in name
                 {"name": "x_ok", "copy_from": "ghost"}): # unknown source
        r = requests.post(url + "/api/profiles/create", json=body,
                          timeout=TIMEOUT)
        assert r.status_code == 400, body
    # duplicate name is refused on the second create
    assert requests.post(url + "/api/profiles/create",
                         json={"name": "dupe_prof"},
                         timeout=TIMEOUT).status_code == 200
    r = requests.post(url + "/api/profiles/create", json={"name": "dupe_prof"},
                      timeout=TIMEOUT)
    assert r.status_code == 400 and "already exists" in r.json()["error"]


def test_delete_custom_profile_and_protections(server):
    url, _, tmp = server
    # create a custom profile (this also switches the project to it)
    requests.post(url + "/api/profiles/create",
                  json={"name": "victim", "copy_from": "ald"}, timeout=TIMEOUT)
    assert (tmp / "profiles" / "victim.toml").exists()

    # the active profile cannot be deleted
    r = requests.post(url + "/api/profiles/delete", json={"name": "victim"},
                      timeout=TIMEOUT)
    assert r.status_code == 400 and "active" in r.json()["error"]

    # switch back to ald, then delete works and the file is gone
    requests.post(url + "/api/config", json={"profile": "ald"},
                  timeout=TIMEOUT)
    r = requests.post(url + "/api/profiles/delete", json={"name": "victim"},
                      timeout=TIMEOUT)
    assert r.status_code == 200
    assert not (tmp / "profiles" / "victim.toml").exists()
    names = [p["name"] for p in requests.get(
        url + "/api/profiles", timeout=TIMEOUT).json()["profiles"]]
    assert "victim" not in names and {"ald", "ale"} <= set(names)

    # built-ins are refused -- even when a project copy shadows the name
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    requests.post(url + "/api/profile",       # in-place save -> shadow ald
                  json={"triage_prompt": j["triage_prompt"],
                        "extraction_prompt": j["extraction_prompt"]},
                  timeout=TIMEOUT)
    assert (tmp / "profiles" / "ald.toml").exists()
    r = requests.post(url + "/api/profiles/delete", json={"name": "ald"},
                      timeout=TIMEOUT)
    assert r.status_code == 400 and "built-in" in r.json()["error"]
    assert (tmp / "profiles" / "ald.toml").exists()   # untouched

    # unknown custom names are a 404
    assert requests.post(url + "/api/profiles/delete", json={"name": "ghost"},
                         timeout=TIMEOUT).status_code == 404


def test_profiles_list_marks_deletable(server):
    url, _, _ = server
    requests.post(url + "/api/profiles/create",
                  json={"name": "mine", "copy_from": "ale"}, timeout=TIMEOUT)
    plist = requests.get(url + "/api/profiles",
                         timeout=TIMEOUT).json()["profiles"]
    flags = {p["name"]: p["deletable"] for p in plist}
    assert flags["mine"] is True
    assert flags["ald"] is False and flags["ale"] is False


def test_profile_save_as_rejects_bad_and_taken_names(server):
    url, _, _ = server
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    body = {"triage_prompt": j["triage_prompt"],
            "extraction_prompt": j["extraction_prompt"]}
    for bad in ("Has Spaces", "1starts_with_digit", "ale"):   # ale = built-in
        r = requests.post(url + "/api/profile",
                          json={**body, "new_name": bad}, timeout=TIMEOUT)
        assert r.status_code == 400, bad
    # a name that already exists in the project is refused too
    assert requests.post(url + "/api/profile",
                         json={**body, "new_name": "dup_prof"},
                         timeout=TIMEOUT).status_code == 200
    r = requests.post(url + "/api/profile",
                      json={**body, "new_name": "dup_prof"}, timeout=TIMEOUT)
    assert r.status_code == 400 and "already exists" in r.json()["error"]


def test_config_rejects_bad_ocr_mode(server):
    url, _, _ = server
    r = requests.post(url + "/api/config", json={"ocr_mode": "sometimes"},
                      timeout=TIMEOUT)
    assert r.status_code == 400
    assert "ocr_mode" in r.json()["error"]
    # the valid values round-trip
    r = requests.post(url + "/api/config", json={"ocr_mode": "deferred"},
                      timeout=TIMEOUT)
    assert r.status_code == 200
    assert r.json()["config"]["ocr_mode"] == "deferred"


def test_profile_rejects_empty_prompt(server):
    url, _, _ = server
    r = requests.post(url + "/api/profile",
                      json={"triage_prompt": "", "extraction_prompt": ""},
                      timeout=TIMEOUT)
    assert r.status_code == 400


def test_key_set_by_env_name_and_status(server, monkeypatch):
    url, _, _ = server
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    j = requests.get(url + "/api/keys?env=OPENAI_API_KEY", timeout=TIMEOUT).json()
    assert j == {"env": "OPENAI_API_KEY", "set": False}
    try:
        r = requests.post(url + "/api/keys",
                          json={"env": "OPENAI_API_KEY", "key": "sk-secret-xyz"},
                          timeout=TIMEOUT)
        assert r.status_code == 200 and r.json()["set"] is True
        j = requests.get(url + "/api/keys?env=OPENAI_API_KEY",
                         timeout=TIMEOUT).json()
        assert j["set"] is True
        # value is never echoed anywhere
        assert "sk-secret-xyz" not in requests.get(
            url + "/api/state", timeout=TIMEOUT).text
    finally:
        requests.post(url + "/api/keys",
                      json={"env": "OPENAI_API_KEY", "key": ""}, timeout=TIMEOUT)
    j = requests.get(url + "/api/keys?env=OPENAI_API_KEY", timeout=TIMEOUT).json()
    assert j["set"] is False


def test_key_rejects_bad_env_name(server):
    url, _, _ = server
    r = requests.post(url + "/api/keys",
                      json={"env": "bad name!", "key": "x"}, timeout=TIMEOUT)
    assert r.status_code == 400


def test_reset_clears_the_database(server):
    url, _, _ = server
    before = requests.get(url + "/api/state", timeout=TIMEOUT).json()["status"]
    assert before["papers"] == 2 and before["recipes"] == 2

    r = requests.post(url + "/api/reset", json={}, timeout=TIMEOUT)
    assert r.status_code == 200

    s = requests.get(url + "/api/state", timeout=TIMEOUT).json()["status"]
    assert s["papers"] == 0
    assert s["recipes"] == 0
    assert s.get("manual_count", 0) == 0


def test_profile_edit_fields_add_and_remove(server):
    url, _, tmp = server
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    assert len(j["fields"]) == 30
    material = next(f for f in j["fields"] if f["name"] == "material")
    assert material["required"] is True
    new_fields = [material,
                  {"name": "my_custom_metric", "type": "number",
                   "required": False, "description": "a user-defined field"}]
    r = requests.post(url + "/api/profile", json={
        "triage_prompt": j["triage_prompt"],
        "extraction_prompt": j["extraction_prompt"],
        "fields": new_fields}, timeout=TIMEOUT)
    assert r.status_code == 200, r.text

    from alpminer import profiles
    prof = profiles.load("ald", tmp)
    assert prof.field_names() == ["material", "my_custom_metric"]
    assert prof.fields[1].type == "number"
    # the edited field set drives the tool schema the model is given
    props = prof.extraction_tool()["input_schema"]["properties"]["records"][
        "items"]["properties"]
    assert set(props) == {"material", "my_custom_metric"}


def test_profile_edit_rejects_fields_without_a_required_key(server):
    url, _, _ = server
    j = requests.get(url + "/api/profile", timeout=TIMEOUT).json()
    r = requests.post(url + "/api/profile", json={
        "triage_prompt": j["triage_prompt"],
        "extraction_prompt": j["extraction_prompt"],
        "fields": [{"name": "just_one", "type": "string", "required": False,
                    "description": ""}]}, timeout=TIMEOUT)
    assert r.status_code == 400   # a profile needs at least one required field


def test_reset_with_files_wipes_downloads(server):
    url, _, tmp = server
    pdf = tmp / "data" / "pdfs" / "junk.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_bytes(b"%PDF-1.7 leftover")
    r = requests.post(url + "/api/reset", json={"files": True}, timeout=TIMEOUT)
    assert r.status_code == 200
    assert not pdf.exists()
