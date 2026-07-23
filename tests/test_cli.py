from alpminer.cli import main


def test_manual_remove_and_restore_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--email", "a@b.edu"]) == 0
    from alpminer import config, db
    from tests.conftest import make_paper
    conn = db.connect(config.load(tmp_path / "alpminer.toml").db_path)
    db.upsert_paper(conn, make_paper(1))          # W1001, pending
    db.upsert_paper(conn, make_paper(2))          # W1002, pending
    conn.close()

    assert main(["manual", "remove", "W1001"]) == 0
    out = capsys.readouterr().out
    assert "Removed 1 paper(s)" in out

    assert main(["manual", "restore"]) == 0       # bare restore lists them
    out = capsys.readouterr().out
    assert "W1001" in out and "1 removed paper(s)" in out

    assert main(["manual", "restore", "W1001"]) == 0
    assert "Restored 1 paper(s)" in capsys.readouterr().out

    # unknown ids are skipped, not fatal; missing ids on remove is usage error
    assert main(["manual", "remove", "nope"]) == 0
    assert "skipped nope" in capsys.readouterr().out
    assert main(["manual", "remove"]) == 2

    # restore --all round-trip
    assert main(["manual", "remove", "W1001", "W1002"]) == 0
    capsys.readouterr()
    assert main(["manual", "restore", "--all"]) == 0
    assert "Restored 2" in capsys.readouterr().out


def test_init_then_status_and_manual_list(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--email", "test@example.edu"]) == 0
    assert (tmp_path / "alpminer.toml").exists()
    assert (tmp_path / "data").is_dir()

    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "papers" in out

    assert main(["manual", "list"]) == 0
    out = capsys.readouterr().out
    assert "Manual queue is empty." in out


def test_missing_config_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["status"]) == 2
    err = capsys.readouterr().err
    assert "alpminer init" in err


def test_init_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--email", "a@b.edu"]) == 0
    assert main(["init", "--email", "a@b.edu"]) == 2
    assert main(["init", "--email", "a@b.edu", "--force"]) == 0


def test_run_continues_when_harvest_is_rate_limited(tmp_path, monkeypatch):
    """A 429 (RetryError) from harvest must not abort the whole run; later
    stages still execute and the run exits cleanly."""
    from alpminer import cli, download, export, extract, harvest, manual
    from alpminer.utils import RetryError

    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--email", "a@b.edu"]) == 0

    def rate_limited(*a, **k):
        raise RetryError("OpenAlex page fetch failed after 6 attempts: HTTP 429")

    ran = []
    monkeypatch.setattr(harvest, "harvest", rate_limited)
    monkeypatch.setattr(download, "download_pending",
                        lambda *a, **k: ran.append("download"))
    monkeypatch.setattr(manual, "ingest_inbox", lambda *a, **k: {"matched": []})
    monkeypatch.setattr(extract, "run_extract",
                        lambda *a, **k: ran.append("extract"))
    monkeypatch.setattr(export, "build_export", lambda *a, **k: ran.append("export"))

    assert cli.main(["run", "--limit", "5"]) == 0     # did not crash on 429
    assert ran == ["download", "extract", "export"]   # continued past harvest


def test_standalone_network_error_is_a_clean_message(tmp_path, monkeypatch, capsys):
    """A bare harvest 429 surfaces as a friendly one-liner, not a traceback."""
    from alpminer import cli, harvest
    from alpminer.utils import RetryError

    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "--email", "a@b.edu"]) == 0
    monkeypatch.setattr(harvest, "harvest", lambda *a, **k: (_ for _ in ()).throw(
        RetryError("OpenAlex page fetch failed after 6 attempts: HTTP 429")))
    assert cli.main(["harvest"]) == 1
    err = capsys.readouterr().err
    assert "network error" in err and "429" in err
    assert "Traceback" not in err
