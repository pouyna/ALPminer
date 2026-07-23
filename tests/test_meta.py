"""Anti-drift checks: the version and docs must move together.

These exist because the README's version string and pyproject.toml once
lagged the package version across several releases.
"""

import tomllib
from pathlib import Path

from alpminer import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package():
    with open(ROOT / "pyproject.toml", "rb") as f:
        assert tomllib.load(f)["project"]["version"] == __version__


def test_legacy_flat_model_keys_still_load():
    """Configs written before the [models.<provider>] tables used flat keys;
    they must keep loading, mapped into the models table."""
    import tempfile

    from alpminer import config
    tmp = Path(tempfile.mkdtemp())
    (tmp / "alpminer.toml").write_text(
        'email = "a@b.edu"\nprovider = "gemini"\n'
        'extraction_model = "claude-x"\ntriage_model = "claude-y"\n'
        'gemini_extraction_model = "gem-x"\n'
        'gemini_triage_model = "gem-y"\n'
        'openai_extraction_model = "gpt-x"\n', encoding="utf-8")
    cfg = config.load(tmp / "alpminer.toml")
    assert cfg.active_extraction_model == "gem-x"
    assert cfg.active_triage_model == "gem-y"
    assert cfg.models_for("anthropic")["extraction"] == "claude-x"
    assert cfg.models_for("openai")["extraction"] == "gpt-x"
    assert cfg.models_for("openai")["triage"] == "gpt-4o-mini"  # default kept
    # a save rewrites the file in the current format and it round-trips
    cfg2 = config.save_config(tmp / "alpminer.toml",
                              {"email": cfg.email, "models": cfg.models,
                               "provider": cfg.provider})
    assert cfg2.active_extraction_model == "gem-x"
    text = (tmp / "alpminer.toml").read_text(encoding="utf-8")
    assert "[models.gemini]" in text and "gemini_extraction_model" not in text


def test_every_builtin_provider_has_equal_model_slots():
    """No provider is privileged: each built-in gets the same two roles."""
    from alpminer.config import DEFAULT_MODELS
    assert set(DEFAULT_MODELS) == {"anthropic", "openai", "gemini",
                                   "openai_compatible"}
    for pair in DEFAULT_MODELS.values():
        assert set(pair) == {"extraction", "triage"}
        assert all(pair.values())


def test_project_folder_is_relocatable(tmp_path):
    """Copying a project folder elsewhere must keep working: config and data
    paths are relative, and stored absolute PDF paths self-heal."""
    from alpminer import config, db
    from tests.conftest import make_paper

    old = tmp_path / "old_home"
    old.mkdir()
    config.write_template(old / config.CONFIG_FILENAME, email="a@b.edu")
    cfg = config.load(old / config.CONFIG_FILENAME)
    cfg.ensure_dirs()
    conn = db.connect(cfg.db_path)
    db.upsert_paper(conn, make_paper(1))
    (cfg.pdf_dir / "W1001.pdf").write_bytes(b"%PDF-1.7 body")
    db.set_fields(conn, "W1001", download_status="downloaded",
                  pdf_path=str(cfg.pdf_dir / "W1001.pdf"))
    conn.close()

    new = tmp_path / "new_computer"
    import shutil
    shutil.move(str(old), str(new))            # "copied" to another machine

    cfg2 = config.load(new / config.CONFIG_FILENAME)
    assert cfg2.db_path.exists()               # relative data_dir followed
    conn = db.connect(cfg2.db_path)
    row = db.get_paper(conn, "W1001")
    assert str(old) in row["pdf_path"]          # stale absolute path...
    from alpminer.pdftext import ensure_text
    ensure_text(conn, cfg2, row)                # ...self-heals on first touch
    assert str(new) in db.get_paper(conn, "W1001")["pdf_path"]
    conn.close()


def test_readme_version_matches_package():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"alpminer {__version__}" in readme, (
        "README's `alpminer --version` example does not match "
        f"alpminer.__version__ ({__version__}) -- update the README"
    )
