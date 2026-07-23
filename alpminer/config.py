"""Configuration: alpminer.toml in the project directory. API keys are never
stored in it; they come from environment variables (ANTHROPIC_API_KEY /
OPENAI_API_KEY / GEMINI_API_KEY / plugin-defined)."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_FILENAME = "alpminer.toml"


def default_path(base_dir: Path) -> Path:
    """The project's config path."""
    return Path(base_dir) / CONFIG_FILENAME

CONFIG_TEMPLATE = '''# alpminer configuration
# API keys are NOT stored here. Set them in your environment, e.g.:
#   setx ANTHROPIC_API_KEY sk-ant-...     (Windows; or export on macOS/Linux)

# Contact email. Required. Used for the OpenAlex and Unpaywall "polite pools"
# (identifies your traffic; both services ask for it and give better rate limits).
email = "{email}"

# Extraction profile: which domain's fields and prompts to use.
# Built-ins: "ald", "ale". Add your own with `alpminer profiles new <name>`.
profile = "ald"

# OpenAlex title/abstract search query. Quotes force an exact phrase; OR is
# supported. Leave empty ('') to use the active profile's default query.
query = ''

# Optional publication-year window. Comment out for no limit.
# from_year = 2000
# to_year = 2026

# Where all pipeline state lives (SQLite DB, PDFs, texts, exports).
data_dir = "data"

# ---- LLM settings -----------------------------------------------------------
# Which LLM provider to use. Every provider is equal: pick one, give it two
# models under [models.<provider>] at the end of this file, and set its API
# key in the environment. Built-ins:
#   "anthropic"          Anthropic API        (ANTHROPIC_API_KEY)
#   "openai"             OpenAI API           (OPENAI_API_KEY)
#   "gemini"             Google Gemini API    (GEMINI_API_KEY)
#   "openai_compatible"  any local/self-hosted OpenAI-style server such as
#                        Ollama, LM Studio, or vLLM (usually no key needed)
# Any other name loads a plugin from llm_providers/<name>.py
# (scaffold one with `alpminer providers new <name>`).
provider = "anthropic"

triage_enabled = true
# Characters of paper text sent to triage (abstract+intro+experimental are early).
triage_chars = 20000
# Character cap on full text sent for extraction (~4 chars/token).
max_paper_chars = 250000
# Max output tokens for the extraction call.
max_output_tokens = 8000
# Concurrent LLM calls during extraction. 1 = one paper at a time (safest).
# 3-4 cuts a large run's wall-clock substantially; higher risks provider
# rate limits (429s are retried, but patience beats hammering).
extract_workers = 1

# OCR fallback for scanned PDFs (no text layer). Needs the optional extra:
#   pip install "alpminer[ocr]"   plus the Tesseract binary on your PATH
# (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki).
# Ignored when OCR is not installed; scanned PDFs are then flagged instead.
ocr_enabled = true
# When to OCR: "deferred" (default) = flag scanned papers (text:
# ocr_pending), finish every text-layer paper first, then OCR the flagged
# ones at the end of the same run, so slow OCR never delays extraction;
# "inline" = OCR each scanned paper the moment it is reached.
ocr_mode = "deferred"

# ---- Network settings -------------------------------------------------------
# Seconds to sleep between outbound requests (be polite to APIs/publishers).
request_delay_s = 1.0
download_timeout_s = 120
max_pdf_mb = 80

# ---- Models per provider ------------------------------------------------------
# The extraction model reads full papers (strong); the triage model only
# classifies relevance (fast, cheap). Each provider keeps its own pair, so
# switching providers never mixes model names. A plugin provider gets its own
# [models.<name>] table the same way.
[models.anthropic]
extraction = "claude-sonnet-4-6"
triage = "claude-haiku-4-5"

[models.openai]
extraction = "gpt-4o"
triage = "gpt-4o-mini"

[models.gemini]
extraction = "gemini-2.5-flash"
triage = "gemini-2.5-flash-lite"

[models.openai_compatible]
extraction = "llama3.1"
triage = "llama3.1"

# Extra settings passed verbatim to openai_compatible or plugin providers.
# [provider_settings]
# base_url = "http://localhost:11434/v1"   # e.g. Ollama
# api_key_env = "OPENAI_API_KEY"           # only if your server needs a key
'''


class ConfigError(RuntimeError):
    pass


# Shipped model pairs. Every provider is a peer: one extraction model (reads
# full papers) and one triage model (cheap relevance classifier) per provider,
# overridable through the [models.<provider>] tables in alpminer.toml.
DEFAULT_MODELS = {
    "anthropic": {"extraction": "claude-sonnet-4-6",
                  "triage": "claude-haiku-4-5"},
    "openai": {"extraction": "gpt-4o", "triage": "gpt-4o-mini"},
    "gemini": {"extraction": "gemini-2.5-flash",
               "triage": "gemini-2.5-flash-lite"},
    "openai_compatible": {"extraction": "llama3.1", "triage": "llama3.1"},
}


@dataclass
class Config:
    email: str = ""
    query: str = ''
    from_year: int | None = None
    to_year: int | None = None
    data_dir: str = "data"
    profile: str = "ald"
    provider: str = "anthropic"
    models: dict = field(default_factory=dict)
    triage_enabled: bool = True
    triage_chars: int = 20_000
    max_paper_chars: int = 250_000
    max_output_tokens: int = 8_000
    extract_workers: int = 1
    ocr_enabled: bool = True
    ocr_mode: str = "deferred"        # "deferred" | "inline"
    request_delay_s: float = 1.0
    download_timeout_s: float = 120.0
    max_pdf_mb: int = 80
    provider_settings: dict = field(default_factory=dict)

    # Base directory the config file lives in (set on load).
    base_dir: Path = field(default_factory=Path.cwd)

    # ---- derived paths ----
    @property
    def root(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else self.base_dir / p

    @property
    def db_path(self) -> Path:
        return self.root / "alpminer.db"

    @property
    def pdf_dir(self) -> Path:
        return self.root / "pdfs"

    @property
    def text_dir(self) -> Path:
        return self.root / "texts"

    @property
    def raw_llm_dir(self) -> Path:
        return self.root / "raw_llm"

    @property
    def inbox_dir(self) -> Path:
        return self.root / "manual_inbox"

    @property
    def export_dir(self) -> Path:
        return self.root / "exports"

    def ensure_dirs(self) -> None:
        for d in (self.root, self.pdf_dir, self.text_dir, self.raw_llm_dir,
                  self.inbox_dir, self.export_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def anthropic_api_key(self) -> str | None:
        return os.environ.get("ANTHROPIC_API_KEY")

    @property
    def gemini_api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def models_for(self, provider: str) -> dict:
        """The {extraction, triage} pair for a provider: shipped defaults
        overlaid with anything the user set under [models.<provider>]."""
        base = dict(DEFAULT_MODELS.get(provider, {}))
        base.update(self.models.get(provider) or {})
        return base

    def _active_model(self, role: str) -> str:
        model = (self.models_for(self.provider).get(role) or "").strip()
        if not model:
            raise ConfigError(
                f"no {role} model configured for provider "
                f"{self.provider!r}: add it under [models.{self.provider}] "
                f"in {CONFIG_FILENAME} (or the Settings tab)."
            )
        return model

    @property
    def active_extraction_model(self) -> str:
        return self._active_model("extraction")

    @property
    def active_triage_model(self) -> str:
        return self._active_model("triage")

    def require_email(self) -> str:
        if not self.email or "@" not in self.email:
            raise ConfigError(
                "A contact email is required for OpenAlex/Unpaywall polite pools. "
                f"Set `email` in {CONFIG_FILENAME} (run `alpminer init` first)."
            )
        return self.email


def write_template(path: Path, email: str = "you@university.edu") -> None:
    if path.exists():
        raise ConfigError(f"{path} already exists (use --force to overwrite).")
    path.write_text(CONFIG_TEMPLATE.format(email=email), encoding="utf-8")


def _toml_str(value: str) -> str:
    """Quote a string for TOML, preferring literal strings so embedded
    double quotes (as in the default phrase query) survive round-trips."""
    if "'" not in value and "\n" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def dump_config(values: dict) -> str:
    """Render a complete alpminer.toml from a dict of Config field values."""
    d = {f.name: getattr(Config, f.name, None) for f in fields(Config)}
    d.pop("base_dir", None)
    d.update({k: v for k, v in values.items() if k in d})
    lines = [
        "# alpminer configuration (edit here or in the GUI settings tab)",
        "# API keys are NOT stored here; they come from ANTHROPIC_API_KEY /",
        "# OPENAI_API_KEY / GEMINI_API_KEY environment variables.",
        "",
        f"email = {_toml_str(d['email'] or '')}",
        f"profile = {_toml_str(d['profile'])}",
        f"query = {_toml_str(d['query'])}",
    ]
    for key in ("from_year", "to_year"):
        if d.get(key) is not None:
            lines.append(f"{key} = {int(d[key])}")
    lines += [
        f"data_dir = {_toml_str(d['data_dir'])}",
        "",
        f"provider = {_toml_str(d['provider'])}",
        f"triage_enabled = {'true' if d['triage_enabled'] else 'false'}",
        f"triage_chars = {int(d['triage_chars'])}",
        f"max_paper_chars = {int(d['max_paper_chars'])}",
        f"max_output_tokens = {int(d['max_output_tokens'])}",
        f"extract_workers = {max(1, int(d['extract_workers'] or 1))}",
        f"ocr_enabled = {'true' if d['ocr_enabled'] else 'false'}",
        f"ocr_mode = {_toml_str(d['ocr_mode'] or 'deferred')}",
        "",
        f"request_delay_s = {float(d['request_delay_s'])}",
        f"download_timeout_s = {float(d['download_timeout_s'])}",
        f"max_pdf_mb = {int(d['max_pdf_mb'])}",
        "",
    ]
    # tables go last: any flat key written after one would belong to it
    models = d.get("models") or {}
    for prov in sorted(set(DEFAULT_MODELS) | set(models)):
        pair = {**DEFAULT_MODELS.get(prov, {}), **(models.get(prov) or {})}
        lines.append(f"[models.{prov}]")
        lines.append(f"extraction = {_toml_str(str(pair.get('extraction', '')))}")
        lines.append(f"triage = {_toml_str(str(pair.get('triage', '')))}")
        lines.append("")
    settings = d.get("provider_settings") or {}
    if settings:
        lines.append("[provider_settings]")
        for k, v in settings.items():
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            else:
                lines.append(f"{k} = {_toml_str(str(v))}")
        lines.append("")
    return "\n".join(lines)


def save_config(path: Path, values: dict) -> Config:
    """Write values to path atomically and return the re-validated Config.
    On validation failure the previous file content is restored."""
    from .utils import atomic_write_text

    previous = path.read_text(encoding="utf-8") if path.exists() else None
    atomic_write_text(path, dump_config(values))
    try:
        return load(path)
    except ConfigError:
        if previous is not None:
            atomic_write_text(path, previous)
        raise


def _migrate_legacy_model_keys(raw: dict) -> None:
    """Configs written before the per-provider [models] tables used flat
    keys (extraction_model, gemini_extraction_model, ...). Fold them into
    the models table so old files keep loading; the next save rewrites the
    file in the current format."""
    legacy = {
        "extraction_model": ("anthropic", "extraction"),
        "triage_model": ("anthropic", "triage"),
        "gemini_extraction_model": ("gemini", "extraction"),
        "gemini_triage_model": ("gemini", "triage"),
        "openai_extraction_model": ("openai", "extraction"),
        "openai_triage_model": ("openai", "triage"),
    }
    models = raw.setdefault("models", {})
    for key, (prov, role) in legacy.items():
        if key in raw:
            value = raw.pop(key)
            if isinstance(models, dict):
                models.setdefault(prov, {}).setdefault(role, value)


def load(config_path: Path | None = None) -> Config:
    """Load alpminer.toml from the given path or the current directory."""
    path = config_path or default_path(Path.cwd())
    if not path.exists():
        raise ConfigError(
            f"No config found at {path}. Run `alpminer init` in your project "
            "directory first."
        )
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    _migrate_legacy_model_keys(raw)
    known = {f.name for f in fields(Config)} - {"base_dir"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown config keys in {path}: {sorted(unknown)}")

    cfg = Config(**{k: v for k, v in raw.items() if k in known})
    cfg.base_dir = path.parent.resolve()
    if not isinstance(cfg.models, dict) or not all(
            isinstance(v, dict) for v in cfg.models.values()):
        raise ConfigError("[models.<provider>] must be tables of "
                          "extraction/triage model names")
    import re as _re
    if not _re.match(r"^[a-z][a-z0-9_]*$", cfg.provider or ""):
        raise ConfigError(
            f"provider must be a snake_case name, got {cfg.provider!r}"
        )
    if not _re.match(r"^[a-z][a-z0-9_]*$", cfg.profile or ""):
        raise ConfigError(
            f"profile must be a snake_case name, got {cfg.profile!r}"
        )
    if not isinstance(cfg.provider_settings, dict):
        raise ConfigError("[provider_settings] must be a table")
    if cfg.ocr_mode not in ("inline", "deferred"):
        raise ConfigError(
            f"ocr_mode must be 'inline' or 'deferred', got {cfg.ocr_mode!r}")
    return cfg
