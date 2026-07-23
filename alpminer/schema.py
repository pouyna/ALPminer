"""Generic LLM result wrappers and unit helpers.

The extraction wrapper is the same for every domain profile: the model
returns ``{relevant, records, notes}`` (triage: ``{relevant, reason}``) and
each record is validated against the active profile's field specs (see
profiles.py). Raw responses cached by alpminer <= 1.x used the ALD-specific
keys ``reports_own_ald_experiment`` / ``recipes``; those are accepted and
mapped transparently so old caches re-parse for free.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from . import profiles

_LEGACY = {"reports_own_ald_experiment": "relevant", "recipes": "records",
           "paper_notes": "notes"}


def _upgrade_legacy(data):
    if isinstance(data, dict):
        for old, new in _LEGACY.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
    return data


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relevant: bool
    records: List[dict] = []
    notes: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_keys(cls, data):
        return _upgrade_legacy(data)


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    relevant: bool
    reason: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def _legacy_keys(cls, data):
        return _upgrade_legacy(data)


# ---- unit helpers ------------------------------------------------------------

_GPC_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(\u00c5|\u212b|A|angstroms?|nm)\s*"
    r"(?:/|per[\s\u00a0]+)?\s*cycle",
    re.IGNORECASE,
)


def parse_gpc(text: Optional[str]) -> Optional[float]:
    """Parse a growth/etch-per-cycle string into angstrom/cycle, else None.

    Examples: '1.1 \u00c5/cycle' -> 1.1 ; '0.11 nm/cycle' -> 1.1 ;
              '0.9 A per cycle' -> 0.9
    """
    if not text:
        return None
    m = _GPC_RE.search(text)
    if not m:
        return None
    value = float(m.group(1))
    if m.group(2).lower() == "nm":
        value *= 10.0
    return round(value, 6)


# Numeric per-cycle fields that can be recovered from their *_as_reported
# twin when the model returned only the raw string (feature-detected, so any
# profile that declares such a pair gets the backfill for free).
PER_CYCLE_BACKFILLS = (
    ("gpc_angstrom_per_cycle", "gpc_as_reported"),
    ("epc_angstrom_per_cycle", "epc_as_reported"),
)


def apply_backfills(record: dict) -> dict:
    for numeric, reported in PER_CYCLE_BACKFILLS:
        if reported in record and record.get(numeric) is None:
            parsed = parse_gpc(record.get(reported))
            if parsed is not None:
                record[numeric] = parsed
    return record


# Per-cycle rates that can be *derived* from a reported total thickness/depth
# (in nm) over a cycle count when the paper states no per-cycle value at all.
# (rate_field, thickness_nm_field, cycles_field). Feature-detected like the
# backfills above: a profile only gets a derivation if it declares all three
# fields, so this stays inert for profiles that don't. 1 nm = 10 angstrom.
PER_CYCLE_DERIVATIONS = (
    ("gpc_angstrom_per_cycle", "film_thickness_nm", "number_of_cycles"),
    ("epc_angstrom_per_cycle", "etch_depth_nm", "number_of_cycles"),
)


def _as_number(value) -> Optional[float]:
    """Return a real number as float, else None (bools are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def apply_derivations(record: dict) -> dict:
    """Fill a per-cycle rate from total thickness/depth divided by cycles.

    Only fires when the rate is still null after extraction and backfill and
    both the thickness (nm) and a positive cycle count are present. The value
    is stored in angstrom/cycle and a short provenance note is appended (when
    the profile has a ``notes`` field) so a derived rate is never mistaken for
    one the paper reported directly.
    """
    for rate_field, thickness_field, cycles_field in PER_CYCLE_DERIVATIONS:
        if rate_field not in record or record.get(rate_field) is not None:
            continue
        thickness = _as_number(record.get(thickness_field))
        cycles = _as_number(record.get(cycles_field))
        if thickness is None or cycles is None or cycles <= 0:
            continue
        value = round(thickness * 10.0 / cycles, 6)
        record[rate_field] = value
        if "notes" in record:
            note = (f"{rate_field} auto-derived as {value:g} A/cycle from "
                    f"{thickness_field} ({thickness:g} nm) / {cycles_field} "
                    f"({cycles:g}); not reported directly.")
            existing = record.get("notes")
            record["notes"] = (f"{existing.rstrip()} {note}"
                               if existing else note)
    return record


# ---- backward-compatible constants (built from the ALD profile) ---------------

_ALD = profiles.load("ald")
EXTRACTION_TOOL = _ALD.extraction_tool()
TRIAGE_TOOL = _ALD.triage_tool()
