"""Generic LLM result wrappers and unit helpers.

The extraction wrapper is the same for every domain profile: the model
returns ``{relevant, records, notes}`` (triage: ``{relevant, reason}``) and
each record is validated against the active profile's field specs (see
profiles.py). Raw responses cached by alpminer <= 1.x used the ALD-specific
keys ``reports_own_ald_experiment`` / ``recipes`` and non-SI quantity fields
(``pressure_torr``, ``gpc_angstrom_per_cycle``, ...); both are mapped
transparently so old caches re-parse for free.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, model_validator

from . import profiles, units

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
    r"(\u00c5|\u212b|A|angstroms?|nm|pm)\s*"
    r"(?:/\s*|per[\s\u00a0]+|\s)?\s*(?:cycles?|cyc|cy)\b"
    r"(?:\s*(?:\^\s*)?[-\u2212\u207b]\s*[1\u00b9])?",
    re.IGNORECASE,
)


def parse_gpc(text: Optional[str]) -> Optional[float]:
    """Parse a growth/etch-per-cycle string into nm/cycle, else None.

    Examples: '1.1 \u00c5/cycle' -> 0.11 ; '0.11 nm/cycle' -> 0.11 ;
              '0.9 A per cycle' -> 0.09 ; '110 pm cycle-1' -> 0.11
    """
    if not text:
        return None
    m = _GPC_RE.search(text)
    if not m:
        return None
    return units.convert(float(m.group(1)), m.group(2) + "/cycle",
                         "nm/cycle")


# Numeric per-cycle fields that can be recovered from their *_as_reported
# twin when the model returned only the raw string (feature-detected, so any
# profile that declares such a pair gets the backfill for free). The third
# item is the field's unit; the 1.x names (angstrom/cycle) still work for
# project profiles written by alpminer 1.x.
PER_CYCLE_BACKFILLS = (
    ("gpc_nm_per_cycle", "gpc_as_reported", "nm/cycle"),
    ("epc_nm_per_cycle", "epc_as_reported", "nm/cycle"),
    ("gpc_angstrom_per_cycle", "gpc_as_reported", "Å/cycle"),
    ("epc_angstrom_per_cycle", "epc_as_reported", "Å/cycle"),
)


def apply_backfills(record: dict, conversions: list | None = None) -> dict:
    """Fill a declared, still-null per-cycle rate from its *_as_reported
    text; a conversion to the field's unit is added to `conversions`."""
    for numeric, reported, unit in PER_CYCLE_BACKFILLS:
        if numeric not in record or record.get(numeric) is not None:
            continue
        parsed = parse_gpc(record.get(reported))
        if parsed is None:
            continue
        record[numeric] = units.convert(parsed, "nm/cycle", unit)
        if conversions is not None:
            source = _GPC_RE.search(record[reported])
            if source and not units.same_unit(source.group(2) + "/cycle",
                                              unit):
                conversions.append(
                    f"{numeric}: {source.group(1)} {source.group(2)}/cycle "
                    f"from {reported} -> {record[numeric]:g} {unit}")
    return record


# Per-cycle rates that can be *derived* from a reported total thickness/depth
# (in nm) over a cycle count when the paper states no per-cycle value at all.
# (rate_field, thickness_nm_field, cycles_field, rate_unit). Feature-detected
# like the backfills above: a profile only gets a derivation if it declares
# all three fields, so this stays inert for profiles that don't.
PER_CYCLE_DERIVATIONS = (
    ("gpc_nm_per_cycle", "film_thickness_nm", "number_of_cycles", "nm/cycle"),
    ("epc_nm_per_cycle", "etch_depth_nm", "number_of_cycles", "nm/cycle"),
    ("gpc_angstrom_per_cycle", "film_thickness_nm", "number_of_cycles",
     "Å/cycle"),
    ("epc_angstrom_per_cycle", "etch_depth_nm", "number_of_cycles",
     "Å/cycle"),
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
    is stored in the rate field's unit and a short provenance note is
    appended (when the profile has a ``notes`` field) so a derived rate is
    never mistaken for one the paper reported directly.
    """
    for rate_field, thickness_field, cycles_field, unit in \
            PER_CYCLE_DERIVATIONS:
        if rate_field not in record or record.get(rate_field) is not None:
            continue
        thickness = _as_number(record.get(thickness_field))
        cycles = _as_number(record.get(cycles_field))
        if thickness is None or cycles is None or cycles <= 0:
            continue
        value = units.convert(thickness / cycles, "nm/cycle", unit)
        record[rate_field] = value
        if "notes" in record:
            note = (f"{rate_field} auto-derived as {value:g} {unit} from "
                    f"{thickness_field} ({thickness:g} nm) / {cycles_field} "
                    f"({cycles:g}); not reported directly.")
            existing = record.get("notes")
            record["notes"] = (f"{existing.rstrip()} {note}"
                               if existing else note)
    return record


# alpminer 1.x stored these quantities under other names in non-SI units.
# Raw LLM caches and stored records from 1.x are upgraded when read, so an
# old project keeps working without re-paying for anything.
LEGACY_FIELDS = {
    "pressure_torr": ("pressure_pa", "Torr", "Pa"),
    "pressure_mtorr": ("pressure_pa", "mTorr", "Pa"),
    "gpc_angstrom_per_cycle": ("gpc_nm_per_cycle", "Å/cycle", "nm/cycle"),
    "epc_angstrom_per_cycle": ("epc_nm_per_cycle", "Å/cycle", "nm/cycle"),
}


def _legacy_number(value) -> Optional[float]:
    """A 1.x value as a number: ints and floats, and numeric strings that
    carry no unit of their own ('1.5')."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parsed = units.parse_quantity(value)
        if parsed is not None and not parsed[1]:
            return parsed[0]
    return None


def upgrade_legacy_fields(record: dict, field_names,
                          conversions: list | None = None) -> dict:
    """Rename and convert 1.x quantity fields when the active profile uses
    the new name, adding each conversion to `conversions`. A value that
    carries its own unit ('1.5 Torr') is moved as is and converted during
    validation. A project profile that still declares the old field keeps
    it unchanged."""
    names = set(field_names)
    for old, (new, old_unit, new_unit) in LEGACY_FIELDS.items():
        if old not in record or old in names or new not in names:
            continue
        value = record.pop(old)
        if record.get(new) is not None:
            continue
        number = _legacy_number(value)
        if number is not None:
            value = units.convert(number, old_unit, new_unit)
            if conversions is not None:
                conversions.append(f"{new}: {number:g} {old_unit} -> "
                                   f"{value:g} {new_unit} (1.x field {old})")
        record[new] = value
    return record


# ---- backward-compatible constants (built from the ALD profile) ---------------

_ALD = profiles.load("ald")
EXTRACTION_TOOL = _ALD.extraction_tool()
TRIAGE_TOOL = _ALD.triage_tool()
