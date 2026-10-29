"""Domain profiles: what to extract, how to ask for it, and how to validate it.

A profile is a TOML file declaring the extraction target for a scientific
domain: the default literature query, the triage and extraction prompts, and
the record fields (name/type/description/limits). A numeric field with a
``unit`` is a quantity: the model reports the value with the unit the paper
used and ALPminer converts it (see units.py). A field marked
``chemical = true`` gets canonical name, formula, and PubChem CID columns
(see chemistry.py).
Two profiles ship with the package (``ald``, ``ale``); users add their own by
dropping ``profiles/<name>.toml`` into the project folder (project profiles
shadow built-ins of the same name) and setting ``profile = "<name>"`` in
``alpminer.toml``. ``alpminer profiles new <name>`` writes a commented
starter file.

The LLM wrapper is fixed across all profiles so the pipeline code never
changes: every extraction returns ``{relevant: bool, records: [...], notes}``
and every triage returns ``{relevant: bool, reason}``. Only the inner record
schema and the prompts vary.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from . import units as _units

FIELD_TYPES = {"string", "number", "integer", "boolean", "array"}
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

RECORD_TOOL_NAME = "record_findings"
TRIAGE_TOOL_NAME = "triage_result"

# Derived column listing every unit conversion ALPminer made in a record.
CONVERSIONS_FIELD = "unit_conversions"


class ProfileError(RuntimeError):
    pass


@dataclass
class FieldSpec:
    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    max_len: int | None = None
    unit: str | None = None            # canonical unit, e.g. "Pa", "°C"
    quantity_kind: str | None = None   # QUDT quantity kind, e.g. "Thickness"
    chemical: bool = False             # normalize to name/formula/CID

    @property
    def is_quantity(self) -> bool:
        return bool(self.unit) and self.type in ("number", "integer")

    def json_schema(self) -> dict:
        if self.is_quantity:
            # the model reports what the paper says; ALPminer converts
            inner: dict = {
                "type": ["object", "null"],
                "properties": {
                    "value": {"type": "number"},
                    "unit": {"type": "string",
                             "description": "The unit exactly as the paper "
                                            "writes it, e.g. 'Torr', 'K', "
                                            "'Å/cycle'."},
                },
                "required": ["value", "unit"],
            }
            desc = (f"{self.description} Give the number with the unit used "
                    f"in the paper; it is converted to {self.unit}.").strip()
            inner["description"] = desc
            return inner
        if self.type == "array":
            inner = {"type": ["array", "null"], "items": {"type": "string"}}
        else:
            inner = {"type": [self.type, "null"]}
        if self.type == "string" and self.required:
            inner = {"type": "string"}
        if self.description:
            inner["description"] = self.description
        return inner


@dataclass
class Profile:
    name: str
    label: str
    description: str = ""
    default_query: str = ""
    record_noun: str = "records"
    triage_prompt: str = ""
    extraction_prompt: str = ""
    units: dict = field(default_factory=dict)
    fields: list[FieldSpec] = field(default_factory=list)
    source_path: str = ""

    # ---- derived artifacts ----
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def has_units(self) -> bool:
        return any(f.unit for f in self.fields)

    def derived_field_names(self) -> list[str]:
        """Columns ALPminer fills itself (never asked of the model): the
        unit-conversion log and the name/formula/CID of chemical fields."""
        from .chemistry import derived_names
        out = [CONVERSIONS_FIELD] if self.has_units() else []
        for f in self.fields:
            if f.chemical:
                out.extend(derived_names(f.name))
        return out

    def export_field_names(self) -> list[str]:
        return self.field_names() + self.derived_field_names()

    def units_metadata(self) -> dict:
        """Per-field unit description for exports (IUPAC symbol and QUDT
        IRIs); profiles without field units fall back to their [units]
        table, as in alpminer 1.x."""
        if not self.has_units():
            return dict(self.units)
        return {f.name: _units.metadata(f.unit, f.quantity_kind)
                for f in self.fields if f.unit}

    def extraction_tool(self) -> dict:
        record_schema = {
            "type": "object",
            "properties": {f.name: f.json_schema() for f in self.fields},
            "required": [f.name for f in self.fields if f.required],
        }
        return {
            "name": RECORD_TOOL_NAME,
            "description": (f"Record every qualifying entry "
                            f"({self.record_noun}) found in this paper. "
                            "Use exactly one call."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "relevant": {
                        "type": "boolean",
                        "description": "True only if this paper contains at "
                                       "least one qualifying entry per the "
                                       "instructions.",
                    },
                    "records": {
                        "type": "array",
                        "items": record_schema,
                        "description": f"One entry per distinct "
                                       f"{self.record_noun[:-1] if self.record_noun.endswith('s') else self.record_noun}. "
                                       "Empty if relevant is false.",
                    },
                    "notes": {
                        "type": ["string", "null"],
                        "description": "Brief extraction caveats, if any.",
                    },
                },
                "required": ["relevant", "records"],
            },
        }

    def triage_tool(self) -> dict:
        return {
            "name": TRIAGE_TOOL_NAME,
            "description": "Classify whether this paper is relevant per the "
                           "instructions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "relevant": {"type": "boolean",
                                 "description": "True if the paper likely "
                                                "contains qualifying entries."},
                    "reason": {"type": ["string", "null"],
                               "description": "One short justifying sentence."},
                },
                "required": ["relevant"],
            },
        }

    # ---- record validation ----
    def validate_record(self, raw: dict,
                        conversions: list[str] | None = None) -> dict:
        """Coerce and validate one extracted record against the field specs.
        Unknown keys are dropped; missing optional fields become null.
        Quantities are converted to their field's unit; each conversion (and
        each value left null because its unit was unknown) is appended to
        `conversions` when given. Raises ProfileError listing every problem
        found."""
        out: dict = {}
        problems: list[str] = []
        for spec in self.fields:
            value = raw.get(spec.name)
            if value is None or value == "":
                if spec.required:
                    problems.append(f"{spec.name}: required but missing")
                out[spec.name] = None
                continue
            try:
                if spec.is_quantity:
                    value = self._to_unit(spec, value, conversions)
                    if value is None:
                        if spec.required:
                            problems.append(f"{spec.name}: required but "
                                            "missing (no usable value or "
                                            "unit)")
                        out[spec.name] = None
                        continue
                elif spec.unit and spec.type == "string":
                    value = self._range_to_unit(spec, value, conversions)
                out[spec.name] = self._coerce(spec, value)
            except (TypeError, ValueError) as exc:
                problems.append(f"{spec.name}: {exc}")
        if problems:
            raise ProfileError("invalid record: " + "; ".join(problems))
        return out

    @staticmethod
    def _to_unit(spec: FieldSpec, value, conversions):
        """A quantity as {value, unit}, a '1.5 Torr' string, or a bare
        number (taken as already in the field's unit) -> number in the
        field's unit, or None when the unit is unknown."""
        if isinstance(value, dict):
            number, unit = value.get("value"), value.get("unit")
            if number is None or number == "":
                return None
        elif isinstance(value, str):
            parsed = _units.parse_quantity(value)
            if parsed is None:
                raise ValueError(f"expected a number with a unit, "
                                 f"got {value!r}")
            number, unit = parsed
        else:
            return value                 # validated as a number below
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            if isinstance(number, str):
                try:
                    number = float(number.replace("−", "-"))
                except ValueError:
                    raise ValueError(f"expected a number, got {number!r}")
            else:
                raise ValueError(f"expected a number, got "
                                 f"{type(number).__name__}")
        unit = str(unit).strip() if unit is not None else ""
        if not unit:
            # a value reported without its unit is kept in the field's unit,
            # and the record says so
            if conversions is not None:
                conversions.append(f"{spec.name}: {number:g} given without "
                                   f"a unit; taken as {spec.unit}")
            return float(number)
        try:
            converted = _units.convert(number, unit, spec.unit)
        except _units.UnitError as exc:
            if conversions is not None:
                conversions.append(f"{spec.name}: {number:g} {unit} left "
                                   f"null ({exc})")
            return None
        if conversions is not None and not _units.same_unit(unit, spec.unit):
            conversions.append(f"{spec.name}: {number:g} {unit} -> "
                               f"{converted:g} {spec.unit}")
        return converted

    @staticmethod
    def _range_to_unit(spec: FieldSpec, value, conversions):
        """A range string (or a single {value, unit}) -> text in the field's
        unit; anything that is not a readable range is kept as written."""
        if isinstance(value, dict):
            number, unit = value.get("value"), value.get("unit") or ""
            if number is None or number == "":
                return None
            value = f"{number} {unit}".strip()
            if _units.parse_quantity(value) is not None and not re.search(
                    r"\d\s*(?:-|–|—|~|to)\s*\d", value):
                n, u = _units.parse_quantity(value)
                try:
                    converted = _units.convert(n, u or None, spec.unit)
                except _units.UnitError:
                    return value
                if conversions is not None and u and not _units.same_unit(
                        u, spec.unit):
                    conversions.append(f"{spec.name}: {n:g} {u} -> "
                                       f"{converted:g} {spec.unit}")
                return f"{converted:g}"
        if not isinstance(value, str):
            return value
        text, note = _units.normalize_range(value, spec.unit)
        if note and conversions is not None:
            conversions.append(f"{spec.name}: {note}")
        return text

    @staticmethod
    def _coerce(spec: FieldSpec, value):
        if spec.type == "string":
            if not isinstance(value, str):
                value = str(value)
            value = value.strip()
            if spec.required and not value:
                raise ValueError("required but empty")
            if spec.max_len and len(value) > spec.max_len:
                value = value[: spec.max_len - 1] + "\u2026"
            return value or None
        if spec.type == "boolean":
            if isinstance(value, bool):
                return value
            raise ValueError(f"expected boolean, got {type(value).__name__}")
        if spec.type in ("number", "integer"):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"expected {spec.type}, "
                                 f"got {type(value).__name__}")
            value = float(value)
            if spec.minimum is not None:
                value = max(spec.minimum, value)
            if spec.maximum is not None:
                value = min(spec.maximum, value)
            return int(round(value)) if spec.type == "integer" else value
        if spec.type == "array":
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                raise ValueError("expected an array of strings")
            return [str(v).strip() for v in value if str(v).strip()] or None
        raise ValueError(f"unsupported field type {spec.type!r}")


# ---- loading ----------------------------------------------------------------------

def _parse(data: dict, source: str) -> Profile:
    try:
        raw_fields = data.pop("field")
    except KeyError:
        raise ProfileError(f"{source}: profile declares no [[field]] entries")
    specs: list[FieldSpec] = []
    seen: set[str] = set()
    for i, f in enumerate(raw_fields):
        name = f.get("name", "")
        if not _NAME_RE.match(name):
            raise ProfileError(f"{source}: field #{i + 1} has invalid name "
                               f"{name!r} (use snake_case)")
        if name in seen:
            raise ProfileError(f"{source}: duplicate field {name!r}")
        seen.add(name)
        ftype = f.get("type", "string")
        if ftype not in FIELD_TYPES:
            raise ProfileError(f"{source}: field {name!r} has unknown type "
                               f"{ftype!r} (use {sorted(FIELD_TYPES)})")
        unit = (f.get("unit") or "").strip() or None
        if unit is not None:
            known = _units.lookup(unit)
            if known is None:
                raise ProfileError(f"{source}: field {name!r} has unknown "
                                   f"unit {unit!r}")
            if ftype not in ("number", "integer", "string"):
                raise ProfileError(f"{source}: field {name!r}: a unit needs "
                                   "a number, integer, or string (range) "
                                   "field")
            unit = known.symbol             # store the IUPAC spelling
        chemical = bool(f.get("chemical", False))
        if chemical and ftype not in ("string", "array"):
            raise ProfileError(f"{source}: field {name!r}: chemical = true "
                               "needs a string or array field")
        specs.append(FieldSpec(
            name=name, type=ftype,
            description=f.get("description", ""),
            required=bool(f.get("required", False)),
            minimum=f.get("minimum"), maximum=f.get("maximum"),
            max_len=f.get("max_len"),
            unit=unit,
            quantity_kind=(f.get("quantity_kind") or "").strip() or None,
            chemical=chemical,
        ))
    derived = set()
    for s in specs:
        if s.chemical:
            derived.update((f"{s.name}_name", f"{s.name}_formula",
                            f"{s.name}_cid"))
    if any(s.unit for s in specs):
        derived.add(CONVERSIONS_FIELD)
    clash = sorted(derived & seen)
    if clash:
        raise ProfileError(f"{source}: field name(s) {clash} are reserved "
                           "for columns ALPminer derives itself")
    profile = Profile(
        name=data.get("name", ""),
        label=data.get("label", data.get("name", "")),
        description=data.get("description", ""),
        default_query=data.get("default_query", ""),
        record_noun=data.get("record_noun", "records"),
        triage_prompt=(data.get("triage_prompt") or "").strip(),
        extraction_prompt=(data.get("extraction_prompt") or "").strip(),
        units=data.get("units", {}),
        fields=specs,
        source_path=source,
    )
    for key, what in (("name", "a name"), ("triage_prompt", "a triage_prompt"),
                      ("extraction_prompt", "an extraction_prompt")):
        if not getattr(profile, key):
            raise ProfileError(f"{source}: profile is missing {what}")
    if not any(s.required for s in specs):
        raise ProfileError(f"{source}: at least one field must be "
                           "required = true (the record's key field)")
    return profile


def _builtin_dir():
    return resources.files("alpminer") / "profiles"


def builtin_names() -> list[str]:
    return sorted(p.name[:-5] for p in _builtin_dir().iterdir()
                  if p.name.endswith(".toml"))


def project_profile_dir(project_dir: Path) -> Path:
    return Path(project_dir) / "profiles"


def list_profiles(project_dir: Path | None = None) -> list[dict]:
    """All available profiles; project files shadow built-ins."""
    found: dict[str, dict] = {}
    for name in builtin_names():
        found[name] = {"name": name, "origin": "built-in"}
    if project_dir:
        pdir = project_profile_dir(project_dir)
        if pdir.is_dir():
            for f in sorted(pdir.glob("*.toml")):
                found[f.stem] = {"name": f.stem, "origin": str(f)}
    out = []
    for name, info in sorted(found.items()):
        try:
            p = load(name, project_dir)
            info.update(label=p.label, fields=len(p.fields),
                        record_noun=p.record_noun)
        except ProfileError as exc:
            info.update(label="(invalid)", error=str(exc))
        out.append(info)
    return out


def load(name: str, project_dir: Path | None = None) -> Profile:
    """Load a profile by name: project profiles/ first, then built-ins."""
    if not _NAME_RE.match(name or ""):
        raise ProfileError(f"invalid profile name {name!r}")
    if project_dir:
        candidate = project_profile_dir(project_dir) / f"{name}.toml"
        if candidate.exists():
            with open(candidate, "rb") as f:
                return _parse(tomllib.load(f), str(candidate))
    builtin = _builtin_dir() / f"{name}.toml"
    try:
        data = tomllib.loads(builtin.read_text(encoding="utf-8"))
    except FileNotFoundError:
        available = ", ".join(n["name"] for n in list_profiles(project_dir))
        raise ProfileError(
            f"no profile named {name!r}. Available: {available}. "
            "Create your own with `alpminer profiles new <name>`."
        ) from None
    return _parse(data, f"built-in:{name}")


NEW_PROFILE_TEMPLATE = '''\
# Custom alpminer extraction profile: {name}
# Set `profile = "{name}"` in alpminer.toml to use it.
# Docs: every [[field]] becomes a column in the database and a slot the LLM
# must fill. Types: string, number, integer, boolean, array (of strings).
# Exactly the fields you declare here are extracted, validated, and exported.
# A number field with `unit = "..."` is a quantity: the model reports the
# value with the unit the paper uses and alpminer converts it to this unit
# (SI or IUPAC-accepted symbols: °C, s, Pa, nm, nm/cycle, W, eV, %).
# `chemical = true` on a string/array field adds canonical name, formula,
# and PubChem CID columns for it.

name = "{name}"
label = "{name} records"
description = "Describe what this profile extracts."
record_noun = "records"

# Default OpenAlex title/abstract query for new projects using this profile.
default_query = '"your search phrase"'

triage_prompt = """You classify scientific papers for a literature mining
pipeline. Decide whether the authors of THIS paper report <your target
result> themselves, with at least some quantitative details. Reviews and
purely computational studies do not qualify. Answer by calling the
triage_result tool exactly once. The text may be truncated; judge from what
is given."""

extraction_prompt = """You are an expert extracting <your target records>
from one journal article for a curated research database.
1. Extract only results reported by the authors of THIS paper; never values
   cited from other work.
2. One record per distinct <thing>.
3. Report every quantity as a number with the unit the paper uses, without
   converting it. Never guess a value that is not stated; use null.
4. evidence_location must be a short pointer (section/table numbers), never
   a quoted passage.
Record your answer by calling the record_findings tool exactly once."""

[[field]]
name = "material"
type = "string"
required = true
description = "The key identity of this record, e.g. a chemical formula."

[[field]]
name = "synthesis_temperature_c"
type = "number"
unit = "°C"
description = "Synthesis temperature. Null if not stated."

[[field]]
name = "notes"
type = "string"
description = "Anything important that does not fit other fields."

[[field]]
name = "evidence_location"
type = "string"
max_len = 200
description = "Short pointer to where the values appear, e.g. 'Table 1'."

[[field]]
name = "confidence"
type = "number"
minimum = 0.0
maximum = 1.0
description = "0-1: how completely and unambiguously this was reported."
'''


def write_new_profile(project_dir: Path, name: str) -> Path:
    if not _NAME_RE.match(name or ""):
        raise ProfileError(f"invalid profile name {name!r} (use snake_case)")
    pdir = project_profile_dir(project_dir)
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / f"{name}.toml"
    if dest.exists():
        raise ProfileError(f"{dest} already exists")
    dest.write_text(NEW_PROFILE_TEMPLATE.format(name=name), encoding="utf-8")
    return dest


# ---- serialization (round-trips a Profile back to editable TOML) ------------------

def _toml_basic(s: str) -> str:
    """Quote a single-line string for TOML: a literal '...' when it has no
    single quote or newline, else an escaped basic "..." string."""
    if "'" not in s and "\n" not in s:
        return f"'{s}'"
    esc = s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{esc}"'


def _toml_multiline(s: str) -> str:
    """A TOML multi-line literal string ''' ... ''' (no escape processing, so
    the prompt text survives verbatim) unless it contains a ''' sequence, in
    which case fall back to an escaped multi-line basic string."""
    if "'''" not in s:
        return "'''\n" + s + "\n'''"
    esc = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return '"""\n' + esc + '\n"""'


def dump_profile(p: Profile) -> str:
    """Render a Profile back to a TOML document that _parse() reads unchanged.
    Used by the GUI to save an edited query / prompts as a project profile."""
    lines = [
        f"# alpminer extraction profile: {p.name}",
        "# Editable here or in the GUI Settings tab. A project copy of this",
        "# name shadows any built-in profile of the same name.",
        "",
        f"name = {_toml_basic(p.name)}",
        f"label = {_toml_basic(p.label)}",
    ]
    if p.description:
        lines.append(f"description = {_toml_basic(p.description)}")
    lines += [
        f"record_noun = {_toml_basic(p.record_noun)}",
        "",
        f"default_query = {_toml_basic(p.default_query)}",
        "",
        f"triage_prompt = {_toml_multiline(p.triage_prompt)}",
        "",
        f"extraction_prompt = {_toml_multiline(p.extraction_prompt)}",
        "",
    ]
    if p.units:
        lines.append("[units]")
        for k, v in p.units.items():
            lines.append(f"{k} = {_toml_basic(str(v))}")
        lines.append("")
    for f in p.fields:
        lines.append("[[field]]")
        lines.append(f"name = {_toml_basic(f.name)}")
        lines.append(f"type = {_toml_basic(f.type)}")
        if f.required:
            lines.append("required = true")
        if f.minimum is not None:
            lines.append(f"minimum = {f.minimum}")
        if f.maximum is not None:
            lines.append(f"maximum = {f.maximum}")
        if f.max_len is not None:
            lines.append(f"max_len = {int(f.max_len)}")
        if f.unit:
            lines.append(f"unit = {_toml_basic(f.unit)}")
        if f.quantity_kind:
            lines.append(f"quantity_kind = {_toml_basic(f.quantity_kind)}")
        if f.chemical:
            lines.append("chemical = true")
        if f.description:
            lines.append(f"description = {_toml_basic(f.description)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_profile(project_dir: Path, profile: Profile) -> Path:
    """Write `profile` to <project>/profiles/<name>.toml atomically, shadowing
    any built-in of the same name. The caller should reload to validate."""
    from .utils import atomic_write_text
    pdir = project_profile_dir(Path(project_dir))
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / f"{profile.name}.toml"
    atomic_write_text(dest, dump_profile(profile))
    return dest
