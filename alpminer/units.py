"""Units: IUPAC symbols, QUDT identifiers, and conversion to canonical units.

Every physical quantity ALPminer stores has one canonical unit per field,
chosen from SI units and the units IUPAC accepts for use with them
("Quantities, Units and Symbols in Physical Chemistry", the Green Book):
degree Celsius, second, pascal, nanometre (also per cycle), watt,
electronvolt, and percent. The extraction model reports each value with the
unit the paper used; this module converts it, so conversions are exact and
documented instead of left to the model.

Each unit also carries its QUDT unit and quantity-kind IRIs. Exports can then
describe a value by its number, unit, and quantity kind, the representation
the AtomicLimits perspective recommends for AI-ready ALD/ALE data (Poupaki
et al., Chem. Mater. 2026, doi:10.1021/acs.chemmater.6c02250).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

QUDT_UNIT = "http://qudt.org/vocab/unit/"
QUDT_KIND = "http://qudt.org/vocab/quantitykind/"

# 1 Torr = 101325/760 Pa exactly; the conventional millimetre of mercury
# (mmHg) is 133.322387415 Pa and differs from the torr by < 0.000015 %.
_TORR_PA = 101325.0 / 760.0


class UnitError(ValueError):
    """Unknown unit, or a unit of the wrong kind for the field."""


@dataclass(frozen=True)
class Unit:
    symbol: str          # IUPAC symbol, e.g. "Pa"
    dimension: str       # temperature, time, pressure, length, per_cycle, ...
    factor: float        # value in the dimension's base unit =
    offset: float = 0.0  #   value * factor + offset
    qudt: str | None = None   # QUDT unit local name, e.g. "PA"


# QUDT quantity kind used when a profile field does not name its own.
DEFAULT_KIND = {"temperature": "Temperature", "time": "Time",
                "pressure": "Pressure", "length": "Length",
                "per_cycle": "Thickness", "power": "Power",
                "energy": "Energy", "voltage": "ElectricPotential",
                "fraction": "DimensionlessRatio"}

# (Unit, aliases). Aliases match exactly first. Written-out aliases of four
# or more characters ("torr", "Kelvin", "MBAR") also match case-insensitively
# when unambiguous; short symbols never do, so "MW" is not read as "mW".
# Base units per dimension: degC, s, Pa, nm, W, eV, V, %.
_TABLE: list[tuple[Unit, tuple[str, ...]]] = [
    (Unit("°C", "temperature", 1.0, 0.0, "DEG_C"),
     ("°C", "° C", "ºC", "˚C", "℃", "degC", "deg C", "deg. C", "C", "oC",
      "celsius", "degree celsius", "degrees celsius", "degree C",
      "degrees C")),
    (Unit("K", "temperature", 1.0, -273.15, "K"), ("K", "kelvin")),
    (Unit("°F", "temperature", 5.0 / 9.0, -160.0 / 9.0, "DEG_F"),
     ("°F", "ºF", "degF", "deg F", "F", "fahrenheit")),

    (Unit("s", "time", 1.0, 0.0, "SEC"),
     ("s", "sec", "secs", "second", "seconds")),
    (Unit("ms", "time", 1e-3, 0.0, "MilliSEC"),
     ("ms", "msec", "millisecond", "milliseconds")),
    (Unit("µs", "time", 1e-6, 0.0, "MicroSEC"),
     ("µs", "μs", "us", "microsecond", "microseconds")),
    (Unit("min", "time", 60.0, 0.0, "MIN"),
     ("min", "mins", "minute", "minutes")),
    (Unit("h", "time", 3600.0, 0.0, "HR"), ("h", "hr", "hrs", "hour", "hours")),

    (Unit("Pa", "pressure", 1.0, 0.0, "PA"), ("Pa", "pascal", "pascals")),
    (Unit("mPa", "pressure", 1e-3, 0.0, "MilliPA"), ("mPa",)),
    (Unit("hPa", "pressure", 100.0, 0.0, "HectoPA"), ("hPa",)),
    (Unit("kPa", "pressure", 1e3, 0.0, "KiloPA"), ("kPa",)),
    (Unit("MPa", "pressure", 1e6, 0.0, "MegaPA"), ("MPa",)),
    (Unit("bar", "pressure", 1e5, 0.0, "BAR"), ("bar",)),
    (Unit("mbar", "pressure", 100.0, 0.0, "MilliBAR"), ("mbar", "millibar")),
    (Unit("µbar", "pressure", 0.1, 0.0, "MicroBAR"), ("µbar", "μbar")),
    (Unit("Torr", "pressure", _TORR_PA, 0.0, "TORR"), ("Torr",)),
    (Unit("mTorr", "pressure", _TORR_PA * 1e-3, 0.0, "MilliTORR"),
     ("mTorr", "millitorr")),
    (Unit("µTorr", "pressure", _TORR_PA * 1e-6, 0.0, "MicroTORR"),
     ("µTorr", "μTorr", "uTorr")),
    (Unit("mmHg", "pressure", 133.322387415, 0.0, "MilliM_HG"),
     ("mmHg", "mm Hg")),
    (Unit("atm", "pressure", 101325.0, 0.0, "ATM"),
     ("atm", "atmosphere", "atmospheres")),
    (Unit("psi", "pressure", 6894.757293168361, 0.0, "PSI"), ("psi",)),

    (Unit("nm", "length", 1.0, 0.0, "NanoM"),
     ("nm", "nanometer", "nanometers", "nanometre", "nanometres")),
    (Unit("Å", "length", 0.1, 0.0, "ANGSTROM"),
     ("Å", "A", "angstrom", "angstroms", "ångström", "ångströms")),
    (Unit("pm", "length", 1e-3, 0.0, "PicoM"),
     ("pm", "picometer", "picometers", "picometre", "picometres")),
    (Unit("µm", "length", 1e3, 0.0, "MicroM"),
     ("µm", "μm", "um", "micron", "microns", "micrometer", "micrometre")),
    (Unit("mm", "length", 1e6, 0.0, "MilliM"), ("mm",)),
    (Unit("m", "length", 1e9, 0.0, "M"), ("m",)),

    (Unit("W", "power", 1.0, 0.0, "W"), ("W", "watt", "watts")),
    (Unit("mW", "power", 1e-3, 0.0, "MilliW"), ("mW",)),
    (Unit("kW", "power", 1e3, 0.0, "KiloW"), ("kW",)),

    (Unit("eV", "energy", 1.0, 0.0, "EV"),
     ("eV", "electronvolt", "electronvolts", "electron volt",
      "electron volts")),
    (Unit("keV", "energy", 1e3, 0.0, "KiloEV"), ("keV",)),
    # A bias or acceleration voltage gives singly charged ions that energy
    # in eV, which is how ALE papers usually state the ion energy (see
    # _compatible); as a field's own unit it stays a voltage.
    (Unit("V", "voltage", 1.0, 0.0, "V"), ("V", "volt", "volts")),

    (Unit("%", "fraction", 1.0, 0.0, "PERCENT"), ("%", "percent", "pct")),
]


def _clean(text) -> str:
    # NFKC folds the angstrom sign, micro sign, and superscripts into their
    # plain forms; the degree sign survives it
    s = unicodedata.normalize("NFKC", str(text)).strip()
    return re.sub(r"\s+", " ", s).rstrip(".").strip()


_EXACT: dict[str, Unit] = {}
_FOLDED: dict[str, Unit | None] = {}
for _unit, _aliases in _TABLE:
    for _alias in _aliases:
        _key = _clean(_alias)
        _EXACT[_key] = _unit
        if len(_key) < 4:
            continue                        # symbols: exact case only
        _low = _key.lower()
        if _low in _FOLDED and _FOLDED[_low] is not _unit:
            _FOLDED[_low] = None            # ambiguous once case-folded
        else:
            _FOLDED[_low] = _unit

# a length unit followed by "/cycle", "per ALD cycle", "cycle-1", "cy⁻¹", ...
_CYCLE = r"(?:(?:ALD|ALE)\s+)?"
_PER_CYCLE = re.compile(
    rf"^(?P<base>.+?)\s*(?:/\s*{_CYCLE}(?:cycles?|cyc|cy|c)"
    rf"|\s+per\s+{_CYCLE}cycles?"
    r"|\s*[·⋅*]?\s*(?:cycles?|cyc|cy)\s*(?:\^\s*)?[-−]\s*1)$",
    re.IGNORECASE)


def _plain(text: str) -> Unit | None:
    if not text:
        return None
    return _EXACT.get(text) or _FOLDED.get(text.lower())


def lookup(symbol) -> Unit | None:
    """The Unit for a symbol as written in a paper, or None if unknown.
    Per-cycle units ("Å/cycle", "nm cycle⁻¹") come back with dimension
    per_cycle; a bare length unit stays a length."""
    if symbol is None:
        return None
    s = _clean(symbol)
    m = _PER_CYCLE.match(s)
    if m:
        base = _plain(m.group("base").strip())
        if base is not None and base.dimension == "length":
            return Unit(f"{base.symbol}/cycle", "per_cycle", base.factor,
                        0.0, base.qudt)
        return None
    return _plain(s)


def _compatible(src: Unit, dst: Unit) -> bool:
    if src.dimension == dst.dimension:
        return True
    # growth/etch per cycle is often written with a bare length ("1.1 Å"),
    # and an ion energy as the bias voltage (singly charged ions: 1 V -> 1 eV)
    return ((src.dimension, dst.dimension)
            in (("length", "per_cycle"), ("voltage", "energy")))


def convert(value: float, from_unit, to_unit: str) -> float:
    """Convert value from from_unit to to_unit (both as written), rounded to
    six significant digits. A missing or blank from_unit means the value is
    already in to_unit. Raises UnitError for unknown or incompatible units."""
    dst = lookup(to_unit)
    if dst is None:
        raise UnitError(f"unknown target unit {to_unit!r}")
    if from_unit is None or not str(from_unit).strip():
        return float(value)
    src = lookup(from_unit)
    if src is None:
        raise UnitError(f"unknown unit {from_unit!r}")
    if not _compatible(src, dst):
        raise UnitError(f"{from_unit!r} is not a unit of {dst.dimension}")
    base = float(value) * src.factor + src.offset
    return float(f"{(base - dst.offset) / dst.factor:.6g}")


def same_unit(a, b: str) -> bool:
    """True when a (as written) is the same unit as b, e.g. 'degC' and '°C'."""
    ua, ub = lookup(a), lookup(b)
    return ua is not None and ub is not None and ua.symbol == ub.symbol


def metadata(symbol: str, quantity_kind: str | None = None) -> dict:
    """Export description of a canonical unit: the IUPAC symbol plus the QUDT
    unit and quantity-kind IRIs."""
    unit = lookup(symbol)
    if unit is None:
        raise UnitError(f"unknown unit {symbol!r}")
    kind = quantity_kind or DEFAULT_KIND[unit.dimension]
    out = {"unit": unit.symbol,
           "qudt_unit": QUDT_UNIT + unit.qudt if unit.qudt else None,
           "quantity_kind": QUDT_KIND + kind}
    if unit.dimension == "per_cycle":
        out["per"] = "cycle"
    return out


# thousands separators ("1,200") are read; a decimal comma ("0,5") is not
# guessed at, so such text is rejected instead of misread
_NUMBER = (r"[-+−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
           r"(?:[eE][-+−]?\d+)?")
_QUANTITY = re.compile(rf"^\s*(?P<num>{_NUMBER})\s*(?P<unit>.*?)\s*$")
_RANGE = re.compile(
    rf"^\s*(?P<lo>{_NUMBER})\s*(?P<u1>[^\d\s\-–—~]*)\s*(?:-|–|—|~|to)\s*"
    rf"(?P<hi>{_NUMBER})\s*(?P<u2>.*?)\s*$", re.IGNORECASE)


def _number(text: str) -> float:
    return float(text.replace("−", "-").replace(",", ""))


def _split_unit(text: str) -> tuple[str, str]:
    """(unit, trailing commentary): '°C (ALD window)' -> ('°C', '(ALD
    window)'); a unit written in brackets is unwrapped: '(Torr)' -> 'Torr'."""
    s = (text or "").strip()
    m = re.match(r"^[(\[]\s*([^()\[\]]*?)\s*[)\]]\s*(.*)$", s)
    if m and lookup(m.group(1)) is not None:
        return m.group(1), m.group(2)
    parts = re.split(r"(?=[(,;])", s, maxsplit=1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")


def parse_quantity(text) -> tuple[float, str] | None:
    """'1.5 Torr' -> (1.5, 'Torr'); '200°C' -> (200.0, '°C'); '1.5 (Torr)'
    -> (1.5, 'Torr'); else None. Text after a comma or bracket that is not
    the unit is dropped; a number glued to more digits ('0,5') is refused."""
    m = _QUANTITY.match(str(text))
    if not m:
        return None
    rest = m.group("unit")
    if re.match(r"^[,.]\s*\d", rest):
        return None
    try:
        return _number(m.group("num")), _split_unit(rest)[0]
    except ValueError:
        return None


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def normalize_range(text, to_unit: str) -> tuple[str, str | None]:
    """Rewrite a range such as '423-573 K' as 'lo-hi' in to_unit; trailing
    commentary is kept ('150-300 °C (window)' -> '150-300 (window)').
    Returns (text, note). The text comes back unchanged when it cannot be read
    as a range; note records the original whenever the text changed through a
    unit conversion."""
    m = _RANGE.match(str(text))
    if not m:
        return text, None
    unit, rest = _split_unit(m.group("u2"))
    if not unit:
        unit = _split_unit(m.group("u1"))[0]
    try:
        lo = _number(m.group("lo"))
        hi = _number(m.group("hi"))
        lo_c = convert(lo, unit or None, to_unit)
        hi_c = convert(hi, unit or None, to_unit)
    except (UnitError, ValueError):
        return text, None
    out = f"{_fmt(lo_c)}-{_fmt(hi_c)}"
    if rest:
        out += rest if rest[0] in ",;" else f" {rest}"
    if unit and not same_unit(unit, to_unit):
        target = lookup(to_unit)
        return out, f"'{str(text).strip()}' -> '{out}' ({target.symbol})"
    return out, None
