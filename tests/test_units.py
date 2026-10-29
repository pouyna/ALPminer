import pytest

from alpminer import units


@pytest.mark.parametrize("value,src,dst,expected", [
    (473.15, "K", "°C", 200.0),
    (300, "kelvin", "°C", 26.85),
    (392, "°F", "°C", 200.0),
    (200, "degC", "°C", 200.0),
    (200, "ºC", "°C", 200.0),            # ordinal indicator, a common typo
    (200, "℃", "°C", 200.0),             # the single-character degree Celsius
    (1, "Torr", "Pa", 133.322),
    (1, "torr", "Pa", 133.322),
    (10, "mTorr", "Pa", 1.33322),
    (1, "mbar", "Pa", 100.0),
    (2, "hPa", "Pa", 200.0),
    (1, "atm", "Pa", 101325.0),
    (1, "mmHg", "Pa", 133.322),
    (1.1, "Å/cycle", "nm/cycle", 0.11),
    (1.1, "Å/cycle", "nm/cycle", 0.11),  # the angstrom sign
    (1.1, "A/cycle", "nm/cycle", 0.11),
    (1.1, "Å", "nm/cycle", 0.11),        # GPC written with a bare length
    (0.11, "nm cycle⁻¹", "nm/cycle", 0.11),
    (110, "pm/cy", "nm/cycle", 0.11),
    (0.9, "angstrom per cycle", "nm/cycle", 0.09),
    (50, "ms", "s", 0.05),
    (2, "min", "s", 120.0),
    (0.3, "kW", "W", 300.0),
    (1, "keV", "eV", 1000.0),
    (100, "V", "eV", 100.0),                 # bias voltage, singly charged ions
    (5, "µm", "nm", 5000.0),
    (95, "%", "%", 95.0),
])
def test_convert(value, src, dst, expected):
    assert units.convert(value, src, dst) == pytest.approx(expected)


def test_blank_unit_means_already_canonical():
    assert units.convert(1.5, None, "Pa") == 1.5
    assert units.convert(1.5, "  ", "Pa") == 1.5


@pytest.mark.parametrize("src,dst", [
    ("Torr", "°C"), ("furlong", "Pa"), ("nm/cycle", "nm"), ("W/cycle", "nm/cycle"),
])
def test_incompatible_or_unknown_units_raise(src, dst):
    with pytest.raises(units.UnitError):
        units.convert(1.0, src, dst)


def test_prefix_case_is_never_guessed():
    # mPa and MPa differ by 10^9: case-folding must not merge them
    assert units.lookup("mPa").symbol == "mPa"
    assert units.lookup("MPa").symbol == "MPa"
    assert units.lookup("mpa") is None


def test_same_unit():
    assert units.same_unit("degC", "°C")
    assert units.same_unit("Å/cycle", "Å/cycle")
    assert not units.same_unit("Torr", "Pa")


@pytest.mark.parametrize("text,expected", [
    ("423-573 K", ("149.85-299.85", "'423-573 K' -> '149.85-299.85' (°C)")),
    ("423-573 (K)", ("149.85-299.85", "'423-573 (K)' -> '149.85-299.85' (°C)")),
    ("150 – 300 °C", ("150-300", None)),
    ("150 °C - 300 °C", ("150-300", None)),
    ("150 to 300 °C (ALD window)", ("150-300 (ALD window)", None)),
    ("150-300, optimum 200", ("150-300, optimum 200", None)),
    ("1,000-1,200 K", ("726.85-926.85", "'1,000-1,200 K' -> '726.85-926.85' (°C)")),
    ("150-300", ("150-300", None)),
    ("about 200", ("about 200", None)),
    ("150-300 furlongs", ("150-300 furlongs", None)),
])
def test_normalize_range(text, expected):
    assert units.normalize_range(text, "°C") == expected


def test_parse_quantity():
    assert units.parse_quantity("1.5 Torr (base)") == (1.5, "Torr")
    assert units.parse_quantity("200°C") == (200.0, "°C")
    assert units.parse_quantity("2e-3 s") == (0.002, "s")
    assert units.parse_quantity("low") is None


def test_metadata_has_iupac_symbol_and_qudt_iris():
    assert units.metadata("Pa") == {
        "unit": "Pa", "qudt_unit": "http://qudt.org/vocab/unit/PA",
        "quantity_kind": "http://qudt.org/vocab/quantitykind/Pressure"}
    gpc = units.metadata("nm/cycle")
    assert gpc["qudt_unit"].endswith("/NanoM") and gpc["per"] == "cycle"
    assert units.metadata("nm", "Thickness")["quantity_kind"].endswith(
        "/Thickness")
    with pytest.raises(units.UnitError):
        units.metadata("parsec")


@pytest.mark.parametrize("text,expected", [
    ("1,200 W", (1200.0, "W")),
    ("1.5 (Torr)", (1.5, "Torr")),
    ("0,5 Torr", None),                 # a decimal comma is not guessed at
    ("12 Torr, base pressure", (12.0, "Torr")),
])
def test_parse_quantity_edge_cases(text, expected):
    assert units.parse_quantity(text) == expected


def test_prefixed_symbols_never_fold_but_words_do():
    assert units.lookup("MW") is None           # megawatt is not milliwatt
    assert units.lookup("mW").symbol == "mW"
    assert units.lookup("TORR").symbol == "Torr"
    assert units.lookup("Kelvin").symbol == "K"


@pytest.mark.parametrize("src", ["° C", "degree C", "Å/ALD cycle",
                                 "nm per ALD cycle", "nm/ALE cycle"])
def test_more_spellings(src):
    assert units.lookup(src) is not None


def test_volt_is_its_own_unit_but_converts_to_ion_energy():
    assert units.metadata("V")["quantity_kind"].endswith("/ElectricPotential")
    assert units.metadata("V")["qudt_unit"].endswith("/unit/V")
    assert units.convert(50, "V", "eV") == 50.0
    with pytest.raises(units.UnitError):
        units.convert(50, "eV", "V")
