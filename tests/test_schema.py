import pytest

from alpminer import profiles
from alpminer.schema import (EXTRACTION_TOOL, TRIAGE_TOOL, ExtractionResult,
                             TriageResult, apply_backfills, apply_derivations,
                             parse_gpc)

ALD = profiles.load("ald")


def test_ald_validate_minimal_record():
    rec = ALD.validate_record({"material": "Al2O3"})
    assert rec["material"] == "Al2O3"
    assert rec["deposition_temperature_c"] is None
    assert set(rec) == set(ALD.field_names())  # every field present


def test_ald_full_roundtrip_and_unknown_keys_dropped():
    rec = ALD.validate_record({
        "material": "TiO2", "technique": "thermal ALD",
        "metal_precursor": "tetrakis(dimethylamido)titanium",
        "metal_precursor_abbrev": "TDMAT", "co_reactant": "ozone",
        "deposition_temperature_c": 200, "pulse_metal_s": 0.1,
        "purge_metal_s": 10, "gpc_as_reported": "0.045 nm/cycle",
        "confidence": 0.9, "bogus_field": "ignored",
    })
    assert rec["metal_precursor_abbrev"] == "TDMAT"
    assert "bogus_field" not in rec
    assert rec == ALD.validate_record(rec)  # idempotent


def test_material_required_and_nonempty():
    with pytest.raises(profiles.ProfileError, match="material"):
        ALD.validate_record({"material": "   "})
    with pytest.raises(profiles.ProfileError, match="material"):
        ALD.validate_record({})


def test_confidence_clamped_and_evidence_truncated():
    rec = ALD.validate_record({"material": "ZnO", "confidence": 1.7,
                               "evidence_location": "x" * 500})
    assert rec["confidence"] == 1.0
    assert len(rec["evidence_location"]) <= 200


def test_type_mismatch_is_rejected_with_field_name():
    with pytest.raises(profiles.ProfileError, match="pulse_metal_s"):
        ALD.validate_record({"material": "ZnO", "pulse_metal_s": "fast"})


def test_extraction_result_accepts_new_and_legacy_keys():
    new = ExtractionResult.model_validate(
        {"relevant": True, "records": [{"material": "HfO2"}]})
    old = ExtractionResult.model_validate(
        {"reports_own_ald_experiment": True,
         "recipes": [{"material": "HfO2"}], "paper_notes": "x"})
    assert new.relevant and old.relevant
    assert old.records[0]["material"] == "HfO2"
    assert old.notes == "x"
    assert TriageResult.model_validate(
        {"reports_own_ald_experiment": False}).relevant is False


@pytest.mark.parametrize("text,expected", [
    ("1.1 \u00c5/cycle", 0.11),
    ("0.11 nm/cycle", 0.11),
    ("growth of 0.9 A per cycle at 150 C", 0.09),
    ("GPC = 1.05 angstrom/cycle", 0.105),
    ("110 pm cycle-1", 0.11),
    ("0.11 nm cy\u207b\u00b9", 0.11),
    ("2 nm thick film", None),
    (None, None),
    ("", None),
])
def test_parse_gpc(text, expected):
    got = parse_gpc(text)
    assert got is None if expected is None else got == expected


def test_apply_backfills_gpc_and_epc():
    rec = {"gpc_nm_per_cycle": None,
           "gpc_as_reported": "1.1 Å/cycle"}
    assert apply_backfills(rec)["gpc_nm_per_cycle"] == 0.11
    rec = {"epc_nm_per_cycle": None,
           "epc_as_reported": "0.61 A/cycle at 300 C"}
    assert apply_backfills(rec)["epc_nm_per_cycle"] == 0.061
    # existing values are never overwritten
    rec = {"gpc_nm_per_cycle": 0.2, "gpc_as_reported": "0.1 nm/cycle"}
    assert apply_backfills(rec)["gpc_nm_per_cycle"] == 0.2


def test_apply_derivations_gpc_from_thickness_and_cycles():
    # 15 nm over 300 cycles -> 0.05 nm/cycle
    rec = ALD.validate_record({"material": "Al2O3", "film_thickness_nm": 15,
                               "number_of_cycles": 300})
    out = apply_derivations(apply_backfills(rec))
    assert out["gpc_nm_per_cycle"] == 0.05
    assert "auto-derived as 0.05 nm/cycle" in out["notes"]


def test_apply_derivations_does_not_override_reported_or_backfilled_gpc():
    # a directly reported GPC is left untouched
    rec = {"gpc_nm_per_cycle": 0.12, "film_thickness_nm": 15.0,
           "number_of_cycles": 300, "notes": None}
    out = apply_derivations(rec)
    assert out["gpc_nm_per_cycle"] == 0.12
    assert out["notes"] is None
    # a value already backfilled from the *_as_reported twin also wins
    rec = {"gpc_nm_per_cycle": None, "gpc_as_reported": "0.11 nm/cycle",
           "film_thickness_nm": 15.0, "number_of_cycles": 300, "notes": None}
    out = apply_derivations(apply_backfills(rec))
    assert out["gpc_nm_per_cycle"] == 0.11  # from the reported string
    assert out["notes"] is None  # nothing was derived


@pytest.mark.parametrize("thickness,cycles", [
    (None, 300), (15.0, None), (15.0, 0), (15.0, -5),
])
def test_apply_derivations_needs_positive_thickness_and_cycles(thickness, cycles):
    rec = {"gpc_nm_per_cycle": None, "film_thickness_nm": thickness,
           "number_of_cycles": cycles, "notes": None}
    out = apply_derivations(rec)
    assert out["gpc_nm_per_cycle"] is None
    assert out["notes"] is None


def test_apply_derivations_inert_when_fields_absent():
    # a profile/record without the thickness or cycle field is untouched
    rec = {"gpc_nm_per_cycle": None, "notes": None}
    assert apply_derivations(dict(rec)) == rec


def test_quantities_are_converted_to_si_and_logged():
    log = []
    rec = ALD.validate_record({
        "material": "Al2O3",
        "deposition_temperature_c": {"value": 473.15, "unit": "K"},
        "pressure_pa": {"value": 1.0, "unit": "Torr"},
        "gpc_nm_per_cycle": {"value": 1.1, "unit": "Å/cycle"},
        "pulse_metal_s": {"value": 50, "unit": "ms"},
        "plasma_power_w": {"value": 300, "unit": "W"},          # already SI
        "film_thickness_nm": 20,                    # bare number: canonical
        "temperature_window_c": "423-573 K",
    }, log)
    assert rec["deposition_temperature_c"] == 200.0
    assert rec["pressure_pa"] == 133.322
    assert rec["gpc_nm_per_cycle"] == 0.11
    assert rec["pulse_metal_s"] == 0.05
    assert rec["plasma_power_w"] == 300
    assert rec["film_thickness_nm"] == 20
    assert rec["temperature_window_c"] == "149.85-299.85"
    assert "pressure_pa: 1 Torr -> 133.322 Pa" in log
    assert not any(line.startswith("plasma_power_w") for line in log)
    assert any(line.startswith("temperature_window_c") for line in log)


def test_unknown_unit_leaves_null_and_explains():
    log = []
    rec = ALD.validate_record({"material": "ZnO", "pressure_pa": {
        "value": 2, "unit": "furlongs"}}, log)
    assert rec["pressure_pa"] is None
    assert "left null" in log[0] and "furlongs" in log[0]
    # a unit of the wrong kind is refused the same way
    rec = ALD.validate_record({"material": "ZnO",
                               "pressure_pa": {"value": 2, "unit": "K"}}, [])
    assert rec["pressure_pa"] is None


def test_quantity_as_text_and_null_value():
    rec = ALD.validate_record({"material": "ZnO",
                               "pressure_pa": "0.5 mbar",
                               "pulse_metal_s": {"value": None, "unit": "s"}})
    assert rec["pressure_pa"] == 50.0
    assert rec["pulse_metal_s"] is None
    with pytest.raises(profiles.ProfileError, match="pressure_pa"):
        ALD.validate_record({"material": "ZnO", "pressure_pa": "low"})


def test_quantity_fields_ask_the_model_for_value_and_unit():
    props = EXTRACTION_TOOL["input_schema"]["properties"]["records"]["items"][
        "properties"]
    assert props["pressure_pa"]["properties"]["unit"]["type"] == "string"
    assert props["pressure_pa"]["required"] == ["value", "unit"]
    assert "converted to Pa" in props["pressure_pa"]["description"]
    assert props["number_of_cycles"]["type"] == ["integer", "null"]


def test_legacy_1x_fields_are_renamed_and_converted():
    from alpminer.schema import upgrade_legacy_fields
    rec = upgrade_legacy_fields(
        {"material": "Al2O3", "pressure_torr": 1.0,
         "gpc_angstrom_per_cycle": 1.1}, ALD.field_names())
    assert rec["pressure_pa"] == 133.322 and rec["gpc_nm_per_cycle"] == 0.11
    assert "pressure_torr" not in rec
    ale = profiles.load("ale")
    rec = upgrade_legacy_fields({"pressure_mtorr": 10.0,
                                 "epc_angstrom_per_cycle": 0.61},
                                ale.field_names())
    assert rec == {"pressure_pa": 1.33322, "epc_nm_per_cycle": 0.061}
    # a profile that still declares the old field keeps it untouched
    assert upgrade_legacy_fields({"pressure_torr": 1.0},
                                 ["pressure_torr"]) == {"pressure_torr": 1.0}


def test_units_metadata_carries_qudt_iris():
    meta = ALD.units_metadata()
    assert meta["pressure_pa"] == {
        "unit": "Pa", "qudt_unit": "http://qudt.org/vocab/unit/PA",
        "quantity_kind": "http://qudt.org/vocab/quantitykind/Pressure"}
    assert meta["gpc_nm_per_cycle"]["per"] == "cycle"
    assert meta["gpc_nm_per_cycle"]["quantity_kind"].endswith("/Thickness")
    assert meta["deposition_temperature_c"]["unit"] == "°C"
    assert "number_of_cycles" not in meta


def test_derived_columns_follow_the_profile_fields():
    derived = ALD.derived_field_names()
    assert derived[0] == "unit_conversions"
    assert {"metal_precursor_cid", "co_reactant_formula",
            "additional_reactants_name"} <= set(derived)
    assert ALD.export_field_names() == ALD.field_names() + derived


def test_backcompat_tool_constants_match_ald_profile():
    schema_props = set(
        EXTRACTION_TOOL["input_schema"]["properties"]["records"]["items"]
        ["properties"])
    assert schema_props == set(ALD.field_names())
    assert TRIAGE_TOOL["input_schema"]["required"] == ["relevant"]


def test_blank_unit_is_kept_and_flagged():
    log = []
    rec = ALD.validate_record({"material": "ZnO", "pressure_pa": {
        "value": 0.5, "unit": ""}}, log)
    assert rec["pressure_pa"] == 0.5
    assert log == ["pressure_pa: 0.5 given without a unit; taken as Pa"]


def _custom_profile(tmp_path, fields):
    pdir = profiles.project_profile_dir(tmp_path)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "c.toml").write_text(
        "name = 'c'\ntriage_prompt='t'\nextraction_prompt='e'\n"
        "[[field]]\nname='m'\nrequired=true\n" + fields, encoding="utf-8")
    return profiles.load("c", tmp_path)


def test_required_quantity_must_be_readable(tmp_path):
    prof = _custom_profile(tmp_path, "[[field]]\nname='t_c'\ntype='number'\n"
                                     "unit='°C'\nrequired=true\n")
    with pytest.raises(profiles.ProfileError, match="t_c: required"):
        prof.validate_record({"m": "x", "t_c": {"value": 2, "unit": "furlong"}})
    assert prof.validate_record({"m": "x", "t_c": {"value": 473.15,
                                                   "unit": "K"}})["t_c"] == 200


def test_integer_quantities_round_instead_of_truncating(tmp_path):
    prof = _custom_profile(tmp_path, "[[field]]\nname='t_c'\ntype='integer'\n"
                                     "unit='°C'\n[[field]]\nname='d_s'\n"
                                     "type='integer'\nunit='s'\n")
    rec = prof.validate_record({"m": "x", "t_c": {"value": 473, "unit": "K"},
                                "d_s": {"value": 999, "unit": "ms"}})
    assert rec["t_c"] == 200 and rec["d_s"] == 1


def test_object_sent_to_a_range_field_is_converted():
    log = []
    rec = ALD.validate_record({"material": "ZnO", "temperature_window_c": {
        "value": 473.15, "unit": "K"}}, log)
    assert rec["temperature_window_c"] == "200"
    assert log == ["temperature_window_c: 473.15 K -> 200 °C"]
    rec = ALD.validate_record({"material": "ZnO", "temperature_window_c": {
        "value": "423-573", "unit": "K"}}, [])
    assert rec["temperature_window_c"] == "149.85-299.85"


def test_legacy_numeric_strings_keep_their_old_unit_and_are_logged():
    from alpminer.schema import upgrade_legacy_fields
    log = []
    rec = upgrade_legacy_fields({"pressure_torr": "1.5",
                                 "gpc_angstrom_per_cycle": "1.1"},
                                ALD.field_names(), log)
    assert rec == {"pressure_pa": 199.984, "gpc_nm_per_cycle": 0.11}
    assert log == [
        "pressure_pa: 1.5 Torr -> 199.984 Pa (1.x field pressure_torr)",
        "gpc_nm_per_cycle: 1.1 Å/cycle -> 0.11 nm/cycle "
        "(1.x field gpc_angstrom_per_cycle)"]
    # a value with its own unit is left for validation to convert
    rec = upgrade_legacy_fields({"pressure_torr": "2 mbar"},
                                ALD.field_names())
    assert ALD.validate_record({"material": "x", **rec})["pressure_pa"] == 200


def test_backfill_conversion_is_logged():
    log = []
    rec = apply_backfills({"gpc_nm_per_cycle": None,
                           "gpc_as_reported": "1.1 Å/cycle at 200 C"}, log)
    assert rec["gpc_nm_per_cycle"] == 0.11
    assert log == ["gpc_nm_per_cycle: 1.1 Å/cycle from gpc_as_reported "
                   "-> 0.11 nm/cycle"]
    log = []
    apply_backfills({"gpc_nm_per_cycle": None,
                     "gpc_as_reported": "0.11 nm/cycle"}, log)
    assert log == []                        # already in the field's unit


def test_1x_profile_copy_keeps_its_angstrom_fields(tmp_path):
    """A project profile written by alpminer 1.x still declares the
    angstrom/cycle field: backfill and derivation fill that field, in its
    unit, and no stray nm/cycle key appears."""
    prof = _custom_profile(
        tmp_path, "[[field]]\nname='gpc_angstrom_per_cycle'\ntype='number'\n"
        "[[field]]\nname='gpc_as_reported'\n[[field]]\n"
        "name='film_thickness_nm'\ntype='number'\n[[field]]\n"
        "name='number_of_cycles'\ntype='integer'\n[[field]]\nname='notes'\n")
    rec = apply_derivations(apply_backfills(prof.validate_record(
        {"m": "Al2O3", "gpc_as_reported": "0.11 nm/cycle at 200 C"})))
    assert rec["gpc_angstrom_per_cycle"] == 1.1 and "gpc_nm_per_cycle" not in rec
    rec = apply_derivations(apply_backfills(prof.validate_record(
        {"m": "Al2O3", "film_thickness_nm": 15, "number_of_cycles": 300})))
    assert rec["gpc_angstrom_per_cycle"] == 0.5
    assert "0.5 Å/cycle" in rec["notes"]
