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
    ("1.1 \u00c5/cycle", 1.1),
    ("0.11 nm/cycle", pytest.approx(1.1)),
    ("growth of 0.9 A per cycle at 150 C", 0.9),
    ("GPC = 1.05 angstrom/cycle", 1.05),
    ("2 nm thick film", None),
    (None, None),
    ("", None),
])
def test_parse_gpc(text, expected):
    got = parse_gpc(text)
    assert got is None if expected is None else got == expected


def test_apply_backfills_gpc_and_epc():
    rec = {"gpc_angstrom_per_cycle": None,
           "gpc_as_reported": "0.11 nm/cycle"}
    assert apply_backfills(rec)["gpc_angstrom_per_cycle"] == 1.1
    rec = {"epc_angstrom_per_cycle": None,
           "epc_as_reported": "0.61 A/cycle at 300 C"}
    assert apply_backfills(rec)["epc_angstrom_per_cycle"] == 0.61
    # existing values are never overwritten
    rec = {"gpc_angstrom_per_cycle": 2.0, "gpc_as_reported": "0.1 nm/cycle"}
    assert apply_backfills(rec)["gpc_angstrom_per_cycle"] == 2.0


def test_apply_derivations_gpc_from_thickness_and_cycles():
    # 15 nm over 300 cycles -> 150 angstrom / 300 = 0.5 A/cycle
    rec = ALD.validate_record({"material": "Al2O3", "film_thickness_nm": 15,
                               "number_of_cycles": 300})
    out = apply_derivations(apply_backfills(rec))
    assert out["gpc_angstrom_per_cycle"] == 0.5
    assert "auto-derived" in out["notes"]


def test_apply_derivations_does_not_override_reported_or_backfilled_gpc():
    # a directly reported GPC is left untouched
    rec = {"gpc_angstrom_per_cycle": 1.2, "film_thickness_nm": 15.0,
           "number_of_cycles": 300, "notes": None}
    out = apply_derivations(rec)
    assert out["gpc_angstrom_per_cycle"] == 1.2
    assert out["notes"] is None
    # a value already backfilled from the *_as_reported twin also wins
    rec = {"gpc_angstrom_per_cycle": None, "gpc_as_reported": "0.11 nm/cycle",
           "film_thickness_nm": 15.0, "number_of_cycles": 300, "notes": None}
    out = apply_derivations(apply_backfills(rec))
    assert out["gpc_angstrom_per_cycle"] == 1.1  # from the reported string
    assert out["notes"] is None  # nothing was derived


@pytest.mark.parametrize("thickness,cycles", [
    (None, 300), (15.0, None), (15.0, 0), (15.0, -5),
])
def test_apply_derivations_needs_positive_thickness_and_cycles(thickness, cycles):
    rec = {"gpc_angstrom_per_cycle": None, "film_thickness_nm": thickness,
           "number_of_cycles": cycles, "notes": None}
    out = apply_derivations(rec)
    assert out["gpc_angstrom_per_cycle"] is None
    assert out["notes"] is None


def test_apply_derivations_inert_when_fields_absent():
    # a profile/record without the thickness or cycle field is untouched
    rec = {"gpc_angstrom_per_cycle": None, "notes": None}
    assert apply_derivations(dict(rec)) == rec


def test_backcompat_tool_constants_match_ald_profile():
    schema_props = set(
        EXTRACTION_TOOL["input_schema"]["properties"]["records"]["items"]
        ["properties"])
    assert schema_props == set(ALD.field_names())
    assert TRIAGE_TOOL["input_schema"]["required"] == ["relevant"]
