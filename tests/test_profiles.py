import pytest

from alpminer import profiles


def test_builtin_profiles_load_and_are_wellformed():
    for name in ("ald", "ale"):
        p = profiles.load(name)
        assert p.name == name
        assert p.triage_prompt and p.extraction_prompt
        assert any(f.required for f in p.fields)
        tool = p.extraction_tool()
        assert tool["name"] == profiles.RECORD_TOOL_NAME
        inner = tool["input_schema"]["properties"]["records"]["items"]
        assert set(inner["properties"]) == set(p.field_names())
        assert "material" in inner["required"]


def test_ale_profile_has_etch_fields():
    p = profiles.load("ale")
    names = p.field_names()
    assert {"modification_reactant", "removal_reactant",
            "epc_angstrom_per_cycle", "ion_energy_ev",
            "synergy_percent"} <= set(names)
    rec = p.validate_record({"material": "Al2O3", "ale_type": "thermal ALE",
                             "epc_as_reported": "0.61 A/cycle"})
    assert rec["material"] == "Al2O3"


def test_list_profiles_includes_builtins():
    names = {p["name"] for p in profiles.list_profiles()}
    assert {"ald", "ale"} <= names


def test_unknown_profile_error_lists_available():
    with pytest.raises(profiles.ProfileError, match="ald"):
        profiles.load("nope")


def test_project_profile_shadows_and_scaffold_loads(tmp_path):
    dest = profiles.write_new_profile(tmp_path, "mof")
    assert dest.exists()
    p = profiles.load("mof", tmp_path)
    assert p.name == "mof"
    assert p.validate_record({"material": "ZIF-8",
                              "synthesis_temperature_c": 120})[
        "synthesis_temperature_c"] == 120
    listed = {x["name"]: x for x in profiles.list_profiles(tmp_path)}
    assert "mof" in listed and listed["mof"]["origin"] != "built-in"
    # shadowing: a project 'ald' wins over the built-in
    (profiles.project_profile_dir(tmp_path) / "ald.toml").write_text(
        profiles.NEW_PROFILE_TEMPLATE.format(name="ald"), encoding="utf-8")
    assert profiles.load("ald", tmp_path).label == "ald records"
    with pytest.raises(profiles.ProfileError, match="already exists"):
        profiles.write_new_profile(tmp_path, "mof")


def test_array_and_integer_coercion():
    p = profiles.load("ald")
    rec = p.validate_record({"material": "Al2O3",
                             "additional_reactants": "O3",
                             "number_of_cycles": 100.0})
    assert rec["additional_reactants"] == ["O3"]
    assert rec["number_of_cycles"] == 100
    assert isinstance(rec["number_of_cycles"], int)


@pytest.mark.parametrize("body,msg", [
    ("name = 'x'\ntriage_prompt='t'\nextraction_prompt='e'",
     "no \\[\\[field\\]\\]"),
    ("name = 'x'\ntriage_prompt='t'\nextraction_prompt='e'\n"
     "[[field]]\nname='Bad-Name'", "invalid name"),
    ("name = 'x'\ntriage_prompt='t'\nextraction_prompt='e'\n"
     "[[field]]\nname='a'\ntype='blob'", "unknown type"),
    ("name = 'x'\ntriage_prompt='t'\nextraction_prompt='e'\n"
     "[[field]]\nname='a'\n[[field]]\nname='a'", "duplicate"),
    ("name = 'x'\ntriage_prompt='t'\nextraction_prompt='e'\n"
     "[[field]]\nname='a'", "required = true"),
    ("name = 'x'\nextraction_prompt='e'\n"
     "[[field]]\nname='a'\nrequired=true", "triage_prompt"),
])
def test_malformed_profiles_fail_with_clear_messages(tmp_path, body, msg):
    pdir = profiles.project_profile_dir(tmp_path)
    pdir.mkdir(parents=True)
    (pdir / "bad.toml").write_text(body, encoding="utf-8")
    with pytest.raises(profiles.ProfileError, match=msg):
        profiles.load("bad", tmp_path)
