import json

import pytest

from alpminer import chemistry, profiles
from alpminer.chemistry import Resolver


@pytest.mark.parametrize("text,formula", [
    ("trimethylaluminum", "AlMe3"),
    ("trimethylaluminium", "AlMe3"),
    ("TMA", "AlMe3"),
    ("Al(CH3)3", "AlMe3"),
    ("trimethylaluminum (TMA)", "AlMe3"),
    ("diethyl zinc", "ZnEt2"),
    ("DEZn", "ZnEt2"),
    ("Zn(C2H5)2", "ZnEt2"),
    ("H₂O", "H2O"),                        # subscript digits
    ("water vapor", "H2O"),
    ("tetrakis(dimethylamino)titanium(IV)", "Ti(NMe2)4"),
    ("TDMAT (Ti(NMe2)4)", "Ti(NMe2)4"),
    ("TEMAH", "Hf(NEtMe)4"),
    ("DMAC", "AlMe2Cl"),
    ("Hhfac", "Hhfac"),
])
def test_builtin_table_maps_variants_to_one_formula(text, formula):
    hit = Resolver().resolve(text)
    assert hit["formula"] == formula and hit["cid"]


def test_same_compound_gets_one_cid():
    r = Resolver()
    cids = {r.resolve(t)["cid"] for t in
            ("diethylzinc", "diethyl zinc", "DEZ", "DEZn", "Zn(C2H5)2")}
    assert cids == {101667988}


def test_element_symbols_do_not_match_abbreviations_by_case():
    r = Resolver()
    assert r.resolve("HF")["name"] == "hydrogen fluoride"
    assert r.resolve("Hf") is None        # hafnium is not hydrogen fluoride
    assert r.resolve("hf") is None


@pytest.mark.parametrize("text,formula", [
    ("O2 plasma", "O2 plasma"),
    ("remote oxygen plasma", "O2 plasma"),
    ("O2-plasma", "O2 plasma"),
    ("N2/H2 plasma", "N2 + H2 plasma"),
    ("NH3 plasma (ICP)", "NH3 plasma"),
    ("Ar plasma", "Ar plasma"),
])
def test_plasmas(text, formula):
    hit = Resolver().resolve(text)
    assert hit["formula"] == formula and hit["cid"] is None


def test_plasma_wins_over_its_gas_hint():
    # co_reactant 'O2 plasma' with co_reactant_abbrev 'O2' is a plasma
    assert Resolver().resolve("O2 plasma", "O2")["formula"] == "O2 plasma"
    # an unknown text falls back to the abbreviation hint
    assert Resolver().resolve("our usual Al source", "TMA")["formula"] == \
        "AlMe3"


class FakeHTTP:
    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def get(self, url, timeout=None, headers=None):
        self.urls.append(url)
        for key, (status, payload) in self.routes.items():
            if key in url:
                return _Resp(status, payload)
        return _Resp(404, {})


class _Resp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture
def fake_pubchem(monkeypatch):
    import requests
    http = FakeHTTP({
        "tris%28dimethylamino%29antimony": (200, {"PropertyTable": {
            "Properties": [{"CID": 4434406, "Title":
                            "Tris(dimethylamino)antimony"}]}}),
    })
    monkeypatch.setattr(requests, "get", http.get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return http


def test_pubchem_lookup_for_unknown_names_is_cached(tmp_path, fake_pubchem):
    cache = tmp_path / "pubchem_cache.json"
    r = Resolver(cache_path=cache, online=True)
    hit = r.resolve("tris(dimethylamino)antimony")
    assert hit == {"name": "Tris(dimethylamino)antimony", "formula": None,
                   "cid": 4434406}
    assert r.resolve("some made-up precursor") is None    # 404, cached
    assert len(fake_pubchem.urls) == 2
    # a second resolver (next run) answers from the cache file, offline
    r2 = Resolver(cache_path=cache, online=False)
    assert r2.resolve("tris(dimethylamino)antimony")["cid"] == 4434406
    assert r2.resolve("some made-up precursor") is None
    assert len(fake_pubchem.urls) == 2
    saved = json.loads(cache.read_text(encoding="utf-8"))
    assert any(v["cid"] is None for v in saved.values())   # misses cached too


def test_abbreviations_formulas_and_steps_never_go_to_pubchem(fake_pubchem):
    r = Resolver(online=True)
    for text in ("TEMAZr2", "XyZ", "Sb(NMe2)3", "Ar+ ions", "thermal desorption",
                 "LED light", "Cu(dmap)2", "Zn(acac)2", "Ti(OiPr)2(dmae)2",
                 "(hfac)Cu(VTMS)", "chlorine (Cl) radicals", "Ar (50 eV ions)",
                 "krypton plasma"):
        assert r.resolve(text) is None, text
    assert r.resolve("Ar plasma with bias")["formula"] == "Ar plasma"
    assert fake_pubchem.urls == []


def test_offline_resolver_makes_no_requests(fake_pubchem):
    assert Resolver(online=False).resolve("tris(dimethylamino)antimony") \
        is None
    assert fake_pubchem.urls == []


def test_network_failures_stop_lookups_for_the_run(monkeypatch):
    import requests
    calls = []

    def boom(url, timeout=None, headers=None):
        calls.append(url)
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", boom)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = Resolver(online=True)
    for i in range(6):
        assert r.resolve(f"methyl{'x' * i}propylsilane") is None
    assert len(calls) == 3                       # then it stops trying


def test_normalize_record_fills_derived_columns():
    ald = profiles.load("ald")
    rec = {"metal_precursor": "trimethylaluminium", "metal_precursor_abbrev":
           "TMA", "co_reactant": "O2 plasma", "co_reactant_abbrev": "O2",
           "additional_reactants": ["DEZ", "mystery"]}
    chemistry.normalize_record(rec, ald, Resolver())
    assert rec["metal_precursor_formula"] == "AlMe3"
    assert rec["metal_precursor_cid"] == 16682925
    assert rec["co_reactant_formula"] == "O2 plasma"
    assert rec["additional_reactants_formula"] == ["ZnEt2", None]
    assert rec["additional_reactants_cid"] == [101667988, None]


def test_normalize_record_all_unknown_array_is_null():
    ale = profiles.load("ale")
    rec = {"modification_reactant": None, "removal_reactant": "Ar+ ions",
           "additional_reactants": ["mystery"]}
    chemistry.normalize_record(rec, ale, Resolver())
    assert rec["removal_reactant_name"] is None
    assert rec["additional_reactants_cid"] is None


def test_table_entries_are_unique_and_complete():
    formulas = [r.formula for r in chemistry.REACTANTS]
    assert len(formulas) == len(set(formulas))
    assert all(isinstance(r.cid, int) and r.cid > 0 for r in chemistry.REACTANTS)
    cids = [r.cid for r in chemistry.REACTANTS]
    assert len(cids) == len(set(cids))


@pytest.mark.parametrize("text,hint", [
    ("O2 plasma, 300 W", "O2"), ("O2 plasma at 300 W", None),
    ("remote O2 plasma at 300 W", None), ("O2 remote plasma", None),
    ("capacitively-coupled O2 plasma", None), ("O2 plasma pulse", "O2"),
    ("oxygen (O2) plasma", None), ("O2 (remote plasma)", None),
])
def test_plasma_wordings_never_become_the_bare_gas(text, hint):
    assert Resolver().resolve(text, hint)["formula"] == "O2 plasma"


def test_an_unreadable_plasma_is_not_its_gas():
    assert Resolver().resolve("plasma", "O2") is None
    assert Resolver().resolve("Ar (50 eV ions)") is None


def test_parentheticals_with_formulas_inside():
    r = Resolver()
    assert r.resolve("titanium precursor (Ti(NMe2)4)")["formula"] == \
        "Ti(NMe2)4"
    assert r.resolve("tin precursor (Sn(acac)2)")["formula"] == "Sn(acac)2"


def test_spaced_oxidation_state_and_abbr_name_pairs_query_whole_names(
        fake_pubchem):
    r = Resolver(online=True)
    r.resolve("cobalt (II) acetylacetonate")
    assert fake_pubchem.urls[-1].split("/name/")[1].startswith(
        "cobalt%28II%29%20acetylacetonate/")
    hit = r.resolve("TDMASb (tris(dimethylamino)antimony)")
    assert hit["cid"] == 4434406
    assert not any("/name/cobalt/" in u for u in fake_pubchem.urls)


def test_cache_key_is_the_query_itself(tmp_path, monkeypatch):
    import requests
    seen = []

    def get(url, timeout=None, headers=None):
        seen.append(url)
        if "/name/ethanol/" in url:
            return _Resp(200, {"PropertyTable": {"Properties": [
                {"CID": 702, "Title": "Ethanol"}]}})
        return _Resp(404, {})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = Resolver(cache_path=tmp_path / "c.json", online=True)
    assert r.resolve("ethanol vapor")["cid"] == 702   # filler word dropped
    assert r.resolve("ethanol")["cid"] == 702
    assert len(seen) == 1


@pytest.mark.parametrize("content", [
    "{\"x\": {\"cid\": 1}}".encode("utf-16"), b"null", b"[1, 2]",
    b"\xef\xbb\xbf{\"tris(dimethylamino)antimony\": {\"cid\": 4434406, "
    b"\"title\": \"T\"}}",
])
def test_cache_files_that_cannot_be_used_are_set_aside(tmp_path, content):
    cache = tmp_path / "pubchem_cache.json"
    cache.write_bytes(content)
    r = Resolver(cache_path=cache, online=False)
    if content.startswith(b"\xef\xbb\xbf"):          # BOM: still readable
        assert r.resolve("tris(dimethylamino)antimony")["cid"] == 4434406
        return
    assert r.resolve("trimethylaluminum")["formula"] == "AlMe3"
    assert (tmp_path / "pubchem_cache.unreadable.json").exists()


def test_unusable_200_answers_are_not_cached(tmp_path, monkeypatch):
    import requests

    class Html:
        status_code = 200

        def json(self):
            raise ValueError("not JSON")

    monkeypatch.setattr(requests, "get",
                        lambda url, timeout=None, headers=None: Html())
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = Resolver(cache_path=tmp_path / "c.json", online=True)
    assert r.resolve("tris(dimethylamino)antimony") is None
    assert not (tmp_path / "c.json").exists()
    assert r._failures == 1


def test_case_only_abbreviations_stay_apart():
    assert chemistry.lookup("DMAc") is None           # dimethylacetamide
    assert chemistry.lookup("DMAC").formula == "AlMe2Cl"
    assert chemistry.lookup("Hf") is None
    assert chemistry.lookup("anhydrous HF").formula == "HF"
