"""Chemical-name normalization for reactant fields.

Papers name the same reactant many ways: diethylzinc appears as diethyl zinc,
DEZ, DEZn, or Zn(C2H5)2, and water as H2O. Profile fields marked
``chemical = true`` (precursors, co-reactants, etchants) are mapped to one
canonical form, following the AtomicLimits ALD/ALE database: element and
compound names per IUPAC recommendations, written as formulas with ligand
abbreviations (AlMe3, Ti(NMe2)4, Hf(NEtMe)4), plus a PubChem compound ID
(Poupaki et al., Chem. Mater. 2026, doi:10.1021/acs.chemmater.6c02250).

Each marked field X gains three derived columns:
    X_name      canonical name (built-in table, else PubChem's title)
    X_formula   AtomicLimits-style formula (built-in table only)
    X_cid       PubChem CID (built-in table, else PubChem lookup)
The text the model extracted stays untouched in X itself.

The built-in table covers the reactants used most in the AtomicLimits
databases; every CID in it was checked against PubChem. Names outside the
table go to PubChem's name search when ``pubchem_lookup`` is on, cached in
data/pubchem_cache.json (misses too). Only full chemical names are sent:
bare abbreviations and formulas are ambiguous there ("TEMAZ" finds
temazepam), and PubChem stores several records for some precursors, so the
table decides those.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .utils import atomic_write_text, log

PUBCHEM_URL = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
               "{name}/property/Title/JSON")
CACHE_FILENAME = "pubchem_cache.json"


@dataclass(frozen=True)
class Reactant:
    formula: str                 # as the AtomicLimits database writes it
    name: str                    # canonical name
    cid: int | None              # PubChem CID
    synonyms: tuple[str, ...] = ()


def _r(formula, name, cid, *synonyms) -> Reactant:
    return Reactant(formula, name, cid, synonyms)


# Ordered roughly by how often the AtomicLimits ALD/ALE databases use them.
REACTANTS: tuple[Reactant, ...] = (
    # co-reactants, gases, and etchants
    _r("H2O", "water", 962, "water", "DI water", "deionized water",
       "water vapor", "water vapour"),
    _r("O3", "ozone", 24823, "ozone"),
    _r("O2", "oxygen", 977, "oxygen", "oxygen gas", "molecular oxygen"),
    _r("N2", "nitrogen", 947, "nitrogen", "nitrogen gas"),
    _r("H2", "hydrogen", 783, "hydrogen", "hydrogen gas", "molecular hydrogen"),
    _r("Ar", "argon", 23968, "argon"),
    _r("He", "helium", 23987, "helium"),
    _r("H2S", "hydrogen sulfide", 402, "hydrogen sulfide", "hydrogen sulphide"),
    _r("NH3", "ammonia", 222, "ammonia"),
    _r("H2O2", "hydrogen peroxide", 784, "hydrogen peroxide"),
    _r("N2H4", "hydrazine", 9321, "hydrazine"),
    _r("N2O", "nitrous oxide", 948, "nitrous oxide", "dinitrogen monoxide"),
    _r("H2Se", "hydrogen selenide", 533, "hydrogen selenide"),
    _r("AsH3", "arsine", 23969, "arsine"),
    _r("PH3", "phosphine", 24404, "phosphine"),
    _r("SiH4", "silane", 23953, "silane", "monosilane"),
    _r("Si2H6", "disilane", 74123, "disilane"),
    _r("B2H6", "diborane", 12544637, "diborane"),
    _r("MeOH", "methanol", 887, "methanol", "CH3OH"),
    _r("HCOOH", "formic acid", 284, "formic acid", "HCO2H"),
    _r("tBuNH2", "tert-butylamine", 6385, "tert-butylamine", "t-butylamine"),
    _r("HF", "hydrogen fluoride", 14917, "hydrogen fluoride", "anhydrous HF"),
    _r("HBr", "hydrogen bromide", 260, "hydrogen bromide"),
    _r("Cl2", "chlorine", 24526, "chlorine", "chlorine gas"),
    _r("BCl3", "boron trichloride", 25135, "boron trichloride"),
    _r("SF6", "sulfur hexafluoride", 17358, "sulfur hexafluoride"),
    _r("SF4", "sulfur tetrafluoride", 24555, "sulfur tetrafluoride"),
    _r("NF3", "nitrogen trifluoride", 24553, "nitrogen trifluoride"),
    _r("CF4", "tetrafluoromethane", 6393, "tetrafluoromethane",
       "carbon tetrafluoride"),
    _r("CHF3", "fluoroform", 6373, "fluoroform", "trifluoromethane"),
    _r("C4F8", "octafluorocyclobutane", 8263, "octafluorocyclobutane",
       "perfluorocyclobutane", "c-C4F8"),
    _r("C4F6", "hexafluoro-1,3-butadiene", 69636, "hexafluoro-1,3-butadiene",
       "hexafluorobutadiene"),
    _r("XeF2", "xenon difluoride", 83674, "xenon difluoride"),
    _r("SO2Cl2", "sulfuryl chloride", 24648, "sulfuryl chloride"),
    _r("SOCl2", "thionyl chloride", 24386, "thionyl chloride"),
    _r("Hacac", "acetylacetone", 31261, "acetylacetone", "acacH",
       "2,4-pentanedione", "pentane-2,4-dione"),
    _r("Hhfac", "hexafluoroacetylacetone", 73706, "hexafluoroacetylacetone",
       "hfacH", "1,1,1,5,5,5-hexafluoro-2,4-pentanedione"),
    _r("PMe3", "trimethylphosphine", 68983, "trimethylphosphine", "P(CH3)3"),
    # metal and semimetal precursors
    _r("AlMe3", "trimethylaluminum", 16682925, "trimethylaluminum",
       "trimethylaluminium", "trimethyl aluminum", "trimethyl aluminium",
       "TMA", "TMAl", "Al(CH3)3", "Al2(CH3)6", "Me3Al", "Al(Me)3"),
    _r("AlMe2Cl", "dimethylaluminum chloride", 79147,
       "dimethylaluminum chloride", "dimethylaluminium chloride",
       "chlorodimethylaluminum", "DMAC", "AlCl(CH3)2", "Al(CH3)2Cl",
       "Me2AlCl", "AlClMe2"),
    _r("AlCl3", "aluminum chloride", 24012, "aluminum chloride",
       "aluminium chloride", "aluminum trichloride", "aluminium trichloride"),
    _r("ZnEt2", "diethylzinc", 101667988, "diethylzinc", "diethyl zinc",
       "DEZ", "DEZn", "Zn(C2H5)2", "Zn(Et)2", "Et2Zn"),
    _r("ZnMe2", "dimethylzinc", 11010, "dimethylzinc", "dimethyl zinc",
       "DMZ", "DMZn", "Zn(CH3)2", "Me2Zn"),
    _r("ZnCl2", "zinc chloride", 5727, "zinc chloride"),
    _r("TiCl4", "titanium tetrachloride", 24193, "titanium tetrachloride",
       "titanium(IV) chloride"),
    _r("Ti(OiPr)4", "titanium(IV) isopropoxide", 11026,
       "titanium(IV) isopropoxide", "titanium isopropoxide",
       "titanium tetraisopropoxide", "TTIP", "Ti(O-iPr)4", "Ti(OCH(CH3)2)4",
       "Ti(OPri)4"),
    _r("Ti(OMe)4", "titanium(IV) methoxide", 5080336,
       "titanium(IV) methoxide", "titanium methoxide", "Ti(OCH3)4"),
    _r("Ti(NMe2)4", "tetrakis(dimethylamido)titanium", 3035369,
       "tetrakis(dimethylamido)titanium", "tetrakis(dimethylamino)titanium",
       "tetrakis(dimethylamido)titanium(IV)",
       "tetrakis(dimethylamino)titanium(IV)", "TDMAT", "Ti(N(CH3)2)4"),
    _r("Ti(NEtMe)4", "tetrakis(ethylmethylamido)titanium", 4439979,
       "tetrakis(ethylmethylamido)titanium",
       "tetrakis(ethylmethylamino)titanium",
       "tetrakis(ethylmethylamido)titanium(IV)", "TEMAT", "Ti(NMeEt)4"),
    _r("TiF4", "titanium tetrafluoride", 121824, "titanium tetrafluoride"),
    _r("GaMe3", "trimethylgallium", 15051, "trimethylgallium", "TMG", "TMGa",
       "Ga(CH3)3"),
    _r("GaEt3", "triethylgallium", 66198, "triethylgallium", "TEG", "TEGa",
       "Ga(C2H5)3"),
    _r("GaCl3", "gallium trichloride", 26010, "gallium trichloride",
       "gallium(III) chloride"),
    _r("InMe3", "trimethylindium", 76919, "trimethylindium", "TMI", "TMIn",
       "In(CH3)3"),
    _r("InEt3", "triethylindium", 101912, "triethylindium", "TEI", "TEIn",
       "In(C2H5)3"),
    _r("Sn(NMe2)4", "tetrakis(dimethylamido)tin", 16684220,
       "tetrakis(dimethylamido)tin", "tetrakis(dimethylamino)tin",
       "TDMASn", "TDMA-Sn", "Sn(N(CH3)2)4"),
    _r("SnCl4", "tin tetrachloride", 24287, "tin tetrachloride",
       "tin(IV) chloride"),
    _r("Sn(acac)2", "tin(II) acetylacetonate", 16687400,
       "tin(II) acetylacetonate", "tin acetylacetonate",
       "bis(acetylacetonato)tin(II)"),
    _r("HfCl4", "hafnium tetrachloride", 37715, "hafnium tetrachloride",
       "hafnium(IV) chloride"),
    _r("Hf(NMe2)4", "tetrakis(dimethylamido)hafnium", 140609,
       "tetrakis(dimethylamido)hafnium", "tetrakis(dimethylamino)hafnium",
       "tetrakis(dimethylamido)hafnium(IV)",
       "tetrakis(dimethylamino)hafnium(IV)", "TDMAH", "TDMAHf",
       "Hf(N(CH3)2)4"),
    _r("Hf(NEtMe)4", "tetrakis(ethylmethylamido)hafnium", 4103752,
       "tetrakis(ethylmethylamido)hafnium",
       "tetrakis(ethylmethylamino)hafnium",
       "tetrakis(ethylmethylamido)hafnium(IV)",
       "tetrakis(ethylmethylamino)hafnium(IV)", "TEMAH", "TEMAHf",
       "Hf(NMeEt)4", "Hf(N(CH3)(C2H5))4"),
    _r("Hf(NEt2)4", "tetrakis(diethylamido)hafnium", 140612,
       "tetrakis(diethylamido)hafnium", "tetrakis(diethylamino)hafnium",
       "tetrakis(diethylamino)hafnium(IV)", "TDEAH", "TDEAHf",
       "Hf(N(C2H5)2)4"),
    _r("ZrCl4", "zirconium tetrachloride", 24817, "zirconium tetrachloride",
       "zirconium(IV) chloride"),
    _r("Zr(NMe2)4", "tetrakis(dimethylamido)zirconium", 140580,
       "tetrakis(dimethylamido)zirconium", "tetrakis(dimethylamino)zirconium",
       "tetrakis(dimethylamino)zirconium(IV)", "TDMAZ", "TDMAZr",
       "Zr(N(CH3)2)4"),
    _r("Zr(NEtMe)4", "tetrakis(ethylmethylamido)zirconium", 4446313,
       "tetrakis(ethylmethylamido)zirconium",
       "tetrakis(ethylmethylamino)zirconium",
       "tetrakis(ethylmethylamido)zirconium(IV)", "TEMAZ", "TEMAZr",
       "Zr(NMeEt)4"),
    _r("Zr(OtBu)4", "zirconium(IV) tert-butoxide", 10221590,
       "zirconium(IV) tert-butoxide", "zirconium tert-butoxide", "ZTB",
       "Zr(OC(CH3)3)4"),
    _r("TaCl5", "tantalum pentachloride", 24394, "tantalum pentachloride",
       "tantalum(V) chloride"),
    _r("Ta(OEt)5", "tantalum(V) ethoxide", 160806, "tantalum(V) ethoxide",
       "tantalum ethoxide", "tantalum pentaethoxide", "Ta(OC2H5)5"),
    _r("Ta(NtBu)(NEt2)3", "tert-butylimidotris(diethylamido)tantalum",
       50911921, "tert-butylimidotris(diethylamido)tantalum",
       "tert-butylimido tris(diethylamido)tantalum", "TBTDET"),
    _r("TaF5", "tantalum pentafluoride", 82218, "tantalum pentafluoride"),
    _r("Nb(OEt)5", "niobium(V) ethoxide", 50897161, "niobium(V) ethoxide",
       "niobium ethoxide", "Nb(OC2H5)5"),
    _r("NbCl5", "niobium pentachloride", 24818, "niobium pentachloride",
       "niobium(V) chloride"),
    _r("NbF5", "niobium pentafluoride", 82217, "niobium pentafluoride"),
    _r("WF6", "tungsten hexafluoride", 522684, "tungsten hexafluoride"),
    _r("W(NtBu)2(NMe2)2", "bis(tert-butylimido)bis(dimethylamido)tungsten",
       16217328, "bis(tert-butylimido)bis(dimethylamido)tungsten",
       "bis(tert-butylimino)bis(dimethylamino)tungsten", "BTBMW"),
    _r("MoCl5", "molybdenum pentachloride", 61497, "molybdenum pentachloride",
       "molybdenum(V) chloride"),
    _r("MoF6", "molybdenum hexafluoride", 82219, "molybdenum hexafluoride"),
    _r("Mo(CO)6", "molybdenum hexacarbonyl", 2724068,
       "molybdenum hexacarbonyl"),
    _r("V(NEtMe)4", "tetrakis(ethylmethylamido)vanadium", 87645524,
       "tetrakis(ethylmethylamido)vanadium",
       "tetrakis(ethylmethylamino)vanadium",
       "tetrakis(ethylmethylamino)vanadium(IV)", "TEMAV", "V(NMeEt)4"),
    _r("VO(OiPr)3", "vanadium(V) oxytriisopropoxide", 79702,
       "vanadium(V) oxytriisopropoxide", "vanadium oxytriisopropoxide",
       "VTOP", "VO(OCH(CH3)2)3"),
    _r("Si(OEt)4", "tetraethyl orthosilicate", 6517,
       "tetraethyl orthosilicate", "tetraethoxysilane", "TEOS",
       "Si(OC2H5)4"),
    _r("SiH(NMe2)3", "tris(dimethylamino)silane", 84795,
       "tris(dimethylamino)silane", "3DMAS", "TDMAS", "SiH(N(CH3)2)3"),
    _r("SiH2(NEt2)2", "bis(diethylamino)silane", 14908161,
       "bis(diethylamino)silane", "BDEAS", "SAM-24", "SiH2(N(C2H5)2)2"),
    _r("SiCl4", "silicon tetrachloride", 24816, "silicon tetrachloride"),
    _r("Si2Cl6", "hexachlorodisilane", 83497, "hexachlorodisilane", "HCDS"),
    _r("Si(OtBu)3OH", "tris(tert-butoxy)silanol", 316157,
       "tris(tert-butoxy)silanol", "tri-tert-butoxysilanol"),
    _r("LiOtBu", "lithium tert-butoxide", 23664764, "lithium tert-butoxide",
       "Li(OtBu)", "LiOC(CH3)3"),
    _r("LiN(SiMe3)2", "lithium bis(trimethylsilyl)amide", 57426343,
       "lithium bis(trimethylsilyl)amide", "LiHMDS"),
    _r("NaOtBu", "sodium tert-butoxide", 45051916, "sodium tert-butoxide",
       "Na(OtBu)"),
    _r("KOtBu", "potassium tert-butoxide", 23665647, "potassium tert-butoxide",
       "K(OtBu)"),
    _r("MgCp2", "bis(cyclopentadienyl)magnesium", 24942211,
       "bis(cyclopentadienyl)magnesium", "magnesocene", "Mg(Cp)2"),
    _r("Mg(CpEt)2", "bis(ethylcyclopentadienyl)magnesium", 45052071,
       "bis(ethylcyclopentadienyl)magnesium", "Mg(EtCp)2"),
    _r("CoCp2", "cobaltocene", 92884, "cobaltocene",
       "bis(cyclopentadienyl)cobalt"),
    _r("NiCp2", "nickelocene", 62390, "nickelocene",
       "bis(cyclopentadienyl)nickel"),
    _r("FeCp2", "ferrocene", 10219726, "ferrocene",
       "bis(cyclopentadienyl)iron"),
    _r("Ni(acac)2", "nickel(II) acetylacetonate", 9834912,
       "nickel(II) acetylacetonate", "nickel acetylacetonate"),
    _r("Cu(acac)2", "copper(II) acetylacetonate", 13669119,
       "copper(II) acetylacetonate", "copper acetylacetonate"),
    _r("Cu(hfac)2", "copper(II) hexafluoroacetylacetonate", 6260710,
       "copper(II) hexafluoroacetylacetonate"),
    _r("Pd(hfac)2", "palladium(II) hexafluoroacetylacetonate", 5831959,
       "palladium(II) hexafluoroacetylacetonate"),
    _r("Pt(CpMe)Me3", "trimethyl(methylcyclopentadienyl)platinum(IV)",
       3024365, "trimethyl(methylcyclopentadienyl)platinum(IV)",
       "trimethyl(methylcyclopentadienyl)platinum",
       "(methylcyclopentadienyl)trimethylplatinum", "MeCpPtMe3",
       "Pt(MeCp)Me3"),
    _r("Ru(CpEt)2", "bis(ethylcyclopentadienyl)ruthenium(II)", 15456827,
       "bis(ethylcyclopentadienyl)ruthenium(II)",
       "bis(ethylcyclopentadienyl)ruthenium", "Ru(EtCp)2"),
    _r("La(thd)3", "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)lanthanum",
       72376397, "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)lanthanum",
       "La(TMHD)3"),
    _r("Y(thd)3", "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)yttrium",
       5829704, "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)yttrium",
       "Y(TMHD)3"),
    _r("Mn(thd)3",
       "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)manganese(III)",
       11456218,
       "tris(2,2,6,6-tetramethyl-3,5-heptanedionato)manganese(III)",
       "Mn(TMHD)3"),
    _r("Sb(OEt)3", "antimony(III) ethoxide", 66326, "antimony(III) ethoxide",
       "antimony ethoxide", "antimony triethoxide", "Sb(OC2H5)3"),
    _r("Te(SiMe3)2", "bis(trimethylsilyl)telluride", 553023,
       "bis(trimethylsilyl)telluride", "(Me3Si)2Te"),
    _r("Se(SiMe3)2", "bis(trimethylsilyl)selenide", 553451,
       "bis(trimethylsilyl)selenide", "(Me3Si)2Se"),
    _r("CdMe2", "dimethylcadmium", 10479, "dimethylcadmium", "DMCd",
       "Cd(CH3)2"),
    _r("PO(OMe)3", "trimethyl phosphate", 10541, "trimethyl phosphate",
       "trimethylphosphate", "TMPO", "P(OMe)3O", "(CH3O)3PO"),
)

# Steps and treatments that are not compounds ("Ar+ ions", "Cl radicals",
# "thermal desorption"); never matched to a gas or sent to PubChem.
_NOT_A_COMPOUND = re.compile(
    r"\b(?:ions?|radicals?|atoms?|neutral|beam|light|anneal(?:ing)?|"
    r"desorption|irradiation|laser|photons?|electrons?|bias|heating|uv|none|"
    r"n/?a|unknown|not stated)\b|\+\s*$", re.IGNORECASE)
_PLASMA_WORD = re.compile(r"\bplasma\b", re.IGNORECASE)
_PLASMA_QUALIFIERS = re.compile(
    r"\b(?:remote|direct|downstream|inductively[- ]coupled|"
    r"capacitively[- ]coupled|icp|ccp|rf|microwave|hollow[- ]cathode|ecr)\b",
    re.IGNORECASE)
_GAS_SPLIT = re.compile(r"\s*(?:/|\+|,|&|\band\b)\s*", re.IGNORECASE)

_STOP_WORDS = ("gas", "vapor", "vapour", "precursor", "anhydrous", "pulse",
               "dose")
_ROMAN_STATE = re.compile(r"\((?:i{1,3}|iv|v|vi{1,3})\)$", re.IGNORECASE)
_SPACED_STATE = re.compile(r"\s+\(((?:i{1,3}|iv|v|vi{1,3}))\)", re.IGNORECASE)


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text)).strip()


def _tidy(text: str) -> str:
    """NFKC (subscripts become digits), single spaces, and oxidation states
    joined to their element: 'cobalt (II) acetylacetonate' ->
    'cobalt(II) acetylacetonate'."""
    s = re.sub(r"\s+", " ", _nfkc(text))
    return _SPACED_STATE.sub(r"(\1)", s)


def _key(text: str) -> str:
    """Case-sensitive lookup key: tidied text, British spellings folded,
    spacing and hyphens removed, a few filler words dropped ("TMA
    precursor" -> "TMA")."""
    s = _tidy(text)
    s = re.sub(r"aluminium", "aluminum", s, flags=re.IGNORECASE)
    s = re.sub(r"sulph", "sulf", s, flags=re.IGNORECASE)
    words = [w for w in s.split(" ") if w.lower() not in _STOP_WORDS]
    return re.sub(r"[\s\-_]+", "", " ".join(words))


def _foldable(alias: str) -> bool:
    # written names fold ("Trimethylaluminum"); formulas and capital-letter
    # abbreviations match exactly, so "DMAc" never becomes "DMAC" and "Hf"
    # never becomes "HF"
    return re.search(r"[a-z]{3}", alias) is not None and not re.search(
        r"(?:^|[^A-Za-z])[A-Z][a-z]?\(", alias)


def _build_index():
    exact: dict[str, Reactant] = {}
    folded: dict[str, Reactant | None] = {}
    for reactant in REACTANTS:
        for alias in (reactant.formula, reactant.name) + reactant.synonyms:
            k = _key(alias)
            exact.setdefault(k, reactant)
            if not _foldable(k):           # judged after filler words go
                continue
            low = k.lower()
            if low in folded and folded[low] is not reactant:
                folded[low] = None                    # ambiguous
            else:
                folded[low] = reactant
    return exact, folded


_EXACT, _FOLDED = _build_index()


def lookup(text) -> Reactant | None:
    """The built-in table entry for a reactant name, formula, or
    abbreviation, or None."""
    if not text or not str(text).strip():
        return None
    k = _key(text)
    hit = _EXACT.get(k) or _FOLDED.get(k.lower())
    if hit is None:
        k2 = _ROMAN_STATE.sub("", k)                 # "...titanium(IV)"
        if k2 != k:
            hit = _EXACT.get(k2) or _FOLDED.get(k2.lower())
    return hit


def _plasma(text: str) -> dict | None:
    """'<gas(es)> plasma' for any wording that names a plasma: 'O2 plasma,
    300 W', 'remote oxygen plasma', 'oxygen (O2) plasma', 'N2/H2 plasma',
    'O2 (remote plasma)'. None when the gas is not in the table."""
    s = _tidy(text)
    m = _PLASMA_WORD.search(s)
    if not m:
        return None
    gas = re.sub(r"\([^()]*\)", " ", s[:m.start()])     # '(O2)', '(50 sccm)'
    gas = _PLASMA_QUALIFIERS.sub(" ", gas.replace("(", " ").replace(")", " "))
    gas = gas.strip(" -,;")
    formulas, names = [], []
    for part in _GAS_SPLIT.split(gas) if gas else []:
        part = part.strip(" -")
        if not part:
            continue
        hit = lookup(part)
        if hit is None:
            return None
        formulas.append(hit.formula)
        names.append(hit.name)
    if not formulas:
        return None
    return {"name": " + ".join(names) + " plasma",
            "formula": " + ".join(formulas) + " plasma", "cid": None}


def _parts(text: str) -> list[str]:
    """[main, inner] when the text ends in a spaced, balanced parenthetical
    ('trimethylaluminum (TMA)', 'TDMAT (Ti(NMe2)4)'); otherwise []. A
    parenthesis glued to a word belongs to a formula or name
    ('Ti(NMe2)4', 'titanium(IV) isopropoxide')."""
    s = _tidy(text)
    if not s.endswith(")"):
        return []
    depth = 0
    for i in range(len(s) - 1, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                break
    else:
        return []
    if i == 0 or s[i - 1] != " ":
        return []
    main, inner = s[:i].strip(), s[i + 1:-1].strip()
    return [main, inner] if main and inner else []


def _query_text(text: str) -> str:
    """What PubChem is asked for: the tidied text without filler words."""
    return " ".join(w for w in _tidy(text).split(" ")
                    if w.lower() not in _STOP_WORDS)


def _looks_like_name(text: str) -> bool:
    """Worth a PubChem name search: a written-out name. Abbreviations,
    formulas with ligand abbreviations ('Cu(dmap)2'), and process steps are
    ambiguous there and are never sent."""
    s = _tidy(text)
    if not 5 <= len(s) <= 120 or _NOT_A_COMPOUND.search(s) \
            or _PLASMA_WORD.search(s) or re.search(r"[;:=<>]", s):
        return False
    if re.search(r"(?:^|[^A-Za-z])[A-Z][a-z]?\(", s) or re.search(r"\)\d", s):
        return False                                 # formula-shaped
    outside = re.sub(r"\([^()]*\)", " ", s)
    return re.search(r"[a-z]{5,}", outside) is not None


class Resolver:
    """Maps reactant text to {name, formula, cid}: the built-in table first,
    then (when online) PubChem's name search, cached on disk. Thread-safe;
    parallel extraction shares one instance."""

    def __init__(self, cache_path: Path | None = None, online: bool = False,
                 delay_s: float = 0.25, timeout_s: float = 15.0,
                 user_agent: str = "alpminer"):
        self.cache_path = Path(cache_path) if cache_path else None
        self.online = online
        self.delay_s = delay_s
        self.timeout_s = timeout_s
        self.user_agent = user_agent
        self._lock = threading.Lock()
        self._last = 0.0
        self._failures = 0
        self._cache: dict[str, dict] = self._load_cache()

    def _load_cache(self) -> dict[str, dict]:
        if not self.cache_path or not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            data = None
        if not isinstance(data, dict):
            aside = self.cache_path.with_suffix(".unreadable.json")
            log.warning("ignoring unreadable %s (kept as %s)",
                        self.cache_path, aside.name)
            try:
                self.cache_path.replace(aside)
            except OSError:
                self.cache_path = None          # never overwrite it
            return {}
        return {k: v for k, v in data.items()
                if isinstance(k, str) and isinstance(v, dict)}

    def resolve(self, text, hint: str | None = None) -> dict | None:
        """{name, formula, cid} for reactant text, or None. The text itself
        decides first: anything naming a plasma is a plasma (or nothing, never
        the bare gas), and process steps such as ions or radicals are not
        compounds. The abbreviation hint is used only when the text is
        unknown."""
        if not isinstance(text, str) or not text.strip():
            return None
        hit = lookup(text)
        if hit is None:
            if _PLASMA_WORD.search(text):
                return _plasma(text)
            if _NOT_A_COMPOUND.search(text):
                return None
        parts = _parts(text)
        for cand in parts + ([hint] if hint else []):
            if hit is not None:
                break
            hit = lookup(cand)
        if hit is not None:
            return {"name": hit.name, "formula": hit.formula, "cid": hit.cid}
        for cand in parts or [text]:
            if not _looks_like_name(cand):
                continue
            found = self._pubchem(_query_text(cand))
            if found and found.get("cid"):
                return {"name": found.get("title"), "formula": None,
                        "cid": found["cid"]}
        return None

    # ---- PubChem -------------------------------------------------------------
    def _pubchem(self, name: str) -> dict | None:
        key = name.casefold()
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            if not self.online or self._failures >= 3:
                return None
            wait = self.delay_s - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            result = self._fetch(name)
            if result is not None:
                self._cache[key] = result
                self._save()
            return result

    def _fetch(self, name: str) -> dict | None:
        """{cid, title} for a hit, {cid: None} for a definite miss (HTTP 404,
        cached), None when PubChem could not answer (not cached)."""
        import requests
        from urllib.parse import quote
        url = PUBCHEM_URL.format(name=quote(name, safe=""))
        try:
            r = requests.get(url, timeout=self.timeout_s,
                             headers={"User-Agent": self.user_agent})
        except requests.RequestException as exc:
            self._failures += 1
            log.debug("PubChem lookup failed for %r: %s", name, exc)
            return None
        if r.status_code == 404:
            self._failures = 0
            return {"cid": None, "title": None}
        try:
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code}")
            props = r.json()["PropertyTable"]["Properties"][0]
            result = {"cid": int(props["CID"]), "title": props.get("Title")}
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            self._failures += 1
            log.debug("PubChem gave no usable answer for %r: %s", name, exc)
            return None
        self._failures = 0
        return result

    def _save(self) -> None:
        if not self.cache_path:
            return
        try:
            atomic_write_text(self.cache_path,
                              json.dumps(self._cache, ensure_ascii=False,
                                         indent=1, sort_keys=True))
        except OSError as exc:
            log.warning("could not write %s: %s", self.cache_path, exc)


def derived_names(field_name: str) -> tuple[str, str, str]:
    return (f"{field_name}_name", f"{field_name}_formula",
            f"{field_name}_cid")


def normalize_record(record: dict, profile, resolver: Resolver) -> dict:
    """Fill the derived name/formula/CID columns of every chemical field."""
    for spec in profile.fields:
        if not spec.chemical:
            continue
        n_name, n_formula, n_cid = derived_names(spec.name)
        value = record.get(spec.name)
        hint = record.get(f"{spec.name}_abbrev")
        if isinstance(value, list):
            hits = [resolver.resolve(v, None) for v in value]
            if any(hits):
                record[n_name] = [h["name"] if h else None for h in hits]
                record[n_formula] = [h["formula"] if h else None
                                     for h in hits]
                record[n_cid] = [h["cid"] if h else None for h in hits]
            else:
                record[n_name] = record[n_formula] = record[n_cid] = None
            continue
        hint = hint if isinstance(hint, str) and hint.strip() else None
        if not (isinstance(value, str) and value.strip()):
            value, hint = hint, None       # only the abbreviation was given
        hit = resolver.resolve(value, hint)
        record[n_name] = hit["name"] if hit else None
        record[n_formula] = hit["formula"] if hit else None
        record[n_cid] = hit["cid"] if hit else None
    return record
