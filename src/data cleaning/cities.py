"""
Match RTE substation names to French communes and assign each to an administrative region.
Uses manual mapping, commune lookup, and neighbor inference for unmatched nodes.
Reads lignes_transmission.csv; outputs an enriched version with region columns.
"""

import pandas as pd
import unicodedata
import re

COMMUNES_PATH    = "data/raw/communes-france-2025.csv"
TRANSMISSION_IN  = "data/final/lignes_transmission.csv"
TRANSMISSION_OUT = "data/final/lignes_transmission.csv"


# ── 1. Normalisation ──────────────────────────────────────────────────────────
_APOSTROPHES = str.maketrans({"’": " ", "’": " ", "‘": " ", "-": " "})

def normalize(s: str) -> str:
    """Uppercase, remove accents, replace apostrophes and hyphens with spaces."""
    s = str(s).upper().translate(_APOSTROPHES)
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(s.split())


_ARTICLE_RE = re.compile(r"^(.*?)\s+\((L[AE]?|LES)\s*\)$", re.IGNORECASE)

def rte_normalize(nom: str) -> str:
    nom = nom.strip()
    # Move trailing article: "ALOUETTES (LES)" → "LES ALOUETTES" (French article inversion)
    m = _ARTICLE_RE.match(nom)
    if m:
        nom = f"{m.group(2).strip()} {m.group(1).strip()}"
    # Expand abbreviated saints
    nom = re.sub(r"\bSTE-", "SAINTE-", nom)
    nom = re.sub(r"\bSTE\b", "SAINTE",  nom)
    nom = re.sub(r"\bST-",  "SAINT-",  nom)
    nom = re.sub(r"\bST\b", "SAINT",   nom)
    return normalize(nom)


# ── 2. Build commune lookup ───────────────────────────────────────────────────
communes = pd.read_csv(COMMUNES_PATH, index_col=0)
communes["_key"] = communes["nom_standard_majuscule"].apply(normalize)

_LEADING_ART = re.compile(r"^(LE|LA|LES|L)\s+(.+)$")

def _add(lkp: dict, amb: set, key: str, val: tuple) -> None:
    if key in lkp:
        if lkp[key][0] != val[0]:
            amb.add(key)
    else:
        lkp[key] = val

lookup:    dict[str, tuple] = {}
ambiguous: set[str]         = set()
for _, row in communes.iterrows():
    key = row["_key"]
    val = (int(row["reg_code"]), row["reg_nom"],
           row["latitude_centre"], row["longitude_centre"])
    _add(lookup, ambiguous, key, val)
    # Index without leading article: "LE GRAND QUEVILLY" → "GRAND QUEVILLY"
    m = _LEADING_ART.match(key)
    if m:
        _add(lookup, ambiguous, m.group(2), val)


# Prefix index for merged-commune matching (e.g. OULLINS → OULLINS-PIERRE-BENITE)
_prefix_index: list[tuple[str, tuple]] = sorted(
    lookup.items(), key=lambda x: len(x[0]), reverse=True
)

def _prefix_match(key: str) -> tuple | None:
    """Find a commune whose normalized name STARTS WITH key (min 5 chars)."""
    if len(key) < 5:
        return None
    candidates = [val for k, val in _prefix_index if k.startswith(key + " ") or k == key]
    if not candidates:
        return None
    # Only accept if all candidates agree on the region (unambiguous)
    regions = {c[0] for c in candidates}
    return candidates[0] if len(regions) == 1 else None


# ── 3. Manual mapping for known substations not matching communes ─────────────
# Format: (reg_code, reg_nom, lat, lon)  |  reg_code=0 → foreign
_IDF  = (11, "Île-de-France")
_CVL  = (24, "Centre-Val de Loire")
_BFC  = (27, "Bourgogne-Franche-Comté")
_NOR  = (28, "Normandie")
_HDF  = (32, "Hauts-de-France")
_GES  = (44, "Grand Est")
_PDL  = (52, "Pays de la Loire")
_BRE  = (53, "Bretagne")
_NAQ  = (75, "Nouvelle-Aquitaine")
_OCC  = (76, "Occitanie")
_ARA  = (84, "Auvergne-Rhône-Alpes")
_PAC  = (93, "Provence-Alpes-Côte d'Azur")
_COR  = (94, "Corse")
_EXT  = (0,  "Étranger")

def _m(reg, lat, lon): return (*reg, lat, lon)

MANUAL: dict[str, tuple] = {
    # ── Île-de-France ──────────────────────────────────────────────────────
    "MORBRAS":              _m(_IDF, 48.76,  2.59),
    "MORBRAS (POSTE SOURCE)": _m(_IDF, 48.76, 2.59),
    "COSSIGNY":             _m(_IDF, 48.74,  2.71),
    "MITRY":                _m(_IDF, 48.99,  2.62),
    "NOISY":                _m(_IDF, 48.90,  2.55),
    "VAIRES":               _m(_IDF, 48.87,  2.65),
    "VAIRES (-SUR-MARNE)":  _m(_IDF, 48.87,  2.65),
    "SENART":               _m(_IDF, 48.64,  2.51),
    "YVELINES-OUEST":       _m(_IDF, 48.77,  1.85),
    "HERBLAY":              _m(_IDF, 48.99,  2.17),
    "CHARENTON":            _m(_IDF, 48.82,  2.41),
    "RUEIL":                _m(_IDF, 48.88,  2.19),
    "LEVALLOIS":            _m(_IDF, 48.90,  2.29),
    "VITRY-NORD":           _m(_IDF, 48.80,  2.40),
    # Paris intra-muros substations
    "AUSTERLITZ":           _m(_IDF, 48.84,  2.37),
    "BATIGNOLLES":          _m(_IDF, 48.88,  2.32),
    "BUTTES CHAUMONT":      _m(_IDF, 48.88,  2.38),
    "BUTTES-CHAUMONT":      _m(_IDF, 48.88,  2.38),
    "CARDINAL LEMOINE":     _m(_IDF, 48.85,  2.35),
    "CARDINET":             _m(_IDF, 48.88,  2.31),
    "CONVENTION":           _m(_IDF, 48.84,  2.30),
    "CRIMEE":               _m(_IDF, 48.89,  2.37),
    "FAIDHERBE":            _m(_IDF, 48.85,  2.38),
    "GAMBETTA":             _m(_IDF, 48.86,  2.40),
    "GOBELINS":             _m(_IDF, 48.84,  2.35),
    "GROS-CAILLOU":         _m(_IDF, 48.86,  2.30),
    "JAVEL":                _m(_IDF, 48.85,  2.28),
    "LONGCHAMP PARIS":      _m(_IDF, 48.86,  2.24),
    "NATION":               _m(_IDF, 48.85,  2.39),
    "NATION III":           _m(_IDF, 48.85,  2.39),
    "ORNANO":               _m(_IDF, 48.89,  2.35),
    "PAPIN":                _m(_IDF, 48.87,  2.35),
    "PERE-LACHAISE":        _m(_IDF, 48.86,  2.39),
    "PYRAMIDES":            _m(_IDF, 48.86,  2.34),
    "TERNES":               _m(_IDF, 48.88,  2.30),
    "TOLBIAC":              _m(_IDF, 48.83,  2.36),
    "TURGOT":               _m(_IDF, 48.87,  2.35),
    # ── Centre-Val de Loire ────────────────────────────────────────────────
    "CHESNOY (LE)":         _m(_CVL, 47.93,  2.77),
    # ── Normandie ─────────────────────────────────────────────────────────
    "GRAND-QUEVILLY":       _m(_NOR, 49.43,  1.02),
    "ROUGEMONTIER":         _m(_NOR, 49.38,  0.85),
    "DIEPPEDALLE":          _m(_NOR, 49.45,  1.01),
    "PORT-JEROME":          _m(_NOR, 49.48,  0.57),
    "BOSCHERVILLE":         _m(_NOR, 49.42,  0.94),
    "HAVRE (LE) (POSTE)":   _m(_NOR, 49.49,  0.11),
    # ── Hauts-de-France ────────────────────────────────────────────────────
    "ARGOEUVES":            _m(_HDF, 49.91,  2.20),
    "HELLEMMES":            _m(_HDF, 50.62,  3.09),
    "LILLE DELIVRANCE":     _m(_HDF, 50.63,  3.03),
    "MOULINS LILLE":        _m(_HDF, 50.60,  3.06),
    "ROUBAIX-NORD":         _m(_HDF, 50.70,  3.18),
    "FLANDRE":              _m(_HDF, 50.90,  2.30),
    "FLANDRE (TR611)":      _m(_HDF, 50.90,  2.30),
    "FLANDRE MARITIME":     _m(_HDF, 50.90,  2.30),
    "WEPPES":               _m(_HDF, 50.60,  2.85),
    # ── Grand Est ─────────────────────────────────────────────────────────
    "CROIX-DE-METZ":        _m(_GES, 48.70,  6.23),
    "DAVID-VILLERS":        _m(_GES, 48.70,  6.30),
    "ILE-NAPOLEON":         _m(_GES, 47.77,  7.30),
    "PORT-DU-RHIN":         _m(_GES, 48.58,  7.79),
    # ── Pays de la Loire ───────────────────────────────────────────────────
    "CORDEMAIS-POSTE":      _m(_PDL, 47.29, -1.86),
    "CORDEMAIS-P":          _m(_PDL, 47.29, -1.86),
    "CHEVIRE":              _m(_PDL, 47.20, -1.59),
    "DOULON":               _m(_PDL, 47.23, -1.52),
    "BLOTTEREAU":           _m(_PDL, 47.21, -1.53),
    # ── Bretagne ──────────────────────────────────────────────────────────
    "PLOUGASTEL":           _m(_BRE, 48.38, -4.39),
    "SQUIVIDAN":            _m(_BRE, 48.00, -4.00),
    "RUMENGOL":             _m(_BRE, 48.28, -4.07),
    "LESTARQUIT":           _m(_BRE, 47.90, -3.87),
    "LOSCOAT":              _m(_BRE, 48.26, -4.66),
    "MORIHAN":              _m(_BRE, 47.64, -2.76),
    # ── Nouvelle-Aquitaine ─────────────────────────────────────────────────
    "BACALAN":              _m(_NAQ, 44.87, -0.55),
    "BOUSCAT":              _m(_NAQ, 44.87, -0.60),
    "FLOIRAC EDF":          _m(_NAQ, 44.83, -0.50),
    "BORDEAUX CENTRE":      _m(_NAQ, 44.84, -0.57),
    "VILLENEUVE-DE-BLAYE":  _m(_NAQ, 45.12, -0.66),
    # ── Occitanie ─────────────────────────────────────────────────────────
    "GINESTOUS":            _m(_OCC, 43.66,  1.44),
    "PORTET-ST-SIMON":      _m(_OCC, 43.53,  1.38),
    "TOULOUSE CENTRE":      _m(_OCC, 43.60,  1.44),
    "BALARUC":              _m(_OCC, 43.44,  3.68),
    # ── Île-de-France (overrides communes in Occitanie / other regions) ──
    "PLAISANCE":            _m(_IDF, 48.84,  2.39),   # substation in Paris 14e (not Plaisance-du-Touch/OCC)
    # ── Grand Est / IDF border ────────────────────────────────────────────
    "FOSSES (LES)":         _m(_IDF, 49.10,  2.52),   # Fosses, Val-d'Oise (connections to Barbuise/GES and Orsonville/IDF)
    # ── Provence-Alpes-Côte d'Azur (communes matched to PDL) ─────────────
    "TRANS":                _m(_PAC, 43.48,  6.48),   # Trans-en-Provence, Var
    "VINS":                 _m(_PAC, 43.41,  6.20),   # Vins-sur-Caramy, Var
    "GUERACHE (LA)":        _m(_PAC, 43.49,  6.47),   # near Trans-en-Provence
    "PIQUAGE A GUERACHE (LA)": _m(_PAC, 43.49,  6.47),
    # ── Auvergne-Rhône-Alpes ───────────────────────────────────────────────
    "GENISSIAT-POSTE":      _m(_ARA, 46.05,  5.80),
    "GENISSIAT(PORTIQUE)":  _m(_ARA, 46.05,  5.80),
    "GRANDE-ILE":           _m(_ARA, 45.73,  6.52),
    "ST-VULBAS-OUEST":      _m(_ARA, 45.84,  5.24),
    "ST-VULBAS-EST":        _m(_ARA, 45.84,  5.26),
    "TRICASTIN-POSTE (LE)": _m(_ARA, 44.33,  4.74),
    "TRICASTIN (LE)":       _m(_ARA, 44.33,  4.74),
    "CROIX-ROUSSE":         _m(_ARA, 45.77,  4.83),
    "VAISE":                _m(_ARA, 45.77,  4.81),
    "PERRACHE":             _m(_ARA, 45.75,  4.83),
    "PIERRE-BENITE":        _m(_ARA, 45.69,  4.82),
    "ISLE D ABEAU":         _m(_ARA, 45.61,  5.23),
    "FLEYRIAT":             _m(_ARA, 46.20,  5.22),
    "GRAND-COEUR":          _m(_ARA, 45.52,  6.52),
    "GRAND-COURBIS":        _m(_ARA, 45.10,  4.85),
    "LONGEFAN":             _m(_ARA, 45.20,  6.61),
    # ── Provence-Alpes-Côte d'Azur ─────────────────────────────────────────
    "ARENC":                _m(_PAC, 43.35,  5.36),
    "CAILLOLS":             _m(_PAC, 43.30,  5.43),
    "MAZARGUES":            _m(_PAC, 43.25,  5.40),
    "SEPTEMES":             _m(_PAC, 43.40,  5.33),
    "BELLE-DE-MAI":         _m(_PAC, 43.31,  5.38),
    "RABATAU":              _m(_PAC, 43.28,  5.41),
    "HENRI-PAUL":           _m(_PAC, 43.30,  5.38),
    "DIGUE-DES-FRANCAIS":   _m(_PAC, 43.71,  7.28),
    "PLAN-DE-GRASSE":       _m(_PAC, 43.68,  6.93),
    "LINGOSTIERE":          _m(_PAC, 43.72,  7.22),
    "REALTOR":              _m(_PAC, 43.43,  5.38),
    "MARTIGUES-PONTEAU":    _m(_PAC, 43.38,  5.04),
    "PONTEAU":              _m(_PAC, 43.37,  5.04),
    "LAVERA":               _m(_PAC, 43.39,  5.04),
    "SAUMATY":              _m(_PAC, 43.36,  5.32),
    "DURANNE (LA)":         _m(_PAC, 43.50,  5.37),
    "ENCO-DE-BOTTE":        _m(_PAC, 43.35,  5.46),
    # ── Corse ─────────────────────────────────────────────────────────────
    "ARRIGHI":              _m(_COR, 42.07,  9.02),
    "LAPARAN":              _m(_COR, 41.85,  9.08),
    "LUCCIANA":             _m(_COR, 42.52,  9.41),
    # ── Normandie (cont.) ─────────────────────────────────────────────────
    "PENLY (POSTE EVACUATION)": _m(_NOR, 49.98,  1.21),
    "PENLY (POSTE CENTRALE)":   _m(_NOR, 49.98,  1.21),
    "TAUTE":                    _m(_NOR, 49.27, -1.28),   # near Saint-Lô, Manche
    "GREPILLES":                _m(_ARA, 44.87,  5.02),   # Isère (near Grenoble) – connections to ARA
    # ── Grand Est (cont.) ─────────────────────────────────────────────────
    "CHOOZ B":                  _m(_GES, 50.09,  4.79),   # Chooz NPP, Ardennes
    "MOULAINE":                 _m(_GES, 49.50,  5.77),   # Meurthe-et-Moselle
    "BOCTOIS":                  _m(_GES, 48.27,  4.12),
    # ── Bourgogne-Franche-Comté ────────────────────────────────────────────
    "VIELMOULIN":               _m(_BFC, 47.36,  4.72),   # Côte-d'Or
    "SEREIN":                   _m(_BFC, 47.60,  3.90),   # Yonne river valley
    "VESLE":                    _m(_GES, 49.25,  4.03),   # Marne/Champagne → Grand Est (not BFC)
    "GATINAIS":                 _m(_CVL, 47.95,  2.50),   # Loiret/Gâtinais
    # ── Auvergne-Rhône-Alpes: communes matched to HDF/BFC in error ────────
    "VIEUX-MOULIN":             _m(_ARA, 45.40,  6.60),   # near Savoie hydro (LA SAUSSAZ, LONGEFAN)
    "PRESSY":                   _m(_ARA, 46.00,  6.60),   # near Cornier/Vallorcine, Haute-Savoie
    "PASSY":                    _m(_ARA, 45.92,  6.68),   # Passy, Haute-Savoie (74), not Seine-et-Marne
    "CHEDDE (S.N.C.F.)":        _m(_ARA, 45.91,  6.67),   # near Passy/Sallanches, Haute-Savoie
    "JUVIGNY":                  _m(_ARA, 46.05,  6.52),   # Juvigny, Haute-Savoie (74)
    "SABLES (LES)":             _m(_ARA, 45.00,  5.70),   # near CHAMPAGNIER/CORDEAC (Isère)
    "THIERS":                   _m(_HDF, 50.39,  3.51),   # substation in Nord (Valenciennes area) – not the Puy-de-Dôme city
    # ── Nouvelle-Aquitaine: communes matched to ARA in error ──────────────
    "BEAULIEU":                 _m(_NAQ, 44.98,  1.83),   # Beaulieu-sur-Dordogne, Corrèze (19)
    "CHASTANG 1":               _m(_NAQ, 45.20,  2.10),   # Chastang Dam, Corrèze (19)
    "CHASTANG (LE)":            _m(_NAQ, 45.20,  2.10),   # same
    "SIRMIERE":                 _m(_NAQ, 46.00,  1.00),   # near border PDL/NAQ; connections to BEAULIEU/MERLATIERE
    # ── Auvergne-Rhône-Alpes: communes matched to NAQ in error ────────────
    "ST-VALLIER":               _m(_ARA, 45.17,  4.82),   # Saint-Vallier, Drôme (26)
    "PRATCLAUX":                _m(_ARA, 44.97,  3.85),   # Haute-Loire (43)
    "ECHALAS":                  _m(_ARA, 45.58,  4.76),   # Rhône (69)
    # ── Pays de la Loire: communes matched to ARA/BFC in error ───────────
    "ST-JOSEPH":                _m(_PDL, 47.27, -1.44),   # near Nantes, Loire-Atlantique (44)
    "ST-BARTHELEMY":            _m(_PDL, 47.29, -1.50),   # near Nantes area
    "RECOUVRANCE":              _m(_PDL, 47.22, -1.57),   # near Nantes; connections to CHEVIRE/CHOLET/MERLATIERE
    # ── Hauts-de-France (cont.) ────────────────────────────────────────────
    "WARANDE":                  _m(_HDF, 50.95,  2.19),   # near Bourbourg, Nord
    "CRECHETS (LES)":           _m(_HDF, 50.55,  2.68),   # near Lestrem/Weppes, Pas-de-Calais
    "ST-BRICE":                 _m(_HDF, 49.52,  3.60),   # near Ormes/Aisne, connected to HDF substations
    "LIMEUX":                   _m(_HDF, 50.06,  2.21),   # Limeux, Somme (80)
    "COQUEREL":                 _m(_HDF, 50.05,  2.20),   # near Limeux, Somme (80)
    "DUNES (LES)":              _m(_HDF, 51.02,  2.49),   # near Grande-Synthe/Dunkerque, Nord
    "VERTEFEUILLE":             _m(_HDF, 50.47,  2.35),   # near Weppes/Pas-de-Calais, Nord
    # ── Auvergne-Rhône-Alpes (cont.) ───────────────────────────────────────
    "MOUCHE (LA)":              _m(_ARA, 45.71,  4.83),   # La Mouche, Oullins/Lyon area (Rhône, 69)
    "BELLE-ETOILE":             _m(_ARA, 45.71,  4.92),   # near Oullins, Rhône (Saint-Priest area)
    "BEC (LE)":                 _m(_ARA, 45.65,  4.87),   # Le Bec near Lyon (connected to Soleil/Oullins cluster)
    "CHAFFARD (LE)":            _m(_ARA, 45.18,  5.87),   # near Grenoble, Isère
    "PIVOZ CORDIER":            _m(_ARA, 45.57,  4.77),   # near Lyon, Rhône
    "PIVOZ-CORDIER":            _m(_ARA, 45.57,  4.77),
    "PRAZ ST ANDRE 1":          _m(_ARA, 45.18,  6.62),   # near Modane, Savoie
    "PRAZ ST ANDRE 2":          _m(_ARA, 45.18,  6.62),
    "PRAZ-ST-ANDRE":            _m(_ARA, 45.18,  6.62),
    "MALGOVERT":                _m(_ARA, 45.54,  6.78),   # Savoie hydro
    "RANDENS":                  _m(_ARA, 45.51,  6.38),   # Savoie
    "TOURS EN SAVOIE":          _m(_ARA, 45.56,  6.47),
    "HERMILLON":                _m(_ARA, 45.33,  6.46),   # Savoie
    # ── Provence-Alpes-Côte d'Azur (cont.) ─────────────────────────────────
    "BOUTRE":                   _m(_PAC, 43.73,  6.12),   # Var hydro
    "LA GAUDIERE":              _m(_PAC, 43.49,  5.45),   # near Aix-en-Provence
    "GAUDIERE (LA)":            _m(_PAC, 43.49,  5.45),
    "SALON":                    _m(_PAC, 43.64,  5.10),   # Salon-de-Provence, Bouches-du-Rhône (13)
    # ── Occitanie (substations matched to GES in error) ─────────────────────
    "TAMAREAU":                 _m(_OCC, 43.64,  3.90),   # near Montpellier (Hérault) – not Metz
    "QUATRE SEIGNEURS":         _m(_OCC, 43.60,  3.85),   # Les Quatre Seigneurs, Hérault
    "CASTELLE (LA)":            _m(_OCC, 43.55,  3.80),   # near Montpellier
    "FOUSCAIS":                 _m(_OCC, 43.57,  3.82),   # near Montpellier
    "SAUMADE":                  _m(_OCC, 43.70,  3.88),   # near Montpellier/Hérault
    # ── Normandie (substations matched to GES in error) ─────────────────────
    "CERNAY":                   _m(_GES, 47.81,  7.17),   # Cernay, Haut-Rhin (68) – NOT NOR
    "MAROLLES":                 _m(_GES, 48.22,  4.17),   # Marolles-sous-Lignières, Aube (10)
    "CHAUSSEE (LA)":            _m(_GES, 48.88,  4.35),   # La Chaussée, Marne (51)
    "CHAPELLE (LA) (CHAPELLE D ARBLAY)": _m(_NOR, 49.67,  1.27),  # La Chapelle-d'Arblay, Seine-Maritime (76)
    # ── Nouvelle-Aquitaine (Corrèze hydro cluster, matched to PAC in error) ──
    "MOLE (LA)":                _m(_NAQ, 45.33,  2.05),   # near Bort-les-Orgues, Corrèze (19)
    "ST-PIERRE-MAREGES":        _m(_NAQ, 45.32,  2.08),   # Marèges Dam, Corrèze (19)
    "BORT":                     _m(_NAQ, 45.39,  2.49),   # Bort-les-Orgues, Corrèze (19)
    # ── Normandie (substations wrongly assigned to NAQ) ──────────────────
    "BARNABOS":                 _m(_NOR, 49.52,  0.97),   # RTE substation Seine-Maritime (all connections to Paluel/Rougemontier)
    "REMISE":                   _m(_NOR, 49.55,  0.65),   # RTE substation near Fécamp; connections to Barnabos/Terrier/Patis
    # ── Overrides: commune ambiguity fixes (wrong region via automatic match) ──
    # Substations near Paris matched to wrong commune (same name, different region)
    "MASSY":                    _m(_IDF, 48.73,  2.27),   # Essonne, near Orly (not NOR)
    "CHEVILLY":                 _m(_IDF, 48.77,  2.34),   # Chevilly-Larue, Val-de-Marne (not CVL)
    "CHATILLON (CLAMART)":      _m(_IDF, 48.80,  2.27),   # Châtillon, Hauts-de-Seine (not ARA)
    "MONTJAY":                  _m(_IDF, 48.88,  2.72),   # Montjay-la-Tour, Seine-et-Marne (not PAC)
    "LIESSE":                   _m(_IDF, 49.05,  3.62),   # actually Aisne → HDF (IDF connections = border)
    "GOUVIEUX":                 _m(_IDF, 49.18,  2.42),   # Oise, near Chantilly (not HDF strictly)
    "NEMOURS":                  _m(_CVL, 48.27,  2.69),   # Seine-et-Marne... but actually IDF
    "GRISOLLES":                _m(_OCC, 43.99,  1.31),   # Tarn-et-Garonne (not HDF)
    "FLAMANVILLE":              _m(_NOR, 49.52, -1.88),   # Manche (nuclear plant, correct)
    "ST-AUBIN":                 _m(_IDF, 48.76,  2.17),   # Saint-Aubin, Essonne (not HDF)
    "MENUEL":                   _m(_NOR, 49.40,  1.05),   # near Criquebeuf/Rouen, Seine-Maritime
    "MENUEL DER TA2":           _m(_GES, 48.57,  4.73),   # Lac du Der, Haute-Marne
    "MENUEL DER TA3":           _m(_GES, 48.57,  4.73),
    "CORBIERE (LA)":            _m(_PDL, 47.43, -1.43),   # near Ancenis, Loire-Atlantique
    "REVIGNY":                  _m(_GES, 48.83,  4.99),   # Revigny-sur-Ornain, Meuse
    "ETIVAL":                   _m(_GES, 48.37,  6.83),   # Étival-Clairefontaine, Vosges
    "VILLERBON":                _m(_CVL, 47.67,  1.43),   # Loir-et-Cher
    "CHANCEAUX":                _m(_CVL, 47.43,  0.68),   # Chanceaux-sur-Choisille, Indre-et-Loire (37)
    "ST-VICTOR":                _m(_OCC, 43.90,  2.61),   # Aveyron area (not ARA)
    "RICHEBOURG":               _m(_HDF, 50.54,  2.69),   # Pas-de-Calais
    "ORMES":                    _m(_HDF, 49.37,  3.52),   # Aisne (not GES)
    "VALENCE":                  _m(_ARA, 44.93,  4.90),   # Drôme (not Charente/NAQ)
    "TARASCON":                 _m(_OCC, 42.85,  1.60),   # Tarascon-sur-Ariège (not PAC)
    "ST JOSEPH":                _m(_PDL, 47.27, -1.44),   # near Nantes, Loire-Atlantique (not ARA)
    # ── Occitanie (cont.) ──────────────────────────────────────────────────
    "HOSPITALET (L)":           _m(_OCC, 42.59,  1.80),   # L'Hospitalet-près-l'Andorre, Ariège
    "REQUISTA -TREBAS":         _m(_OCC, 44.01,  2.54),   # Aveyron
    "CROUX":                    _m(_OCC, 44.04,  2.52),   # near Réquista, Aveyron
    # ── Auvergne-Rhône-Alpes (cont.) ───────────────────────────────────────
    "CORDEAC":                  _m(_ARA, 44.86,  5.73),   # Isère, Drac valley
    "SAUTET (LE)":              _m(_ARA, 44.83,  5.74),   # Isère, Drac valley hydro
    "PONT-ESCOFFIER":           _m(_ARA, 44.80,  4.62),   # Ardèche
    "ST-GUILLERME":             _m(_ARA, 44.79,  4.63),   # Ardèche
    # ── Provence-Alpes-Côte d'Azur (cont.) ─────────────────────────────────
    "SERRE-PONCON":             _m(_PAC, 44.53,  6.31),   # Hautes-Alpes hydro reservoir
    "ARGENTIERE (L)":           _m(_PAC, 44.79,  6.55),   # L'Argentière-la-Bessée, Hautes-Alpes
    # ── Foreign ───────────────────────────────────────────────────────────
    "ACHENE":               _m(_EXT, None, None),
    "AUBANGE":              _m(_EXT, None, None),
    "AVELGEM":              _m(_EXT, None, None),
    "BASSECOURT":           _m(_EXT, None, None),
    "BIESCAS":              _m(_EXT, None, None),
    "CAMPOROSSO":           _m(_EXT, None, None),
    "EICHSTETTEN":          _m(_EXT, None, None),
    "ENSDORF":              _m(_EXT, None, None),
    "GRAU-ROIG":            _m(_EXT, None, None),
    "HERNANI":              _m(_EXT, None, None),
    "LAUFENBOURG":          _m(_EXT, None, None),
    "RONDISSONE":           _m(_EXT, None, None),
    "ROMANEL":              _m(_EXT, None, None),
    "ROZEBOOM":             _m(_EXT, None, None),
    "SAINT-TRIPHON":        _m(_EXT, None, None),
    "ST-TRIPHON":           _m(_EXT, None, None),
    "VERBOIS":              _m(_EXT, None, None),
    "VICH":                 _m(_EXT, None, None),
    "VENAUS":               _m(_EXT, None, None),
    "WOESTYNE":             _m(_EXT, None, None),
    "C.E.R.N.":             _m(_EXT, None, None),
    "RIDDES":               _m(_EXT, None, None),
    "NENTILLA":             _m(_EXT, None, None),
    "LOEWERT":              _m(_EXT, None, None),
    "SPORENINSEL":          _m(_EXT, None, None),
    "BRUNNENWASSER":        _m(_EXT, None, None),
    "BERGHOLZ":             _m(_EXT, None, None),
    "SCHEER":               _m(_EXT, None, None),
}


# ── 4. Match RTE city names ───────────────────────────────────────────────────
df = pd.read_csv(TRANSMISSION_IN)
villes = pd.concat([df["Ville_Depart"], df["Ville_Arrivee"]]).unique()

matched:   dict[str, tuple] = {}
unmatched: list[str]        = []

for ville in villes:
    stripped = re.sub(r"\s*\([^)]*\)", "", ville).strip()
    # Step 1: manual mapping takes priority (overrides ambiguous commune matches)
    if ville in MANUAL:
        matched[ville] = MANUAL[ville]
        continue
    if stripped in MANUAL:
        matched[ville] = MANUAL[stripped]
        continue
    key  = rte_normalize(ville)
    key2 = rte_normalize(stripped)
    # Step 2: exact commune match
    if key in lookup:
        matched[ville] = lookup[key]
        continue
    # Step 3: strip parentheticals, retry exact
    if key2 in lookup:
        matched[ville] = lookup[key2]
        continue
    # Step 4: prefix match (catches merged communes)
    val = _prefix_match(key) or _prefix_match(key2)
    if val:
        matched[ville] = val
    else:
        unmatched.append(ville)

n = len(villes)
print(f"Matched (direct):  {len(matched)}/{n} ({100*len(matched)/n:.1f}%)")

# ── 5. Neighbor inference ─────────────────────────────────────────────────────
# For each unmatched node, vote on its region from the regions of connected nodes.
# Iterate until stable (chains like A-B-C where only C is matched propagate inward).

def _apply(df_: pd.DataFrame, m: dict) -> pd.DataFrame:
    def get_(v, i): return m[v][i] if v in m else None
    df_ = df_.copy()
    df_["Reg_Depart_Code"]  = df_["Ville_Depart"].map(lambda v: get_(v, 0))
    df_["Reg_Depart"]       = df_["Ville_Depart"].map(lambda v: get_(v, 1))
    df_["Reg_Arrivee_Code"] = df_["Ville_Arrivee"].map(lambda v: get_(v, 0))
    df_["Reg_Arrivee"]      = df_["Ville_Arrivee"].map(lambda v: get_(v, 1))
    return df_

df = _apply(df, matched)
still_unmatched = set(unmatched)

for iteration in range(10):
    newly = {}
    for name in list(still_unmatched):
        rows = df[(df["Ville_Depart"] == name) | (df["Ville_Arrivee"] == name)]
        # Collect neighbor regions (exclude foreign nodes — don't propagate from them)
        neighbor_regs = []
        for _, r in rows.iterrows():
            other_reg  = r["Reg_Arrivee"]  if r["Ville_Depart"]  == name else r["Reg_Depart"]
            other_code = r["Reg_Arrivee_Code"] if r["Ville_Depart"] == name else r["Reg_Depart_Code"]
            if pd.notna(other_reg) and other_reg != "Étranger":
                neighbor_regs.append((int(other_code), other_reg))
        if not neighbor_regs:
            continue
        # Majority vote (simple majority, ≥1 vote minimum)
        from collections import Counter
        votes = Counter(neighbor_regs)
        (best_code, best_reg), best_n = votes.most_common(1)[0]
        if best_n / len(neighbor_regs) >= 0.5:   # simple majority
            newly[name] = (best_code, best_reg, None, None)

    if not newly:
        break
    matched.update(newly)
    still_unmatched -= set(newly)
    df = _apply(df, matched)
    print(f"  Iteration {iteration+1}: +{len(newly)} inferred  "
          f"({len(still_unmatched)} still unmatched)")

print(f"Matched (total):   {len(matched)}/{n} ({100*len(matched)/n:.1f}%)")
print(f"Unmatched final:   {len(still_unmatched)}")

both_matched = df["Reg_Depart_Code"].notna() & df["Reg_Arrivee_Code"].notna()
print(f"Matched segments:  {both_matched.sum()}/{len(df)} ({100*both_matched.mean():.1f}%)")

df.to_csv(TRANSMISSION_OUT, index=False)
print(f"Saved -> {TRANSMISSION_OUT}")

if still_unmatched:
    print("\nUnmatched names after inference:")
    for u in sorted(still_unmatched):
        print(f"  {u!r}")
