import pandas as pd
import re

RAW_AERIEN     = "data/raw/lignes-aeriennes-rte-nv.csv"
RAW_SOUTERRAIN = "data/raw/lignes-souterraines-rte-nv.csv"
OUT_PATH       = "data/processed/lignes_transmission.csv"


# ── 1. Load and clean both raw CSVs ──────────────────────────────────────────
def _load_raw(path: str, type_ouvrage: str, has_source_col: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    assert (df["Type ouvrage"] == type_ouvrage).all(), f"Type inattendu dans {path}"
    assert (df["Etat"] == "EN EXPLOITATION").all(),    f"Etat inattendu dans {path}"
    if has_source_col:
        src_col = "Source donn\xe9e"
        assert (df[src_col] == "RTE").all(), f"Source non RTE dans {path}"
        df = df.drop(columns=["Etat", src_col])
    else:
        df = df.drop(columns=["Etat"])

    rows = []
    for _, row in df.iterrows():
        for i in range(1, int(row["Nombre circuit"]) + 1):
            nom = row[f"Nom ligne {i}"]
            if pd.isna(nom):
                continue
            rows.append({
                "Type_Ouvrage": type_ouvrage,
                "Code_Ligne":   row[f"Code ligne {i}"],
                "Nom_Ligne":    nom,
                "Tension":      row["TENSION"],
            })
    return pd.DataFrame(rows)


df_aerien     = _load_raw(RAW_AERIEN,     "AERIEN",     has_source_col=True)
df_souterrain = _load_raw(RAW_SOUTERRAIN, "SOUTERRAIN", has_source_col=False)
df_clean      = pd.concat([df_aerien, df_souterrain], ignore_index=True)

print(f"Aérien:     {len(df_aerien)} circuits")
print(f"Souterrain: {len(df_souterrain)} circuits")
print(f"Total:      {len(df_clean)} circuits")


# ── 2. Build known hyphenated place-names ────────────────────────────────────
def _strip_prefix(nom: str) -> str:
    m = re.match(r'(?:RES\. )?LIAISON \S+ N[O0] \d+ +(.+)', nom)
    return m.group(1).strip() if m else nom


KNOWN: set[str] = set()
for nom in df_clean["Nom_Ligne"]:
    city_part = _strip_prefix(nom)
    if " - " in city_part:
        for part in city_part.split(" - "):
            part = part.strip()
            if "-" in part:
                KNOWN.add(part)

# Supplement verified via web research / geographical knowledge
KNOWN.update({
    "CROIX-DE-METZ",                 # poste RTE près de Nancy (54)
    "PONT-SUR-SAMBRE",               # commune Nord (59)
    "CAGNES-SUR-MER",                # commune Alpes-Maritimes (06)
    "ILE-NAPOLEON",                  # zone industrielle Mulhouse (68)
    "PLAN-DE-GRASSE",                # poste RTE près de Grasse (06)
    "DIGUE-DES-FRANCAIS",            # poste RTE Nice area (06)
    "PORT-DE-BOUC",                  # commune Bouches-du-Rhône (13)
    "PORT-DU-RHIN",                  # zone portuaire Strasbourg (67)
    "GRAND-COEUR",                   # lieu-dit Savoie (73)
    "GRAND-QUEVILLY",                # commune Seine-Maritime (76)
    "STE-MAXENCE",                   # poste Oise (60)
    "ST-VALLIER",                    # commune Drôme (26)
    "ST-NAZAIRE",                    # commune Loire-Atlantique (44)
    "ST-CESAIRE",                    # commune Gard (30)
    "ST-CHAMAS",                     # commune Bouches-du-Rhône (13)
    "ST-ORENS",                      # commune Haute-Garonne (31)
    "ST-BRICE",                      # commune
    "ST-ESTEVE",                     # commune Pyrénées-Orientales (66)
    "ST-BARTHELEMY",                 # commune
    "ST-GUILLERME",                  # poste RTE Ardèche
    "ST-SYLVAIN-D'ANJOU",            # commune Maine-et-Loire (49)
    "ST-PIERRE-DE-BAILLEUL",         # commune Eure (27)
    "ST-ETIENNE-DU-ROUVRAY",         # commune Seine-Maritime (76)
    "SAINT-TRIPHON",                 # lieu-dit frontière franco-suisse
    "ST-TRIPHON",                    # idem abrégé
    "VILLENEUVE-DE-BLAYE",           # commune Gironde (33)
    "PETITE-ROSSELLE",               # commune Moselle (57)
    "DAMPIERRE-EN-BURLY",            # commune Loiret – centrale nucléaire (45)
    "PONT-EVEQUE",                   # commune Isère (38)
    "CROIX-ROUSSE",                  # quartier Lyon / poste RTE (69)
    "DAVID-VILLERS",                 # commune Meurthe-et-Moselle (54)
    "LONG-CHAMP (LE)",               # lieu-dit (poste RTE)
    "GAULT-SAINT-DENIS",             # commune Eure-et-Loir (28)
    "CHARITE-SUR-LOIRE (LA)",        # commune Nièvre (58)
    "AIR-LIQUIDE(A GRANDE SYNTHE)",  # poste industriel Grande-Synthe (59)
    "ST-PIERRE-D ALBIGNY",           # commune Savoie (73) – apostrophe → espace
    "ST-AVRE",                       # commune Savoie (73)
    "ST-CALAIS",                     # commune Sarthe (72)
    "ST-SULPICE",                    # commune Tarn (81)
    "ST-AMOUR",                      # commune / poste Lyon (69)
    "ST-ETIENNE DU ROUVRAY",         # commune Seine-Maritime – écrit avec espaces dans les données
    "BELLE-DE-MAI",                  # quartier Marseille (13)
    "PONT-ESCOFFIER",                # pont / poste RTE Ardèche
    "VITRY-NORD",                    # poste RTE Vitry-sur-Seine (94)
    "PERE-LACHAISE",                 # quartier / poste RTE Paris 20e
})


# ── 3. City-name parser ───────────────────────────────────────────────────────
def _split_outside_parens(s: str) -> list[str]:
    """Split on '-' except when inside parentheses."""
    parts, depth, current = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "-" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _greedy_split(s: str, known: set[str]) -> list[str]:
    """Greedily merge tokens into longest known compound name."""
    tokens = [t.strip() for t in _split_outside_parens(s)]
    result, i = [], 0
    while i < len(tokens):
        matched = False
        for j in range(len(tokens), i, -1):
            candidate = "-".join(tokens[i:j])
            if candidate in known:
                result.append(candidate)
                i = j
                matched = True
                break
        if not matched:
            result.append(tokens[i])
            i += 1
    return [r for r in result if r]


def parse_cities(nom: str, known: set[str]) -> list[str]:
    city_part = _strip_prefix(nom)
    city_part = re.sub(r"^\([^)]+\)\s*", "", city_part)  # strip (AUX1-2) prefix
    if " - " in city_part:
        return [p.strip() for p in city_part.split(" - ") if p.strip()]
    return _greedy_split(city_part, known)


# ── 4. Split into tronçons (one row per consecutive city pair) ────────────────
parsed = df_clean["Nom_Ligne"].apply(lambda n: parse_cities(n, KNOWN))

troncons = []
for (_, row), villes in zip(df_clean.iterrows(), parsed):
    for i in range(len(villes) - 1):
        troncons.append({
            "Type_Ouvrage": row["Type_Ouvrage"],
            "Code_Ligne":   row["Code_Ligne"],
            "Tension":      row["Tension"],
            "Ville_Depart": villes[i],
            "Ville_Arrivee": villes[i + 1],
        })

CAPACITE_MW = {
    ("400kV", "AERIEN"):     1500,
    ("400kV", "SOUTERRAIN"): 1000,
    ("225kV", "AERIEN"):      400,
    ("225kV", "SOUTERRAIN"):  300,
}

df_out = pd.DataFrame(troncons)
df_out["Capacite_MW"] = df_out.apply(
    lambda r: CAPACITE_MW[(r["Tension"], r["Type_Ouvrage"])], axis=1
)
df_out.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(df_out)} tronçons -> {OUT_PATH}")
print(f"Columns: {list(df_out.columns)}")
print(df_out.head(10).to_string())
