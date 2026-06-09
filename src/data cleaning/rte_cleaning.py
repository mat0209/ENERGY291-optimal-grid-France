"""
Cleaning script for RTE mix data. Reads raw CSV, drops first two columns, removes intermediate quarter-hour rows, normalizes column names (removing accents, standardizing
names), and writes cleaned CSV to disk. Designed to be run once and produce a clean version of the data for analysis and modeling.
"""

import os
import re
import unicodedata
import pandas as pd


def normalize_column(name: str) -> str:
    # remove accents, lowercase, replace spaces and special chars with underscore
    nf = unicodedata.normalize("NFKD", str(name))
    no_accent = "".join([c for c in nf if not unicodedata.combining(c)])
    s = no_accent.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def clean_mix_rte_2024(in_path: str, out_path: str = None) -> str:
    """Read mix_rte_2024.csv, drop first two columns, remove intermediate quarter-hour rows
    and normalize column names (no accents, simple names). Writes cleaned CSV and
    returns output path.

    Behavior: keeps every second data row starting from the second row of the file
    (i.e. df.iloc[1::2]) which corresponds to removing rows 3,5,7... in 1-based indexing.
    """
    if out_path is None:
        dirname = os.path.dirname(in_path)
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(dirname, f"{base}_clean.csv")

    # file uses semicolon separator and latin-1 encoding (contains accents)
    df = pd.read_csv(in_path, sep=";", encoding="latin-1")

    if df.shape[1] <= 2:
        raise ValueError("Input file has <=2 columns; cannot drop first two columns")

    # drop first two columns by position
    df = df.iloc[:, 2:]

    # remove intermediate quarter-hour rows: keep the main rows (every 2nd row starting at 0)
    df = df.iloc[::2].reset_index(drop=True)

    # nicer, stable column name mapping
    def pretty_column_name(name: str) -> str:
        nf = unicodedata.normalize("NFKD", str(name))
        no_accent = "".join([c for c in nf if not unicodedata.combining(c)])
        s = no_accent.lower()
        s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")

        # common mappings
        if "consommation" in s:
            return "consommation"
        if "prevision" in s and ("j_1" in s or "j-1" in name or "-1" in name):
            return "prevision_j_1"
        if "prevision" in s:
            return "prevision_j"
        if s in ("date", "jour"):
            return "date"
        if "heure" in s:
            return "heures"
        if "nucl" in s:
            return "nucleaire"
        if "eolien" in s:
            if "offshore" in s:
                return "eolien_offshore"
            if "terrestre" in s:
                return "eolien_terrestre"
            return "eolien_total"
        if "solair" in s:
            return "solaire"
        if "hydraul" in s:
            if "fil" in s or "eau" in s:
                return "hydraulique_fil_de_l_eau"
            if "lacs" in s:
                return "hydraulique_lacs"
            if "step" in s or "turbinage" in s:
                return "hydraulique_step_turbinage"
            return "hydraulique"
        if "pompage" in s:
            return "pompage"
        if "bio" in s:
            if "dechet" in s:
                return "bio_dechets"
            if "biomass" in s or "biomasse" in s:
                return "bio_biomasse"
            if "biogaz" in s:
                return "bio_biogaz"
            return "bioenergies"
        if "taux" in s and "co2" in s:
            return "taux_co2"
        # exchanges by country
        if "angleterre" in s:
            return "ech_angleterre"
        if "espagne" in s:
            return "ech_espagne"
        if "italie" in s:
            return "ech_italie"
        if "suisse" in s:
            return "ech_suisse"
        if "allemagne" in s or "belgique" in s:
            return "ech_allemagne_belgique"
        if "ech" in s or "echange" in s:
            return "ech_physiques"
        if "fioul" in s:
            if "tac" in s:
                return "fioul_tac"
            if "cog" in s or "cogi" in s:
                return "fioul_cogen"
            if "autre" in s:
                return "fioul_autres"
            return "fioul"
        if "gaz" in s:
            if "tac" in s:
                return "gaz_tac"
            if "cog" in s or "cogi" in s:
                return "gaz_cogen"
            if "ccg" in s:
                return "gaz_ccg"
            if "autre" in s:
                return "gaz_autres"
            return "gaz"
        if "stock" in s and "batterie" in s:
            if "de" in s or "desto" in s or "destock" in s:
                return "destockage_batterie"
            return "stockage_batterie"

        # fallback: normalized form
        return re.sub(r"_+", "_", s).strip("_")

    df.columns = [pretty_column_name(c) for c in df.columns]

    # post-process common corrupted patterns and disambiguate duplicates
    fixed = []
    for c in df.columns:
        c2 = c
        c2 = c2.replace('pra_vision', 'prevision')
        c2 = c2.replace('pri_1_2vision', 'prevision')
        c2 = c2.replace('pri__vision', 'prevision')
        c2 = c2.replace('pra__vision', 'prevision')
        c2 = re.sub(r'1_2', '', c2)
        c2 = re.sub(r'_+', '_', c2).strip('_')
        fixed.append(c2)

    from collections import Counter
    cnt = Counter(fixed)
    if cnt.get('stockage_batterie', 0) > 1:
        new = []
        seen = 0
        for c in fixed:
            if c == 'stockage_batterie':
                seen += 1
                if seen == 2:
                    new.append('destockage_batterie')
                    continue
            new.append(c)
        fixed = new

    df.columns = fixed

    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    # determine project root (two levels up from this file: src/data -> src -> project)
    here = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(here, "..", ".."))

    default_in = os.path.join(project_root, "data", "raw", "mix_rte_2024.csv")
    default_out_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(default_out_dir, exist_ok=True)
    default_out = os.path.join(default_out_dir, "mix_rte_2024_clean.csv")

    inp = default_in
    if not os.path.exists(inp):
        raise SystemExit(f"Input file not found: {inp}")

    out = clean_mix_rte_2024(inp, out_path=default_out)
    print(out)
