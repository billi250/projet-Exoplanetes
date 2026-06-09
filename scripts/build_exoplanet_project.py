import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
NOTEBOOK_PATH = ROOT / "notebooks" / "Projet_E_Exoplanetes.ipynb"
REPORT_PATH = ROOT / "reports" / "rapport_projet_exoplanetes.md"


def source(text):
    text = text.strip("\n")
    return [line + "\n" for line in text.splitlines()]


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source(text),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source(text),
    }


cells = [
    md(
        r"""
# Projet E - Analyse des Exoplanetes

## NASA Exoplanet Archive

Ce notebook realise une analyse complete d'un jeu de donnees d'exoplanetes :

- chargement et filtrage des observations de reference ;
- controle de qualite et nettoyage ;
- analyses univariees et bivariees ;
- feature engineering ;
- analyses croisees par methode et par annee de decouverte ;
- regression lineaire ;
- ACP (PCA) ;
- synthese scientifique.

Le filtre obligatoire du sujet est applique :

```python
df = df[df["default_flag"] == 1]
```

Ce filtre conserve une seule ligne de reference par planete et evite de melanger plusieurs jeux de parametres pour le meme objet.
"""
    ),
    md(
        r"""
---
## 0. Mise en place

Les bibliotheques utilisees sont les bibliotheques classiques de data science disponibles dans Google Colab.
"""
    ),
    code(
        r"""
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
"""
    ),
    md(
        r"""
### 0.1 Chargement du fichier CSV

Le fichier attendu s'appelle `projet_E_dataset_exoplanets.csv`.

Dans Colab, deux options fonctionnent :

1. placer le CSV dans les fichiers de la session Colab ;
2. laisser la cellule demander l'upload si le fichier n'est pas trouve.

Les lignes d'en-tete de la NASA commencent par `#`, donc le chargement utilise `comment="#"`.
"""
    ),
    code(
        r"""
DATASET_NAME = "projet_E_dataset_exoplanets.csv"

candidate_paths = [
    Path(DATASET_NAME),
    Path("data") / DATASET_NAME,
    Path("..") / "data" / DATASET_NAME,
    Path("/content") / DATASET_NAME,
    Path("/content/data") / DATASET_NAME,
    Path("/content/drive/MyDrive") / DATASET_NAME,
    Path("/content/drive/MyDrive/projet_exoplanetes") / DATASET_NAME,
    Path("/content/drive/MyDrive/projet_exoplanetes/data") / DATASET_NAME,
]

data_path = next((p for p in candidate_paths if p.exists()), None)

if data_path is None:
    try:
        from google.colab import files

        print("Fichier CSV non trouve automatiquement. Selectionnez le fichier :", DATASET_NAME)
        uploaded = files.upload()
        csv_files = [name for name in uploaded if name.lower().endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("Aucun fichier CSV n'a ete uploade.")
        data_path = Path(csv_files[0])
    except Exception as exc:
        raise FileNotFoundError(
            "Placez le fichier projet_E_dataset_exoplanets.csv dans Colab, "
            "ou dans Google Drive puis relancez cette cellule."
        ) from exc

df_raw = pd.read_csv(data_path, comment="#", low_memory=False)
print(f"Fichier charge : {data_path}")
print(f"Donnees brutes : {df_raw.shape[0]:,} lignes x {df_raw.shape[1]:,} colonnes")

df = df_raw[df_raw["default_flag"] == 1].copy()
print(f"Apres filtre default_flag == 1 : {df.shape[0]:,} lignes x {df.shape[1]:,} colonnes")
df.head()
"""
    ),
    md(
        r"""
### 0.2 Dictionnaire des variables retenues

Le dataset contient beaucoup de colonnes techniques. Pour garder une analyse claire, on conserve les variables les plus utiles pour la physique des planetes, des etoiles et les biais de detection.
"""
    ),
    code(
        r"""
column_dictionary = {
    "pl_name": "Nom de la planete",
    "hostname": "Nom de l'etoile hote",
    "default_flag": "Parametres de reference",
    "sy_snum": "Nombre d'etoiles dans le systeme",
    "sy_pnum": "Nombre de planetes connues dans le systeme",
    "discoverymethod": "Methode de decouverte",
    "disc_year": "Annee de decouverte",
    "disc_facility": "Observatoire ou mission",
    "pl_orbper": "Periode orbitale en jours",
    "pl_orbsmax": "Demi-grand axe orbital en unite astronomique",
    "pl_rade": "Rayon planetaire en rayons terrestres",
    "pl_bmasse": "Masse planetaire en masses terrestres",
    "pl_orbeccen": "Excentricite orbitale",
    "pl_insol": "Flux d'irradiation recu, en flux terrestre",
    "pl_eqt": "Temperature d'equilibre en K",
    "st_spectype": "Type spectral de l'etoile",
    "st_teff": "Temperature effective de l'etoile en K",
    "st_rad": "Rayon stellaire en rayons solaires",
    "st_mass": "Masse stellaire en masses solaires",
    "st_met": "Metallicite stellaire",
    "st_logg": "Gravite de surface stellaire",
    "sy_dist": "Distance du systeme en parsecs",
    "sy_vmag": "Magnitude apparente V",
    "sy_kmag": "Magnitude apparente K",
}

selected_cols = [col for col in column_dictionary if col in df.columns]
df_work = df[selected_cols].copy()

dictionary_table = pd.DataFrame(
    {"colonne": list(column_dictionary.keys()), "description": list(column_dictionary.values())}
)
display(dictionary_table[dictionary_table["colonne"].isin(selected_cols)])
"""
    ),
    md(
        r"""
---
## 1. Prise en main, nettoyage et qualite des donnees

L'objectif est de verifier la structure du dataset, les doublons et les valeurs manquantes avant toute interpretation.
"""
    ),
    code(
        r"""
print("Dimensions :", df_work.shape)
print("\nTypes de donnees :")
display(df_work.dtypes.to_frame("type"))

print("\nDoublons exacts :", df_work.duplicated().sum())
print("Doublons sur le nom de planete :", df_work["pl_name"].duplicated().sum())

missing = df_work.isna().sum().to_frame("valeurs_manquantes")
missing["pourcentage"] = (missing["valeurs_manquantes"] / len(df_work) * 100).round(2)
missing = missing.sort_values("pourcentage", ascending=False)
display(missing)

plt.figure(figsize=(10, 7))
sns.barplot(
    data=missing.reset_index().rename(columns={"index": "colonne"}),
    y="colonne",
    x="pourcentage",
    color="#4C78A8",
)
plt.title("Part des valeurs manquantes par variable")
plt.xlabel("Valeurs manquantes (%)")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
numeric_cols = [
    "sy_snum", "sy_pnum", "disc_year", "pl_orbper", "pl_orbsmax",
    "pl_rade", "pl_bmasse", "pl_orbeccen", "pl_insol", "pl_eqt",
    "st_teff", "st_rad", "st_mass", "st_met", "st_logg",
    "sy_dist", "sy_vmag", "sy_kmag",
]
numeric_cols = [col for col in numeric_cols if col in df_work.columns]

for col in numeric_cols:
    df_work[col] = pd.to_numeric(df_work[col], errors="coerce")

def iqr_outlier_summary(data, columns):
    rows = []
    for col in columns:
        values = data[col].dropna()
        if len(values) == 0:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((values < lower) | (values > upper)).sum()
        rows.append(
            {
                "variable": col,
                "n": len(values),
                "q1": q1,
                "median": values.median(),
                "q3": q3,
                "outliers_iqr": int(outliers),
                "outliers_%": round(outliers / len(values) * 100, 2),
            }
        )
    return pd.DataFrame(rows).sort_values("outliers_%", ascending=False)

outlier_table = iqr_outlier_summary(
    df_work,
    ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_rad", "st_mass", "sy_dist"],
)
display(outlier_table)
"""
    ),
    md(
        r"""
**Interpretation.** Le dataset filtre contient une ligne par planete de reference. Les doublons ne sont donc pas le probleme principal. La vraie difficulte vient des valeurs manquantes : la masse, la temperature d'equilibre, l'insolation et le type spectral sont absents pour une partie importante des planetes. Ce manque est normal en astronomie, car toutes les methodes de detection ne mesurent pas les memes grandeurs. On ne supprime donc pas toutes les lignes incompletes : on adapte le sous-ensemble de donnees a chaque analyse.
"""
    ),
    md(
        r"""
---
## 2. Analyses univariees

Les variables physiques sont souvent tres asymetriques : quelques planetes ont des periodes, masses ou distances beaucoup plus grandes que la majorite. Les graphiques utilisent donc souvent une transformation logarithmique.
"""
    ),
    code(
        r"""
key_numeric = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_rad", "st_mass", "sy_dist"]
key_numeric = [col for col in key_numeric if col in df_work.columns]

summary = df_work[key_numeric].describe(percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]).T
summary["missing_%"] = (df_work[key_numeric].isna().mean() * 100).round(2)
display(summary.round(3))

log_variables = {"pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "st_rad", "st_mass", "sy_dist"}
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.ravel()

for ax, col in zip(axes, key_numeric):
    values = df_work[col].dropna()
    if col in log_variables:
        values = values[values > 0]
        sns.histplot(np.log10(values), kde=True, ax=ax, color="#4C78A8")
        ax.set_xlabel(f"log10({col})")
    else:
        sns.histplot(values, kde=True, ax=ax, color="#F58518")
        ax.set_xlabel(col)
    ax.set_title(f"Distribution de {col}")

for ax in axes[len(key_numeric):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
box_cols = ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_eqt", "sy_dist"]
box_cols = [col for col in box_cols if col in df_work.columns]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.ravel()

for ax, col in zip(axes, box_cols):
    values = df_work[col].dropna()
    if col in log_variables:
        values = np.log10(values[values > 0])
        x_label = f"log10({col})"
    else:
        x_label = col
    sns.boxplot(x=values, ax=ax, color="#54A24B")
    ax.set_title(f"Boxplot de {col}")
    ax.set_xlabel(x_label)

for ax in axes[len(box_cols):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
**Interpretation.** Les distributions confirment que les exoplanetes connues ne forment pas un echantillon uniforme. Les petites periodes orbitales et les planetes proches de leur etoile sont tres nombreuses, car elles sont plus faciles a detecter par transit ou vitesse radiale. Les masses et rayons presentent aussi une forte asymetrie : quelques objets tres massifs ou tres grands tirent les moyennes vers le haut. Les medianes sont donc plus representatives que les moyennes.
"""
    ),
    md(
        r"""
---
## 3. Feature engineering

On cree des variables derivees pour faciliter l'interpretation :

- une densite apparente en unite terrestre ;
- une densite approximee en g/cm3 ;
- une classe de planete selon le rayon ;
- une classe thermique ;
- une classe stellaire approximee a partir de la temperature de l'etoile ;
- des versions logarithmiques des variables tres asymetriques.
"""
    ),
    code(
        r"""
df_feat = df_work.copy()

valid_density = (df_feat["pl_bmasse"] > 0) & (df_feat["pl_rade"] > 0)
df_feat["density_earth_units"] = np.nan
df_feat.loc[valid_density, "density_earth_units"] = (
    df_feat.loc[valid_density, "pl_bmasse"] / (df_feat.loc[valid_density, "pl_rade"] ** 3)
)
df_feat["density_g_cm3"] = df_feat["density_earth_units"] * 5.51

def classify_planet(row):
    radius = row.get("pl_rade")
    mass = row.get("pl_bmasse")
    if pd.notna(radius):
        if radius < 1.25:
            return "Rocheuse"
        if radius < 2.0:
            return "Super-Terre"
        if radius < 4.0:
            return "Sub-Neptune"
        if radius < 8.0:
            return "Neptune-like"
        return "Geante gazeuse"
    if pd.notna(mass):
        if mass < 2:
            return "Rocheuse"
        if mass < 10:
            return "Super-Terre"
        if mass < 50:
            return "Sub-Neptune"
        if mass < 150:
            return "Neptune-like"
        return "Geante gazeuse"
    return "Non classee"

df_feat["planet_type"] = df_feat.apply(classify_planet, axis=1)

df_feat["temperature_class"] = pd.cut(
    df_feat["pl_eqt"],
    bins=[0, 250, 320, np.inf],
    labels=["Froide", "Temperee", "Chaude"],
)

def stellar_class(teff):
    if pd.isna(teff):
        return "Inconnue"
    if teff < 3700:
        return "M"
    if teff < 5200:
        return "K"
    if teff < 6000:
        return "G"
    if teff < 7500:
        return "F"
    if teff < 10000:
        return "A"
    return "Tres chaude"

df_feat["stellar_class"] = df_feat["st_teff"].apply(stellar_class)

for col in ["pl_orbper", "pl_orbsmax", "pl_rade", "pl_bmasse", "pl_insol", "st_rad", "st_mass", "sy_dist"]:
    if col in df_feat.columns:
        df_feat[f"log_{col}"] = np.where(df_feat[col] > 0, np.log10(df_feat[col]), np.nan)

display(
    df_feat[
        [
            "pl_name", "pl_rade", "pl_bmasse", "density_g_cm3",
            "planet_type", "pl_eqt", "temperature_class", "st_teff", "stellar_class",
        ]
    ].head(10)
)
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

planet_order = df_feat["planet_type"].value_counts().index
sns.countplot(data=df_feat, y="planet_type", order=planet_order, ax=axes[0], palette="viridis")
axes[0].set_title("Nombre de planetes par classe")
axes[0].set_xlabel("Nombre")
axes[0].set_ylabel("")

temp_order = ["Froide", "Temperee", "Chaude"]
sns.countplot(data=df_feat, x="temperature_class", order=temp_order, ax=axes[1], palette="magma")
axes[1].set_title("Classes thermiques")
axes[1].set_xlabel("")
axes[1].set_ylabel("Nombre")

stellar_order = df_feat["stellar_class"].value_counts().index
sns.countplot(data=df_feat, x="stellar_class", order=stellar_order, ax=axes[2], palette="crest")
axes[2].set_title("Classes stellaires approximees")
axes[2].set_xlabel("Classe")
axes[2].set_ylabel("Nombre")

plt.tight_layout()
plt.show()

type_summary = (
    df_feat.groupby("planet_type")
    .agg(
        n=("pl_name", "count"),
        rayon_median=("pl_rade", "median"),
        masse_mediane=("pl_bmasse", "median"),
        densite_mediane=("density_g_cm3", "median"),
        temperature_mediane=("pl_eqt", "median"),
    )
    .sort_values("n", ascending=False)
)
display(type_summary.round(2))
"""
    ),
    md(
        r"""
### 3.1 Recherche de candidates temperees

On isole un petit groupe de planetes potentiellement interessantes : rayon compatible avec une planete rocheuse ou super-Terre, et temperature d'equilibre situee dans une zone temperee. Ce n'est pas une preuve d'habitabilite, mais un filtre exploratoire utile.
"""
    ),
    code(
        r"""
temperate_candidates = df_feat[
    df_feat["pl_eqt"].between(250, 320, inclusive="both")
    & df_feat["pl_rade"].between(0.5, 2.0, inclusive="both")
].copy()

temperate_candidates["data_completeness_score"] = temperate_candidates[
    ["pl_rade", "pl_bmasse", "pl_eqt", "pl_orbper", "sy_dist"]
].notna().sum(axis=1)

candidate_cols = [
    "pl_name", "hostname", "planet_type", "pl_rade", "pl_bmasse",
    "density_g_cm3", "pl_eqt", "pl_orbper", "sy_dist", "discoverymethod",
    "data_completeness_score",
]

print(
    "Nombre de candidates temperees avec rayon entre 0.5 et 2 rayons terrestres :",
    len(temperate_candidates),
)
display(
    temperate_candidates[candidate_cols]
    .sort_values(["data_completeness_score", "sy_dist"], ascending=[False, True])
    .head(15)
    .round(3)
)
"""
    ),
    md(
        r"""
**Interpretation.** La classification montre la diversite des exoplanetes : beaucoup d'objets sont des super-Terres, sub-Neptunes ou geantes gazeuses, mais cette repartition est influencee par les methodes de detection. La densite apparente aide a distinguer les planetes compactes des planetes tres volumineuses. Elle doit toutefois etre interpretee avec prudence car masse et rayon ne sont pas toujours mesures simultanement avec la meme precision. Le tableau des candidates temperees donne une piste scientifique concrete, mais il ne suffit pas a conclure sur l'habitabilite, car l'atmosphere et l'albedo ne sont pas connus.
"""
    ),
    md(
        r"""
---
## 4. Correlations et analyses bivariees

On utilise des correlations de Spearman, plus robustes pour des variables asymetriques et non lineaires. Les variables logarithmiques sont preferees quand les ordres de grandeur sont tres differents.
"""
    ),
    code(
        r"""
corr_features = [
    "log_pl_orbper", "log_pl_orbsmax", "log_pl_rade", "log_pl_bmasse",
    "pl_orbeccen", "log_pl_insol", "pl_eqt", "st_teff", "log_st_rad",
    "log_st_mass", "st_met", "st_logg", "log_sy_dist", "density_g_cm3",
]
corr_features = [col for col in corr_features if col in df_feat.columns]

corr = df_feat[corr_features].corr(method="spearman", numeric_only=True)

plt.figure(figsize=(13, 10))
sns.heatmap(corr, cmap="vlag", center=0, annot=False, linewidths=0.4)
plt.title("Matrice de correlation de Spearman")
plt.tight_layout()
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
top_corr = (
    upper.stack()
    .reset_index()
    .rename(columns={"level_0": "variable_1", "level_1": "variable_2", 0: "correlation"})
)
top_corr["abs_correlation"] = top_corr["correlation"].abs()
display(top_corr.sort_values("abs_correlation", ascending=False).head(12).round(3))
"""
    ),
    code(
        r"""
top_methods = df_feat["discoverymethod"].value_counts().head(4).index
df_feat["method_group"] = np.where(
    df_feat["discoverymethod"].isin(top_methods),
    df_feat["discoverymethod"],
    "Autres",
)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.scatterplot(
    data=df_feat,
    x="pl_rade",
    y="pl_bmasse",
    hue="method_group",
    alpha=0.65,
    ax=axes[0, 0],
)
axes[0, 0].set_xscale("log")
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("Masse planetaire vs rayon planetaire")
axes[0, 0].set_xlabel("Rayon (Terre, log)")
axes[0, 0].set_ylabel("Masse (Terre, log)")

sns.scatterplot(
    data=df_feat,
    x="pl_orbsmax",
    y="pl_eqt",
    hue="method_group",
    alpha=0.65,
    ax=axes[0, 1],
    legend=False,
)
axes[0, 1].set_xscale("log")
axes[0, 1].set_title("Temperature d'equilibre vs distance orbitale")
axes[0, 1].set_xlabel("Demi-grand axe (UA, log)")
axes[0, 1].set_ylabel("Temperature d'equilibre (K)")

sns.scatterplot(
    data=df_feat,
    x="st_mass",
    y="pl_bmasse",
    hue="method_group",
    alpha=0.65,
    ax=axes[1, 0],
    legend=False,
)
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("Masse stellaire vs masse planetaire")
axes[1, 0].set_xlabel("Masse stellaire (Soleil)")
axes[1, 0].set_ylabel("Masse planetaire (Terre, log)")

sns.scatterplot(
    data=df_feat,
    x="st_rad",
    y="pl_rade",
    hue="method_group",
    alpha=0.65,
    ax=axes[1, 1],
    legend=False,
)
axes[1, 1].set_xscale("log")
axes[1, 1].set_yscale("log")
axes[1, 1].set_title("Rayon stellaire vs rayon planetaire")
axes[1, 1].set_xlabel("Rayon stellaire (Soleil, log)")
axes[1, 1].set_ylabel("Rayon planetaire (Terre, log)")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
**Interpretation.** La relation masse-rayon est positive mais dispersee : deux planetes de meme rayon peuvent avoir des masses tres differentes selon leur composition. La temperature d'equilibre diminue globalement quand la distance orbitale augmente, ce qui est coherent avec la physique du rayonnement. Les graphiques montrent aussi des biais de methode : les transits dominent les planetes proches, tandis que la vitesse radiale et l'imagerie apparaissent davantage pour certains objets massifs ou eloignes.
"""
    ),
    md(
        r"""
---
## 5. Analyse croisee

Cette partie relie les proprietes physiques aux methodes de decouverte et a l'evolution historique des detections.
"""
    ),
    code(
        r"""
method_counts = df_feat["discoverymethod"].value_counts()
display(method_counts.to_frame("nombre_de_planetes"))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(
    x=method_counts.head(10).values,
    y=method_counts.head(10).index,
    ax=axes[0],
    palette="cubehelix",
)
axes[0].set_title("Top 10 des methodes de decouverte")
axes[0].set_xlabel("Nombre de planetes")
axes[0].set_ylabel("")

year_counts = df_feat.groupby("disc_year").size()
year_counts.plot(ax=axes[1], color="#4C78A8", linewidth=2)
axes[1].set_title("Nombre de decouvertes par annee")
axes[1].set_xlabel("Annee")
axes[1].set_ylabel("Nombre de planetes")
axes[1].axvline(2009, color="#E45756", linestyle="--", linewidth=1, label="Lancement Kepler")
axes[1].legend()

plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
major_methods = method_counts[method_counts >= 30].index
method_df = df_feat[df_feat["discoverymethod"].isin(major_methods)].copy()

method_summary = (
    method_df.groupby("discoverymethod")
    .agg(
        n=("pl_name", "count"),
        rayon_median=("pl_rade", "median"),
        masse_mediane=("pl_bmasse", "median"),
        periode_mediane=("pl_orbper", "median"),
        distance_mediane=("sy_dist", "median"),
        temperature_mediane=("pl_eqt", "median"),
    )
    .sort_values("n", ascending=False)
)
display(method_summary.round(2))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.boxplot(data=method_df, y="discoverymethod", x="pl_rade", ax=axes[0], palette="viridis")
axes[0].set_xscale("log")
axes[0].set_title("Rayon par methode de decouverte")
axes[0].set_xlabel("Rayon planetaire (Terre, log)")
axes[0].set_ylabel("")

sns.boxplot(data=method_df, y="discoverymethod", x="pl_orbper", ax=axes[1], palette="magma")
axes[1].set_xscale("log")
axes[1].set_title("Periode orbitale par methode de decouverte")
axes[1].set_xlabel("Periode orbitale (jours, log)")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
method_year = pd.crosstab(df_feat["disc_year"], df_feat["method_group"])
method_year = method_year.sort_index()

plt.figure(figsize=(14, 7))
method_year.plot.area(ax=plt.gca(), alpha=0.85)
plt.title("Evolution des decouvertes par methode")
plt.xlabel("Annee")
plt.ylabel("Nombre de decouvertes")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
**Interpretation.** La methode du transit domine fortement l'echantillon, surtout depuis les missions spatiales comme Kepler. Ce n'est pas seulement une information astronomique : c'est aussi un biais observationnel. Les planetes proches de leur etoile produisent plus de transits et sont donc surrepresentees. Les methodes comme l'imagerie directe detectent moins d'objets, mais elles sont plus adaptees a des planetes massives et eloignees.
"""
    ),
    md(
        r"""
---
## 6. Regression lineaire

On cherche a predire la temperature d'equilibre `pl_eqt` a partir de variables stellaires et orbitales. Le choix est physiquement coherent : la temperature d'une planete depend de son eloignement et des proprietes de son etoile.

Le modele reste volontairement simple, car l'objectif est d'interpreter les resultats et les limites d'une regression lineaire.
"""
    ),
    code(
        r"""
reg_cols = ["pl_eqt", "pl_orbsmax", "st_teff", "st_rad", "st_mass", "st_logg"]
reg_df = df_feat[reg_cols].dropna().copy()
reg_df = reg_df[(reg_df["pl_eqt"] > 0) & (reg_df["pl_orbsmax"] > 0) & (reg_df["st_rad"] > 0)]

reg_df["log_pl_orbsmax"] = np.log10(reg_df["pl_orbsmax"])
reg_df["log_st_rad"] = np.log10(reg_df["st_rad"])

features = ["log_pl_orbsmax", "st_teff", "log_st_rad", "st_mass", "st_logg"]
target = "pl_eqt"

print(f"Nombre de lignes utilisables pour la regression : {len(reg_df):,}")

X = reg_df[features]
y = reg_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

baseline = np.repeat(y_train.mean(), len(y_test))

metrics = pd.DataFrame(
    {
        "modele": ["Regression lineaire", "Baseline moyenne"],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, y_pred)),
            np.sqrt(mean_squared_error(y_test, baseline)),
        ],
        "MAE": [
            mean_absolute_error(y_test, y_pred),
            mean_absolute_error(y_test, baseline),
        ],
        "R2": [
            r2_score(y_test, y_pred),
            r2_score(y_test, baseline),
        ],
    }
)
display(metrics.round(3))

coef_table = pd.DataFrame(
    {
        "variable": features,
        "coefficient_standardise": model.coef_,
    }
).sort_values("coefficient_standardise", key=lambda s: s.abs(), ascending=False)
display(coef_table.round(3))
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], alpha=0.7, color="#4C78A8")
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
axes[0].plot([min_val, max_val], [min_val, max_val], color="#E45756", linestyle="--")
axes[0].set_title("Temperature observee vs predite")
axes[0].set_xlabel("Temperature observee (K)")
axes[0].set_ylabel("Temperature predite (K)")

residuals = y_test - y_pred
sns.scatterplot(x=y_pred, y=residuals, ax=axes[1], alpha=0.7, color="#54A24B")
axes[1].axhline(0, color="#E45756", linestyle="--")
axes[1].set_title("Residus du modele")
axes[1].set_xlabel("Temperature predite (K)")
axes[1].set_ylabel("Residus (K)")

plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### 6.1 Variante log-lineaire plus physique

La temperature d'equilibre est liee approximativement a la temperature de l'etoile, au rayon de l'etoile et a la distance orbitale. Une regression lineaire sur `log10(pl_eqt)` permet donc d'obtenir un modele plus proche de la relation physique attendue.
"""
    ),
    code(
        r"""
reg_log_cols = ["pl_eqt", "pl_orbsmax", "st_teff", "st_rad"]
reg_log_df = df_feat[reg_log_cols].dropna().copy()
reg_log_df = reg_log_df[
    (reg_log_df["pl_eqt"] > 0)
    & (reg_log_df["pl_orbsmax"] > 0)
    & (reg_log_df["st_teff"] > 0)
    & (reg_log_df["st_rad"] > 0)
]

reg_log_df["log_pl_eqt"] = np.log10(reg_log_df["pl_eqt"])
reg_log_df["log_pl_orbsmax"] = np.log10(reg_log_df["pl_orbsmax"])
reg_log_df["log_st_teff"] = np.log10(reg_log_df["st_teff"])
reg_log_df["log_st_rad"] = np.log10(reg_log_df["st_rad"])

features_log = ["log_pl_orbsmax", "log_st_teff", "log_st_rad"]
X_log = reg_log_df[features_log]
y_log = reg_log_df["log_pl_eqt"]
y_kelvin = reg_log_df["pl_eqt"]

(
    X_train_log,
    X_test_log,
    y_train_log,
    y_test_log,
    y_train_kelvin,
    y_test_kelvin,
) = train_test_split(X_log, y_log, y_kelvin, test_size=0.2, random_state=42)

scaler_log = StandardScaler()
X_train_log_scaled = scaler_log.fit_transform(X_train_log)
X_test_log_scaled = scaler_log.transform(X_test_log)

model_log = LinearRegression()
model_log.fit(X_train_log_scaled, y_train_log)

y_pred_log = model_log.predict(X_test_log_scaled)
y_pred_kelvin = 10 ** y_pred_log
y_baseline_kelvin = np.repeat(y_train_kelvin.mean(), len(y_test_kelvin))

metrics_log = pd.DataFrame(
    {
        "modele": ["Regression log-lineaire", "Baseline moyenne"],
        "RMSE_K": [
            np.sqrt(mean_squared_error(y_test_kelvin, y_pred_kelvin)),
            np.sqrt(mean_squared_error(y_test_kelvin, y_baseline_kelvin)),
        ],
        "MAE_K": [
            mean_absolute_error(y_test_kelvin, y_pred_kelvin),
            mean_absolute_error(y_test_kelvin, y_baseline_kelvin),
        ],
        "R2_sur_temperature_K": [
            r2_score(y_test_kelvin, y_pred_kelvin),
            r2_score(y_test_kelvin, y_baseline_kelvin),
        ],
        "R2_sur_log_temperature": [
            r2_score(y_test_log, y_pred_log),
            np.nan,
        ],
    }
)
display(metrics_log.round(3))

coef_log = pd.DataFrame(
    {
        "variable": features_log,
        "coefficient_standardise": model_log.coef_,
        "sens_attendu": ["negatif", "positif", "positif"],
    }
)
display(coef_log.round(3))

plt.figure(figsize=(7, 6))
sns.scatterplot(x=y_test_kelvin, y=y_pred_kelvin, alpha=0.7, color="#4C78A8")
min_val = min(y_test_kelvin.min(), y_pred_kelvin.min())
max_val = max(y_test_kelvin.max(), y_pred_kelvin.max())
plt.plot([min_val, max_val], [min_val, max_val], color="#E45756", linestyle="--")
plt.title("Modele log-lineaire : temperature observee vs predite")
plt.xlabel("Temperature observee (K)")
plt.ylabel("Temperature predite (K)")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
**Interpretation.** Le signe attendu le plus important est celui du demi-grand axe orbital : plus une planete est eloignee, plus sa temperature d'equilibre tend a diminuer. Dans le modele brut, certains coefficients peuvent devenir contre-intuitifs a cause de la multicolinearite entre masse, rayon et gravite de l'etoile. Le modele log-lineaire est donc plus facile a defendre scientifiquement : il garde les variables directement liees au bilan radiatif et donne des signes plus conformes a la physique. Les deux modeles restent limites, car l'albedo, l'atmosphere, l'excentricite, l'inclinaison et les incertitudes de mesure ne sont pas modelises.
"""
    ),
    md(
        r"""
---
## 7. ACP (PCA)

L'ACP resume plusieurs variables physiques en quelques axes. Les variables sont imputees par la mediane puis standardisees, car elles n'ont pas les memes unites.
"""
    ),
    code(
        r"""
pca_cols = [
    "log_pl_orbper", "log_pl_orbsmax", "log_pl_rade", "log_pl_bmasse",
    "pl_orbeccen", "log_pl_insol", "pl_eqt", "st_teff", "log_st_rad",
    "log_st_mass", "st_met", "st_logg", "log_sy_dist", "density_g_cm3",
]
pca_cols = [col for col in pca_cols if col in df_feat.columns]

pca_input = df_feat[pca_cols].copy()

min_non_missing = int(0.25 * len(pca_input))
pca_input = pca_input.dropna(axis=1, thresh=min_non_missing)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_imputed = imputer.fit_transform(pca_input)
X_scaled = scaler.fit_transform(X_imputed)

pca_full = PCA()
pca_full.fit(X_scaled)

explained = pd.DataFrame(
    {
        "composante": np.arange(1, len(pca_full.explained_variance_ratio_) + 1),
        "variance_expliquee": pca_full.explained_variance_ratio_,
        "variance_cumulee": np.cumsum(pca_full.explained_variance_ratio_),
    }
)
display(explained.head(10).round(3))

n_80 = int((explained["variance_cumulee"] < 0.80).sum() + 1)
print(f"Nombre de composantes necessaires pour atteindre au moins 80% de variance : {n_80}")

plt.figure(figsize=(10, 5))
sns.lineplot(data=explained, x="composante", y="variance_cumulee", marker="o", color="#4C78A8")
plt.axhline(0.80, color="#E45756", linestyle="--", label="80%")
plt.title("Variance cumulee expliquee par l'ACP")
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulee")
plt.legend()
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
pca_2 = PCA(n_components=2)
coords = pca_2.fit_transform(X_scaled)

pca_plot = pd.DataFrame(coords, columns=["PC1", "PC2"], index=pca_input.index)
pca_plot["planet_type"] = df_feat.loc[pca_input.index, "planet_type"].values
pca_plot["method_group"] = df_feat.loc[pca_input.index, "method_group"].values

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="planet_type",
    alpha=0.65,
    ax=axes[0],
)
axes[0].set_title("ACP coloree par type de planete")

sns.scatterplot(
    data=pca_plot,
    x="PC1",
    y="PC2",
    hue="method_group",
    alpha=0.65,
    ax=axes[1],
)
axes[1].set_title("ACP coloree par methode de decouverte")

plt.tight_layout()
plt.show()

loadings = pd.DataFrame(
    pca_2.components_.T,
    index=pca_input.columns,
    columns=["PC1", "PC2"],
)
display(loadings.sort_values("PC1", key=lambda s: s.abs(), ascending=False).round(3))
"""
    ),
    md(
        r"""
**Interpretation.** Les premieres composantes separent generalement les dimensions liees a l'orbite, aux proprietes planetaires et aux proprietes stellaires. Si les groupes ne sont pas parfaitement separes, c'est normal : les classes de planete sont simplifiees, et les donnees melangent plusieurs techniques de detection. L'ACP reste utile pour visualiser la structure globale et reperer les variables qui portent le plus d'information.
"""
    ),
    md(
        r"""
---
## 8. Synthese et recommandations

### 8.1 Insights scientifiques

1. Les detections sont dominees par la methode du transit. Cela reflete l'impact des missions spatiales et explique la surrepresentation des planetes proches de leur etoile.
2. Les grandeurs planetaires sont tres asymetriques. Les masses, rayons, periodes et distances doivent etre lues en echelle logarithmique pour eviter que quelques valeurs extremes dominent l'analyse.
3. La temperature d'equilibre est reliee a la distance orbitale et aux proprietes de l'etoile. La regression lineaire confirme cette tendance, tout en montrant les limites d'un modele trop simple.

### 8.2 Limites

- Les valeurs manquantes ne sont pas aleatoires : elles dependent des instruments et des methodes de detection.
- Les categories de planetes sont des approximations basees sur le rayon ou la masse.
- La regression ne modelise pas l'atmosphere, l'albedo, l'inclinaison, ni toutes les incertitudes physiques.
- Le dataset represente les exoplanetes detectees, pas toutes les exoplanetes existantes dans la galaxie.

### 8.3 Conclusion

L'analyse montre que les exoplanetes connues sont fortement marquees par les biais d'observation. Les tendances physiques attendues sont visibles, notamment le lien entre temperature et distance orbitale, mais elles doivent toujours etre interpretees avec les limites des donnees. Une suite logique serait de comparer plusieurs modeles de regression et de traiter explicitement les incertitudes de mesure.
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.x",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")

report = r"""
# Rapport - Projet E : Analyse des Exoplanetes

## Objectif du projet

Le projet analyse un dataset de la NASA Exoplanet Archive afin de comprendre les proprietes des exoplanetes connues et les biais lies aux methodes de detection. Le notebook applique les etapes demandees : nettoyage, analyse exploratoire, feature engineering, visualisations, regression lineaire, ACP et conclusion scientifique.

## Fichiers du rendu

- `notebooks/Projet_E_Exoplanetes.ipynb` : notebook final a executer dans Google Colab.
- `data/projet_E_dataset_exoplanets.csv` : dataset NASA utilise par le notebook.
- `reports/rapport_projet_exoplanetes.md` : ce rapport explicatif.
- `docs/Projet_E_Exoplanetes_original.ipynb` : copie du notebook de depart, conservee comme sauvegarde.

## Etapes realisees dans le notebook

### 1. Chargement des donnees

Le notebook charge le fichier `projet_E_dataset_exoplanets.csv` avec :

```python
pd.read_csv(data_path, comment="#", low_memory=False)
```

Le parametre `comment="#"` est indispensable, car le CSV contient des lignes descriptives NASA avant la vraie table. Le filtre impose par le sujet est ensuite applique :

```python
df = df_raw[df_raw["default_flag"] == 1].copy()
```

Sur le fichier fourni, le dataset passe de 38 449 lignes a 5 903 lignes de reference.

### 2. Qualite et nettoyage

Le notebook verifie :

- les dimensions du dataset ;
- les types de donnees ;
- les doublons ;
- les valeurs manquantes ;
- les valeurs extremes par la methode IQR.

Conclusion importante : le probleme principal n'est pas la duplication, mais les valeurs manquantes. Elles sont normales dans ce type de dataset, car toutes les methodes de detection ne mesurent pas les memes grandeurs physiques.

### 3. Analyse univariee

Les distributions des variables principales sont visualisees :

- periode orbitale ;
- demi-grand axe orbital ;
- rayon planetaire ;
- masse planetaire ;
- temperature d'equilibre ;
- temperature, rayon et masse stellaires ;
- distance du systeme.

Plusieurs variables sont affichees en `log10`, car elles couvrent plusieurs ordres de grandeur. Cela rend les graphiques plus lisibles et evite que quelques valeurs extremes masquent la structure de l'echantillon.

### 4. Feature engineering

Le notebook cree plusieurs variables :

- `density_earth_units` : densite apparente en unite terrestre ;
- `density_g_cm3` : densite approximee en g/cm3 ;
- `planet_type` : classe de planete selon le rayon ou la masse ;
- `temperature_class` : froide, temperee ou chaude ;
- `stellar_class` : classe stellaire approximee a partir de la temperature effective ;
- variables logarithmiques pour les grandeurs tres asymetriques.

Ces variables servent a rendre les analyses plus interpretable scientifiquement.

Une analyse supplementaire filtre aussi les candidates temperees : rayon entre 0,5 et 2 rayons terrestres, temperature d'equilibre entre 250 K et 320 K. Ce filtre ne prouve pas l'habitabilite, mais il donne une piste scientifique concrete a commenter.

### 5. Correlations et analyses bivariees

Le notebook calcule une matrice de correlation de Spearman, plus adaptee que Pearson pour des donnees non lineaires et tres asymetriques.

Il produit aussi des scatterplots :

- masse planetaire vs rayon planetaire ;
- temperature d'equilibre vs distance orbitale ;
- masse stellaire vs masse planetaire ;
- rayon stellaire vs rayon planetaire.

Les graphiques mettent en evidence des tendances physiques, mais aussi des biais lies aux methodes de detection.

### 6. Analyses croisees

Le notebook compare les exoplanetes selon :

- la methode de decouverte ;
- l'annee de decouverte ;
- les distributions de rayon et de periode par methode.

La methode du transit domine nettement l'echantillon. Cette domination est liee a l'efficacite des missions comme Kepler pour detecter des planetes proches de leur etoile.

### 7. Regression lineaire

Le modele predit la temperature d'equilibre `pl_eqt` avec :

- le demi-grand axe orbital en log ;
- la temperature de l'etoile ;
- le rayon stellaire en log ;
- la masse stellaire ;
- la gravite de surface stellaire.

Ce choix est coherent physiquement, car la temperature d'une planete depend de l'energie recue de son etoile et de sa distance orbitale.

Le notebook affiche :

- RMSE ;
- MAE ;
- R2 ;
- comparaison avec une baseline ;
- coefficients standardises ;
- graphique observe vs predit ;
- graphique des residus.

Une deuxieme regression log-lineaire est ajoutee pour ameliorer l'interpretation physique. Elle predit `log10(pl_eqt)` avec `log10(pl_orbsmax)`, `log10(st_teff)` et `log10(st_rad)`. Ce modele est plus facile a defendre, car les signes attendus correspondent directement au bilan radiatif : eloignement negatif, temperature stellaire positive, rayon stellaire positif.

### 8. ACP / PCA

L'ACP est appliquee apres :

- imputation mediane des valeurs manquantes ;
- standardisation des variables.

Le notebook affiche :

- la variance expliquee ;
- le nombre de composantes necessaires pour atteindre environ 80 % de variance ;
- la projection sur PC1 et PC2 ;
- les loadings des variables.

L'ACP permet de visualiser les grands axes de variation des planetes, mais ne separe pas parfaitement toutes les classes, car les categories sont simplifiees et les donnees sont observationnelles.

## Principaux resultats observes

- Le dataset filtre contient 5 903 planetes de reference.
- Les decouvertes vont de 1992 a 2025.
- La methode `Transit` est dominante avec plus de 4 000 planetes dans le fichier filtre.
- Beaucoup de variables physiques ont des valeurs manquantes, notamment l'insolation, le type spectral, la temperature d'equilibre et la masse planetaire.
- Les periodes orbitales, masses, rayons et distances sont tres asymetriques : les transformations logarithmiques sont justifiees.
- Les planetes proches de leur etoile sont plus faciles a detecter et souvent mieux representees dans le dataset.
- Un petit groupe de planetes temperees et de rayon compatible avec des planetes rocheuses/super-Terres peut etre identifie pour enrichir la discussion scientifique.

## Comment executer dans Google Colab

1. Ouvrir Google Colab.
2. Importer le notebook `notebooks/Projet_E_Exoplanetes.ipynb`.
3. Importer le fichier `data/projet_E_dataset_exoplanets.csv` dans les fichiers de la session Colab, ou le placer dans Google Drive.
4. Cliquer sur `Exécution > Tout exécuter`.
5. Si Colab demande le fichier CSV, selectionner `projet_E_dataset_exoplanets.csv`.
6. Lire les sorties et les graphiques dans l'ordre.

## Points a comprendre pour la presentation orale

- Le filtre `default_flag == 1` evite de garder plusieurs jeux de parametres pour la meme planete.
- Les valeurs manquantes ne doivent pas etre supprimees aveuglement, car elles sont liees aux methodes d'observation.
- Les graphiques en log sont necessaires parce que les variables astronomiques couvrent de tres grands ecarts.
- La regression lineaire sert a interpreter une tendance simple, pas a reproduire toute la physique d'une atmosphere planetaire.
- L'ACP resume la structure des donnees, mais ses axes sont des combinaisons statistiques, pas des variables physiques directes.
"""

REPORT_PATH.write_text(report.strip() + "\n", encoding="utf-8")

print(f"Notebook ecrit : {NOTEBOOK_PATH}")
print(f"Rapport ecrit : {REPORT_PATH}")
