# 🌌 Projet Data Science : Analyse des Exoplanètes (NASA Exoplanet Dataset)

## 📌 Contexte du Projet

Ce projet consiste à réaliser une analyse complète d’un dataset d’exoplanètes provenant de la NASA.  
Une exoplanète est une planète située en dehors du système solaire et orbitant autour d’une autre étoile que le Soleil.

Le but du projet est d’utiliser les techniques de :
- Data Cleaning,
- Analyse Exploratoire de Données (EDA/AED),
- Feature Engineering,
- Visualisation,
- Régression,
- ACP (PCA),
- Analyse Statistique.

Le projet doit être réalisé en Python avec Jupyter Notebook.

---

# 🎯 Objectifs du Projet

Le projet doit permettre de :

- Comprendre la structure des données astronomiques.
- Nettoyer les données.
- Étudier les caractéristiques des exoplanètes.
- Identifier des relations entre les variables.
- Créer de nouvelles variables pertinentes.
- Réaliser des visualisations scientifiques.
- Construire un modèle de régression.
- Réduire la dimension des données avec PCA.
- Interpréter scientifiquement les résultats.

---

# 📂 Informations sur le Dataset

Le dataset est un CSV très volumineux provenant de la NASA Exoplanet Archive.

Le notebook impose de filtrer les données avec :

```python
df = df[df["default_flag"] == 1]
```

Cela permet de garder uniquement les observations validées et de référence.

---

# 🛠️ Technologies et Librairies à Utiliser

## Langage
- Python 3

## Environnement
- Jupyter Notebook

## Librairies principales

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
```

---

# 📋 Plan Global du Projet

## 1️⃣ Chargement et Compréhension des Données

### Objectifs

- Charger le CSV.
- Comprendre la structure des données.
- Explorer les colonnes.
- Identifier les types de variables.

### Étapes à réaliser

```python
df = pd.read_csv("nom_du_dataset.csv")
```

```python
df = df[df["default_flag"] == 1]
```

```python
df.head()
df.info()
df.describe()
df.shape
df.columns
```

---

## 2️⃣ Data Cleaning (Nettoyage des Données)

### Objectifs

Préparer un dataset propre avant toute analyse.

### Étapes à réaliser

```python
df.isnull().sum()
```

```python
df.duplicated().sum()
```

```python
df["colonne"] = df["colonne"].fillna(df["colonne"].median())
```

```python
sns.boxplot(x=df["variable"])
```

---

## 3️⃣ Analyse Univariée

### Variables importantes à analyser

- masse des planètes,
- rayon,
- température,
- période orbitale,
- distance,
- masse de l’étoile,
- rayon de l’étoile.

### Visualisations

```python
sns.histplot(df["variable"], kde=True)
```

```python
sns.boxplot(x=df["variable"])
```

```python
np.log10(df["variable"])
```

---

## 4️⃣ Analyse Bivariée et Corrélations

### Analyses importantes

- masse vs rayon,
- température vs distance,
- masse étoile vs masse planète,
- rayon étoile vs rayon planète.

### Visualisations

```python
sns.scatterplot(x="x", y="y", data=df)
```

```python
corr = df.corr(numeric_only=True)

plt.figure(figsize=(15,10))
sns.heatmap(corr, cmap="coolwarm")
```

---

## 5️⃣ Feature Engineering

### Variables possibles à créer

#### Densité planétaire

Formule :

ρ = m / ((4/3) × π × r³)

```python
df["density"] = df["mass"] / ((4/3) * np.pi * (df["radius"]**3))
```

#### Classification des planètes

Créer des catégories :
- planète rocheuse,
- géante gazeuse,
- super-Terre,
- Neptune-like.

#### Température catégorisée

- froide,
- tempérée,
- chaude.

---

## 6️⃣ Analyses Croisées

### Comparaisons

- méthode de découverte,
- année de découverte,
- type d’étoile.

```python
df.groupby("disc_year").size()
```

---

## 7️⃣ Régression Linéaire

### Exemple recommandé

Prédire :
- la température,
- le rayon,
- ou la masse.

```python
X = df[["distance"]]
y = df["temperature"]
```

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

```python
r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)
```

---

## 8️⃣ PCA (Analyse en Composantes Principales)

### Objectifs

Réduire les dimensions du dataset.

### Étapes

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

```python
plt.scatter(X_pca[:,0], X_pca[:,1])
```

---

# 📊 Visualisations Importantes à Produire

- histogrammes,
- boxplots,
- scatterplots,
- heatmaps,
- courbes temporelles,
- visualisation PCA,
- graphiques de catégories.

---

# 🧠 Interprétation Scientifique

Chaque graphique doit être accompagné d’une interprétation scientifique.

Exemples :
- Les planètes proches de leur étoile semblent plus chaudes.
- Les géantes gazeuses sont détectées plus facilement.
- Le nombre de découvertes augmente après certaines missions spatiales.

---

# 📌 Structure Finale Conseillée du Notebook

1. Introduction
2. Chargement des données
3. Data Cleaning
4. Analyse Univariée
5. Analyse Bivariée
6. Feature Engineering
7. Analyses Croisées
8. Régression Linéaire
9. PCA
10. Conclusion

---

# 🎯 Objectif Final

Construire une étude complète et professionnelle des exoplanètes à partir des données NASA en appliquant des techniques modernes de Data Science et d’analyse statistique.
