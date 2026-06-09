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
