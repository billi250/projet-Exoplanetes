# Projet E - Analyse des Exoplanetes

Analyse exploratoire et modelisation simple d'un dataset issu de la NASA Exoplanet Archive.

Le projet etudie les caracteristiques physiques des exoplanetes confirmees, les biais de detection, puis applique une regression lineaire et une ACP pour resumer les relations entre variables astronomiques.

## Objectifs

- Charger et filtrer le dataset NASA avec `default_flag == 1`.
- Nettoyer les donnees et analyser les valeurs manquantes.
- Explorer les distributions des variables physiques principales.
- Creer des variables derivees : densite, type de planete, classe thermique, classe stellaire.
- Comparer les exoplanetes selon la methode et l'annee de decouverte.
- Construire des modeles de regression pour predire la temperature d'equilibre.
- Appliquer une ACP sur les variables numeriques physiques.
- Produire une interpretation scientifique claire des resultats.

## Structure du depot

```text
.
├── data/
│   ├── README.md
│   └── projet_E_dataset_exoplanets.csv
├── docs/
│   ├── Projet_E_Exoplanetes_original.ipynb
│   └── projet_exoplanetes_guide.md
├── notebooks/
│   └── Projet_E_Exoplanetes.ipynb
├── reports/
│   └── rapport_projet_exoplanetes.md
├── scripts/
│   └── build_exoplanet_project.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Dataset

Le fichier utilise est `data/projet_E_dataset_exoplanets.csv`.

Il provient de la NASA Exoplanet Archive et contient des lignes de metadonnees commencant par `#`. Le notebook le charge donc avec :

```python
pd.read_csv(data_path, comment="#", low_memory=False)
```

Le filtre obligatoire du sujet est ensuite applique :

```python
df = df_raw[df_raw["default_flag"] == 1].copy()
```

Dans le fichier fourni, le dataset passe de 38 449 lignes brutes a 5 903 lignes de reference.

## Execution dans Google Colab

1. Ouvrir Google Colab.
2. Importer `notebooks/Projet_E_Exoplanetes.ipynb`.
3. Importer le CSV `data/projet_E_dataset_exoplanets.csv` dans les fichiers Colab, ou placer le dossier du projet dans Google Drive.
4. Executer toutes les cellules avec `Execution > Tout executer`.

Le notebook cherche automatiquement le CSV dans plusieurs emplacements :

- `projet_E_dataset_exoplanets.csv`
- `data/projet_E_dataset_exoplanets.csv`
- `/content/projet_E_dataset_exoplanets.csv`
- `/content/data/projet_E_dataset_exoplanets.csv`
- `/content/drive/MyDrive/projet_exoplanetes/projet_E_dataset_exoplanets.csv`
- `/content/drive/MyDrive/projet_exoplanetes/data/projet_E_dataset_exoplanets.csv`

Si le fichier n'est pas trouve, Colab propose un upload manuel.

## Execution locale

Creer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement :

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Installer les dependances :

```bash
pip install -r requirements.txt
```

Lancer Jupyter :

```bash
jupyter lab
```

Puis ouvrir `notebooks/Projet_E_Exoplanetes.ipynb`.

## Dependances principales

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyterlab

## Resultats principaux

- Le dataset filtre contient 5 903 exoplanetes de reference.
- Les decouvertes couvrent la periode 1992-2025.
- La methode de detection dominante est le transit.
- Les distributions des masses, rayons, periodes et distances sont tres asymetriques, ce qui justifie les transformations logarithmiques.
- La temperature d'equilibre est fortement liee a la distance orbitale et aux proprietes de l'etoile.
- L'ACP montre que plusieurs axes sont necessaires pour resumer correctement la diversite des exoplanetes.

## Notes de securite et publication

Le depot ignore les environnements virtuels, fichiers temporaires, caches Python, checkpoints Jupyter, exports ZIP et fichiers `.env`.

Aucun mot de passe, token ou variable d'environnement ne doit etre ajoute au depot.

## Regeneration du notebook

Le script `scripts/build_exoplanet_project.py` regenere le notebook final et le rapport :

```bash
python scripts/build_exoplanet_project.py
```

