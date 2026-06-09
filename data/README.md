# Dataset

Ce dossier contient le CSV utilise par le notebook :

`projet_E_dataset_exoplanets.csv`

Le fichier provient de la NASA Exoplanet Archive. Il est charge avec `comment="#"` car les premieres lignes du CSV contiennent des metadonnees.

Le notebook applique ensuite le filtre de reference :

```python
df = df_raw[df_raw["default_flag"] == 1].copy()
```

