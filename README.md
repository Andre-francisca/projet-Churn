# Projet_3_Churn

Ce dépôt contient un projet de prédiction du churn client sur le jeu de données Telco.

## Lancer le projet

Depuis la racine du projet, installer les dépendances Python :

```powershell
python -m pip install pandas numpy scikit-learn matplotlib plotly nbformat
```

Exécuter le pipeline reproductible :

```powershell
python .\src\churn_model.py
```

Le script régénère `reports/churn_model_metrics.json` et `reports/churn_confusion_matrix.png`.
Le dashboard est un fichier HTML autonome : ouvrir `reports/churn_dashboard.html`, ou exécuter
`Start-Process .\reports\churn_dashboard.html` sous PowerShell.

## Structure du projet

- `data/` : jeu de données brut et fichier source du projet.
- `notebooks/` : carnet de préparation de données.
- `src/` : scripts de traitement et de modélisation.
- `reports/` : rapports HTML, image de matrice de confusion et métriques JSON.

Les livrables rédigés sont séparés dans les trois documents suivants :

1. `reports/01_data_quality_cleaning.md` : qualité, statistiques descriptives et nettoyage ;
2. `reports/02_algorithm_selection_performance.md` : modèles, hyperparamètres et performances ;
3. `reports/03_test_generalization_conclusions.md` : test final, généralisation, limites et conclusions.

## Livrables du brief

1. Analyse et préparation des données :
   - notebook principal `notebooks/01_data_preparation.ipynb` ;
   - contrôle de qualité des colonnes, des types, des valeurs manquantes et des doublons.

2. Modélisation et validation :
   - script Python dans `src/churn_model.py` ;
   - modèle de régression logistique pour la classification binaire du churn ;
   - métriques enregistrées dans `reports/churn_model_metrics.json`.

3. Visualisations :
   - `reports/churn_dashboard.html`
   - `reports/churn_pie.html`
   - `reports/churn_contract_histogram.html`
   - `reports/churn_confusion_matrix.png`

## Modèle

Le script de machine learning applique un pipeline de preprocessing complet sur les variables numériques et catégorielles, puis lance un GridSearchCV sur la constante de régularisation `C` du modèle logistic.

Les métriques de test sont calculées dans le script et écrites dans le JSON de rapport de projet.
Le seuil de décision est sélectionné par validation croisée sur le jeu d'entraînement uniquement ;
le jeu de test reste réservé à l'évaluation finale.

## Référence des résultats

Les résultats livrés doivent être régénérés avec `src/churn_model.py`. Les anciennes cellules
exploratoires du notebook peuvent présenter des métriques différentes car elles testent plusieurs
pipelines, seuils et jeux de paramètres. Le JSON généré par le script est la référence finale.
