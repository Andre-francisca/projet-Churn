# Projet_3_Churn

Ce dépôt contient un projet de prédiction du churn client à partir du jeu de données Telco Customer Churn.
L’objectif est de construire, évaluer et documenter un modèle de classification binaire capable de prédire si un client va quitter le service (`Churn = Yes`) ou rester (`Churn = No`).

## 1. Description du projet

Le projet couvre l’ensemble du flux de travail de machine learning :

1. Chargement et nettoyage du dataset.
2. Analyse de qualité des données.
3. Préparation des variables numériques et catégorielles.
4. Entraînement d’un modèle de régression logistique.
5. Sélection des hyperparamètres par validation croisée.
6. Évaluation sur un jeu de test réservé.
7. Production de rapports et de visualisations.

## 2. Pré-requis

Le projet a été développé avec Python 3.10+.

Les dépendances principales sont :

```powershell
python -m pip install pandas numpy scikit-learn matplotlib plotly nbformat streamlit
```

Si vous voulez réinstaller complètement les dépendances de manière reproductible, vous pouvez utiliser :

```powershell
python -m pip install -r requirements.txt
```

Si le fichier `requirements.txt` n’existe pas encore, vous pouvez créer une liste minimale à partir des imports du code :

```powershell
python -m pip install pandas numpy scikit-learn matplotlib plotly nbformat streamlit
```

## 3. Installation

Cloner le dépôt :

```powershell
git clone https://github.com/Andre-francisca/projet-Churn.git
cd projet-Churn
```

Créer un environnement virtuel, puis installer les dépendances :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn matplotlib plotly nbformat streamlit
```

## 4. Lancer le projet

### 4.1 Générer les métriques et le graphe de confusion

Depuis la racine du projet :

```powershell
python .\src\churn_model.py
```

Ce script régénère :

- `reports/churn_model_metrics.json`
- `reports/churn_confusion_matrix.png`

### 4.2 Ouvrir le dashboard HTML

Le dashboard HTML est préparé dans le dossier `reports/` :

```powershell
Start-Process .\reports\churn_dashboard.html
```

Vous pouvez aussi l’ouvrir directement via le navigateur en ouvrant le fichier HTML.

### 4.3 Lancer le dashboard Streamlit interactif

Le projet contient aussi un dashboard interactif généré depuis `src/dashboard.py` :

```powershell
streamlit run .\src\dashboard.py
```

Cette version affiche une interface visuelle de type KPI et de visualisations à partir du dataset et des métriques calculées.

## 5. Structure du dépôt

- `data/` : jeu de données brut, source principale du projet.
- `notebooks/` : carnets d’analyse et de préparation des données.
- `src/` : scripts Python de preprocessing, modélisation et dashboard.
- `reports/` : livrables de rendu, visualisations, métriques JSON et rapports markdown.
- `scripts/` : scripts utilitaires de maintenance du dépôt.

## 6. Livrables de rapport

Les livrables rédigés sont séparés dans les trois documents suivants :

1. `reports/01_data_quality_cleaning.md` : qualité des données, statistiques descriptives et nettoyage.
2. `reports/02_algorithm_selection_performance.md` : modèles, hyperparamètres et performances comparées.
3. `reports/03_test_generalization_conclusions.md` : test final, généralisation, limites et conclusions.

## 7. Livrables du brief

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

## 8. Modèle et approche

Le script de machine learning applique un pipeline de preprocessing complet sur les variables numériques et catégorielles, puis lance un `GridSearchCV` sur la constante de régularisation `C` du modèle de régression logistique.

Le pipeline implémente :

- imputation des valeurs manquantes ;
- encodage OneHot des variables catégorielles ;
- normalisation des variables numériques ;
- entraînement du classifieur avec validation croisée.

Les métriques de test sont calculées dans le script et écrites dans le fichier JSON de rapport du projet.
Le seuil de décision est sélectionné par validation croisée uniquement sur le jeu d’entraînement ;
le jeu de test est conservé pour l’évaluation finale.

## 9. Référence des résultats

Les résultats livrés doivent être régénérés avec `src/churn_model.py`.
Les anciennes cellules exploratoires du notebook peuvent afficher des métriques différentes car elles testent plusieurs pipelines, seuils et jeux de paramètres.
Le JSON généré par le script est la référence finale.

## 10. Note de publication

Le dépôt contient les fichiers utiles pour reproduire le projet, les rapports produits et les visualisations. Les artefacts générés par l’exécution Python et les fichiers système sont ignorés via `.gitignore`.
