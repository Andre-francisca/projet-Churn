# Projet_3_Churn

Ce dépôt présente un projet de prédiction du churn client à partir du jeu de données Telco Customer Churn.
L’objectif principal est de construire un modèle de classification binaire capable de distinguer les clients qui vont rester (`No`) de ceux qui vont résilier le contrat (`Yes`).

Le projet montre le workflow complet d’un projet de data science : exploration, nettoyage, préparation, modélisation, validation, visualisation et rédaction de rapports.

## 1. Contexte métier

Le churn correspond au risque qu’un client décide de quitter une entreprise ou un service.
Dans un contexte télécom, l’identification des clients à risque permet à l’entreprise de mieux cibler ses actions de fidélisation, de réduire les pertes de revenus et de mieux comprendre la relation entre la qualité de service et la fidélité des abonnés.

## 2. Jeu de données

Le dataset utilisé est disponible dans le dossier `data/` et porte le nom :

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`

Il contient des variables décrivant les abonnements, les contrats, les services, les facturations et le statut final de churn du client.

## 3. Pipeline du projet

Le projet couvre les étapes suivantes :

1. Chargement et nettoyage du dataset.
2. Analyse de qualité des données.
3. Préparation des colonnes numériques et catégorielles.
4. Entraînement d’un modèle de régression logistique.
5. Recherche d’hyperparamètres via `GridSearchCV`.
6. Évaluation sur un test final.
7. Génération de rapports et visualisations.

## 4. Pré-requis

Le projet a été développé avec Python 3.10+.

Les dépendances principales sont listées dans le fichier `requirements.txt` du dépôt :

```powershell
python -m pip install -r requirements.txt
```

Les bibliothèques utilisées sont : `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `plotly`, `nbformat` et `streamlit`.

## 5. Installation locale

Cloner le dépôt :

```powershell
git clone https://github.com/Andre-francisca/projet-Churn.git
cd projet-Churn
```

Créer et activer un environnement virtuel :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 6. Exécution du projet

### 6.1 Générer les métriques du modèle

Depuis la racine du projet :

```powershell
python .\src\churn_model.py
```

Ce script régénère les artefacts suivants :

- `reports/churn_model_metrics.json`
- `reports/churn_confusion_matrix.png`

### 6.2 Ouvrir le dashboard HTML

Le dashboard HTML est fourni dans le dossier `reports/` :

```powershell
Start-Process .\reports\churn_dashboard.html
```

### 6.3 Lancer le dashboard Streamlit

Le repository contient aussi un dashboard interactif construit avec Streamlit :

```powershell
streamlit run .\src\dashboard.py
```

Cette interface permet d’explorer les clients, la répartition du churn et les indicateurs de performance.

## 7. Structure du dépôt

- `data/` : jeu de données brut utilisé pour l’analyse.
- `notebooks/` : notebook de préparation et de modélisation.
- `src/` : scripts de preprocessing, modélisation et dashboard.
- `reports/` : rapports écrits, visualisations HTML, matrice de confusion et métriques JSON.
- `scripts/` : script de nettoyage des artefacts notebook.

## 8. Livrables de travail

Les livrables écrits du projet sont contenus dans :

1. `reports/01_data_quality_cleaning.md` : qualité des données, statistiques descriptives et nettoyage.
2. `reports/02_algorithm_selection_performance.md` : modèles, hyperparamètres et performances comparées.
3. `reports/03_test_generalization_conclusions.md` : résultats de validation, généralisation et conclusion.

## 9. Livrables techniques

Le projet produit les livrables suivants :

1. Analyse et préparation des données :
   - `notebooks/01_data_preparation.ipynb`
   - contrôles sur les colonnes, types, valeurs manquantes et doublons.

2. Modélisation et validation :
   - `src/churn_model.py`
   - modèle logistique binaire pour la prédiction du churn.
   - métriques sauvegardées dans `reports/churn_model_metrics.json`.

3. Visualisations :
   - `reports/churn_dashboard.html`
   - `reports/churn_pie.html`
   - `reports/churn_contract_histogram.html`
   - `reports/churn_confusion_matrix.png`

## 10. Méthode de modélisation

Le script principal applique un pipeline complet de preprocessing pour les variables numériques et catégorielles, puis lance un `GridSearchCV` sur la constante de régularisation `C` du modèle de régression logistique.

Le traitement comprend :

- imputation des valeurs manquantes ;
- encodage OneHot pour les variables catégorielles ;
- normalisation des variables numériques ;
- entraînement du modèle sur les données préparées.

Les métriques de test sont calculées dans le script puis écrites dans le JSON final du projet.
Le seuil de décision est ajusté par validation croisée sur le jeu d’entraînement uniquement ;
le jeu de test reste réservé à l’évaluation finale.

## 11. Référence des résultats

Les résultats comparables du projet doivent être régénérés à partir de `src/churn_model.py`.
Les cellules exploratoires du notebook peuvent produire des vues temporaires et parfois différentes, parce qu’elles testent plusieurs pipelines, seuils et combinaisons de paramètres.
Le fichier JSON généré par le script constitue la source de vérité officielle du rapport final.

## 12. Résumé

Ce dépôt vise à produire une solution simple, reproductible et documentée pour prédire le churn télécom à partir de variables métier et de service client.
Il montre comment transformer une base de données brute en un modèle exploitable, en visualisations et en livrables de synthèse pour une analyse de décision.
