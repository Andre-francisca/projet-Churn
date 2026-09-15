# Rapport de modélisation – Churn prediction

> Ce rapport est un résumé. Les trois livrables détaillés sont `01_data_quality_cleaning.md`, `02_algorithm_selection_performance.md` et `03_test_generalization_conclusions.md`. Les métriques ci-dessous sont celles générées par `src/churn_model.py`.

## Objectif

Prédire si un client va partir du service télécom (variable cible `Churn`).

## Données

Le fichier source est : `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Préparation

- suppression de la colonne `customerID` car identifiant unique non prédictif ;
- conversion de la cible `Churn` en 0/1 : `No -> 0`, `Yes -> 1` ;
- conversion de `TotalCharges` en numérique avec gestion des valeurs invalides ;
- séparation `train/test` stratifiée avec `test_size=0.25` et `random_state=42`.

## Algorithme

Le modèle retenu est une régression logistique avec `LogisticRegression` et `class_weight="balanced"`.

Cette méthode est adaptée au cas de classification binaire, peu coûteuse en calcul, facile à interpréter, et compatible avec un prétraitement par OneHotEncoder + StandardScaler.

## Hyperparamètres

Le script de référence utilise un `GridSearchCV` sur le paramètre `C` :

- `C = 0.1`
- `C = 1`
- `C = 5`
- `C = 10`

Le meilleur paramètre retenu sur la validation croisée est `C = 10`.

## Prétraitement

- colonnes numériques : `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`
- colonnes catégorielles : `gender`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

Pipeline de preprocessing :

- imputation médiane pour les numériques ;
- standardisation des variables numériques ;
- imputation par la valeur la plus fréquente pour les colonnes catégorielles ;
- encodage one-hot catégoriel.

## Mesures de performance

Sur le jeu de test, le modèle atteint :

- accuracy : 0.7723
- precision : 0.5548
- recall : 0.7152
- f1 : 0.6249
- roc_auc : 0.8455

Le seuil de décision retenu est `0.60`. Il a été choisi sur des prédictions hors-fold du jeu d'entraînement, sans utiliser le jeu de test.

La matrice de confusion est sauvegardée dans `reports/churn_confusion_matrix.png`.

## Conclusion

Le modèle présente une bonne capacité de détection des clients au risque de churn (`recall` de 0.7152), mais la précision reste modérée. Les modèles arbre et Random Forest sont documentés dans le notebook exploratoire ; le pipeline reproductible final reste la régression logistique, car elle offre une bonne discrimination et une interprétation plus directe.
