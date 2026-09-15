# Document 1 — Qualité des données, statistiques et nettoyage

## Source et contrôle qualité

La source est `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Le contrôle réalisé dans `notebooks/01_data_preparation.ipynb` vérifie les dimensions, les types, les valeurs manquantes, les doublons de lignes et les doublons de `customerID`.

Le jeu initial contient 7 043 lignes et 21 colonnes. `customerID` est un identifiant unique et n'est pas utilisé comme variable prédictive.

## Valeurs manquantes et `TotalCharges`

`TotalCharges` est lu comme texte dans le fichier brut. Les espaces sont convertis en valeurs manquantes, puis la colonne est convertie en numérique. Les 11 lignes correspondant à une ancienneté nulle sont retirées dans le notebook exploratoire.

Le script reproductible conserve une stratégie plus robuste : conversion avec `errors="coerce"`, puis imputation médiane dans le pipeline. Cette stratégie permet de reconstruire le traitement sans apprendre la médiane sur le jeu de test.

## Statistiques descriptives et déséquilibre

Après nettoyage exploratoire, la cible présente environ 73,4 % de clients restés et 26,6 % de clients churners. Le déséquilibre est pris en compte par `class_weight="balanced"` et par l'utilisation de la précision, du rappel, du F1 et de la ROC-AUC en complément de l'accuracy.

L'exploration met en évidence des différences de churn selon le contrat, l'ancienneté, les frais mensuels et le service Internet. Ces observations orientent l'analyse, mais ne constituent pas à elles seules une preuve de causalité.

## Encodage et normalisation

Les variables catégorielles sont imputées par la modalité la plus fréquente puis encodées avec `OneHotEncoder(handle_unknown="ignore")`. Les variables numériques sont imputées par la médiane puis standardisées avec `StandardScaler`.

## Procédure reproductible

La procédure de référence est `src/churn_model.py` :

1. lecture du fichier brut et conversion de `Churn` en 0/1 ;
2. conversion de `TotalCharges` ;
3. séparation stratifiée train/test ;
4. ajustement du prétraitement uniquement dans un `Pipeline` sur les données d'entraînement ;
5. recherche des hyperparamètres par validation croisée ;
6. évaluation finale sur le test conservé.

La sélection du seuil de décision utilise des prédictions hors-fold du jeu d'entraînement. Le jeu de test n'est pas utilisé pour choisir ce seuil.
