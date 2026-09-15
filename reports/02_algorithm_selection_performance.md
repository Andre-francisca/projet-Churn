# Document 2 — Choix des algorithmes, hyperparamètres et performances

## Algorithmes étudiés

Le notebook `notebooks/01_data_preparation.ipynb` compare une régression logistique, un arbre de décision et un Random Forest. La régression logistique est retenue comme modèle de référence car elle est adaptée à une classification binaire, interprétable et compatible avec les variables encodées et standardisées.

L'arbre de décision sert de modèle non linéaire de comparaison. Le Random Forest complète cette comparaison et fournit une importance des variables. Les résultats exploratoires indiquent que `tenure`, `TotalCharges`, `Contract` et `MonthlyCharges` sont parmi les variables les plus utiles.

## Interprétation des règles de l'arbre

Une version volontairement bornée de l'arbre (`max_depth=3`, `min_samples_leaf=50`) donne des règles lisibles. Le chemin le plus directement associé au churn est : contrat `Month-to-month`, service Internet `Fiber optic`, puis `TotalCharges <= 1 150,60` ; cette feuille prédit la classe churn. À l'inverse, les contrats non mensuels ou une ancienneté plus élevée conduisent plus souvent vers la classe non churn dans cet arbre.

Ces règles décrivent les associations apprises par le modèle et ne prouvent pas que le contrat ou le service Internet causent le churn. L'arbre complet du notebook est évalué séparément ; cette version bornée sert à rendre son interprétation intelligible.

## Hyperparamètres et validation

Le pipeline reproductible utilise :

- `LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")` ;
- recherche de `model__C` parmi `[0.1, 1, 5, 10]` ;
- validation croisée stratifiée en 3 folds ;
- optimisation selon la ROC-AUC.

Le meilleur paramètre obtenu est `C = 10`.

Le seuil de classification est ensuite choisi parmi 0,20 à 0,70 par pas de 0,05, en maximisant le F1 sur des prédictions hors-fold du jeu d'entraînement. Le seuil retenu lors de la génération de ce rapport est `0,60`.

## Résultats de référence sur le test

| Indicateur | Valeur |
|---|---:|
| Accuracy | 0,7723 |
| Précision, classe churn | 0,5548 |
| Rappel, classe churn | 0,7152 |
| F1, classe churn | 0,6249 |
| ROC-AUC | 0,8455 |
| Seuil | 0,60 |

La matrice de confusion correspondante est : TN = 1 026, FP = 268, FN = 133, TP = 334.

## Choix final

Le modèle final est la régression logistique optimisée. Le rappel de la classe churn est privilégié comme métrique métier, car un faux négatif représente un client à risque non ciblé par une action de fidélisation. La précision reste nécessaire pour contrôler le coût des campagnes et le F1 donne un compromis lisible entre les deux.

Les sorties exploratoires du notebook ne doivent pas être mélangées avec ces chiffres : elles utilisent notamment d'autres seuils et d'autres configurations. Le fichier `reports/churn_model_metrics.json`, régénéré par `src/churn_model.py`, est la référence finale.
