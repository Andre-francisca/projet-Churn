# Document 3 — Test final, généralisation et conclusions

## Protocole de test

Le jeu de données est séparé avec `test_size=0.25`, `random_state=42` et `stratify=y`. Le jeu de test contient 1 761 observations et n'intervient ni dans l'apprentissage du prétraitement, ni dans la recherche de `C`, ni dans la sélection du seuil.

Le prétraitement est encapsulé dans un `Pipeline` et un `ColumnTransformer`. Les statistiques d'imputation, les catégories encodées et les paramètres du scaler sont donc appris à l'intérieur des folds d'entraînement et jamais à partir du test.

## Capacité de généralisation

Sur le jeu de test conservé, le modèle obtient une ROC-AUC de 0,8455. Il détecte 334 churners sur 467, soit un rappel de 0,7152. Il produit également 268 faux positifs et 133 faux négatifs.

Ces résultats indiquent une bonne capacité de discrimination sur cet échantillon, mais ne constituent pas une garantie de performance en production. Le jeu de données est historique, les variables disponibles sont limitées et le seuil doit être recalibré si les coûts métier changent.

## Limites

- La validation croisée porte sur un seul jeu de données historique.
- Les performances peuvent varier sur une autre période ou une autre population de clients.
- Le rappel élevé est obtenu au prix de faux positifs qui peuvent entraîner des campagnes inutiles.
- Une validation temporelle, un suivi de dérive et une analyse du coût des erreurs seraient nécessaires avant déploiement.

## Conclusion et usage métier

Le modèle doit être utilisé comme outil d'aide à la décision. Les clients signalés comme à risque peuvent être priorisés pour une action de fidélisation, avec une attention particulière aux contrats mensuels, à la faible ancienneté et aux frais élevés. Une campagne pilote doit mesurer le coût des faux positifs et le nombre de churns effectivement évités.

La commande reproductible est :

```powershell
python .\\src\\churn_model.py
```

Elle régénère les métriques et la matrice de confusion à partir de la même procédure.
