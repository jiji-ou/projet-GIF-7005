from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd

np.random.seed(42)

# Lire le csv et le convertir en pytorch
ds = pd.read_csv('data/donnees_traitees_classification.csv')
ds = ds.to_numpy()

# Définir les données et les targets
X = ds[:, :-1]
y = ds[:, -1]

# Splitter pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reprendre le meilleur modèle et calculer les erreurs
model = GradientBoostingClassifier(n_estimators=1000)
model.fit(X_train, y_train)
erreur_train = 1 - model.score(X_train, y_train)
erreur_test = 1 - model.score(X_test, y_test)

# Sauvegarder les résultats
resultats = pd.DataFrame({'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
resultats.to_csv('resultats/gradient-boosting.csv', index=False)