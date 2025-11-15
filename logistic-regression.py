from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

np.random.seed(42)

# Lire le csv et le convertir en pytorch
ds = pd.read_csv('data/donnees_traitees_binaire.csv')
ds = ds.to_numpy()

# Définir les données et les targets
X = ds[:, :-1]
y = ds[:, -1]

# Splitter pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

valeurs_C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
best_C, best_erreur = 0, 1

for C in valeurs_C:
    # Reprendre le meilleur modèle et calculer les erreurs
    model = LogisticRegression(C=C, solver='newton-cholesky')
    model.fit(X_train, y_train)
    erreur_train = 1 - model.score(X_train, y_train)
    erreur_test = 1 - model.score(X_test, y_test)

    if erreur_test < best_erreur:
        best_C, best_erreur = C, erreur_test

model = LogisticRegression(C=best_C, solver='newton-cholesky')
model.fit(X_train, y_train)
erreur_train = 1 - model.score(X_train, y_train)
erreur_test = 1 - model.score(X_test, y_test)

# Sauvegarder les résultats
resultats = pd.DataFrame({'C': [best_C], 'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
resultats.to_csv('resultats/logistic-regression.csv', index=False)