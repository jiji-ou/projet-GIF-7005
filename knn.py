from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

k_values = [i for i in range(1, 101)]
best_k, best_erreur = 0, 1

for k in k_values:
    # Reprendre le meilleur modèle et calculer les erreurs
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    erreur_train = 1 - model.score(X_train, y_train)
    erreur_test = 1 - model.score(X_test, y_test)

    if erreur_test < best_erreur:
        best_k, best_erreur = k, erreur_test

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)
erreur_train = 1 - model.score(X_train, y_train)
erreur_test = 1 - model.score(X_test, y_test)

# Sauvegarder les résultats
resultats = pd.DataFrame({'k': [best_k], 'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
resultats.to_csv('resultats/knn.csv', index=False)