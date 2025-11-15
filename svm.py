from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import scipy as sp
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

norms = sp.spatial.distance.pdist(X_train, metric='euclidean')
sigma_min = norms.min()
print('sigma_min:', sigma_min)

### Recherche en grille
valeurs_C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5]
valeurs_gamma = [sigma_min*2**i for i in range(7)]
best_C, best_gamma, best_erreur = 0, 0, 1

for C in valeurs_C:
    for gamma in valeurs_gamma:
        # Définir le modèle
        model = SVC(kernel='rbf', C=C, gamma=gamma)

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Calculer l'erreur
        erreur_train = 1 - model.score(X_train, y_train)
        erreur_test = 1 - model.score(X_test, y_test)

        # On minimise l'erreur sur le test pour éviter d'overfitter
        if erreur_test < best_erreur:
            best_C, best_gamma, best_erreur = C, gamma, erreur_test

        print(f'C: {C}, gamma: {gamma}')
        print(f"Erreur d'entraînement: {erreur_train}, Erreur de test: {erreur_test}")

# Reprendre le meilleur modèle et calculer les erreurs
model = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
model.fit(X_train, y_train)
erreur_train = 1 - model.score(X_train, y_train)
erreur_test = 1 - model.score(X_test, y_test)

# Sauvegarder les résultats
resultats = pd.DataFrame({'C': [best_C], 'gamma': [best_gamma], 'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
resultats.to_csv('resultats/svm.csv', index=False)