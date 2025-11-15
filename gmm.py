from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

np.random.seed(42)

# Lire le csv et le convertir en pytorch
ds = pd.read_csv('data/donnees_traitees_classification.csv')
ds = ds.to_numpy()

# Définir les données et les targets
X = ds[:, :-1]
y = ds[:, -1]

nclasses = 3

# Analyser les prédictions sur l'ensemble du jeu de données pour trouver
# quelle classe correspond à quelle dans le clustering
def get_y_map():
    y_map = np.zeros(3, dtype=int)
    y_pred = model.predict(X)
    C = confusion_matrix(y, y_pred)
    row_idx, col_idx = linear_sum_assignment(-C)
    for c in row_idx:
        y_map[c] = col_idx[c]
    return y_map

# Calculer l'exactitude du clustering
def get_accuracy(model, X, y, y_map):
    y_pred = model.predict(X)
    accuracy = np.mean(y_pred == y_map[y.astype(int)])
    return accuracy

# Splitter pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reprendre le meilleur modèle et calculer les erreurs
model = GaussianMixture(nclasses)
model.fit(X_train)
y_map = get_y_map()
score_train = 1 - get_accuracy(model, X_train, y_train, y_map)
score_test = 1 - get_accuracy(model, X_test, y_test, y_map)

# Sauvegarder les résultats
resultats = pd.DataFrame({'erreur entraînement': [score_train], 'erreur test': [score_test]})
resultats.to_csv('resultats/gmm.csv', index=False)