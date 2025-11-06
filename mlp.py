import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from sklearn.model_selection import train_test_split

# Lire le csv et le convertir en pytorch
ds = pd.read_csv('data/donnees_traitees_classification.csv')
ds = ds.to_numpy()
ds = torch.from_numpy(ds).float()

# Définir les données et les targets
X = ds[:, :-1]
y = ds[:, -1].long()

# Splitter pour le test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Régler la device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Définition du MLP
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(X.shape[1], 300) # Couche d'entrée
        self.l2 = nn.Linear(300, 300) # Couche cachée
        self.l3 = nn.Linear(300, 300) # Couche cachée
        self.output = nn.Linear(300, 3) # Couche de sortie

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.output(x)
        return x

# Définir le modèle
net = Net()

# Mettre sur la GPU si possible
net = net.to(device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

# Entraînement du modèle
optim = SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
epochs = 500
loss = None
for epoch in range(epochs):
    optim.zero_grad()

    output = net(X_train)
    loss = criterion(output, y_train)
    loss.backward()

    optim.step()

    print(f"epoch: {epoch}, perte: {loss.item()}")

# Sauvegarder le réseau entraîné
torch.save(net.state_dict(), 'models/heart-attack-mlp.pt2')

# Calculer l'erreur d'entraînement
output = net(X_train)
correct = output.argmax(1).eq(y_train).sum().item()
erreur_train = 1 - correct/X_train.shape[0]

# Calculer l'erreur en test
output = net(X_test)
correct = output.argmax(1).eq(y_test).sum().item()
erreur_test = 1 - correct/X_test.shape[0]

print(f"Erreur d'entraînement: {erreur_train}, erreur de test: {erreur_test}")

results = pd.DataFrame({'epochs': [epochs], 'perte finale': [loss.item()], 'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
results.to_csv('resultats/mlp.csv', index=False)