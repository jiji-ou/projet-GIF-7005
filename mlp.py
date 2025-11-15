import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

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
class ResBlock(nn.Module):
    def __init__(self, n):
        super(ResBlock, self).__init__()
        self.l1 = nn.Linear(n, n)
        self.l2 = nn.Linear(n, n)

    def forward(self, x):
        x2 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x2))
        return x + x2

class Net(nn.Module):
    def __init__(self, n, l, res=False):
        super(Net, self).__init__()
        self.input = nn.Linear(X.shape[1], n) # Couche d'entrée
        if res:
            self.blocks = nn.ModuleList([ResBlock(n) for _ in range(l)])
        else:
            self.blocks = nn.ModuleList([nn.Linear(n, n) for _ in range(l)])
        self.output = nn.Linear(n, 3) # Couche de sortie

    def forward(self, x):
        x = F.relu(self.input(x))
        for block in self.blocks:
            x = F.relu(block(x))
        x = self.output(x)
        return x

# Mettre sur la GPU si possible
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_and_test(net, name):
    # Entraînement du modèle
    optim = SGD(net.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 500
    loss = None
    for epoch in range(epochs):
        losses = []
        for data, targets in train_loader:
            optim.zero_grad()

            output = net(data)
            loss = criterion(output, targets)
            losses.append(loss.item())
            loss.backward()

            optim.step()

        print(f"{name} - epoch: {epoch}, perte: {np.array(losses).mean()}")

    # Sauvegarder le réseau entraîné
    torch.save(net.state_dict(), f'models/heart-attack-{name}.pt2')

    # Calculer l'erreur d'entraînement
    output = net(X_train)
    correct = output.argmax(1).eq(y_train).sum().item()
    erreur_train = 1 - correct/X_train.shape[0]

    # Calculer l'erreur en test
    output = net(X_test)
    correct = output.argmax(1).eq(y_test).sum().item()
    erreur_test = 1 - correct/X_test.shape[0]

    print(f"{name} - Erreur d'entraînement: {erreur_train}, erreur de test: {erreur_test}")

    results = pd.DataFrame({'epochs': [epochs], 'perte finale': [loss.item()], 'erreur entraînement': [erreur_train], 'erreur test': [erreur_test]})
    results.to_csv(f'resultats/{name}.csv', index=False)

# Définir le modèle
resnet = Net(300, 20, res=True).to(device)
mlp = Net(300, 5).to(device)

train_and_test(mlp, 'mlp')
train_and_test(resnet, 'resnet')