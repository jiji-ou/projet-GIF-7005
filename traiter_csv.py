import pandas as pd

ds = pd.read_csv('data/heart-attack-risk-prediction-dataset.csv')

# Transformer le sexe en données numériques
ds[ds == 'Male'] = 1 # J'ai choisi 1 pour les hommes parce que les hommes ont un plus haut risque de crises de coeur
ds[ds == 'Female'] = 0

# Normaliser le nombre de jours d'activité physique par semaine
ds.loc[:, 'Physical Activity Days Per Week'] /= ds.loc[:, 'Physical Activity Days Per Week'].max()

# Retirer les lignes avec des données manquantes
ds = ds.dropna()

# Mettre la colonne cible comme dernière colonne
ds_binaire = ds[['Age','Cholesterol','Heart rate','Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Exercise Hours Per Week','Diet','Previous Heart Problems','Medication Use','Stress Level','Sedentary Hours Per Day','Income','BMI','Triglycerides','Physical Activity Days Per Week','Sleep Hours Per Day','Blood sugar','CK-MB','Troponin','Gender','Systolic blood pressure','Diastolic blood pressure','Heart Attack Risk (Binary)']]
ds_texte = ds[['Age','Cholesterol','Heart rate','Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Exercise Hours Per Week','Diet','Previous Heart Problems','Medication Use','Stress Level','Sedentary Hours Per Day','Income','BMI','Triglycerides','Physical Activity Days Per Week','Sleep Hours Per Day','Blood sugar','CK-MB','Troponin','Gender','Systolic blood pressure','Diastolic blood pressure','Heart Attack Risk (Text)']]

ds_binaire.to_csv('data/donnees_traitees_binaire.csv', index=False)
ds_texte.to_csv('data/donnees_traitees_classification.csv', index=False)