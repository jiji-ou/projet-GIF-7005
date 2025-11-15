import pandas as pd

ds = pd.read_csv('data/heart-attack-risk-prediction-dataset.csv')

# Transformer le sexe en données numériques
ds[ds == 'Male'] = 1 # J'ai choisi 1 pour les hommes parce que les hommes ont un plus haut risque de crises de coeur
ds[ds == 'Female'] = 0

# Retirer les lignes avec des données manquantes
ds = ds.dropna()

# Séparer Diet en trois variables binaires
ds = pd.get_dummies(ds, columns=['Diet'], dtype=int)

# Mettre les variables binaires comme des int
ds[['Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Diet_0','Diet_1','Diet_2','Previous Heart Problems','Medication Use','Gender','Heart Attack Risk (Binary)']] = ds[['Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Diet_0','Diet_1','Diet_2','Previous Heart Problems','Medication Use','Gender','Heart Attack Risk (Binary)']].astype(int)

# Normaliser le nombre de jours d'activité physique par semaine
ds.loc[:, 'Physical Activity Days Per Week'] /= ds.loc[:, 'Physical Activity Days Per Week'].max()
ds.loc[:, 'Stress Level'] /= ds.loc[:, 'Stress Level'].max()

# Mettre la colonne cible comme dernière colonne
ds_binaire = ds[['Age','Cholesterol','Heart rate','Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Exercise Hours Per Week','Diet_0','Diet_1','Diet_2','Previous Heart Problems','Medication Use','Stress Level','Sedentary Hours Per Day','Income','BMI','Triglycerides','Physical Activity Days Per Week','Sleep Hours Per Day','Blood sugar','CK-MB','Troponin','Gender','Systolic blood pressure','Diastolic blood pressure','Heart Attack Risk (Binary)']]
ds_texte = ds[['Age','Cholesterol','Heart rate','Diabetes','Family History','Smoking','Obesity','Alcohol Consumption','Exercise Hours Per Week','Diet_0','Diet_1','Diet_2','Previous Heart Problems','Medication Use','Stress Level','Sedentary Hours Per Day','Income','BMI','Triglycerides','Physical Activity Days Per Week','Sleep Hours Per Day','Blood sugar','CK-MB','Troponin','Gender','Systolic blood pressure','Diastolic blood pressure','Heart Attack Risk (Text)']]

ds_binaire.to_csv('data/donnees_traitees_binaire.csv', index=False)
ds_texte.to_csv('data/donnees_traitees_classification.csv', index=False)