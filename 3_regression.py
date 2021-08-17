#Dans un problème de régression , l'objectif est de prédire la sortie d'une valeur continue, comme un prix ou une probabilité. Comparez cela avec un problème de classification ,
#où le but est de sélectionner une classe dans une liste de classes (par exemple, où une image contient une pomme ou une orange, en reconnaissant quel fruit est dans l'image).

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)


url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

print(dataset.tail())

#On supprime les valeurs 
print(dataset.isna().sum())
dataset = dataset.dropna()

#On split la colonne Origin en trois colonne pour les 3 valeurs possibles : la valeur était catégorique et pas numérique
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
print(dataset.tail())

#Séparation des données en train et test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#Inspection des données
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde', hue='MPG')
plt.show()

    #Affiche les stats générales
print(train_dataset.describe().transpose())

    #Séparer des entités via étiquettes : Séparer la valeur cible, le "label", des caractéristiques. Cette étiquette est la valeur que vous entraînerez le modèle à prédire.
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#Normalisation
train_dataset.describe().transpose()[['mean', 'std']]
 #Les plages de chaque caractéristique sont différentes : on veut normaliser les fonctionnalités d'echelles et/ou de plages différentes
 #L'une des raisons pour lesquelles cela est important est que les caractéristiques sont multipliées par les poids du modèle. 
 #Ainsi, l'échelle des sorties et l'échelle des gradients sont affectées par l'échelle des entrées.
 #Bien qu'un modèle pourrait converger sans normalisation des fonctionnalités, la normalisation rend la formation beaucoup plus stable.

    #La couche de normalisation
normalizer = preprocessing.Normalization(axis=-1)
#Création d'un calque
normalizer.adapt(np.array(train_features))
#Cela calcule la moyenne et la variance, et les stocke dans la couche.
print(normalizer.mean.numpy())

first = np.array(train_features[:1])
with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())

#Lorsque la couche est appelée, elle renvoie les données d'entrée, chaque entité étant normalisée indépendamment

#Régression linéaire
    #Une variable (MPG)
        #La formation d' un modèle avec tf.keras commence généralement en définissant l'architecture du modèle.
        #Dans ce cas , utilisez un keras.Sequential modèle. Ce modèle représente une séquence d'étapes. Dans ce cas, il y a deux étapes :
            # 1) Normaliser l'entrée de horsepower .
            # 2)Appliquer une transformation linéaire (y=ax+b) pour produire une sortie en utilisant layers.Dense .
        #Le nombre d'entrées peut être soit fixée par le input_shape argument ou automatiquement lorsque le modèle est exécuté pour la première fois.

        #Tout d' abord créer la couche Normalization:
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = preprocessing.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

        #Puis le modèle sequentiel pour prédire MPG de Horsepower
horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
print(horsepower_model.summary())

        #Execution du modele non-train sur les 10 première val de Horsepower. Sortie pas bonne mais de la forme attendue (10,1)
print(horsepower_model.predict(horsepower[:10]))

        #Training
horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

history = horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_loss(history)

    #Stockage des résultats de test
test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

    #Regression à 1 variable : facile d'examiner les predictions du modele en fct de l'entrée

x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

def plot_horsepower(x, y):
    plt.scatter(train_features['Horsepower'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

plot_horsepower(x,y)

    #Variables multiples (n variables, tjr linéaire, mais y=ax+b avec a qui est une matrice n*n et b un vecteur)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
]) 
print(linear_model.predict(train_features[:10]))

    #Lorsque vous appelez le modèle, ses matrices de poids seront construites. Maintenant , vous pouvez voir une forme en (9,1) pour les 9 variables d'entrées 
    # (8 var, MPG est la sortie, donc 7, Origine est sep en 3, donc 6+3=9 variables d'entrée)
linear_model.layers[1].kernel

    #Les .compil et .fit ne changent pas en multi-var
linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')
    
history = linear_model.fit(
    train_features, train_labels, 
    epochs=100,
    # suppress logging
    verbose=0,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

    #Utiliser toutes les entrées permet d'obtenir une erreur beaucoup plus faible et une erreur de validation que la horsepower modèle:
plot_loss(history)

    #Collecte des résultats
test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)

#Regression DNN (Deep Neural Network)
#Code assez identique que pour la reg linéaire, sauf que lme modele s'étant à des couches non-linéaires "cachées" (= pas de moyen direct de connection avec les entrées et les sorties)
#Contiennent plus de couches que pour le modele linéaire:
    #La couche de normalisation
    #2 couches Dense "cachées" qui utilisent le relu nonlinearity
    #Une couche mono-entrée linéaire
#Les deux modeles utiliseront la même procédure d'entrainement -> compile methode incluse dans la fonction suivante

def build_and_compile_model(norm):
    modele = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    modele.compile(loss='mean_absolute_error',
        optimizer=tf.keras.optimizers.Adam(0.001))
    return modele

    #Une variable (Horsepower)

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
        #Ce modèle a plusieurs paramètre d'entrainement supp par rapport aux modeles linéaires
print(dnn_horsepower_model.summary())

history = dnn_horsepower_model.fit(
    train_features['Horsepower'], train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history) #Fait un peu mieux que le modele une var linéaire

        #Les predictions montrent l'avantage de ce modele non-linéaire sur le modele linéaire
x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)
plot_horsepower(x, y)

        #Collecte des resultats
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0)

    #Modèle complet
        #L'utilisation de toutes les entrées augmente la perf de la validation
dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

#Performance
#On compare ici l'erreur moyenne via les resultats sauvés prédemment
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)

    #Faire des prédictions
        #Ici on regarde les erreurs du modele quand il fait des predictions sur le test
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

        #Il semble que le modele prédise assez bien
        #Regardons l'erreur de distribution

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [MPG]')
_ = plt.ylabel('Count')
plt.show()

        #Centré globalement sur 0 : la prédiction est assez correcte !
#On sauvegarde le modele pour une utilisation future
dnn_model.save('AI-TF-Learning-main\dnn_model')

#En rechargeant le modele, on obtient un résultat identique
reloaded = tf.keras.models.load_model('AI-TF-Learning-main\dnn_model')

test_results['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0)
print(pd.DataFrame(test_results, index=['Mean absolute error [MPG]']).T)