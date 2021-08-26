#Dans les exemples précdent, on a vu que la précision du modele sur les données de validation atteint sa limite après N epochs, puis stagne ou décroit.
#Dans ce cas, notre modele est surajusté/en sur-apprentissage ("overfit") aux données d'entrainement. Il faut apprendre à gérer cet overfit. 
#Il est souvent possible d'obtenir une grande précision sur l'ensemble de train, mais on veut plutôt max cette valeur pour des ensembles de tst (aka des données inconnues jusqu'à présent)
#A l'inverse, il existe le sous-apprentissage/sousajusté ("underfit"), qui se produit lorsque le train peut encore être amélioré. 
    #Peut se produire car modele pas assez puissant, trop régularisé ou pas assez entrainé. Cela signifie que le réseau n'a pas appris les modeles pertinent des données du train.
#Mais si l'entrainement est trop long, le modele commencera a s'overfit et a apprendre des modeles à partie de données d'entrainement qui ne se généraliseront pas aux données de test.
#Le but est donc de trouver un équilibre : comprendre comment s'entrainer pour un bon nombre d'epochs.
#Pour éviter l'overfit, la meilleure solution est d'utiliser des données d'entrainement plus complètes : l'ensemble des données doit couvrir TOUTE la gamme d'entrées que le modèle est censé gérer.
#Les nouvelles données ne peuvent être utiles que si elles couvrent des cas nouveaux et interrésants pour le modele.
#Un modele entrainé sur des données plus complètes se généralisera naturellement mieux. Quand cela n'est plus possible, le mieux est d'utiliser des techniques comme la régularisation
#Ces méthodes imposent des contraintes sur la qté et le type d'infos que le modele peut stocker. Si un réseau ne peut mémoriser qu'un petit nb de motifs, le process d'optimisation le forcera à se concentrer sur les plus importants, qui pourraient mieux se généraliser

from inspect import formatannotation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
from  IPython import display
from matplotlib import pyplot as plt
import numpy as np
import pathlib
import shutil
import tempfile

logdir = pathlib.Path(tempfile.mkdtemp())/"tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

#L'ensemble de données de Higgs

#Le jeu de données contient 11 000 000 exemples, chacun avec 28 fonctionnalités, et une étiquette de classe binaire.
gz = tf.keras.utils.get_file('HIGGS.csv.gz', 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz')
FEATURES = 28
# La classe tf.data.experimental.CsvDataset peut être utilisée pour lire des enregistrements csv directement à partir d'un fichier gzip sans étape de décompression intermédiaire.
ds = tf.data.experimental.CsvDataset(gz,[float(),]*(FEATURES+1), compression_type="GZIP")

#Cette classe de lecteur csv renvoie une liste de scalaires pour chaque enregistrement. 
#La fonction suivante reconditionne cette liste de scalaires dans une paire (feature_vector, label).
def pack_row(*row):
  label = row[0]
  features = tf.stack(row[1:],1)
  return features, label

#TensorFlow est plus efficace lorsqu'il fonctionne sur de gros lots de données.
#Ainsi, au lieu de reconditionner chaque ligne individuellement, créez un nouvel ensemble de Dataset qui prend des lots de 10 000 exemples, applique la fonction pack_row à chaque lot, puis divise les lots en enregistrements individuels

packed_ds = ds.batch(10000).map(pack_row).unbatch()

for features,label in packed_ds.batch(1000).take(1):
    print(features[0])
    plt.hist(features.numpy().flatten(), bins = 101)
    plt.show()

#Pour garder ce didacticiel relativement court, utilisez uniquement les 1000 premiers échantillons pour la validation et les 10 000 suivants pour la formation
#Les méthodes Dataset.skip et Dataset.take facilitent cette tâche.
#Dans le même temps, utilisez la méthode Dataset.cache pour vous assurer que le chargeur n'a pas besoin de relire les données du fichier à chaque époque 

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN//BATCH_SIZE

validate_ds = packed_ds.take(N_VALIDATION).cache()
train_ds = packed_ds.skip(N_VALIDATION).take(N_TRAIN).cache()
print(train_ds)

#Ces ensembles de données renvoient des exemples individuels. Utilisez la méthode .batch pour créer des lots d'une taille appropriée pour l'entraînement.
#Avant le traitement par lots, n'oubliez pas .shuffle et de .repeat l'ensemble d'apprentissage.

validate_ds = validate_ds.batch(BATCH_SIZE)
train_ds = train_ds.shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)


#Démontrer le surapprentissage

#Le plus simple est de commencer avec un petit modèle, avec un petit nb de paramètres pouvant être appris (determiné par le nb de couches et d'unités/couche). En DL, le nb de param est appelé "capacité" du modèle
#Un modèle avec + de param aura plus de capacité de mémorisation,  et sera donc capable d'apprendre facilement un mapping parfait de type dictionnaire entre les échantillons d'apprentissage et leurs cibles, un mapping sans aucun pouvoir de généralisation, mais cela serait inutile pour faire des prédictions sur des données inédites.
#Les modèles d'apprentissage en profondeur ont tendance à bien s'adapter aux données d'entraînement, mais le vrai défi est la généralisation, pas l'ajustement.
#A l'inverse, peu de capacité de mémorisation implique que l'apprentissage sera difficile.  Pour minimiser sa perte, il devra apprendre des représentations compressées qui ont plus de pouvoir prédictif. En même temps, si vous rendez votre modèle trop petit, il aura du mal à s'adapter aux données d'entraînement.
#Il y a un équilibre entre "trop ​​de capacité" et "pas assez de capacité".
#Malheureusement, il n'y a pas de formule magique pour déterminer la bonne taille ou l'architecture de votre modèle (en termes de nombre de couches, ou la bonne taille pour chaque couche). Vous devrez expérimenter en utilisant une série d'architectures différentes.
#Pour trouver une taille de modèle appropriée, il est préférable de commencer avec relativement peu de couches et de paramètres, puis de commencer à augmenter la taille des couches ou à ajouter de nouvelles couches jusqu'à ce que vous voyiez des retours décroissants sur la perte de validation.
#Commencez avec un modèle simple en utilisant uniquement des layers.Dense comme référence, puis créez des versions plus grandes et comparez-les.

    #Procédure de formation

    #De nombreux modèles s'entraînent mieux si vous réduisez progressivement le taux d'apprentissage pendant l'entraînement. Utilisez optimizers.schedules pour réduire le taux d'apprentissage au fil du temps
    #Ce code définit un schedules.InverseTimeDecay pour diminuer hyperboliquement le taux d'apprentissage à 1/2 du taux de base à 1000 époques, 1/3 à 2000 époques... (cf courbe)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

def get_optimizer():
  return tf.keras.optimizers.Adam(lr_schedule)

step = np.linspace(0,100000)
lr = lr_schedule(step)
plt.figure(figsize = (8,6))
plt.suptitle("Taux apprentissage en fonction du nb d'epochs")
plt.plot(step/STEPS_PER_EPOCH, lr)
plt.ylim([0,max(plt.ylim())])
plt.xlabel('Epoch')
_ = plt.ylabel('Learning Rate')
plt.show()

    #Chaque modèle de ce didacticiel utilisera la même configuration d'entraînement. Configurez-les donc de manière réutilisable, en commençant par la liste des rappels.
    #La formation pour ce didacticiel s'étend sur de nombreuses périodes courtes. Pour réduire le bruit de journalisation, utilisez le tfdocs.EpochDots qui imprime simplement un tfdocs.EpochDots . pour chaque époque, et un ensemble complet de métriques toutes les 100 époques.
    #Ensuite, incluez les callbacks.EarlyStopping pour éviter des temps d'entraînement longs et inutiles. Notez que ce rappel est défini pour surveiller le val_binary_crossentropy , pas le val_loss . Cette différence sera importante plus tard.
    #Utilisez callbacks.TensorBoard pour générer des journaux TensorBoard pour la formation.

def get_callbacks(name):
  return [
    tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir/name),
  ]

    #De même, chaque modèle utilisera les mêmes paramètres Model.compile et Model.fit

def compile_and_fit(model, name, optimizer=None, max_epochs=10000):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[
                  tf.keras.losses.BinaryCrossentropy(
                      from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(
    train_ds,
    steps_per_epoch = STEPS_PER_EPOCH,
    epochs=max_epochs,
    validation_data=validate_ds,
    callbacks=get_callbacks(name),
    verbose=0)
  return history

    #Entrainement d'un petit modèle (1 couche de 16)

tiny_model = tf.keras.Sequential([
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(1)
])
size_histories = {}
size_histories['Tiny'] = compile_and_fit(tiny_model, 'sizes/Tiny')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

    #On augmente un peu la taille (2 couches de 16 unités chacunes)

small_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(16, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(16, activation='elu'),
    layers.Dense(1)
])
size_histories['Small'] = compile_and_fit(small_model, 'sizes/Small')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

    #Encore un peu (3 couches de 64 unités)

medium_model = tf.keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    layers.Dense(64, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(64, activation='elu'),
    layers.Dense(64, activation='elu'),
    layers.Dense(1)
])
size_histories['Medium'] = compile_and_fit(medium_model, 'sizes/Medium')

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)
plotter.plot(size_histories)
plt.ylim([0.5, 0.7])

    #Un modele plus grand

large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model, "sizes/large")
