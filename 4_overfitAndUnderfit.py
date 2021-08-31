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

#from tensorflow.core.protobuf import rewriter_config_pb2
#config_proto = tf.compat.v1.ConfigProto()
#off = rewriter_config_pb2.RewriterConfig.OFF
#config_proto.graph_options.rewrite_options.arithmetic_optimization = off
#session = tf.compat.v1.Session(config=config_proto)

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

#from tensorflow.core.protobuf import rewriter_config_pb2
#from tensorflow.keras.backend import set_session
#tf.keras.backend.clear_session()  # For easy reset of notebook state.

#config_proto = tf.ConfigProto()
#off = rewriter_config_pb2.RewriterConfig.OFF
#config_proto.graph_options.rewrite_options.arithmetic_optimization = off
#session = tf.Session(config=config_proto)
#set_session(session)

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
plt.show()

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
plt.show()

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
plt.show()

    #Un modele plus grand

large_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(512, activation='elu'),
    layers.Dense(1)
])
size_histories['large'] = compile_and_fit(large_model, "sizes/large")

#Tracé des pertes de form & val

  #Plein = formation, pointillé = val
  #Bien que la construction d'un modèle plus grand lui donne plus de puissance, si cette puissance n'est pas limitée d'une manière ou d'une autre, elle peut facilement s'adapter à l'ensemble d'entraînement.

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0.5, 0.7])
plt.xlabel("Epochs [Log Scale]")
plt.show()

  #Dans cet exemple, en général, seul le modèle "Tiny" parvient à éviter complètement le surajustement, et chacun des modèles plus grands suradapte les données plus rapidement. 
  #Cela devient si grave pour le "large" modèle que vous devez passer le tracé à une échelle logarithmique pour vraiment voir ce qui se passe.
  #Cela est évident si vous tracez et comparez les métriques de validation aux métriques d'entraînement :
    #Une différence entre les deux est normale
    #Les deux évoluent dans la même direction = OK (Cas Tiny)
    #Si la validation stagne alors que l'entrainement continue d'augmenter, le surapprentissage est proche (Cas Small)
    #Si la validation va dans la mauvaise direction = modèle surajusté (Cas Medium et Huge)


#Affichage dans tensorboard

  #cf https://www.tensorflow.org/tensorboard/get_started

#Eviter le surapprentissage

  #Copie les journaux d'entraînement du modèle "Tiny"
shutil.rmtree(logdir/'regularizers/Tiny', ignore_errors=True)
shutil.copytree(logdir/'sizes/Tiny', logdir/'regularizers/Tiny')

regularizer_histories = {}
regularizer_histories['Tiny'] = size_histories['Tiny']

  #Ajouter une régularisation de poids
    #Etant donné certaines données d'entraînement et une architecture de réseau, il existe plusieurs ensembles de valeurs de pondération (plusieurs modèles) qui pourraient expliquer les données, et les modèles plus simples sont moins susceptibles de surajuster que les modèles complexes.
    #Un « modèle simple » dans ce contexte est un modèle où la distribution des valeurs des paramètres a moins d'entropie (ou un modèle avec moins de paramètres au total, comme nous l'avons vu dans la section ci-dessus).
    #Un moyen courant d'atténuer le surapprentissage consiste à imposer des contraintes sur la complexité d'un réseau en forçant ses poids à ne prendre que de petites valeurs, ce qui rend la distribution des valeurs de poids plus « régulière ».
    #C'est ce qu'on appelle la « régularisation des poids », et cela se fait en ajoutant à la fonction de perte du réseau un coût associé au fait d'avoir des poids importants.
    #2 types :
      #Le coût ajouté est proportionnel à la valeur absolue des coefficients de pondération (c'est-à-dire à ce qu'on appelle la « norme L1 » des pondérations). (https://developers.google.com/machine-learning/glossary/#L1_regularization)
      #Le coût ajouté est proportionnel au carré de la valeur des coefficients de poids (c'est-à-dire à ce qu'on appelle le carré "norme L2" des poids). La régularisation L2 est également appelée perte de poids dans le contexte des réseaux de neurones. La décroissance du poids est mathématiquement exactement la même que la régularisation L2. (https://developers.google.com/machine-learning/glossary/#L2_regularization)
    #La régularisation L1 pousse les poids vers exactement zéro, encourageant un modèle clairsemé. 
    #La régularisation L2 pénalisera les paramètres de poids sans les rendre clairsemés puisque la pénalité passe à zéro pour les petits poids - une des raisons pour lesquelles L2 est plus courant.

    #Dans tf.keras , la régularisation de poids est ajoutée en passant des instances de régularisation de poids aux couches en tant qu'arguments de mot-clé. Ajoutons maintenant la régularisation du poids L2.

l2_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001), #l2(0.001) signifie que chaque coefficient dans la matrice de pondération de la couche ajoutera 0.001 * weight_coefficient_value**2 à la perte totale du réseau.
                 input_shape=(FEATURES,)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)), #A appliquer à chaque couche
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(512, activation='elu',
                 kernel_regularizer=regularizers.l2(0.001)),
    layers.Dense(1)
])

regularizer_histories['large_l2'] = compile_and_fit(l2_model, "regularizers/large_l2")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.xlim([5, max(plt.xlim())])
plt.xlabel("Epochs [Log Scale]")
plt.show()

    #Le modèle régularisé "L2" est désormais beaucoup plus compétitif que le modèle "Tiny" . 
    #Ce modèle "L2" est également beaucoup plus résistant au surapprentissage que le modèle "Large" sur lequel il était basé malgré le même nombre de paramètres."

  #Plus d'infos sur les régul
    #2 choses importantes pour la regul
      #1) Si vous écrivez votre propre boucle d'entraînement, alors vous devez vous assurer de demander au modèle ses pertes de régularisation.

result = l2_model(features)
regularization_loss=tf.add_n(l2_model.losses)

      #2) Cette implémentation fonctionne en ajoutant les pénalités de poids à la perte du modèle, puis en appliquant une procédure d'optimisation standard par la suite.
      #Il existe une deuxième approche qui, à la place, n'exécute l'optimiseur que sur la perte brute, puis lors de l'application du pas calculé, l'optimiseur applique également une certaine perte de poids. 
      #Cette "déclinaison de poids découplée" est observée dans les optimiseurs tels que optimizers.FTRL et optimizers.AdamW.

#Ajouter un abandon

  #L'abandon est l'une des techniques de régularisation les plus efficaces et les plus couramment utilisées pour les réseaux de neurones, développée par Hinton et ses étudiants de l'Université de Toronto.
  #L'explication intuitive de l'abandon est que, étant donné que les nœuds individuels du réseau ne peuvent pas compter sur la sortie des autres, chaque nœud doit générer des fonctionnalités utiles en soi.
  #L'abandon, appliqué à une couche, consiste à "abandonner" aléatoirement (c'est-à-dire mis à zéro) un certain nombre de caractéristiques de sortie de la couche pendant l'entraînement. 
  #Disons qu'une couche donnée aurait normalement renvoyé un vecteur [0.2, 0.5, 1.3, 0.8, 1.1] pour un échantillon d'entrée donné pendant l'apprentissage ; après avoir appliqué la suppression, ce vecteur aura quelques entrées nulles distribuées au hasard, par exemple [0, 0.5, 1.3, 0, 1.1].
  #Le « taux d'abandon » est la fraction des caractéristiques qui sont mises à zéro ; il est généralement fixé entre 0,2 et 0,5.
  #Au moment du test, aucune unité n'est abandonnée, et à la place, les valeurs de sortie de la couche sont réduites d'un facteur égal au taux d'abandon, afin de compenser le fait que plus d'unités sont actives qu'au moment de l'entraînement.
  #Dans tf.keras vous pouvez introduire l'abandon dans un réseau via la couche Dropout, qui est appliquée à la sortie de la couche juste avant.
  
dropout_model = tf.keras.Sequential([
    layers.Dense(512, activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5), #Ici
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5), #A répéter pour toutes les couches
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['large_dropout'] = compile_and_fit(dropout_model, "regularizers/large_dropout")

plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.xlim([5, max(plt.xlim())])
plt.xlabel("Epochs [Log Scale]")
plt.show()

  #Il ressort clairement de ce graphique que ces deux approches de régularisation améliorent le comportement du modèle "Large" . Mais cela ne bat toujours pas même la ligne de base "Tiny" 

#Combinaison de la regul + abandon

combined_model = tf.keras.Sequential([
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu', input_shape=(FEATURES,)),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(512, kernel_regularizer=regularizers.l2(0.0001),
                 activation='elu'),
    layers.Dropout(0.5),
    layers.Dense(1)
])

regularizer_histories['combined'] = compile_and_fit(combined_model, "regularizers/combined")
plotter.plot(regularizer_histories)
plt.ylim([0.5, 0.7])
plt.xlim([5, max(plt.xlim())])
plt.xlabel("Epochs [Log Scale]")
plt.show()

  #Ce modèle avec la régularisation "Combined" est évidemment le meilleur à ce jour.

#Conclusion : 

  #Eviter le surapprentissage :
    #Avoir + de données d'entrainement
    #Réduire la capacité du réseau
    #Ajouter une regul de poids
    #Ajouter un abandon
  #Autres façons non couvertes ici:
    #Batch normalization
    #Data-augmentation