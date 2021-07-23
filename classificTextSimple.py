#https://www.tensorflow.org/tutorials/keras/text_classification

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import re
import shutil
import string
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# DL data

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# Explore data
    #Arborescence
print(os.listdir(dataset_dir))
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

    #Fichiers
sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
  print(f.read())

#Charger le dataset

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed) #25000 exemples, 20000 utilisés pour l'entrainement

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Review", text_batch.numpy()[i]) #Le texte brut du commentaire
    print("Label", label_batch.numpy()[i]) #Donne l'info du type du comm (+ ou -)

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("\n")

# Validation (On utilisera les 5000 restants pour la validation)

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, #20% des data serviront à la validation
    subset='validation', 
    seed=seed) #Penser à spécifier une seed aléatoire, ou shuffle=False pour éviter une superposition dans les éléments de la validation et de l'entrainement !

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)

#Préparation du dataset pour l'entrainement en 3 phases

    #Standadisation : suppression du texte "parasite" : ici, ponctuation + balises HTML par ex

    #Echantillonage (tokenization) : séparer les phrases en token (mots par ex)

    #Vectorisation : Transformer les tokens en nombres pour le réseau de neurones

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization, # Défini par la fonction de standardisation ci-dessus
    max_tokens=max_features,
    output_mode='int', # Permet d'avoir un index unique pour chaque token
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
# It's important to only use your training data when calling adapt (using the test set would leak information).
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label): # Permet de voir le résultat de l'utilisation de ce layer pour preporcess des données
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287]) #Donne le string qui correspond à cet int 
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print(" 2 ---> ",vectorize_layer.get_vocabulary()[2])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

#Application du TextVectorization sur les dataset d'entrainement, de validation et de test
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

#Configuration du dataset pour performances

    # 2 méthodes à appliquer pour faire en sorte que I/O ne devienne pas bloquant :
    # .cache() pour garder la mémoire en cache pour éviter la perte de données en entrainement
    # .prefetch() overlaps data preprocessing and model execution while training.
     
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Création du modèle 

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim), # Recoit les token int et cherche un vecteur d'encastrement pour chaque. ces vecteurs sont appris pendant l'entrainement du modèle, et ajoutent une dimension au tableau de sortie de la forme (batch, sequence, embedding). 
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()
