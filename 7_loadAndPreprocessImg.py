import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


#On va apprendre à charger et pretraiter un dataset d'images de trois façons. 
    #1) Utilisation de Keras qui a des capacité de praitraitement de haut niveau pour lire un répertoire d'images sur le disque.
    #2) On écrira notre propre pipeline d'input de zéro en utilisant "tf.data"
    #3) On téléchargera un dataset du large catalogue de TF Datasets 

#Dtaset d'image de fleurs (3670) 5 classes : daisy/dandelion/roses/sunflowers/tulips/

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url, 
                                   fname='flower_photos', 
                                   untar=True)
data_dir = pathlib.Path(data_dir)
print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count) #3670

roses = list(data_dir.glob('roses/*'))
for i in range(2):
    with PIL.Image.open(str(roses[i]))as im:
        im.show() #On affiche les deux premières images de rose

#Chargement avec tf.keras.preprocessing
    #Créer un dataset

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory( #On recup les images directement depuis le répertoire
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names #Contient le nom des différentes classes (ici daisy/dandelion/roses/sunflowers/tulips)
print(class_names)

    #Visualiser les données

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

        #The image_batch is a tensor of the shape (32, 180, 180, 3). This is a batch of 32 images of shape 180x180x3 (the last dimension referes to color channels RGB). 
        #The label_batch is a tensor of the shape (32,), these are the corresponding labels to the 32 images.
        #NB : you can call .numpy() on either of these tensors to convert them to a numpy.ndarray.

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

    #Standardisation

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #Methode de sytrandardisation (ici, pixel € [0,255] -> [0,1])
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))
#NB: si on voulait des valeurs [-1,1] --> Rescaling(1./127.5, offset=-1)

    #Configurer pour performances
        #Let's make sure to use buffered prefetching, so you can yield data from disk without having I/O become blocking. These are two important methods you should use when loading data.
        #1).cache() keeps the images in memory after they're loaded off disk during the first epoch. This will ensure the dataset does not become a bottleneck while training your model. If your dataset is too large to fit into memory, you can also use this method to create a performant on-disk cache.
        #2).prefetch() overlaps data preprocessing and model execution while training. 
        #Plus d'inofs : https://www.tensorflow.org/guide/data_performance#prefetching

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #Entrainement du modèle
        #Infos sur la classification des images : https://www.tensorflow.org/tutorials/images/classification
num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=3
)
        #Ecrire une boucle d'entrainement au lieu d'utiliser model.fit : https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch

#Utiliser tf.data pour un controle plus fin
    #