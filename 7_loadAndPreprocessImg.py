import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import tensorflow_datasets as tfds


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
PIL.Image.open(str(roses[0])) #On affiche la première image de rose
