import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras

#Dataset d'exemple

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000] #On prends que les 1000 permiers ex, pour accélérer
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#Définir le modèle

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.SparseCategoricalAccuracy()])

  return model

    # Create a basic model instance
model = create_model()
    # Display the model's architecture
model.summary()

#Checkpoint de sauvegarde pendant l'entrainement
    #Un modèle entrainé peut être réutilisé sans avoir à le ré-entrainer, ou en court d'entrainement.
    #La méthode " tf.keras.callbacks.ModelCheckpoint" permet de sauver continuellement le modèle pendant et à la fin de l'entrainement.

    #Créer un checkpoint pendant l'entrainement

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

        # Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

        # Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])  # Pass callback to training

        # This may generate warnings related to saving the state of the optimizer.
        # These warnings (and similar warnings throughout this notebook) are in place to discourage outdated usage, and can be ignored.

        #This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:

print(os.listdir(checkpoint_dir))

        #Tant que deux modèles ont la même architecture, les poids sont partageables.
        #Donc restaurer un modèle depuis ses poids = créer un nouveau avec la même architecture que l'original et définir ses poids
    
        #Un modèle non entrainé aura ~10% accuracy :

# Create a basic model instance
model = create_model()
# Evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

        #Alors que lorsqu'on charge les poids :
# Loads the weights
model.load_weights(checkpoint_path)
# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


    #Option de checkpoint
        #Il est possible de faire un nommage unique pour chaque checkpoint et d'ajuster leur fréquence :

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
batch_size = 32
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq=5*batch_size)
# Create a new model instance
model = create_model()
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))
# Train the model with the new callback
model.fit(train_images, 
          train_labels,
          epochs=50, 
          batch_size=batch_size, 
          callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)
print(os.listdir(checkpoint_dir))
latest = tf.train.latest_checkpoint(checkpoint_dir) #Donne l'enregistrement le plus récent
print(latest)

        #Testons en utilisant cette sauvegarde sur un nouveau modèle

model = create_model()
model.load_weights(latest)
# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#Que sont ces fichiers ?
    #Ce sont des fichiers checkpoint-formated binaires qui contiennent les poids entrainés. Les checkpoint contiennent:
        #1) Un ou plusiers fragment qui contiennent les poids du modele
        #2) Un index qui indique quel poids est stocké dans quel fragment
    #Si on entraine le model sur une seule machine, il n'y aura qu'un seul éclat avec le suffixe .data-00000-of-00001
    
#Sauvegarde manuelle
    #Possible avec la methode "model.save_weights". Par défaut, keras et sav_weights utilisent le format .ckpt de tensorflow (sauvegarde en HDF5 (ou format .h5) --> https://www.tensorflow.org/guide/keras/save_and_serialize#weights_only_saving_in_savedmodel_format)

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')
model = create_model()
# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')
loss, acc = model.evaluate(test_images, test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

#Sauvegarde du modèle entier

    #La méthode "model.save" permet de sauver l'architecture, les poids et l'entrainement d'un modèle en un seul fichier/dossier. Cela permet d'exporter des modèles sans avoir accès au code originel. (NB : Custom objects (e.g. subclassed models or layers) require special attention when saving and loading. See the Saving custom objects section below)
    #Deux formats de sauvegarde existent : HDF5 et SaveModel. Le dernier cospond au format standard TF2;X.
    #Sauver un modèle fonctionnel est très utile, notamment pour les utiliser avec TF.js ou TFLite pour mobile avec un modèle créé sous python initiallement
    
    #Format SavedModel
        #Restoration via "tf.keras.models.load_model", compatiblke avec TF Serving (Plus d'infos : https://www.tensorflow.org/guide/saved_model)

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
os.mkdir("saved_model")
model.save('saved_model/my_model')
        #Contient un fichier binaire protobuf et un checkpoint TF
print(os.listdir("saved_model"))
print(os.listdir("saved_model/my_model"))

        #Recharge d'un nouveau modele via le sauvé :

new_model = tf.keras.models.load_model('saved_model/my_model')
# Check its architecture
new_model.summary()

        #Le modèle recréé ets compilé avec les mêmes arguments que le modele original.
        # Evaluate the restored model
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

print(new_model.predict(test_images).shape)

    #Format HDF5
        #Fourni de base par keras (HDF5 standard)

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model should be saved to HDF5.
model.save('my_model.h5')

        #Puis on réutilise la sauvegarde pour recréer un modèle
# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()
loss, acc = new_model.evaluate(test_images, test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        #Keras sauve les modele en ispectant leur architecture, ce qui sauvegarde tout :
            #1)La valmeur des poids
            #2) L'architecture du modele
            #3) La configuration de l'entrainement du modele (ce qui est donné à la méthode .compile())
            #4) L'optimizer et son état, s'il y en a un (ce qui permet de reprendre un entrainement à tout point d'enregistrement)
        #Keras n'est pas capable de sauver les optimizer v1.x (via tf.compat.v1.train) car non compatible avec les checkpoints. Dans ce cas, il faut re-compiler le model après le chargement, ce qui perd l'état de l'optimizer.

    #Sauvegarde d'objets customs
        #Seulement en HDF5 car il utilise les configs objet pour sauver l'architecture du modele, alors que SavedModel sauve le graph d'execution, ce qui lui permet de sauver des objets customs comme des subclassed model ou des customs layers sans avoir besoin du code original.
        #Pour sauver des obejts customs en HDF5, il faut :
            #1) Définir une méthode "get_config(self)" dans l'objet, et eventuellement une methode de classe"from_config"
                #get_config(self) doit renvoyer un dico sérializable JSON de parametre requis pour recréer l'objet
                #from_config(cls, config) qui utilise ce que renvoie get_configue pour créer un nouvel objet. Par defaut, cette fonction la config en kwargs d'init (return cls(**config))
            #2) Passer l'objet à l'attribut "custom_objects" quand on charge le modele. Cet argument doit etre un dico qui mappe le string class name to the Python class (Ex : tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer}))
        
        #Pour des exemples d'objets customs : https://www.tensorflow.org/guide/keras/custom_layers_and_models