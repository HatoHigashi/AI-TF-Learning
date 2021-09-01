#Keras-tuner est une library permettant de choisir le meilleur set d'hyperparametres pour le programme. Le fait de selectionner le bon set d'hyperpram pour le ML est appelé "hypertuning"
#Les hyperpara sont des variables qui décident le trainig process et la toology d'un model ML. Elles restent contantes pendant la phase d'entrainementet impacte la perf du programme.
#Il y en a 2 types :
    #1)Les hyperpara du modele qui influencent la selection du modele comme le nombre et/ou la largeur des couchers cachées
    #2) Les hyperpara d'algorithme qui influenecent la vitesse et la qualité de l'algo d'appentissage que le taux d'apprentissage du SGD ("Stochastic Gradient Descent") et le nombre de voisins proches pour un classifier KNN ("k nearest neighboor")


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from tensorflow import keras

import keras_tuner as kt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data() #Même dataset que pour 0

# Normalize pixel values between 0 and 1
img_train = img_train.astype('float32') / 255.0
img_test = img_test.astype('float32') / 255.0

#Définition du modele

    #Lors de la def d'un modele hyper-accordé, on défini aussi le search space des hyperpara en plus de l'achitecture du modèle
    #Il existe 2 approches :
        #1) Utiliser une fonction de création de modèle
        #2) En subclassant la classe "HyperModel" de l'API Keras Tuner
        # Il est aussi possible d'utiliser deux classes HyperModel prédéfinies : HyperXception et HyperResNet pour des application de computer vision.

def model_builder(hp): #Renvoie un modèle compilé et utilise les hyperpara choisi pour hypertune le modèle
  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape=(28, 28)))

  # Tune the number of units in the first Dense layer
  # Choose an optimal value between 32-512
  hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
  model.add(keras.layers.Dense(units=hp_units, activation='relu'))
  model.add(keras.layers.Dense(10))

  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

#Instancier le tuner et faire de l'hypertuning

    #Keras Tuner a 4 tuners disponibles : RandomSearch, Hyperband, BayesianOptimization, et Sklearn. On utilisera le second ici.
    #Pour l'instancier, on doit spécifier l'hypermodele, l'obectif d'optimisation et le nb max d'epochs.

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')

    #L'algo d'hyperband utilise des ressources adaptatives et un early-stopping pour converger rapidement vers un modele avec des hautes perf, via une selection d type compet sportive.
    #L'algo entraine un grand nombre de modeles pour peu d'epochs et continue seulement avec la moitié des modeles les + performants. Hyperband determine le nombre de models à entrainer avec la formule "1+log(max_epochs) base factor" arrondi à l'enteir le plus proche.
    
    #Fonction de callback pour stopper l'entrainement après qu'une certaine valeur de perte est atteinte
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    #Ici on cherche les hyperpara . Les arguments sont les mêmes que pour un model.fit, avec la fonction de callback précédentement définie
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

    # --> Trial 30 Complete [00h 00m 44s]
        #val_accuracy: 0.8840833306312561

        #Best val_accuracy So Far: 0.8882499933242798
        #Total elapsed time: 00h 12m 17s

        #The hyperparameter search is complete. The optimal number of units in the first densely-connected
        #layer is 288 and the optimal learning rate for the optimizer
        #is 0.001.

#Entrainer le modèle
    #On cherche le nombre optimal d'epochs 

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

    #Puis on ré-instancie l'hypermodel et on l'entraine avec le nb d'epochs trouvé
hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(img_train, label_train, epochs=best_epoch, validation_split=0.2)

    #Enfin, on évalue l'hypermodele sur l'ensemble de test
eval_result = hypermodel.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

    #Le répertoir "my_dir/intro_to_kt" contient les logs détaillé et les checkpoints de chaque trial efféctué pendant la recherche des hyperparametres.
    #Si l'on refait cette recherche, KT utilise les logs pour continuer la recherche.
    #Pour désactiver ce comportement, on utilise l'argument "overwrite=True" en instanciant le tuner.

#+ d'infos : https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html / https://keras-team.github.io/keras-tuner/ / https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams