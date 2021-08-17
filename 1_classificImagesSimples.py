# https://www.tensorflow.org/tutorials/keras/classification?hl=fr

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

test = 0 #Nombre d'images testées

#Chargement, creation et mise en forme des données

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)  #renvoie (nombre d'images, largeur (px), longeur (px)) ici (60000,28,28)
print(len(train_labels)) #renvoie le nombre de labels dans le training set ici 60000
print(train_labels) #renvoie l'array des labels
print(test_images.shape) ##renvoie (nombre d'images, largeur (px), longeur (px)) ici (10000,28,28)

train_images = train_images / 255.0 #Pour rendre la valeur du pixel entre 0 et 1
test_images = test_images / 255.0 

# Test d'intégrité des données

if test == 1: #On affiche juste la première image pour tester
    plt.figure()
    plt.imshow(train_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
elif test == 25: #On affiche les 25 premières images
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


#Construction du modèle

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), #Transforme une image 28x28 en un tableau monodim de 28x28 = 784
    tf.keras.layers.Dense(128, activation='relu'), #Un layer de 128 nodes
    tf.keras.layers.Dense(10)
])

#Compilation du modéle

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Optimizer —This is how the model is updated based on the data it sees and its loss function.
# Loss function —This measures how accurate the model is during training. You want to minimize this function to "steer" the model in the right direction.
# Metrics —Used to monitor the training and testing steps. The following example uses accuracy, the fraction of the images that are correctly classified.

# Entrainer le modèle
    # Nourrir le modéle 

model.fit(train_images, train_labels, epochs = 25) #epochs = nb de repet

    #Evaluer la précision

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

    #Prédire

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0]) #Renvoie un tableau d'entier entre 0 et 1
print(np.argmax(predictions[0])) #Donne la clé de la plus haute valeur du tableau de prédiction, à comparer avec test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

    #Vérifier les prédictions
wrongInput = True
while wrongInput == True:
    i = int(input("Choose a number between 1 and 60000 :"))-1
    if i>=0 & i<60000:
        wrongInput = False
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 8
num_cols = 8
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

    #Utiliser le modèle

img = test_images[1]
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))
print(img.shape)
predictions_single = probability_model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))
