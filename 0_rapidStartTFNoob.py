#Pour cacher les W et I

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

# Charge et prépare l' ensemble de données MNIST . Convertit les échantillons d'entiers en nombres à virgule flottante :
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

# Construit le modèle tf.keras.Sequential en empilant des couches. Choisit un optimiseur et une fonction de perte pour l'entraînement :

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
    ])

# Pour chaque exemple, le modèle renvoie un vecteur de scores " logits " ou " log-odds ", un pour chaque classe.

predictions = model(x_train[:1]).numpy()
print(predictions)

# La fonction tf.nn.softmax convertit ces logits en "probabilités" pour chaque classe :

proba = tf.nn.softmax(predictions).numpy()
print(proba)

# La perte losses.SparseCategoricalCrossentropy prend un vecteur de logits et un indice True et renvoie une perte scalaire pour chaque exemple.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Cette perte est égale au log de probabilité négatif de la vraie classe : elle est nulle si le modèle est sûr de la bonne classe.
# Ce modèle non entraîné donne des probabilités proches du hasard (1/10 pour chaque classe), donc la perte initiale devrait être proche de -tf.math.log(1/10) ~= 2.3 .

probaloss = loss_fn(y_train[:1], predictions).numpy()
print(probaloss)

# La méthode Model.fit ajuste les paramètres du modèle pour minimiser la perte :

model.compile(optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

# La méthode Model.evaluate vérifie les performances des modèles, généralement sur un " Validation-set " ou " Test-set ".

model.evaluate(x_test, y_test, verbose=2)

#  Le classificateur d'images est maintenant entraîné à une précision d'environ 98 % sur cet ensemble de données. Pour en savoir plus, lisez les didacticiels TensorFlow .
# Si vous souhaitez que votre modèle renvoie une probabilité, vous pouvez envelopper le modèle entraîné et lui attacher le softmax :

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probamodel = probability_model(x_test[:5])
print(probamodel)
