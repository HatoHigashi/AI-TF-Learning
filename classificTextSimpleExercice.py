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

# Variables

    #Charger le dataset
batch_size = 32
seed = 42   
val_split = 0.4 #Pourcentage de fichiers utilisés dans la phase de validation
    #Vectorisation
max_features = 10000
sequence_length = 250

    #Modèle
epochs = 30
minValueAcceptable = 0.65

# DL data

url = "http://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

dataset = tf.keras.utils.get_file("aclStackOv\stack_overflow_16k", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), '')

# Explore data
    #Arborescence
print(os.listdir(dataset_dir))
train_dir = os.path.join(dataset_dir, 'train')
print(os.listdir(train_dir))

    #Fichiers
sample_file = os.path.join(train_dir, 'javascript/0.txt')
with open(sample_file) as f:
  print(f.read())

# ------------------DATASET---------------------

#Charger le dataset

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclStackOv/train', 
    batch_size=batch_size, 
    validation_split=val_split, 
    subset='training', 
    seed=seed) 

for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(3):
    print("Question : ", text_batch.numpy()[i]) #Le texte brut de la question
    print("-------------------------------------")
    print("Label : ", label_batch.numpy()[i]) #Donne l'info du type du comm (+ ou -)
    print("-------------------------------------\n")

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])
print("\n")

# Validation

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclStackOv/train', 
    batch_size=batch_size, 
    validation_split=val_split, #20% des data serviront à la validation
    subset='validation', 
    seed=seed) #Penser à spécifier une seed aléatoire, ou shuffle=False pour éviter une superposition dans les éléments de la validation et de l'entrainement !

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclStackOv/test', 
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

#-------------MODELE-------------

# Création du modèle 

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim), # Recoit les token int et cherche un vecteur d'encastrement pour chaque. ces vecteurs sont appris pendant l'entrainement du modèle, et ajoutent une dimension au tableau de sortie de la forme (batch, sequence, embedding). 
  layers.Dropout(0.1),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.4),
  layers.Dense(4)]) # il y a maintenant quatre classes de sortie

model.summary()

#Conf de la fonction de pertes : SparseCategoricalCrossentropy = fonction de perte dans le cas multiclasse avec étiquettes en int
model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer='adam',
              metrics=['accuracy']) #et pas BinaryAccuracy en multiclasse

#Entrainement du modèle
history = model.fit(   # model.fit() renvoie un objet 'History' qui contient un dictionnaire avec tout ce qui s'est passé pendant l'entraînement
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

#Evaluation du modèle
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)


#--------------STATISTIQUES-----------

history_dict = history.history 
history_dict.keys() # dict_keys(['loss', 'binary_accuracy', 'val_loss', 'val_binary_accuracy'])
                    #Il y a quatre entrées : une pour chaque métrique surveillée pendant la formation et la validation

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show() #Loss/epochs

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()#Accuracy/epochs
          #points = entrainement ligne = validation 
          #Notez que la perte d'entraînement diminue à chaque époque et que la précision de l'entraînement augmente à chaque époque. Ceci est attendu lors de l'utilisation d'une optimisation de descente de gradient - elle devrait minimiser la quantité souhaitée à chaque itération.
          #Ce n'est pas le cas pour la perte de validation et la précision, elles semblent culminer avant la précision de l'entraînement. Il s'agit d'un exemple de surapprentissage : le modèle fonctionne mieux sur les données d'apprentissage que sur des données qu'il n'a jamais vues auparavant. Après ce point, le modèle sur-optimise et apprend des représentations spécifiques aux données d'apprentissage qui ne se généralisent pas aux données de test.
          #Dans ce cas particulier, vous pouvez éviter le surapprentissage en arrêtant simplement l'entraînement lorsque la précision de validation n'augmente plus. Une façon de le faire est d'utiliser le rappel tf.keras.callbacks.EarlyStopping .

#Exporter le modèle

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print(accuracy)

#Inférence sur de nouvelles données
csharp = "Someone please assist,I configured Hangfire to run my jobs and when i click on Jobs and Servers the page loads fine and shows jobs and servers but when i click scheduled/Processing/Succeeded/Failed jobs it returns 'Page Isi'nt working,localhost is currently unable to handle this request HTTP 500' please see attached screenshot and configuration code below Startup.csusing Hangfire;using Hangfire.MySql;using Hangfire.SqlServer;public void ConfigureServices(IServiceCollection services){      services.AddHangfire(configuration => {                configuration.UseStorage(                    new MySqlStorage(                        mySqlConnectionStr,                        new MySqlStorageOptions                        {                            TransactionIsolationLevel = IsolationLevel.ReadCommitted,                            QueuePollInterval = TimeSpan.FromSeconds(15),                            JobExpirationCheckInterval = TimeSpan.FromHours(1),                            CountersAggregateInterval = TimeSpan.FromMinutes(5),                            PrepareSchemaIfNecessary = true,                            DashboardJobListLimit = 50000,                            TransactionTimeout = TimeSpan.FromMinutes(1),                            TablesPrefix = 'Hangfire'                        }                    )                );            });            // Add the processing server as IHostedService            services.AddHangfireServer();}public void Configure(IApplicationBuilder app, IWebHostEnvironment env){   app.UseHangfireServer(               new BackgroundJobServerOptions               {                   WorkerCount = 1               }); app.UseEndpoints(endpoints =>            {                endpoints.MapControllerRoute(                name: 'myconn',                pattern: '{controller=Home}/{action=Index}/{id?}');                endpoints.MapControllers();                endpoints.MapRazorPages();                endpoints.MapHangfireDashboard();            });}can someone please suggest a work around or solution for the pages to load?Thanks"
java = "I am doing non-blocking IO using selectors, and here's how I process requestswhile (keys.hasNext()) {        SelectionKey key = keys.next();        LOGGER.debug('Thread : {} iterating over key : {}', Thread.currentThread().getName(), key);        keys.remove();        if (!key.isValid()) {            //This message is warn because keys are normally supposed to be valid            LOGGER.warn('Invalid key : {}', key);            continue;        }        try {            if (key !=null && key.isAcceptable()) {                LOGGER.debug('acceptable key : {}', key);                this.accept(key);            }            if (key != null && key.isReadable()) {                LOGGER.debug('readable key : {}', key);                this.read(key, threadPoolTaskExecutor);            } else if (key == null) {                LOGGER.error('Got a null key!');            }        } catch (Exception ex) {            LOGGER.error('Got exception while reading key: {}', key);        }    }}My read method looks like thisprivate void read(SelectionKey key, ThreadPoolTaskExecutor threadPoolTaskExecutor) throws IOException {    Instant startOfRead = Instant.now();    SocketChannel channel = (SocketChannel) key.channel();    if (channel.isOpen()) {        List<ByteBuffer> messageList = new ArrayList<>();        List<Integer> lengthOfMessages = new ArrayList<>();        int numRead = 0;        int totalNumRead = 0;        boolean firstTime = true;        while (channel.isOpen() && (numRead > 0 || firstTime)) {            LOGGER.debug('channel.isOpen() -------------Message_Processor--->>>>numRead={}', numRead);            firstTime = false;            ByteBuffer buffer = ByteBuffer.allocate(4096);            try {                numRead = channel.read(buffer);                if (numRead > 0) {                    messageList.add(buffer);                    lengthOfMessages.add(numRead);                    totalNumRead += numRead;                } else if (numRead == -1) {               //What do I need to do here?                 }            } catch (IOException exception) {                LOGGER.info('Exception while reading from buffer, this usually happens when a device closes ' +                        'connection. Exception is:{}', exception.getMessage());                        //Usually this is connection reset by peer                        //What do I need to do here?                                    return;            }        }        /* Some more trivial processing commented out for brevity        */            LOGGER.debug('inputmessage : {} for socket:{} for client:{}', inputMessage, socket, client);            ProcessRead processRead = new ProcessRead(gpsHistoryKafkaTopic, gpsHistoryKafkaTopicNew, selector, deviceType, axesTrackParser, listenerPort,                    idealConnectionThreshold, kafkaHandlerExecutor,                    gpsKafkaTopic, gpsKafkaTopicNew, kafkaTemplateOld,                    kafkaTemplateNew, kafkaHandlerExecutorNew, this.parseStat, channel, inputMessage, key, rawData,                    threadPoolTaskExecutor,responseTopic);            threadPoolTaskExecutor.execute(processRead);        }I have the following questions:What do I need to do in case I read <=0 bytes?What do I need to do when I get the IOException? In this case, it is almost always 'connection reset by peer' In the above cases, I need to process the messages sent by the client, and, in some cases, send a message back to the client.In some cases, the client may not close the connection, and yet create a fresh connection (i.e. new socket) In this case, I'd want to close the connection. I'm okay closing the connections in a periodic cron job, but I need to know if I need to close the socketchannel with a socketchannel.close()I need to close the socket corresponding to the client withif(!socket.isClosed()) { socket.shutdownOutput(); socket.shutdownInput(); socket.close(); }Both, and if so, which one first?Neither, the resources will be released on thier own.Here's the code for sending data back to the channeltry {        if (channel.isOpen()) {            ByteBuffer ackBuffer = ByteBuffer.wrap(ack);            channel.write(ackBuffer);            ackBuffer.clear();        } else {            LOGGER.info('Warning: Connection is closed');        }    } catch (IOException ex) {        LOGGER.error('Exception occurred while sending the data back to client : {} with client details as : {}  ===  {}', ex.getMessage(), channel.getLocalAddress(), channel.getRemoteAddress());    }I am also confused as to the difference between a Socket and a SocketChannel. While I know that I can get the socket by using channel.socket(), and I have read other articles, it is not clear when either should be closed, and in what order.If there are serious flaws with the code, I'd be happy if they're pointed out!Thanks in advance!"
ruby = "I'm trying to add a new function to an old puppet module and added a function with the following command.pdk new function --type=v4 sumthis adds a file lib/puppet/functions/test/sum.rb# frozen_string_literal: true# https://github.com/puppetlabs/puppet-specifications/blob/master/language/func-api.md#the-4x-apiPuppet::Functions.create_function(:'test::sum') do  dispatch :sum do    param 'Numeric', :a    return_type 'Numeric'  end  # the function below is called by puppet and and must match  # the name of the puppet function above. You can set your  # required parameters below and puppet will enforce these  # so change x to suit your needs although only one parameter is required  # as defined in the dispatch method.  def sum(x)    x * 2  end  # you can define other helper methods in this code block as wellendI can also see the function in pdk console when I run the functions command.but when I try to call the function as followstest::sum(5) # Evaluation Error: Unknown function: 'test::sum'test::sum.call(5) # Evaluation Error: Unknown function: 'test::sum'Deferred(test::sum, [5]).call # Evaluation Error: Unknown function: 'test::sum'Why it is not working? Am I doing something wrong?Note: I'm quite new to the puppet."
python = "I'm trying to make a function that will compare multiple variables to an integer and output a string of three letters. I was wondering if there was a way to translate this into Python. So say:x = 0y = 1z = 3mylist = []if x or y or z == 0 :    mylist.append('c')if x or y or z == 1 :    mylist.append('d')if x or y or z == 2 :    mylist.append('e')if x or y or z == 3 :     mylist.append('f')which would return a list of:['c', 'd', 'f']Is something like this possible?"
js = "i need select value form my local storage . this can save value but can not read value form storage and run statei think save selector in local storage first and add event listener for read and return value or fix localStorage.getItem() code for get value or use console                      <option value='R'>currency</option>                      <option value='USD'> USD</option>                      <option value='R'> ریال</option>                      <option value='UTC'> الدینار</option>                      <option value='RUB'> RUB</option>                      <option value='CHI'> 角</option>                      <option value='HI'> INR</option>                      <option value='BIT'> bitcoin</option>                      <option value='TET'> tether</option>                      <input type='submit'   style='width:0%;'></button>                      <script id='jsbin-javascript'>                          var                              selector = document.getElementById('currencySelector');                          var                              currencyElements = document.getElementsByClassName('notranslate');                              var lastSelected = localStorage.getItem('curensy');                              if(lastSelected) {                                  selector.value = lastSelected;                              }                          var                              usdChangeRate = {                                  R: 1, // 1AUD = 1.0490 USD                                  UTC: 24000, // 1EUR = 1.4407 USD                                  CHI: 1.6424,                                  USD:22000,                                  RUB:1000,                                  HI: 1.4407, // 1EUR = 1.4407 USD                                  BIT: 55000000,                                  TET: 45000000                              };                          selector.onchange = function () {                              var                                  toCurrency = selector.value.toUpperCase();                              for (var i=0,l=currencyElements.length; i<l; ++i) {                                  var                                      el = currencyElements[i];                                  var                                      fromCurrency = el.getAttribute('data-currencyName').toUpperCase();                                  if (fromCurrency in usdChangeRate) {                                      var                                          // currency change to usd                                          fromCurrencyToUsdAmount = parseFloat(el.innerHTML) * usdChangeRate[fromCurrency];                                      console.log(parseInt(el.innerHTML,10) + fromCurrency + '=' + fromCurrencyToUsdAmount + 'USD');                                      var                                          // change to currency unit selected                                          toCurrenyAmount = fromCurrencyToUsdAmount / usdChangeRate[toCurrency];                                      el.innerHTML =   Math.round(toCurrenyAmount) + '<span>' + toCurrency.toUpperCase() + '</span>';                                      el.setAttribute('data-currencyName',toCurrency);                                  }window.localStorage.setItem('currencySelector', this.value);window.onload = function() {                                  if (localStorage.getItem('  el.innerHTML')) {                                      $('#currencySelector').val(localStorage.getItem('currencySelector')).trigger('change');                                return this.value }                              }                          }};                          </script>"
c = "#include <stdio.h>int main(){   printf('Hello');   fflush(stdout);   return 0;}#include <stdio.h>int main(){   printf('Hello');   return 0;}I'm trying to understand the use of fflush(stdout) and what is the difference between the 2 programs above?"

examples = [
  python,
  csharp,
  js,
  java,
  ruby,
  c
]

def maxLabel(results,raw_label):
    labels = []
    results = results.tolist()
    for elem in results:
        max = np.max(elem)
        if float(max)>=minValueAcceptable:
            labMax = elem.index(float(max))
            label = raw_label.class_names[labMax]
        else:
            label = "Inderterminé"
        labels.append(label)
    return labels

#Ici, on est sensé avoir les valeurs les plus elevées en 3 0 2 1 respectivement, les deux derniers ne correspondent à aucun ex du modèle
print(export_model.predict(examples),raw_train_ds)
print(maxLabel(export_model.predict(examples),raw_train_ds)) #Donne les prédictions sur les exemples
print("Attendu : [python, csharp, javascript, java, ???, ???]")