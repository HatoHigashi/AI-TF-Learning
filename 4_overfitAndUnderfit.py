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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import regularizers

