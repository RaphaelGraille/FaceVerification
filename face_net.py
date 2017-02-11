from __future__ import print_function

import cPickle as pickle
import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from theano import shared
import lasagne
print('importing face_test')
import face_test as test

from random import shuffle

from PIL import Image

# Cette fonction renvoie la base de donnée située dans le dossier data_dir sous forme
# d'une liste de tableau. La liste a autant d'élément que de personnes dans la base.
# Chaque élément de la liste est un tableau de taille :
# [nombre_img_pour_personne, 3, 64, 64]
def load_database(data_dir = 'aligned_64_train'):

    # Pour chaque personne dans le repertoire d'apprentissage
    persons = os.listdir(data_dir)[0:]

    # Cette ligne supprime des fichiers caches genere par les system mac osX
    if persons.count('.DS_Store') > 0:
        persons.remove('.DS_Store')

    nb_pers = len(persons)
    print(nb_pers)

    # Initialisation de la liste DataBase
    DataBase = np.zeros(nb_pers, dtype='float32').tolist()

    i = 0
    # Pour chaque personne
    for person in persons:
        # On recupere le path
        fullPersonPath = os.path.join(data_dir, person)
        # On recupere les images
        pictures = os.listdir(fullPersonPath)
        if pictures.count('.DS_Store') > 0:
            pictures.remove('.DS_Store')
        print(person)
        nb_pics = len(pictures)
        print(str(i))

        # On cree le tableau
        person_pics = np.zeros((nb_pics,3,64,64),dtype='float32')
        j = 0
        # Pour chaque image de la personne
        for pic in pictures:
            # On en recupere le path
            fullPicPath = os.path.join(fullPersonPath,pic)

            # On recupere l'image
            img = Image.open(fullPicPath)
            # On la convertie en tableau
            img = np.asarray(img, dtype='float32') / np.float32(256.)
            # On rearrange les canaux
            person_pics[j, 0,: , :] = img[:, :, 0]
            person_pics[j, 1, :, :] = img[:, :, 1]
            person_pics[j, 2, :, :] = img[:, :, 2]
            j = j + 1
        # On set la ieme valeur du tableau
        DataBase[i] = person_pics
        i = i + 1

    return DataBase

# Cette fonction renvoie un sous-ensemble aléatoire de database contenant nb_pers_sel personnes
def select_random_person(nb_pers_sel, DataBase):

    nb_pers_total = len(DataBase)

    # on cree un tableau d'indices ordonnes
    indices = range(nb_pers_total)
    # On les melange
    shuffle(indices)
    # On choisi les nb_pers_sel premiers
    selected_indices = indices[0:nb_pers_sel]

    # renvoie la list des personnes selectionnees
    return list(DataBase[i] for i in selected_indices)

# Renvoie DataBase avec nb_pic par personne
def select_random_pic_for_each_pers(nb_pic, DataBase):
    nb_pers = len(DataBase)

    # On initizlise la structure batch
    batch = np.zeros((nb_pers*nb_pic,3,64,64),dtype='float32')

    i = 0
    # Pour chaque personne dans la Database
    for person in DataBase:
        nb_pic_total_pers = len(person)
        # on cree un tableau d'indices ordonnes
        indices = range(nb_pic_total_pers)
        # On les melanges
        shuffle(indices)
        selected_indices = indices[0:nb_pic]
        # On set la valeur du batch
        batch[i*nb_pic:(i+1)*nb_pic,:,:,:] = person[selected_indices]
        i = i + 1
    return batch

# Selectionne les triplets dans batch_features selon la méthode Semi Hard Negative
def choose_triplet(batch_features,nb_pic_per_pers,nb_pers):

    size_batch = len(batch_features)

    # On initialise la structure triplets comme un tableau [nb_triplet,3]
    triplets = np.zeros((nb_pic_per_pers*(nb_pic_per_pers-1)*nb_pers,3),dtype='int32')

    # On initialise une matrice de distance entre les images
    distances = np.zeros((size_batch,size_batch),dtype='float32')

    i = 0
    current_triplet_index = 0
    # Pour chaque feature d'image
    for features1 in batch_features:
        j = 0
        # Pour chaque feture d'image
        for features2 in batch_features:
            # Si les images sont de la même personne
            if i/nb_pic_per_pers == j/nb_pic_per_pers:
                # Si les deux images ne sont pas identiques
                if i != j:
                    # On ajoute un nouveau triplet
                    triplets[current_triplet_index,0] = i
                    triplets[current_triplet_index,1] = j
                    # On calcul la distance entre les deux representation
                    dist = np.linalg.norm(features1 - features2)
                    # On met a jour l'indice du triplet courant
                    current_triplet_index = current_triplet_index + 1
                    # On met a jour la matrice de distances
                    distances[i,j] = dist
                j = j + 1
            else:
                # On calcul la distance entre les deux representation
                dist = np.linalg.norm(features1 - features2)
                # On met a jour la matrice de distances (qui est symétrique)
                distances[i,j] = dist
                distances[j,i] = dist
                j = j + 1
        i = i + 1

    # On parcours toutes le pairs créee (On souhaite ajouté l'image negative)
    for t in range(len(triplets)):

        anchor = triplets[t,0]
        positive = triplets[t,1]
        # Distance entre les deux représentation des images de la même personne
        dist_anch_pos = distances[anchor,positive]
        best_semi_hard_dist = 10000
        best_semi_hard_pic = 0
        # On parcours toutes les image
        for neg in range(size_batch):
            # On ne considere que les images qui ne sont pas de la meme personne que anchor
            #(et positive)
            if not neg/nb_pic_per_pers == anchor/nb_pic_per_pers:
                # On calcul la distance entre la representation de anchor et negative
                dist_anch_neg = distances[anchor, neg]
                # Condition pour la semi hard negative
                if dist_anch_neg > dist_anch_pos:
                    # Si la distance entre anchor et negative est plus petite que la
                    # meilleurs distance
                    if dist_anch_neg < best_semi_hard_dist:
                        best_semi_hard_pic = neg
                        best_semi_hard_dist = dist_anch_neg

        # On set la valeur de l'image negative
        triplets[t,2] = best_semi_hard_pic

    return triplets

# Calcul la triplet loss sur les triplets
def triplet_loss(predictions, triplets):

    # Valeur de alpha
    a = np.float32(0.2)

    # On calcul les distances entre les representations de anchor/positive et anchor/negative
    dist1 = ((predictions[triplets[:,0]] - predictions[triplets[:,1]])**2).sum(axis=1)
    dist2 = ((predictions[triplets[:, 0]] - predictions[triplets[:, 2]]) ** 2).sum(axis=1)
    s = dist1 - dist2 + a
    # On calcul la loss
    loss = s * T.gt(s, 0.0)

    return loss

# Construit le CNN avec Lasagne (inspiré de l'exemple mnist.py de lasagne)
def build_cnn(input_var=None, filename = 'model.npz'):

    # Couche d'entree : les entrees sont de la forme [nb_exemple,3,64,64]
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)

    # Couche de convolutions avec 32 noyeaux de 5x5
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal())

    # Max-pooling avec un facteur 2 dans chaque dimension
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Normalisation des Maps
    network = lasagne.layers.BatchNormLayer(network)

    # Couche de convolutions avec 64 noyeaux de 3x3
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.BatchNormLayer(network)

    # Convolutions avec 128 noyeaux de 3x3
    network = lasagne.layers.Conv2DLayer(
         network, num_filters=128, filter_size=(3, 3),
         nonlinearity=lasagne.nonlinearities.rectify,
         W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    

    #  Couche fully-connected avec 128 neurones:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=128,
            W=lasagne.init.GlorotNormal(),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    #  Couche de sorti fully-connected avec 128 neurones:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=128,
            W=lasagne.init.GlorotNormal(),
            nonlinearity=None)

    return network

# Fonction principale du program
# num_epochs : nombre d'itérations
# nb_pic_per_pers : nombre d'image par personne dans chaque batch
# nb_pers_per_batch : nombre de personnes par batch
# filenameIn : fichier d'entrée si l'on souhaite chargé un model sauvegardé
def main(num_epochs = 10000, nb_pic_per_pers = 20, nb_pers_per_batch = 20, filenameIn = 'model1400.npz'):

    print("Loading training database")
    Database_train = load_database('aligned_64_train')

    input_var = T.tensor4('inputs')

    print("building the model ...")
    network = build_cnn(input_var, filenameIn)

    # On definit ici des expression symbolics (qui n'ont pas de valeur relle) et qui vont être compilé
    # pour donner des fonctions
    # Les lignes suivantes (jusqu'a la boucle d'itération) ne calcul donc rien. Elles definissent des fonctions
    # qui vont etre compilé par Theano de maniere optimise.

    # On recupere la sorti du reseau pour un batch passe en entree
    prediction = lasagne.layers.get_output(network)

    # On normalise les prediction
    pred_norm, updates0 = theano.scan(lambda x_i: x_i / T.sqrt((x_i ** 2).sum()), sequences=[prediction])
    # On compile cette fonction. Theano va donc creer une fonction prenant
    # en argument la variable d'entree (le batch) et renvoyer la prediction normalise.
    # Il s'agit des representations des images
    get_network_output = theano.function([input_var], [pred_norm])

    # On initialise la structure des triplets
    triplets = np.zeros((nb_pic_per_pers*(nb_pic_per_pers-1)*nb_pers_per_batch,3),dtype='int32')

    # On initialise la structure des triplets comme une variable Theano de type "shared"
    # Il s'agit de variable Theano qui ont a la fois une valeur symbolic ET une valeur numerique
    # (alors que les autres variables Theano n'ont qu'une valeur symbolique)
    # Cela permet de pouvoir utiliser triplet_shared dans une fonction compilé Theano tout en lui
    # conférent des valeurs numeriques.
    triplets_shared = shared(triplets)

    # On calcule la triplet loss
    loss = triplet_loss(pred_norm, triplets_shared)
    # On en calcul la moyenne sur tous les exemples du batch
    loss = loss.mean()

    # On recupere les parametres du model
    params = lasagne.layers.get_all_params(network, trainable=True)
    # On les met à jour avec une descente de gradient
    updates = lasagne.updates.adagrad(loss, params, learning_rate=0.06)

    # On compile la fontion theano qui va calculer la loss et mettre a jour les parametres
    # a chaque iteration.
    train_fn = theano.function([input_var], [loss], updates=updates)

    print("Start training !")

    # Les lignes suivantes utilisent les fonction compilées ci-dessus
    # a chaque itération
    for epoch in range(num_epochs):

        start_time = time.time()

        # On cree le batch
        selected_person = select_random_person(nb_pers_per_batch, Database_train)
        batch = select_random_pic_for_each_pers(nb_pic_per_pers, selected_person)

        # On calcule les predictions sur le batch en utilisnt la fonction definit au dessus
        [predictions] = get_network_output(batch)

        # On selectionne les triplets en fonction de la représentation des images du batch
        triplets = choose_triplet(predictions,nb_pic_per_pers,nb_pers_per_batch)

        # On affecte ces valeurs a la variable
        triplets_shared.set_value(triplets)

        # On met a jour les parametres du reseau (entrainement) et on recupere la loss
        [train_err] = train_fn(batch)

        # On l'affiche avec le temps de calcul
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err /1))
        
        # Toutes les 50 itérations on sauvegarde le model dans un fichier
        if epoch % 50 == 0:
            precision = test.launchTestsBuilt(get_network_output, test.Database_test, threshold = 1.21)
            np.savez('v3model'+str(epoch)+'_score'+str(precision)+'.npz', *lasagne.layers.get_all_param_values(network))

    print("Training done !")

main()
