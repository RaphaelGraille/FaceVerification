from __future__ import print_function

import cPickle as pickle
import sys
import os
import time
import random 

import numpy as np
import theano
import theano.tensor as T
from theano import shared

import lasagne

#from lasagne.layers.normalization import BatchNormLayer

from random import shuffle

from PIL import Image

def load_database(data_dir):

    persons = os.listdir(data_dir)[:]
    if persons.count('.DS_Store') > 0:
        persons.remove('.DS_Store')
    nb_pers = len(persons)
    print(nb_pers)
    DataBase = np.zeros(nb_pers, dtype='float32').tolist()

    #DataBase[0] = np.zeros((100,3,96,96))
    #DataBase[1] = np.zeros((50, 3, 96, 96))
    #print(DataBase[1][0,0,:,:])
    #print(len(DataBase[1]))

    # Nombre min de photo pour une personne = 23

    i = 0
    for person in persons:
        fullPersonPath = os.path.join(data_dir, person)
        pictures = os.listdir(fullPersonPath)
        if pictures.count('.DS_Store') > 0:
            pictures.remove('.DS_Store')
        print(person)
        nb_pics = len(pictures)
        print(str(i))

        person_pics = np.zeros((nb_pics,3,64,64),dtype='float32')
        j = 0
        for pic in pictures:
            fullPicPath = os.path.join(fullPersonPath,pic)
            img = Image.open(fullPicPath)
            img = np.asarray(img, dtype='float32') / np.float32(256.)
            person_pics[j, 0,: , :] = img[:, :, 0]
            person_pics[j, 1, :, :] = img[:, :, 1]
            person_pics[j, 2, :, :] = img[:, :, 2]
            j = j + 1
        DataBase[i] = person_pics
        i = i + 1

    return DataBase

def select_random_pic_for_each_pers(nb_pic, DataBase):
    nb_pers = len(DataBase)
    #print(DataBase[0])
    batch = np.zeros((nb_pers*nb_pic,3,64,64),dtype='float32')

    i = 0
    for person in DataBase:
        nb_pic_total_pers = len(person)
        indices = range(nb_pic_total_pers)
        shuffle(indices)
        selected_indices = indices[0:nb_pic]
        batch[i*nb_pic:(i+1)*nb_pic,:,:,:] = person[selected_indices]
        i = i + 1
        #print(selected_indices)
    return batch

def select_random_person(nb_pers_sel, DataBase):

    nb_pers_total = len(DataBase)
    #nb_pers_total = 45

    indices = range(nb_pers_total)
    
    selected_indices = indices[0:nb_pers_sel]
    #print(selected_indices)

    return list(DataBase[i] for i in selected_indices)


def build_cnn(input_var=None, filename = 'model.npz'):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 64, 64),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.BatchNormLayer(network)

    # Another convolution with 64 3x3 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.BatchNormLayer(network)

    #A third convolution with 128 3x3 kernels, and a last 2x2 pooling
    network = lasagne.layers.Conv2DLayer(
         network, num_filters=128, filter_size=(3, 3),
         nonlinearity=lasagne.nonlinearities.rectify,
         W=lasagne.init.GlorotNormal())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    

    # A fully-connected layer of 128 units:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=128,
            W=lasagne.init.GlorotNormal(),
            nonlinearity=lasagne.nonlinearities.leaky_rectify)

    # And, finally, a last fully connected layer:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=128,
            W=lasagne.init.GlorotNormal(),
            nonlinearity=None)

    #with np.load(filename) as f:
    #    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    #    lasagne.layers.set_all_param_values(network, param_values)

    return network



def build(filenameIn = 'model1500.npz'):

    input_var = T.tensor4('inputs')

    print("building the model ...")
    network = build_cnn(input_var, filenameIn)

    prediction = lasagne.layers.get_output(network)
    pred_norm, updates0 = theano.scan(lambda x_i: x_i / T.sqrt((x_i ** 2).sum()), sequences=[prediction])
    get_network_output = theano.function([input_var], [pred_norm])

    return(get_network_output)



