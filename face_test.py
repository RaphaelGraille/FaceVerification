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

import face_build as build
print("Import done")

#from lasagne.layers.normalization import BatchNormLayer
from random import shuffle
from PIL import Image

def test_general(img1, img2, samePers, threshold, get_network_output):
    batch = np.zeros((2,3,64,64),dtype='float32')
    batch[0] = img1
    batch[1] = img2
    [predictions] = get_network_output(batch)
    norm2 = np.linalg.norm(predictions[0] - predictions[1])
    #print("    Distance : {:.4f}".format(norm2))
    if norm2 <= threshold and samePers:
        return True
    elif norm2 >= threshold and not(samePers):
        return True
    else:
        return False

#test for same person
def samePersonTest(get_network_output, database, threshold):
    print("Start testing ! (Same person)")
    right_predictions = 0
    start = time.time()
    nb_tests = 0
    for persons in database:
        for i in range(persons.shape[0]-1):
            for j in range(i+1,persons.shape[0]):
                right_predictions += test_general(persons[i,:,:,:],persons[j,:,:,:], True, threshold, get_network_output)
                nb_tests +=1

    print("Test on same person ended in {:.3f}s.".format(time.time()-start))
    precisionS = float(right_predictions)/float(nb_tests)
    print("Precision of {:.3f}% on test set 'same'.".format(precisionS*100))
    
    return(precisionS)


#testing on different persons
def differentPersonTest(get_network_output, database, threshold):
    print("Start testing ! (Different person)")
    right_predictions = 0
    nb_tests = 0
    start = time.time()
    for i in range(len(database)-1):
        persi = database[i]    
        for k in range(persi.shape[0]):
            for j in range(i+1,len(database)):
                persj = database[j]
                for l in range(persj.shape[0]):
                    img1 = persi[k,:,:,:]
                    img2 = persj[l,:,:,:]
                    right_predictions += test_general(img1,img2, False,threshold , get_network_output)
                    nb_tests +=1
                   

    print("Test on different person ended in {:.3f}s.".format(time.time()-start))
    precisionD = float(right_predictions)/float(nb_tests)
    print("Precision of {:.3f}% on test set 'different'.".format(precisionD*100))

    return(precisionD)



#for re-building a stored model and testing it
def launchTests(filename , dbs, dbd, threshold = 1.185):
    
    get_network_output = build.build(filename)

    precisionS = samePersonTest(get_network_output, dbs, threshold)
    precisionD = differentPersonTest(get_network_output, dbd, threshold)

    return(precisionS, precisionD, np.sqrt(precisionS*precisionD))
    
#for testing the current model
def launchTestsBuilt(get_network_output, dbs,dbd, threshold = 1.2):
    
    precisionS = samePersonTest(get_network_output, dbs, threshold)
    precisionD = differentPersonTest(get_network_output, dbd, threshold)

    return(precisionS, precisionD, np.sqrt(precisionS*precisionD))


print("loading test database...")
dbs = build.load_database(data_dir = 'aligned_64_testSame')
dbd = build.load_database(data_dir = 'aligned_64_testDifferent')

#exemple:
#launchTests('model800.npz',dbs,dbd, 1.2)
