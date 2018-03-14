'''
Created on Mar 10, 2018

@author: estuche & jocelyn
'''

import pickle
import numpy as np
import os
from sklearn.datasets import load_iris

class Classificator:
    
    def __init__(self, W):
        self.W = W
    
    def hingeLoss(self):
        #Esto es un comentario bien loquillo

def load_cifar_batch(fileName):
    with open(fileName, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'].astype(np.float64), np.array(datadict['labels'])

def load_cifar(folder):
    with open(os.path.join(folder, 'batches.meta'), 'rb') as f:
        names = pickle.load(f, encoding='latin1')
    training_data = np.empty([50000, 3072], dtype=np.float64)
    training_labels = np.empty([50000], dtype=np.uint8)
    for i in range(1, 6):
        start = (i - 1) * 10000
        end = i * 10000
        training_data[start:end], training_labels[start:end] = \
            load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    testing_data, testing_labels = load_cifar_batch(os.path.join(folder, 'test_batch'))
    training_data_grayscale = training_data.reshape((50000, 3, 1024)).transpose((0, 2, 1))
    training_data_grayscale = np.mean(training_data_grayscale, axis=2)
    testing_data_grayscale = testing_data.reshape((10000, 3, 1024)).transpose((0, 2, 1))
    testing_data_grayscale = np.mean(testing_data_grayscale, axis=2)
    
    
    
    return training_data_grayscale, training_labels, testing_data_grayscale, testing_labels, names['label_names']

def main():
    
    data = load_cifar("cifar-10-batches-py")
	print(data)
    
    classificator = Classificator(data)
    #Conflicto en la misma linea
def datosIris():
	iris = load_iris()

	# Arreglos de numpy
	train_data = iris.data
	train_labels = iris.target
    
##lalalallala
main()
    
    