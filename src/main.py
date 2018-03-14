'''
Created on Mar 13, 2018

@author: estuche
'''

import numpy as np

def unpickle(fileName):
    import pickle
    with open(fileName, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary