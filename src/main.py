'''
Created on Mar 10, 2018

@author: estuche & jocelyn
'''

from sklearn.datasets import load_iris
import pickle
import numpy as np
import os
import random

#Batch files available: CIFAR | IRIS
batch_file_name = 'IRIS'

normalization = 0
generation_size = 10
generation_count = 1
pressure = 3
largoIndividuo = 150
mutation_chance = 0.3

def hinge_loss(W, X, y):
    #Based on https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    num_train = X.shape[0]
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]),y] 
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    margins[np.arange(num_train),y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    
    return loss

def load_cifar_batch(fileName):
    with open(fileName, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'], np.array(datadict['labels'])

def load_cifar(folder):     
    training_data = np.empty([50000, 3072], dtype=np.uint8)
    training_labels = np.empty([50000], dtype=np.uint8)
    
    for i in range(1, 6):
        start = (i - 1) * 10000
        end = i * 10000
        training_data[start:end], training_labels[start:end] = \
            load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    
    '''
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
     We are only interested in 'airplane', 'cat', 'frog', 'horse' classes
    '''
    label_index = np.append(np.where(training_labels == 0)[0], \
                            np.where(training_labels == 3)[0])
    
    label_index = np.append(label_index, \
                            np.where(training_labels == 6)[0])
    
    label_index = np.append(label_index, \
                            np.where(training_labels == 7)[0])
    
    training_data = training_data[label_index]
    training_labels = training_labels[label_index]
    
    training_data_grayscale = training_data.reshape((20000, 3, 1024)).transpose((0, 2, 1))
    training_data_grayscale = np.mean(training_data_grayscale, axis=2)
        
    return training_data_grayscale, training_labels

def load_data(dataset):
    global normalization
    if (dataset == 'CIFAR'):
        normalization = 255
        data = load_cifar("cifar-10-batches-py")
        return data[0], data[1]
    elif (dataset == 'IRIS'):
        normalization = 10
        data = load_iris()
        return data.data, data.target
    

#W poblacion
def selection(data,labels,population):
    
    puntuados = [(hinge_loss(i,data,labels),i) for i in population] #Calcula el fitness de cada individuo, y lo guarda en pares ordenados de la forma (5 , [1,2,1,1,4,1,8,9,4,1])
    puntuados = [i[1] for i in sorted(puntuados)] #Ordena los pares ordenados y se queda solo con el array de valores
    population = puntuados
    selected =  puntuados[(len(puntuados)-pressure):] #Esta linea selecciona los 'n' individuos del final, donde n viene dado por 'pressure'

    return selected

def cross(selected,population):
    for i in range(len(population)-pressure):
        punto = random.randint(1,largoIndividuo-1) #Se elige un punto para hacer el intercambio
        parents = random.sample(selected, 2) #Se eligen dos padres
        #print("padre1",parents[0])
        population[i][:punto] = parents[0][:punto] #Se mezcla el material genetico de los padres en cada nuevo individuo
        population[i][punto:] = parents[1][punto:]
  
    return population #El array 'population' tiene ahora una nueva poblacion 
def mutation(population):
    """
        Se mutan los individuos al azar. Sin la mutacion de nuevos genes nunca podria
        alcanzarse la solucion.
    """
    print(len(population))
    for i in range(len(population)-pressure):
        if random.random() <= mutation_chance: #Cada individuo de la poblacion (menos los padres) tienen una probabilidad de mutar
            punto = random.randint(0,largoIndividuo-1) #Se elgie un punto al azar
            nuevo_valor = random.randint(1,255) #y un nuevo valor para este punto
            #print(population[i])
            
  
            #Es importante mirar que el nuevo valor no sea igual al viejo
            while nuevo_valor == population[i][punto]:
                nuevo_valor = random.randint(1,9)
  
            #Se aplica la mutacion
            population[i][punto] = nuevo_valor
  
    return population


def main():
    
    data, labels = load_data(batch_file_name)
    
    data_size = data.shape[1]
    class_count = labels.shape[0]
    
    print(labels)
        
    for i in range(0, generation_count):
        generation = np.random.rand(generation_size, data_size, class_count)
        generation *= normalization
        #print(generation)
  
    #print("")
    #print(len(generation))
    #print(len(generation[0]))
    #print(len(generation[0][0]))
    #print(data_size)
    
    selec = selection(data,labels,generation)
    nuevapop=cross(selec,generation)
    #print(mutation(nuevapop))
    
    
    
main()
    
