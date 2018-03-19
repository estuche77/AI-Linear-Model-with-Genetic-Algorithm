'''
Created on Mar 10, 2018

@author: estuche & jocelyn
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import random

#Batch files available: CIFAR | IRIS
batch_name = 'IRIS'
normalization = 0

#Genetic algorithm parameters
generation_size = 20
generation_count = 50
pressure = 3
largoIndividuo = 5
mutation_chance = 0.3

def hinge_loss_and_accuracy(W, X, y):
    #Based on https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    num_train = X.shape[0]
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]),y] 
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    
    margins[np.arange(num_train),y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    
    predicted_classes = np.argmax(margins, axis = 1).T - y
    
    correct_count = np.count_nonzero(predicted_classes == 0)
    accuracy = correct_count / num_train
    
    return loss, accuracy

def load_cifar_batch(fileName):
    with open(fileName, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'], np.array(datadict['labels'])

#Load CIFAR-10 train data, chooses 4 classes and convert images to gray scale 
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

#Load data generalization for CIFAR-10 and IRIS
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

def evaluation(data,labels,population):
    
    #Calcula el fitness de cada individuo, y lo guarda en pares ordenados de la forma (loss , W)
    evaluaton = [hinge_loss_and_accuracy(i,data,labels) + (i,) for i in population]
    return evaluaton
    
#data = all X values
#labels = all Y values
#population = a W list
def selection(population):
    
    #Calcula el fitness de cada individuo, y lo guarda en pares ordenados de la forma (loss , W)
    #puntuados = [(hinge_loss_and_accuracy(i,data,labels),i) for i in population]
    
    #Ordena los pares ordenados y se queda solo con el W
    puntuados = [i[2] for i in sorted(population, key=lambda tup: tup[0])]
    population = puntuados
       
    #Esta linea selecciona los 'n' individuos del final, donde n viene dado por 'pressure'
    selected =  puntuados[:pressure]

    return selected

def cross(selected,population):
    for i in range(len(population)-pressure):
        
       
        punto = random.randint(1,largoIndividuo-1)
        
        #Se eligen dos padres
        parents = random.sample(selected, 2)
       
        population[i][:punto] = parents[0][:punto]
        population[i][punto:] = parents[1][punto:]
        
    #El array 'population' tiene ahora una nueva poblacion
    return population

def mutation(population,class_count):
    
    for i in range(len(population)-pressure):
        
        #Cada individuo de la poblacion (menos los padres) tienen una probabilidad de mutar
        if random.random() <= mutation_chance:
            
            #Se elgie un punto al azar
            punto = random.randint(0,largoIndividuo-1)
            
            #y un nuevo valor para este punto
            nuevo_valor = np.random.rand(class_count)
                        
            #Es importante mirar que el nuevo valor no sea igual al viejo           
            while np.array_equal(nuevo_valor, population [i][punto])==True:
                np.random.rand(class_count)
  
            #Se aplica la mutacion
            population[i][punto] = nuevo_valor
  
    return population


def main():
    
    #The data is loaded
    data, labels = load_data(batch_name)
    
    #The plot variables are created
    generation_iteration = []
    best_loss = []
    best_accuracy = []
    
    #The class count and the data dimension is obtained
    data_dimension = data.shape[1]
    class_count = np.unique(labels).shape[0]
    
    #This is done because the bias trick
    data = np.insert(data, data.shape[1], 1, axis = 1)
    
    #The first generation is created
    generation = np.random.rand(generation_size, data_dimension + 1, class_count)
    generation *= normalization
    
    for i in range(generation_count):
        #The iteration is listed
        generation_iteration.append(i)
        
        #The generation is evaluated
        evaluated = evaluation(data, labels, generation)
        
        '''
        Las siguientes dos instrucciones hay que corregirlas 
        para buscar el mejor de la generacion
        '''
        #Here the best evaluated individual should be inserted
        best_loss.append(evaluated[0][0])
        
        best_accuracy.append(evaluated[0][1])
        
        #Now we select individuals for the following breed
        selected = selection(evaluated)
        
        #The chosen individuals are crossed
        crossed = cross(selected, generation)
        
        #The crossed individuals are mutated
        mutated = mutation(crossed,class_count)
        
        #The result is now a new generation for the following iteration
        generation = mutated
    
    #The plot shows the population behavior        
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(generation_iteration, best_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    
    color = 'tab:red'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(generation_iteration, best_accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.show()
    
main()
    
