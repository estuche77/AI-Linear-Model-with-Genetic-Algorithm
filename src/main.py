'''
Created on Mar 10, 2018

@author: jason & jocelyn
'''

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import random

#Batch files available: CIFAR | IRIS
batch_name = 'CIFAR'
normalization = 0

#Genetic algorithm parameters
generation_size = 20
generation_count = 50
pressure = 3
mutation_rate = 0.1
mutation_chance = 0.3

#This loads a single CIFAR file
def load_cifar_batch(fileName):
    with open(fileName, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
    return datadict['data'], np.array(datadict['labels'])

#Load CIFAR-10 train data, chooses 4 classes and convert images to gray scale 
def load_cifar(folder):     
    training_data = np.empty([50000, 3072], dtype=np.float64)
    training_labels = np.empty([50000], dtype=np.uint8)
    
    for i in range(1, 6):
        start = (i - 1) * 10000
        end = i * 10000
        training_data[start:end], training_labels[start:end] = \
            load_cifar_batch(os.path.join(folder, 'data_batch_%d' % i))
    
    '''
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
     We are taking only the first 4 classes
    '''
    
    label_index = np.append(np.where(training_labels == 0)[0], \
                            np.where(training_labels == 1)[0])
    
    label_index = np.append(label_index, \
                            np.where(training_labels == 2)[0])
    
    label_index = np.append(label_index, \
                            np.where(training_labels == 3)[0])
    
    training_data = training_data[label_index]
    training_labels = training_labels[label_index]
    
    training_data_grayscale = training_data.reshape((20000, 3, 1024)).transpose((0, 2, 1))
    training_data_grayscale = np.mean(training_data_grayscale, axis=2)
        
    return training_data_grayscale, training_labels

#Load data from CIFAR-10 or IRIS datasets
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

def hinge_loss_and_accuracy(W, X, y):
    #Based on https://mlxai.github.io/2017/01/06/vectorized-implementation-of-svm-loss-and-gradient-update.html
    num_train = X.shape[0]
    scores = X.dot(W)
    yi_scores = scores[np.arange(scores.shape[0]),y] 
    margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)
    
    #Here the accuracy is calculated based on hinge loss margins
    predicted_classes = np.argmax(scores, axis = 1).T - y
    correct_count = np.count_nonzero(predicted_classes == 0)
    accuracy = correct_count / num_train
    
    #The loss is calculated after because is necessary to mute the correct class
    #by changing the value by 0 and then mean the vector for each data
    
    #The correct classes are ignored
    margins[np.arange(num_train),y] = 0
    loss = np.mean(np.sum(margins, axis=1))
    
    return loss, accuracy

def evaluation(data,labels,population):
    #Calculate fitness value for every individual of the
    #population and returns as a list of (loss, accuracy, W)
    evaluaton = [hinge_loss_and_accuracy(i,data,labels) + (i,) for i in population]
    return evaluaton
    
#data = all X values
#labels = all Y values
#population = a W list
def selection(population):
        
    #Ordena los pares ordenados y se queda solo con el W
    puntuados = [i[2] for i in sorted(population, key=lambda tup: tup[0])]
    #population = puntuados
       
    #Esta linea selecciona los 'n' individuos del final, donde n viene dado por 'pressure'
    selected =  puntuados[:pressure]

    return selected

def cross(selected,population):
    for i in range(len(population)-pressure):
        
        punto = random.randint(1,population.shape[1]-1)
        
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
            punto = random.randint(0,population.shape[1]-1)
            
            #y un nuevo valor para este punto
            nuevo_valor = np.random.rand(class_count)
                        
            #Es importante mirar que el nuevo valor no sea igual al viejo           
            while np.array_equal(nuevo_valor, population [i][punto])==True:
                np.random.rand(class_count)
  
            #Se aplica la mutacion
            population[i][punto] = nuevo_valor
  
    return population

#This function does a selection, crossover and mutation from a generation's population
def creative_ga(population):
    
    #Here the population are sorted
    sorted_population = np.array([i[2] for i in sorted(population, key=lambda tup: tup[0])])
    sorted_loss = np.array([i[0] for i in sorted(population, key=lambda tup: tup[0])])
    
    #This data is obtained
    data_dimension = sorted_population.shape[1]
    class_count = sorted_population.shape[2] 
    
    new_population = []
    
    #This represent the elitism
    new_population.append(sorted_population[0])
    
    for i in range(1, sorted_population.shape[0]):
        
        #An alpha coefficient is calculated
        totalLoss = sorted_loss[i] + sorted_loss[i-1]
        alpha = sorted_loss[i] / totalLoss
        new_individual = sorted_population[i] * alpha + sorted_population[i-1] * (1 - alpha)
        
        #Random indexes and values are generated
        random_index = np.random.randint(data_dimension, size=int(round(mutation_rate*data_dimension)))
        random_values = np.random.rand(int(round(mutation_rate*data_dimension))) * normalization
        
        #This code flatten the individual in order to chance randomly its values
        new_individual = new_individual.reshape(new_individual.shape[0] * new_individual.shape[1])
        new_individual[random_index] = random_values
        new_individual = new_individual.reshape(data_dimension, class_count)
        
        #The new individual is added
        new_population.append(new_individual)
    
    return new_population

#This function will save a log to a file
def log(file, string):
    with open(file, "a") as myfile:
        myfile.write(string + "\n")
    
    print(string)

#Visualize an W with a title, loss and ith generation count 
def visualize_image(W,loss,title,i):
    #Based on: https://www.quora.com/How-can-l-visualize-cifar-10-data-RGB-using-python-matplotlib
    element = W[:,i]
    img = element.reshape(32,32)
    plt.imshow(img, cmap='gray')
    plt.title("W " + str(i) + "th with loss of " + str(loss))
    
    #Uncomment this to show the image
    #plt.show()
    
    directory = os.path.abspath("output/" + title)
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory+"/img"+str(i))

#Last simulation created
def simulation_2():
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
    
    #The first generation is created (data_dimension + 1 because the bias trick)
    generation = np.random.rand(generation_size, data_dimension + 1, class_count)
    generation *= normalization
    
    for i in range(1, generation_count + 1):
        
        print("Generation: " + str(i))
        
        #The iteration is listed
        generation_iteration.append(i)
        
        #The generation is evaluated
        evaluated = evaluation(data, labels, generation)
        
        #Here the best evaluated individual should be inserted
        #No reverse since we want the lowest loss
        sorted(evaluated, key=lambda tup: tup[0])
        best_loss.append(evaluated[0][0])
        best_accuracy.append(evaluated[0][1])
        
        #A new generation is created
        generation = creative_ga(evaluated)
        
    #The plot shows the population behavior        
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Generations')
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
    
    if batch_name == 'CIFAR':
        #This line of code visualizes every photo that contains W
        #The bias trick should be removed from the W in order to reshape it to 32x32
        [visualize_image(evaluated[0][2][:-1,:], evaluated[0][0], "Generation" + str(i), j) for j in range(class_count)]
    
    log("Simulation2.txt", "End of execution")
    log("Simulation2.txt", "Last accuracy: " + str(evaluated[0][1]))
    log("Simulation2.txt", "Last loss: " + str(evaluated[0][0]))
    log("Simulation2.txt", "Generation count: " + str(generation_count))
    log("Simulation2.txt", "Generation size: " + str(generation_size))
    log("Simulation2.txt", "Mutation rate: " + str(mutation_rate))

#First simulation created
def simulation():
    
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
    
    #The first generation is created (data_dimension + 1 because the bias trick)
    generation = np.random.rand(generation_size, data_dimension + 1, class_count)
    generation *= normalization
    
    for i in range(1, generation_count + 1):
        
        print("Generation: " + str(i))
        
        #The iteration is listed
        generation_iteration.append(i)
        
        #The generation is evaluated
        evaluated = evaluation(data, labels, generation)
        
        #Here the best evaluated individual should be inserted
        #No reverse since we want the lowest loss
        sorted(evaluated, key=lambda tup: tup[0])
        best_loss.append(evaluated[0][0])
        best_accuracy.append(evaluated[0][1])
        
        #This line of code visualizes every photo that contains W
        #The bias trick should be removed from the W in order to reshape it to 32x32
        #[visualize_image(evaluated[0][2][:-1,:], evaluated[0][0], "Generation" + str(i), j) for j in range(class_count)]
        
        #Now we select individuals to use in the following breed
        selected = selection(evaluated)
        
        #The chosen individuals are crossed
        crossed = cross(selected, generation)
        
        #The crossed individuals are mutated
        mutated = mutation(crossed,class_count)
        
        #The result is now the new generation for the following iteration
        generation = mutated
    
    #The plot shows the population behavior        
    fig, ax1 = plt.subplots()
    
    color = 'tab:blue'
    ax1.set_xlabel('Generations')
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
    
    if batch_name == 'CIFAR':
        #This line of code visualizes every photo that contains W
        #The bias trick should be removed from the W in order to reshape it to 32x32
        #This [0] first element, [2] the W, [:-1,:] without bias trick
        [visualize_image(evaluated[0][2][:-1,:], evaluated[0][0], "Generation" + str(i), j) for j in range(class_count)]
    
    log("Simulation.txt", "End of execution")
    log("Simulation.txt", "Last accuracy: " + str(evaluated[0][1]))
    log("Simulation.txt", "Last loss: " + str(evaluated[0][0]))
    log("Simulation.txt", "Generation count: " + str(generation_count))
    log("Simulation.txt", "Generation size: " + str(generation_size))
    log("Simulation.txt", "Mutation chance: " + str(mutation_chance))
    log("Simulation.txt", "Pressure: " + str(pressure))
    
#For all night simulation xD
def main():
    
    global generation_size
    global generation_count
    global pressure
    global mutation_rate
    global mutation_chance
    
    generation_size = 2048
    generation_count = 50
    pressure = 3
    mutation_rate = 0.3
    mutation_chance = 0.3
    
    simulation_2()

#main()
#simulation()
simulation_2()

