import random
from keras.datasets import mnist
from matplotlib import pyplot as plt
from pylab import *
import numpy as np
import pycxsimulator

# Image set
(train_X, train_y), (test_X, test_y) = mnist.load_data()
# Get total pixels in an image
PIX = train_X.shape[1] * train_X.shape[2]


# Neural spiking network
class Neuron:
    def __init__(self, id, threshold, weight, leakage):
        self.id = id
        self.threshold = threshold
        self.weight = weight
        self.potential = 0
        self.fired = False
        self.spike = 0
        self.leakage = leakage
        self.connections = []


    # Function to fire the neuron
    def fire(self):
        self.fired = True
        self.spike = .5 * self.weight
        self.potential = 0

    # Function to reset the neuron
    def reset(self):
        self.fired = False
        self.spike = 0

    # Function to update the neuron
    # If the neuron has fired, reset it
    # If the neuron has not fired, update its potential with leakage
    def update(self, input):
        # had += input * self.weight, but moved weight to spike in def fire()
        self.potential += (input * self.weight) - self.leakage*self.potential
        # Just for testing
        pot = self.potential
        if self.potential < 0: self.potential = 0
        if self.potential >= self.threshold:
            self.fire()
        else:
            self.reset()
        # Returning potential for testing
        return pot

    # Get the output of the neuron
    def get_spike(self):
        return self.spike

    def get_potential(self):
        return self.potential


class Layer:
    def __init__(self):
        self.neurons = []





# TEST METHODS
def test_neuron():
    threshold=1.5
    n1 = Neuron(1, threshold, 1, 0.1)
    response_list = []
    potential_list = []

    spike_list = [0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]
    for element in spike_list:
        potential_list.append(n1.update(element))
        # potential_list.append(n1.get_potential())
        response_list.append(n1.get_spike())

    plt.plot(response_list, label='Spike')
    plt.plot(spike_list, label='Input')
    plt.plot(potential_list, label='Potential')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.legend()
    plt.show()



def test_layer():
    inputLayer = Layer()
    for i in range(2):
        inputLayer.neurons.append(Neuron(id=i, threshold=.8, weight=0.5, leakage=.01))
    # Creating hidden layer
    hiddenLayer = Layer()
    for i in range(1):
        hiddenLayer.neurons.append(Neuron(id=i, threshold=.6, weight=.5, leakage=.05))

    response_list = []
    potential_list = []
    l1 = []
    l2 = []

    spike_list1 = [0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]
    spike_list2 = [0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1]
    # spike_list2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(spike_list1)):
        inputLayer.neurons[0].update(spike_list1[i])
        inputLayer.neurons[1].update(spike_list2[i])
        l = inputLayer.neurons[0].get_spike()
        l1.append(l)
        u = inputLayer.neurons[1].get_spike()
        l2.append(u)
        potential_list.append(hiddenLayer.neurons[0].update( u+l ))
        # potential_list.append(n1.get_potential())
        response_list.append(hiddenLayer.neurons[0].get_spike())

    plt.plot(response_list, label='Spike')
    plt.plot(spike_list1, label='Input1')
    plt.plot(spike_list2, label='Input2')
    plt.plot(l1, label='1 pot')
    plt.plot(l2, label='2 pot')
    plt.plot(potential_list, label='Potential')
    plt.axhline(y=0.6, color='r', linestyle='-')
    plt.legend()
    plt.show()


def initialize():
    global spike_train, inputLayer, hiddenLayer, outputLayer
    spike_train = np.zeros((PIX, 1))
    inputLayer = Layer()
    for i in range(PIX):
        inputLayer.neurons.append(Neuron(id=i, threshold=.5, weight=.5, leakage=.01))
    # Creating hidden layer
    hiddenLayer = Layer()
    for i in range(20):
        hiddenLayer.neurons.append(Neuron(id=i, threshold=.5, weight=.5, leakage=.01))
    # Creating output layer
    outputLayer = Layer()
    for i in range(10):
        outputLayer.neurons.append(Neuron(id=i, threshold=.5, weight=.5, leakage=.01))

    # Creating network
    network = [inputLayer, hiddenLayer, outputLayer]
    # Creating connections between layers
    for i in range(len(inputLayer.neurons)):
        for j in range(len(hiddenLayer.neurons)):
            inputLayer.neurons[i].weight = np.random.uniform(0, 1)
            hiddenLayer.neurons[j].weight = np.random.uniform(0, 1)
            inputLayer.neurons[i].connections.append(hiddenLayer.neurons[j])
            hiddenLayer.neurons[j].connections.append(inputLayer.neurons[i])

    for i in range(len(hiddenLayer.neurons)):
        for j in range(len(outputLayer.neurons)):
            hiddenLayer.neurons[i].weight = np.random.uniform(0, 1)
            outputLayer.neurons[j].weight = np.random.uniform(0, 1)
            hiddenLayer.neurons[i].connections.append(outputLayer.neurons[j])
            outputLayer.neurons[j].connections.append(hiddenLayer.neurons[i])


def observe():
    pass

def update():
    #Send one bit of spike_train to each neuron in input layer
    for i in range(PIX):
        inputLayer.neurons[i].update(spike_train[i])



test_layer()
test_neuron()
# pycxsimulator.GUI().start(func=[initialize, observe, update])
initialize()
# Creating input layer






# for i in range(9):
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
# pyplot.show()


# for x in range(10):
#     for i in range(50000):
#         if train_y[i]==x:
#             plt.imshow(train_X[i], cmap=plt.get_cmap('gray'), alpha=0.1)
#     plt.show()


# Class
# 0 - low / no spikes - low frequency
# 255 - high / many spikes - high frequency
