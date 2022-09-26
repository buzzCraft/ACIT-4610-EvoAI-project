import random
from keras.datasets import mnist
from matplotlib import pyplot as plt
# from pylab import *
import numpy as np
import spikeGen

# Image set
(train_X, train_y), (test_X, test_y) = mnist.load_data()



# Neural spiking network
class Neuron:
    def __init__(self, id, threshold, leakage, weight):
        self.id = id
        self.threshold = threshold
        self.weight = weight
        self.potential = 0
        self.fired = False
        self.spike = 0
        self.leakage = leakage
        # self.connections = []


    # Function to fire the neuron
    def fire(self):
        self.fired = True
        self.spike = 1 #TODO endres til 1 eller noe sånn etterhvet
        self.potential = 0

    # Function to reset the neuron
    def reset(self):
        self.fired = False
        self.spike = 0


    # Function to update the neuron
    # If the neuron has fired, reset it
    # If the neuron has not fired, update its potential with leakage
    def update(self, input):

        self.potential += (input) - self.leakage*self.potential #(input * self.weight) - self.leakage*self.potential

        if self.potential >= self.threshold:
            self.fire()
        else:
            self.reset()

    # Get the output of the neuron
    def get_spike(self):
        return self.spike

    def get_spike_and_weight(self, next_neuron):
        return self.spike, self.weight[next_neuron.id]

    def get_potential(self):
        return self.potential


class Layer:
    """
    Base class for Layers
    """
    def __init__(self):
        self.neurons = []

    def get_spikes(self):
        """
        :return: list of spikes
        """
        return [n.get_spike() for n in self.neurons]
    def get_weights(self, id):
        """
        :return: list of weights
        """
        return [n.weight[id] for n in self.neurons]




class inputLayer(Layer):
    """
    Input layer with a update function that takes a list of inputs
    """
    def __init__(self):
        super().__init__()
    # Take a list of inputs (one for each layer and update the neurons)
    def update(self, input_list):
        for i in range(len(self.neurons)):
            self.neurons[i].update(input_list[i])



class hiddenLayer(Layer):
    """
    Hidden layer with a update function that takes the previous layer as input
    """
    def __init__(self, prev_layer):
        super().__init__()
        self.prev_layer = prev_layer

    # Take the previous layer as input and update the neurons
    def update(self):
        for n in self.neurons:
            # Get the spikes from the previous layer for current neuron
            weights = self.prev_layer.get_weights(n.id)
            spikes = self.prev_layer.get_spikes()
            # Sum the weighted spikes from the previous layer for the current neuron
            n.update(sum([w*s for w,s in zip(weights, spikes)]))
            # print(sum([w*s for w,s in zip(weights, spikes)]))
            # Means the same as:
            # [1,3,4] [2,4,5] -> 1*2 + 3*4 + 4*5 = 26


class outputLayer(Layer):
    """
    Output layer with a update function that takes the previous layer as input
    And has a get_output function that returns the output of the layer
    """
    def __init__(self, prev_layer):
        super().__init__()
        self.prev_layer = prev_layer

    # Take the previous layer as input and update the neurons
    def update(self):
        for n in self.neurons:
            # Get the spikes from the previous layer for current neuron
            weights = self.prev_layer.get_weights(n.id)
            spikes = self.prev_layer.get_spikes()
            # Sum the weighted spikes from the previous layer for the current neuron
            n.update(sum([w*s for w,s in zip(weights, spikes)]))
            pot = n.get_potential()
        # print(self.get_output())




    def get_output(self):
        return [n.get_spike() for n in self.neurons]


class Network:
    """
    A class to represent a network
    Calling update will update each layer in the network
    """
    def __init__(self, input_l, hidden_l, output_l):
        self.inputLayer = input_l
        self.hiddenLayer = hidden_l
        self.outputLayer = output_l

    def update(self, input):
        self.inputLayer.update(input)
        self.hiddenLayer.update()
        self.outputLayer.update()

    def observe(self):
        pass









def initialize(nr_input, nr_hidden, nr_output, threshold = 1.5, leakage = 0.01, number_of_networks = 1):
    """
    :param nr_input: number of input neurons
    :param nr_hidden: number of hidden neurons
    :param nr_output: number of output neurons
    :param threshold: threshold for firing
    :param leakage: leakage for updating potential
    """
    global network_list
    network_list = []
    for i in range(number_of_networks):
        input_layer = inputLayer()
        for i in range(nr_input):
            # Generate a list with length of nr_hidden, with random values between 0 and 1
            weight_list = [random.random() for _ in range(nr_hidden)]
            # The line above is the same as:
            # weight_list = []
            # for i in range(nr_hidden):
            #     weight_list.append(random.random())
            input_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage))
        # Creating hidden layer
        hidden_layer = hiddenLayer(input_layer)
        for i in range(nr_hidden):
            # Generate a list with length of nr_output, with random values between 0 and 1
            weight_list = [random.random() for _ in range(nr_output)]
            hidden_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage))
        # Creating output layer
        output_layer = outputLayer(hidden_layer)
        for i in range(nr_output):
            output_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=[1], leakage=leakage))

        # Creating network
        network = Network(input_layer, hidden_layer, output_layer)
        network_list.append(network)
    return network_list


def observe():
    pass

def update():
    pass


if __name__ == '__main__':
    # Get total pixels in an image
    nr_pix = train_X.shape[1] * train_X.shape[2]
    initialize(nr_pix, 20, 10, number_of_networks=1)


    # Create a spiketrain for a image
    output_arr = []
    spikeTrain = spikeGen.rateCodingRand2D(train_X[0], T = 50)
    num_rows, num_cols = spikeTrain.shape
    for i in range(num_cols):
        inp = (spikeTrain[:,i])
        for network in network_list:
            network.update(inp)
            # print(network.outputLayer.get_output())
            output_arr.append(network.outputLayer.get_output())
    arr = np.array(output_arr)
    neuralData = np.random.randint([10, 50])

    arr = np.swapaxes(arr,0,1)
    plt.eventplot(arr)
    plt.show()
    # print(output_arr)








