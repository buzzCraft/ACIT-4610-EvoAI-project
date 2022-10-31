import copy

import numpy as np
from scipy.special import expit as sigmoid
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm
from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

def norm2D(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))

def sigmoid(x):
    return 1 / (1 + np.exp(x))

class NeuralNetwork:
    def __init__(self, input_size, output_size, hidden_size):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.weights1 = np.random.rand(self.input_size, self.hidden_size) * 2 - 1
        self.weights2 = np.random.rand(self.hidden_size, self.output_size) * 2 - 1

    def feedforward(self, X):
        self.hidden = sigmoid(np.dot(X, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def mutate(self):
        for w in self.weights1:
            for w1 in w:
                if random.random() < 0.1:
                    w1 += np.random.normal(0,0.2)
        for w in self.weights2:
            for w1 in w:
                if random.random() < 0.1:
                    w1 += np.random.normal(0, 0.2)

    def calculate_predicton_score(self, X, y):
        self.feedforward(X)
        return np.sum(np.abs(self.output - y))


def evolve_population(networks):
    predictions = []
    accuracy = []

    for network in networks:
        network.calculate_prediction_score()
    # Sort the networks by their prediction score
    networks.sort()
    best_acc_network = networks[0]
    best_score = best_acc_network.current_prediction_score
    # best_score_history.append(best_score)
    # best_acc_history.append(best_acc_network.current_accuracy)
    # Get the best networks
    best_networks = copy.deepcopy(networks[:5])
    for network in best_networks:
        print(network.get_log_loss())

    for i in range(int(len(networks) / 2)):
        net1 = random.choice(best_networks)
        net2 = random.choice(best_networks)
        if random.uniform(0, 1) > 0.2:
            net3 = random.choice(best_networks)
            net4 = random.choice(best_networks)
            # Get the best of net1 and net2
            if net1.current_accuracy < net2.current_accuracy:
                net1 = net2
            # Get the best of net3 and net4
            if net3.current_accuracy > net4.current_accuracy:
                net2 = net3
            else:
                net2 = net4

            # Do crossover on net1 and net2
            net1, net2 = crossover(net1.get_genome(), net2.get_genome())
            networks[i].set_genome(net1)
            networks[-i].set_genome(net2)

    for net in networks:
        # if random.uniform(0, 1) < 0.5:
        net.mutate_network(0.5)
    networks.remove(random.choice(networks))
    networks.append(best_networks[0])
    print(best_networks[0])

def crossover(net1, net2):
    # Do crossover between the networks
    for i in range(len(net1.weights1)):
        for j in range(len(net1.weights1[i])):
            if random.random() < 0.5:
                tmp = net1.weights1[i][j]
                net1.weights1[i][j] = net2.weights1[i][j]
                net2.weights1[i][j] = tmp






if __name__ == "__main__":
    networks = []
    # Add a population of networks
    for i in range(10):
        networks.append(NeuralNetwork(784, 10, 100))
    # Train the networks
    for i in range(100):
        evolve_population(networks)
        print("Epoch: ", i)
    out = net.feedforward(train_X[0].flatten())
    print(out)