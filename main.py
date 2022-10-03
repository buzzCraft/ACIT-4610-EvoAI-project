import random
from keras.datasets import mnist
from matplotlib import pyplot as plt
# from pylab import *
import numpy as np
import spikeGen
import copy
from network import *
import net_classes as nc
# Image set
(train_X, train_y), (test_X, test_y) = mnist.load_data()



# Neural spiking network


def init(nr_input, nr_hidden, nr_output, threshold = 1, leakage = 0.01, number_of_networks = 1, train_length = 100):
    population = Population()
    for n in range(number_of_networks):
        input_layer = inputLayer()
        for i in range(nr_input):
            threshold = np.array(round(random.uniform(0, 5), 3))
            # Input weights will be stored in the next layer
            input_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=[1], leakage=leakage,spikeLength=train_length))
        hidden_layer = hiddenLayer(input_layer)
        for i in range(nr_hidden):
            threshold = round(random.uniform(0, 5), 3)
            # Generate a list with length of nr_input, with random values between 0 and 1
            weight_list = np.array([random.gauss(0,1) for _ in range(nr_input)])
            hidden_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage,spikeLength=train_length))

        output_layer = outputLayer(hidden_layer)
        for i in range(nr_output):
            threshold = round(random.uniform(0, 5), 3)
            weight_list = np.array([random.gauss(0, 1) for _ in range(nr_hidden)])
            output_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage,spikeLength=train_length))

        # Creating network
        population.add_network(Network(n, input_layer, hidden_layer, output_layer))

    return population








def initialize(nr_input, nr_hidden, nr_output, threshold = 1, leakage = 0.01, number_of_networks = 1):
    """
    :param nr_input: number of input neurons
    :param nr_hidden: number of hidden neurons
    :param nr_output: number of output neurons
    :param threshold: threshold for firing
    :param leakage: leakage for updating potential
    """
    global network_list
    network_list = []
    for n in range(number_of_networks):
        input_layer = inputLayer()
        for i in range(nr_input):
            # Generate a list with length of nr_hidden, with random values between 0 and 1
            # weight_list = [random.uniform(0, 1) for i in range(nr_hidden)]
            weight_list = [random.gauss(0,1) for _ in range(nr_hidden)]

            # The line above is the same as:
            # weight_list = []
            # for i in range(nr_hidden):
            #     weight_list.append(random.random())
            # Setting random threshold for each neuron
            input_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage))
        # Creating hidden layer
        hidden_layer = hiddenLayer(input_layer)
        for i in range(nr_hidden):
            threshold = round(random.uniform(0, 5), 3)
            # Generate a list with length of nr_output, with random values between 0 and 1
            weight_list = [random.random() for _ in range(nr_output)]
            hidden_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage))
        # Creating output layer
        output_layer = outputLayer(hidden_layer)
        for i in range(nr_output):
            threshold = round(random.uniform(0, 5), 3)
            output_layer.neurons.append(Neuron(id=i, threshold=threshold, weight=[1], leakage=leakage))

        # Creating network
        network = Network(n, input_layer, hidden_layer, output_layer)
        network_list.append(network)
    return network_list


def observe():
    pass

def update():
    pass
def plot_spike_train(spike_train, title):
    i=0
    for train in spike_train:
        train = train + i
        plt.plot(train, label='Neuron {}'.format(i))
        i +=1
    plt.title(title)
    plt.yticks([0,1,2,3,4,5,6,7,8,9])
    # plt.legend()
    plt.show()

#TODO NOT DONE!!!
def tournament_selection(network_list, pred_list):
    """
    Take in a list of networks and return the one with the best prediction score
    :param network_list: list of networks
    :param pred_list: list of prediction scores
    :return: network with the best prediction score
    """
    # Find index of the highest value in the array (eg the best network
    index = np.argmax(pred_list)
    # Return the network with the best prediction score
    return network_list[index]




def evolve():
    predict = []
    n = network_list
    # Training epochs
    for ep in range(100):
        print(f'Epoch {ep}')
        # Images to train on
        for p in range(20):
            print(f'Image {p}')
            # Genereating a spike train for each image
            spikeTrain = spikeGen.rateCodingRand2D(train_X[p], T = 100)
            # Get shape of the spike train
            num_rows, num_cols = spikeTrain.shape
            # For each timestep
            start = time.time()
            for i in range(num_cols):
                # Get the input for the current timestep
                inp = (spikeTrain[:,i])
                # For each network

                for network in n:

                    # Send the input to the network
                    network.update(inp)
                    # Save the output of the network
                    network.store_output()
            end = time.time()
            delta = end - start
            print("took %.2f seconds to process" % delta)
            for network in n:
                # Finding prediction score for each network
                pred_score = network.get_prediction(train_y[p])
                print(pred_score)
                predict.append(pred_score)
            ######
            # Find index of the highest value in the array (eg the best network
            index = np.argmax(predict)
            print(f'Best network: {index}, the prediction score is {predict[index]}')
            print(n[index].prediction_history)
            result = [predict.index(i) for i in sorted(predict, reverse=True)][:3]
            keepers = []
            # Implement tournament selection
            # Select
            # 1 vs 1

            predict = []
            for index in result:
                keepers.append(n[index])
            n = []

            plot_spike_train(keepers[0].get_output(), f'Spike train for the number: {train_y[p]}, epoch: {ep}')
            for network in keepers:
                network.reset()
            for _ in range(17):
                new_network = copy.deepcopy(keepers[random.randint(0,2)])
                new_network.mutate(w=.5, t=.5, l=.5)
                n.append(new_network)
            for net in keepers:
                n.append(net)

def evolve2():
    predict = []
    n = network_list
    # Training epochs
    for ep in range(100):
        print(f'Epoch {ep}')
        # Images to train on
        for p in range(2):
            for net in n:
                net.reset()
            print(f'Image {p}')
            # Genereating a spike train for each image
            spikeTrain = spikeGen.rateCodingRand2D(train_X[p], T = 100)
            # Get shape of the spike train
            num_rows, num_cols = spikeTrain.shape
            # For each timestep
            start = time.time()
            for i in range(num_cols):
                # Get the input for the current timestep
                inp = (spikeTrain[:,i])
                # For each network

                for network in n:

                    # Send the input to the network
                    network.update(inp)
                    # Save the output of the network
                    network.store_output()
            end = time.time()
            delta = end - start
            print("took %.2f seconds to process" % delta)
            for network in n:
                # Finding prediction score for each network
                pred_score = network.get_prediction(train_y[p])


            ######
            # Find index of the highest value in the array (eg the best network
        for network in n:
            predict.append(network.get_prediction_score())
        i = predict
        print(i)
        index = np.argmax(predict)
        print(f'Best network: {index}, the prediction score is {predict[index]}')
        print(n[index].prediction_history[-4:])
        result = [predict.index(i) for i in sorted(predict, reverse=True)][:3]
        keepers = []
        # Implement tournament selection
        # Select
        # 1 vs 1

        predict = []
        for index in result:
            keepers.append(n[index])
        n = []

        plot_spike_train(keepers[0].get_output(), f'Spike train for the number: {train_y[p]}, epoch: {ep}')
        for network in keepers:
            network.reset()
        for _ in range(17):
            new_network = copy.deepcopy(keepers[random.randint(0,2)])
            new_network.mutate(w=.5, t=.5, l=.5)
            n.append(new_network)
        for net in keepers:
            n.append(net)

def evolve3(spike_train_length, network_list):
    for ep in range(100):
        print(f'Epoch {ep}')
        # Images to train on
        for p in range(10):
            print(f'Image {p}')
            # Reset spike train history for all networks
            for net in network_list:
                net.reset()
                # for lay in net.layers:
                #     print(lay.get_number_of_neurons())
            start = time.time()
            # Genereating a spike train for image p
            spikeTrain = spikeGen.rateCodingRand2D(train_X[p], T=spike_train_length)

            print("took %.2f seconds to process" % delta)
            # Update the networks with the spike train
            for net in network_list:
                net.network_update(spikeTrain)
            end = time.time()
            delta = end - start


def init2(nr_input, nr_hidden, nr_output, threshold = 1, leakage = 0.01, number_of_networks = 1, train_length = 10):
    network_list = []
    for n in range(number_of_networks):
        input_layer = nc.Layer(neurons=[],spike_train_length=train_length)
        # print(input_layer.get_number_of_neurons())
        for i in range(nr_input):
            # print(i)
            threshold = np.array(round(random.uniform(0, 5), 3))
            # Input weights will be stored in the next layer
            input_layer.add_neuron(nc.Neuron(id=i, threshold=threshold, weight=np.array([1]), leakage=leakage, spike_train_length=train_length))
            # Create the correct numpy array after adding number of neurons

        # print(input_layer.get_number_of_neurons())

        hidden_layer = nc.Layer(neurons=[],spike_train_length=train_length)
        # print(hidden_layer.get_number_of_neurons())
        for i in range(nr_hidden):
            threshold = round(random.uniform(0, 5), 3)
            # Generate a list with length of nr_input, with random values between 0 and 1
            weight_list = np.array([random.gauss(0, 1) for _ in range(nr_input)])
            hidden_layer.add_neuron(
                nc.Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage, spike_train_length=train_length))

        # print(input_layer.get_number_of_neurons())
        # print(hidden_layer.get_number_of_neurons())
        output_layer = nc.Layer(neurons=[],spike_train_length=train_length)
        for i in range(nr_output):
            threshold = round(random.uniform(0, 5), 3)
            weight_list = np.array([random.gauss(0, 1) for _ in range(nr_hidden)])
            output_layer.add_neuron(
                nc.Neuron(id=i, threshold=threshold, weight=weight_list, leakage=leakage, spike_train_length=train_length))
        # print(input_layer.get_number_of_neurons())
        input_layer.update_output_array()
        hidden_layer.update_output_array()
        output_layer.update_output_array()
        net = nc.Network(nr_outputs=nr_output,spike_train_length=train_length)
        net.add_layer(input_layer)
        net.add_layer(hidden_layer)
        net.add_layer(output_layer)
        network_list.append(net)
    return network_list


#### NEW START THINGY ####
# Create the population
import population

def init_n(nr_input, nr_hidden, nr_output, threshold = 1, leakage = 0.01, number_of_networks = 1, train_length = 10):
    pop = population.Population(nr_inputs=nr_input, nr_hidden=nr_hidden, nr_outputs=nr_output, threshold=threshold, leakage=leakage, size=number_of_networks, spike_train_length=train_length)
    pop.create_population()
    return pop
    # print(pop)

def pop_eve(spike_train_length, population):
    for ep in range(100):
        print(f'Epoch {ep}')
        # Images to train on
        for p in range(10):
            print(f'Image {p}')
            # Reset spike train history for all networks
            population.reset_population()

            start = time.time()
            # TODO Generate all spike trains at once and store them in a text file
            # Then read it in when needed
            # Genereating a spike train for image p
            spikeTrain = spikeGen.rateCodingRand2D(train_X[p], T=spike_train_length)


            # Update the networks with the spike train
            population.update_population(spikeTrain)
            end = time.time()
            delta = end - start
            print("took %.2f seconds to process" % delta)


if __name__ == '__main__':
    import time

    # Get total pixels in an image
    nr_pix = train_X.shape[1] * train_X.shape[2]

    spike_train_length = 100
    # Nr_input neurons -> number of pixels in an image
    # pop = init(nr_input=nr_pix, nr_output=10, nr_hidden=10, number_of_networks=20, train_length=spike_train_length)
    # evolve3(spike_train_length)
    # initialize(nr_input=nr_pix, nr_hidden=20, nr_output=10, threshold=5, number_of_networks=20, leakage=0.05)
    # evolve2()

    # Bruker en ny måte å regne på.. Ikke implementert evolusjon enda
    # n = init2(nr_input=nr_pix, nr_hidden=20, nr_output=10, threshold=5, number_of_networks=20, leakage=0.05, train_length=spike_train_length)
    # evolve3(spike_train_length,n)
    p = init_n(nr_pix, [20], 10, 5, 0.05, 20, spike_train_length)
    pop_eve(spike_train_length, p)










