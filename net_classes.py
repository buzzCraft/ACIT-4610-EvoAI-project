import numpy as np
import time
import random
class Neuron():

    def __init__(self, id, threshold=0.6, weight = [1,1], leakage=0.5, spike_train_length=100):
        self.id = id
        self.threshold = threshold
        self.leakage = leakage
        self.spike_train = np.zeros(spike_train_length)
        self.weight = np.array(weight)
        self.potential = 0

    def _update(self, input, step):
        self.potential += input - self.leakage*self.potential
        if self.potential > self.threshold and not self.fired:
            self.spike_train[step] = 1
            self.potential = 0
            self.fired = True
        else:
            self.spike_train[step] = 0
            self.fired = False

    def neuron_update(self, input):
        # Take the input array, multiply it with the weight array and sum the result
        input_array = np.sum(input * self.weight[:, None], axis=0)
        # Send each timestep of the input array to the neuron
        # Save the output to spike_train
        for step in range(len(input_array)):
            self._update(input_array[step], step)

    def get_spike_train(self):
        # Return the spike train
        return self.spike_train

    def reset(self):
        # Reset the neuron for a new input
        self.spike_train = np.zeros(len(self.spike_train))
        self.potential = 0

    def mutate(self, mutation_rate=0.5, weight_mutation_rate=0.5, threshold_mutation_rate=0.5, leakage_mutation_rate=0.5):
        # Mutate the neuron
        if np.random.rand() < mutation_rate:
            self.__mutate_weight(weight_mutation_rate)
        if np.radom.rand() < mutation_rate:
            self.__mutate_threshold(threshold_mutation_rate)
        if np.random.rand() < mutation_rate:
            self.__mutate_leakage(leakage_mutation_rate)


    def __mutate_weight(self, mutation_rate):
        # Mutate the weight array
        for i in range(len(self.weight)):
            if np.random.rand() < mutation_rate:
                self.weight[i] += np.random.uniform(-.5, .5)
    def __mutate_threshold(self, mutation_rate):
        # Mutate the threshold
        if np.random.rand() < mutation_rate:
            self.threshold += np.random.uniform(-.5, .5)

    def __mutate_leakage(self, mutation_rate):
        # Mutate the leakage
        if np.random.rand() < mutation_rate:
            self.leakage += np.random.uniform(-.05, .05)

    def return_genome(self):
        # Return the genome of the neuron
        return [self.weight, self.threshold, self.leakage]


class Layer:
    def __init__(self, neurons=[], spike_train_length=100):
        self.neurons = neurons
        self.spike_train_length = spike_train_length
        self.output = np.zeros((len(neurons), self.spike_train_length))

    def add_neuron(self, neuron):
        # Add a neuron to the layer
        self.neurons.append(neuron)

    def update_output_array(self):
        self.output = np.zeros((len(self.neurons), self.spike_train_length))
        # print(f'Output array updated to {self.output.shape}')

    def layer_update(self, input):
        # Send the input to each neuron in the layer
        i = 0
        for neuron in self.neurons:
            # Update neuron
            neuron.neuron_update(input)
            # Store neuron output in output array
            self.output[i] = neuron.get_spike_train()
            i += 1
        # Return the array of neuron outputs
        return self.output

    def reset(self):
        # Reset the layer for a new input
        for neuron in self.neurons:
            neuron.reset()

    def get_number_of_neurons(self):
        return len(self.neurons)

    def get_genome(self):
        genome = []
        for neuron in self.neurons:
            genome.append(neuron.return_genome())
        return genome


    def mutate(self, mutation_rate=0.5, weight_mutation_rate=0.5, threshold_mutation_rate=0.5, leakage_mutation_rate=0.5):
        for neuron in self.neurons:
            neuron.mutate(mutation_rate, weight_mutation_rate, threshold_mutation_rate, leakage_mutation_rate)

class Network:
    def __init__(self, layers=[], nr_outputs=10, spike_train_length=100, id=0):
        self.layers = layers
        self.id = id
        # Create a empty array for the output
        self.output = np.zeros((nr_outputs, spike_train_length))
        self.prediction_history = []
        self.genome = []

    def add_layer(self, layer):
        # Add a layer to the network
        self.layers.append(layer)

    def network_update(self, input):
        # Send the input to each layer in the network
        # TODO ALL COMPUTING TIME IS HERE
        for layer in self.layers:
            # print("Starting with layer", layer)
            # Copy the output from the previous layer to the input of the next layer
            input = layer.layer_update(input)
            # print("Done with layer", layer)
        # TODO END OF COMPUTING TIME

        start = time.time()
        # TODO Set np.array to self.output
        self.output = input
        # Check time to set np.array

        for i in range(len(input)):
            self.output[i] = input[i]
        # print(self.output)

    def reset(self):
        # Reset the network for a new input
        for layer in self.layers:
            layer.reset()

    def get_output(self):
        return self.output

    def mutate_network(self, mutation_rate=0.5, weight_mutation_rate=0.5, threshold_mutation_rate=0.5, leakage_mutation_rate=0.5):
        for layer in self.layers:
            layer.mutate(mutation_rate, weight_mutation_rate, threshold_mutation_rate, leakage_mutation_rate)

    def get_genome(self):
        for layer in self.layers:
            self.genome.append(layer.get_genome())
        return self.genome

    def get_prediction(self, answer):
        """
        Get the prediction of the network
        And save it to self.prediction_history
        :param answer: The correct answer
        :return: The prediction
        """
        # Get the prediction of the network
        output = self.get_output()
        prediction = [sum(neuron) for neuron in output]
        if sum(prediction) == 0:
            return 0.0
        else:
            predicted = prediction.index(max(prediction))
            return self.__get_prediction_score(predicted, answer)

    def __get_prediction_score(self, predicted, answer):
        """
        Get the score of the prediction
        and save it in self.prediction_history
        :param predicted: The predicted answer
        :param answer: The correct answer
        :return: The score of the prediction
        """
        # Get the prediction score of the network
        prediction = self.get_prediction()
        pred_score = prediction[answer] / sum(prediction)
        self.prediction_history.append([pred_score, answer])
        return pred_score

    def get_prediction_score(self):
        """
        Get the average score of the prediction
        """
        return sum([x[0] for x in self.prediction_history])

    def get_genome(self):
        genome = []
        for layer in self.layers:
            genome.append(layer.get_genome())
        return genome

    def __str__(self):
        return f'Network {self.id} with {len(self.layers)} layers'

# class Population():
#     def __init__(self,nr_inputs=100, nr_hidden=[20], nr_outputs=10, size=100, spike_train_length=100, leakage=0.1, threshold=0.5):
#         """
#         Create a population of networks
#         :param size: The size of the population
#         :param nr_outputs: The number of outputs of the network
#         :param spike_train_length: The length of the spike train
#         """
#         self.size = size
#         self.nr_outputs = nr_outputs
#         self.nr_inputs = nr_inputs
#         self.nr_hidden = nr_hidden
#         self.spike_train_length = spike_train_length
#         self.networks = []
#         self.leakage = leakage
#         self.threshold = threshold
#         ####
#         self.genomes = []
#         self.fitness = []
#         self.best_network = None
#         self.best_score = 0.0
#         self.best_genome = None
#
#     def create_population(self, input_size=784, nr_neurons=100, nr_layers=2):
#         """
#         Create a population of networks
#         :param input_size: The size of the input
#         :param nr_neurons: The number of neurons in each layer
#         :param nr_layers: The number of layers in the network
#         """
#         # Create a population of networks
#         for i in range(self.size):
#             self.networks.append(self.__create_network())
#
#     def __create_network(self):
#         # Creating input layer
#         input_layer = Layer(neurons=[],spike_train_length=self.spike_train_length)
#         for i in range(self.nr_inputs):
#             # print(i)
#             threshold = np.array(round(random.uniform(0, 5), 3))
#             # Input weights will be stored in the next layer
#             input_layer.add_neuron(Neuron(id=i, threshold=self.threshold, weight=np.array([1]), leakage=self.leakage,spike_train_length=self.spike_train_length))
#         hidden_layers = []
#         for nr_of_neurons in self.nr_hidden:
#             h_l = (Layer(neurons=[], spike_train_length=self.spike_train_length))
#             # print(hidden_layer.get_number_of_neurons())
#             weight_list = np.array([random.gauss(0, 1) for _ in range(self.nr_inputs)])
#             for i in range(nr_of_neurons):
#                 h_l.add_neuron(
#                     Neuron(id=i, threshold=self.threshold, weight=weight_list, leakage=self.leakage, spike_train_length=self.spike_train_length))
#             hidden_layers.append(h_l)
#
#         output_layer = Layer(neurons=[], spike_train_length=self.spike_train_length)
#         for i in range(self.nr_outputs):
#             weight_list = np.array([random.gauss(0, 1) for _ in range(self.nr_hidden[-1])])
#             output_layer.add_neuron(
#                 Neuron(id=i, threshold=self.threshold, weight=weight_list, leakage=self.leakage, spike_train_length=self.spike_train_length))
#         return Network(layers=[input_layer] + hidden_layers + [output_layer])
#     def update_population(self, input):
#         # Update the population
#         for network in self.networks:
#             network.network_update(input)
#
#     def reset_population(self):
#         # Reset the population
#         for network in self.networks:
#             network.reset()
#
#     def get_population_output(self):
#         # Get the output of the population
#         output = []
#         for network in self.networks:
#             output.append(network.get_output())
#         return output
#
#     def get_population_prediction(self, answer):
#         """
#         Get the prediction of the population
#         :param answer: The correct answer
#         :return: The prediction
#         """
#         for network in self.networks:
#             network.get_prediction(answer)

if __name__ == '__main__':

    neuron1 = Neuron(id = 1, threshold=3, weight = [0.1, 0.3], leakage = 0.1, spike_train_length=10)
    neuron2 = Neuron(id=1, threshold=1, weight=[1, 2], leakage=0.01, spike_train_length=10)
    neuron3 = Neuron(id=1, threshold=1, weight=[1, 3], leakage=0.1, spike_train_length=10)
    neuron4 = Neuron(id=1, threshold=1, weight=[1, -2], leakage=0.1, spike_train_length=10)
    neuron5 = Neuron(id=1, threshold=5, weight=[1, 0.1], leakage=0.5, spike_train_length=10)

    layer1 = Layer(neurons=[neuron1, neuron2], spike_train_length=10)
    layer2 = Layer(neurons=[neuron3, neuron4,neuron5], spike_train_length=10)
    layer2.add_neuron(neuron5)
    layer2.update_output_array()
    net = Network(layers=[layer1, layer2], nr_outputs=4, spike_train_length=10)
    print(layer2.get_number_of_neurons())
    print(layer1.get_number_of_neurons())



    spike_train = np.array([[0, 0, 0, 1, 1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 1, 1, 1, 1, 0]])

    # layer1.layer_update(spike_train)
    net.network_update(spike_train)



