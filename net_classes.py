import numpy as np
import time
import random
from numba import jit
class Neuron():

    def __init__(self, id, threshold=0.6, weight = [1,1], leakage=0.5, spike_train_length=100):
        self.id = id
        self.threshold = threshold
        self.leakage = leakage
        self.spike_train = np.zeros(spike_train_length)
        self.weight = np.array(weight)
        self.potential = 0
        self.fired = False

    def _update(self, input, step):
        #TODO Check if (1-self.fired) is needed
        self.potential += (input*(1-self.fired) - self.leakage*self.potential)
        if self.potential > self.threshold and not self.fired:
            self.spike_train[step] = 1
            self.potential = 0
            self.fired = True
        else:
            self.spike_train[step] = 0
            self.fired = False

    # @jit(forceobj=True)
    def neuron_update(self, input):
        # Take the input array, multiply it with the weight array and sum the result
        x = (input * self.weight[:, None])
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
        if np.random.rand() < mutation_rate:
            self.__mutate_threshold(threshold_mutation_rate)
        if np.random.rand() < mutation_rate:
            self.__mutate_leakage(leakage_mutation_rate)


    def __mutate_weight(self, mutation_rate):
        # Mutate the weight array
        for i in range(len(self.weight)):
            if np.random.rand() < 1:
                self.weight[i] += np.random.uniform(-.5, .5)
    def __mutate_threshold(self, mutation_rate):
        # Mutate the threshold
        if np.random.rand() < mutation_rate:
            self.threshold += np.random.uniform(-.5, .5)

    def __mutate_leakage(self, mutation_rate):
        # Mutate the leakage
        if np.random.rand() < mutation_rate:
            self.leakage += np.random.uniform(-.05, .05)

    def get_genome_neuron(self):
        # Return the genome of the neuron
        return [np.copy(self.weight), self.threshold, self.leakage]

    def set_genome(self, genome):
        # Set the genome of the neuron
        self.weight = genome[0]
        self.threshold = genome[1]
        self.leakage = genome[2]


class Layer:
    def __init__(self, neurons=[], spike_train_length=100):
        self.neurons = neurons
        self.spike_train_length = spike_train_length
        self.output = np.zeros((len(neurons), self.spike_train_length))
        self.input_layer = False

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

            if self.input_layer:
                neuron.neuron_update(input[i])
            else:
                neuron.neuron_update(input)
            # Store neuron output in output array
            self.output[i] = neuron.get_spike_train()
            i += 1
        # print(i)
        # Return the array of neuron outputs
        return self.output

    def reset(self):
        # Reset the layer for a new input
        self.output = np.zeros((len(self.neurons), self.spike_train_length))
        for neuron in self.neurons:
            neuron.reset()

    def get_number_of_neurons(self):
        return len(self.neurons)

    def get_genome_layer(self):
        genome = []
        for neuron in self.neurons:
            genome.append(neuron.get_genome_neuron())
        return genome

    def set_genome(self, genome):
        for i in range(len(self.neurons)):
            self.neurons[i].set_genome(genome[i])


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
        self.current_prediction_score = 0

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

    def mutate_network(self, mutation_rate=0.5, weight_mutation_rate=0.5, threshold_mutation_rate=0.1, leakage_mutation_rate=0.1):
        for layer in self.layers:
            layer.mutate(mutation_rate, weight_mutation_rate, threshold_mutation_rate, leakage_mutation_rate)

    def get_genome(self):
        for layer in self.layers:
            self.genome.append(layer.get_genome_layer())
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
            self.prediction_history.append([0,answer])
            return 0.0
        else:

            pred_score = prediction[answer] / sum(prediction)
            self.prediction_history.append([pred_score, answer])
            return pred_score

    def get_prediction_score(self, batch_size):
        """
        Get the average score of the prediction
        """
        self.current_prediction_score = sum([x[0] for x in self.prediction_history[-batch_size:]])/batch_size
        return self.current_prediction_score
        # return sum([x[0] for x in self.prediction_history[-batch_size:]])

    def get_genome(self):
        genome = []
        for layer in self.layers:
            genome.append(layer.get_genome_layer())
        return genome

    def set_genome(self, genome):
        for i in range(len(self.layers)):
            self.layers[i].set_genome(genome[i])

    def __str__(self):
        return f'Network {self.id} ' \
               f'\nwith a prediction score of {self.current_prediction_score}' \




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



