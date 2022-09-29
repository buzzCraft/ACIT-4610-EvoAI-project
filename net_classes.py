import numpy as np
import time
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

class Network:
    def __init__(self, layers=[], nr_outputs=10, spike_train_length=100):
        self.layers = layers
        # Create a empty array for the output
        self.layers = []
        self.output = np.zeros((nr_outputs, spike_train_length))

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
        for i in range(len(input)):
            self.output[i] = input[i]
        # print(self.output)

    def reset(self):
        # Reset the network for a new input
        for layer in self.layers:
            layer.reset()

    def get_output(self):
        return self.output




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



