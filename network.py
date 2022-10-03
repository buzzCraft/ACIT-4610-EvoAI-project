import random
import numpy as np
import threading
import multiprocessing as mp


class Neuron:
    def __init__(self, id, threshold, leakage, weight, spikeLength=0):
        self.id = id
        self.threshold = threshold
        self.weight = weight
        self.potential = 0
        self.fired = False
        self.spike = 0
        self.leakage = leakage
        self.spikeVal = 1
        self.spike_history = np.zeros(spikeLength)


    # Function to fire the neuron
    def fire(self):
        self.fired = True
        self.spike = self.spikeVal #TODO endres til 1 eller noe sånn etterhvet
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
        if self.fired:
            self.reset()
        elif self.potential >= self.threshold:
            self.fire()
    def update_enitre(self, input):
        self.potential += input - self.leakage*self.potential
        if self.potential >= self.threshold:
            return 1
        else: return 0
    # Rewriting the update function to update entire spiketrain at once
    def update_list(self, input):
        for i in range(len(input)):
            # Update and get store spike status
            x = self.update_enitre(i)
            self.spike_history[i] = self.update_enitre(i)

    def get_spike_history(self):
        return self.spike_history

    def reset_spike_history(self):
        self.spike_history = np.zero(len(self.spike_history))



    # Get the output of the neuron
    def get_spike(self):
        return self.spike

    def get_spike_and_weight(self, next_neuron):
        return self.spike, self.weight[next_neuron.id]

    def get_potential(self):
        return self.potential

    def get_weights(self):
        return self.weight


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

    def get_spike_train(self):
        """
        :return: list of spikes
        """
        return [n.get_spike_history() for n in self.neurons]

    def total_update(self):
        spikeTrain = np.array(self.prev_layer.get_spike_train())
        for n in self.neurons:
            w = n.get_weights()
            input_train = spikeTrain * n.get_weights()[:, None]
            input = np.sum(input_train, axis=0)
            n.update_list(input)




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

    def total_input(self, input_list):
        """
        This list will be the entire input list for the network
        input_list = [[Network 1 spiketrain],
                        [Network 2 spiketrain],
                        ...
                        [Network n spiketrain]]
        """
        for i in range(len(self.neurons)):
            self.neurons[i].update_list(input_list[i])




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

            spikes = self.prev_layer.get_spikes()
            weights = self.prev_layer.get_weights(n.id)
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
            # pot = n.get_potential()
        # print(self.get_output())




    def get_output(self):
        return [n.get_spike() for n in self.neurons]


class Network():
    """
    A class to represent a network
    Calling update will update each layer in the network
    """
    def __init__(self, id, input_l, hidden_l, output_l):

        self.id = id
        self.inputLayer = input_l
        self.hiddenLayer = hidden_l
        self.outputLayer = output_l
        self.output = []
        self.prediction_history = []

    def update(self, input):
        self.inputLayer.update(input)
        self.hiddenLayer.update()
        self.outputLayer.update()

    def spike_list_update(self, input):
        self.inputLayer.total_input(input)
        self.hiddenLayer.total_update()
        self.outputLayer.total_update()
        self.output = self.outputLayer.get_output()
        self.prediction_history.append(self.output)
        return self.output

    #TODO
    # Save predscore
    # Weights, threshold, leakage for every neuron in the network



    def store_output(self):
        """
        Get the output of the network in current timestep and store it
        """
        self.output.append(self.outputLayer.get_output())

    def store_prediction_score(self, pred_score, answer):
        self.prediction_history.append([pred_score, answer])

    def get_prediction_score(self):
        return sum([x[0] for x in self.prediction_history])



    def get_output(self):
        """
        :return: list of outputs, with swaped axes
        """
        output = np.swapaxes(self.output,0,1)
        return output


    def get_prediction(self, answer):
        """
        :return: the prediction of the network
        Get the output from the network
        Sum the number of spikes for each output neuron
        Return the index of the neuron with the highest sum
        """
        output = self.get_output()
        prediction = [sum(o) for o in output]
        if sum(prediction)==0:
            return 0.0
        pred_score=prediction[answer]/sum(prediction)
        self.store_prediction_score(pred_score, answer)
        return pred_score



    def get_prediction_history(self):
        return self.prediction_history




    def mutate(self, weights=True, threshold=True, leakage=True, w = 0.5, t = 0.5, l = 0.01):
        """
        Mutate the network
        :param weights: if True, mutate the weights
        :param threshold: if True, mutate the threshold
        :param leakage: if True, mutate the leakage
        :param w: the probability of mutating a weight
        :param t: the probability of mutating a threshold
        :param l: the probability of mutating a leakage
        """

        if weights:
            for neuron in self.inputLayer.neurons:
                if random.random() < w:
                    neuron.weight = neuron.weight + np.random.uniform(-.5, .5, len(neuron.weight))

            for neuron in self.hiddenLayer.neurons:
                if random.random() < w:
                    neuron.weight = neuron.weight + np.random.uniform(-.5, .5, len(neuron.weight))

        if threshold:
            for neuron in self.inputLayer.neurons:
                if random.random() < t:
                    neuron.threshold = neuron.threshold + np.random.uniform(-.1, .1)
                    if neuron.threshold < 0:
                        neuron.threshold = 0
            for neuron in self.hiddenLayer.neurons:
                if random.random() < t:
                    neuron.threshold = neuron.threshold + np.random.uniform(-.1, .1)
                    if neuron.threshold < 0:
                        neuron.threshold = 0

        if leakage:
            for neuron in self.inputLayer.neurons:
                if random.random() < l:
                    neuron.leakage = neuron.leakage + np.random.uniform(-.05, .05)
                    if neuron.leakage < 0:
                        neuron.leakage = 0
            for neuron in self.hiddenLayer.neurons:
                if random.random() < l:
                    neuron.leakage = neuron.leakage + np.random.uniform(-.05, .05)
                    if neuron.leakage < 0:
                        neuron.leakage = 0

    def reset(self):
        self.output = []


    def save(self):
        """
        Save the network to a file
        """
        input_l = []
        hidden_l = []
        output_l = []
        for neuron in self.inputLayer.neurons:
            input_l.append(neuron.weight)
            input_l.append(neuron.threshold)
            input_l.append(neuron.leakage)
        for neuron in self.hiddenLayer.neurons:
            hidden_l.append(neuron.weight)
            hidden_l.append(neuron.threshold)
            hidden_l.append(neuron.leakage)
        for neuron in self.outputLayer.neurons:
            hidden_l.append(neuron.weight)
            hidden_l.append(neuron.threshold)
            hidden_l.append(neuron.leakage)
        #TODO
        # Save the network to a file
        pass

