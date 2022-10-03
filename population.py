from net_classes import *

class Population():
    def __init__(self,nr_inputs, nr_hidden, nr_outputs, size, spike_train_length, leakage=0.1, threshold=0.5):
        """
        Create a population of networks
        :param nr_inputs: Number of inputs neurons
        :param nr_hidden: Number of hidden neurons -> list [x1,x2,x3] where x1 is number of
        neurons in first hidden layer, x2 is number of neurons in second hidden layer and so on
        :param size: The size of the population
        :param nr_outputs: The number of outputs of the network
        :param spike_train_length: The length of the spike train
        """
        self.size = size
        self.nr_outputs = nr_outputs
        self.nr_inputs = nr_inputs
        self.nr_hidden = nr_hidden
        self.spike_train_length = spike_train_length
        self.networks = []
        self.leakage = leakage
        self.threshold = threshold
        ####
        self.genomes = []
        self.fitness = []
        self.best_network = None
        self.best_score = 0.0
        self.best_genome = None

    def create_population(self):
        """
        Create a population of networks
        """
        # Create a population of networks
        for i in range(self.size):
            self.networks.append(self.__create_network(i))

    def __create_network(self, network_id):
        # Creating input layer
        input_layer = Layer(neurons=[],spike_train_length=self.spike_train_length)
        for i in range(self.nr_inputs):
            # print(i)
            # threshold = np.array(round(random.uniform(0, 5), 3))
            # Input weights will be stored in the next layer
            input_layer.add_neuron(Neuron(id=i, threshold=self.threshold, weight=np.array([1]), leakage=self.leakage,spike_train_length=self.spike_train_length))
        input_layer.update_output_array()
        hidden_layers = []
        for nr_of_neurons in self.nr_hidden:
            h_l = (Layer(neurons=[], spike_train_length=self.spike_train_length))
            # print(hidden_layer.get_number_of_neurons())
            weight_list = np.array([random.gauss(0, 1) for _ in range(self.nr_inputs)])
            for i in range(nr_of_neurons):
                h_l.add_neuron(
                    Neuron(id=i, threshold=self.threshold, weight=weight_list, leakage=self.leakage, spike_train_length=self.spike_train_length))
            hidden_layers.append(h_l)
            h_l.update_output_array()

        output_layer = Layer(neurons=[], spike_train_length=self.spike_train_length)
        for i in range(self.nr_outputs):
            weight_list = np.array([random.gauss(0, 1) for _ in range(self.nr_hidden[-1])])
            output_layer.add_neuron(
                Neuron(id=i, threshold=self.threshold, weight=weight_list, leakage=self.leakage, spike_train_length=self.spike_train_length))
        output_layer.update_output_array()
        return Network(id = network_id, layers=[input_layer] + hidden_layers + [output_layer])

    def update_population(self, input):
        # Update the population
        for network in self.networks:
            network.network_update(input)

    def reset_population(self):
        # Reset the spike train history of the population
        # keep weights and thresholds
        for network in self.networks:
            network.reset()

    def get_population_output(self):
        # Get the output of the population
        output = []
        for network in self.networks:
            output.append(network.get_output())
        return output

    def get_population_prediction(self, answer):
        """
        Get the prediction of the population
        :param answer: The correct answer
        :return: The prediction
        """
        for network in self.networks:
            network.get_prediction(answer)

    def __str__(self):
        # Print the population
        s = ""
        for network in self.networks:
            s += str(network) + "\n"
        return s


if __name__ == "__main__":
    population = Population(nr_inputs=10, nr_hidden=[10], nr_outputs=10, size=10, spike_train_length=5, leakage=0.1, threshold=0.5)
    population.create_population()
    print(population)

