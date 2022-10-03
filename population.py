from net_classes import *
import random
from matplotlib import pyplot as plt
import copy
class Population():
    def __init__(self,nr_inputs, nr_hidden, nr_outputs, size, spike_train_length, batch_size, leakage=0.1, threshold=0.5, tournament_size=5):
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
        self.batch_size = batch_size
        self.tournament_size = tournament_size
        self.mutation_rate = 0.3
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
            # For each input neuron create a neuron
            input_layer.add_neuron(Neuron(id=i, threshold=self.threshold, weight=np.array([1]), leakage=self.leakage,spike_train_length=self.spike_train_length))
        # Update the size of the input layer
        input_layer.update_output_array()
        # Create a list to store hidden layers
        hidden_layers = []
        # Create the hidden layers
        for nr_of_neurons in self.nr_hidden:
            #
            h_l = (Layer(neurons=[], spike_train_length=self.spike_train_length))
            # weight = np.random.uniform(-1, 1, self.nr_inputs)
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

    def update_population(self, input, answer):
        # Update the population
        for network in self.networks:
            network.network_update(input)
            network.get_prediction(answer)

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

    def get_population_prediction(self):
        """
        Get the prediction of the population
        :param answer: The correct answer
        :return: The prediction
        """
        predictions = []
        for network in self.networks:
            # p = network.get_prediction_score()
            predictions.append(network.get_prediction_score(self.batch_size)/self.batch_size)
        return predictions

    def evolve_population(self):
        predictions = self.get_population_prediction()
        # Find index of best score in predictions list

        index_of_best = predictions.index(max(predictions))
        keeper = []
        keeper.append(self.networks.pop(index_of_best))
        print(keeper[0])
        print("Best score: ", max(predictions))
        keeper.extend(self.tournament_selection(self.tournament_size))
        self.mutate_population(keeper)
        self.best_network= keeper[0]

    def tournament_selection(self, nr_of_tournaments):
        """
        Tournament selection
        :return: The survivors of the tournament
        """
        # First shuffle the population

        # Create sublists of size nr_of_tournaments
        tournament_list = list(self.__generate_tournament_lists(nr_of_tournaments))
        survivors = []
        for pop in tournament_list:

            to = []
            for network in pop:
                to.append(network.get_prediction_score(self.batch_size))
            survivors.append(pop[to.index(max(to))])
        return survivors

    def __generate_tournament_lists(self, nr_of_tournaments):
        lst = self.networks
        # First shuffle the population
        random.shuffle(lst)
        # # Create sublists of size nr_of_tournaments
        subListLength = len(lst) // nr_of_tournaments
        for i in range(0, len(lst), subListLength):
            yield lst[i:i + subListLength]

    def mutate_population(self, pop):
        """
        Mutate the population
        """
        # for network in pop:
        #     network.get_genome()

        for network in self.networks:
            if network != pop:
                network.set_genome(random.choice(pop).get_genome())
                network.mutate_network(self.mutation_rate)
        self.networks.extend(pop)



    def plot_best_network(self, image, ep):
        spike_train = self.best_network.get_output()
        i = 0
        for train in spike_train:
            train = train + i
            plt.plot(train, label='Neuron {}'.format(i))
            i += 1
        plt.title(f'Spike train for the number: {image}, epoch: {ep}')
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # plt.legend()
        plt.show()

    def __str__(self):
        # Print the population
        return f'Population size: {self.size}\n'


if __name__ == "__main__":
    population = Population(nr_inputs=10, nr_hidden=[10], nr_outputs=10, size=10, spike_train_length=10, batch_size=2, leakage=0.1, threshold=0.5)
    population.create_population()
    population.tournament_size=2
    population.update_population(np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), answer=2)
    population.update_population(np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 1]), answer= 1)
    population.evolve_population()

    # print(population)

