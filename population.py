from net_classes import *
import random
from matplotlib import pyplot as plt
from numba import jit, cuda
class Population():
    def __init__(self,nr_inputs, nr_hidden, nr_outputs, size, spike_train_length, batch_size, leakage=0.1, threshold=0.5, tournament_size=2):
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
        self.best_acc_network = None
        self.best_score = 0.0
        self.best_score_history = []
        self.best_acc_history = []
        self.best_genome = None

    def create_population(self):
        """
        Create a population of networks
        """
        # Create a population of networks
        for i in range(self.size):
            self.networks.append(self.__create_network(i))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        for network in self.networks:
            network.batch_size = batch_size

    def __create_network(self, network_id):
        # Creating input layer
        input_layer = Layer(neurons=[],spike_train_length=self.spike_train_length)
        for i in range(self.nr_inputs):
            # For each input neuron create a neuron
            input_layer.add_neuron(Neuron(id=i, threshold=self.threshold, weight=np.array([1]), leakage=self.leakage,spike_train_length=self.spike_train_length))
        # Update the size of the input layer
        input_layer.input_layer=True
        input_layer.update_output_array()
        # Create a list to store hidden layers
        hidden_layers = []
        # Create the hidden layers
        layer_couter = 0
        first_hidden = True
        for nr_of_neurons in self.nr_hidden:
            if first_hidden:
                first_hidden = False
                nr_of_weights = self.nr_inputs
            else:
                nr_of_weights = self.nr_hidden[layer_couter-1]
            layer_couter += 1
            h_l = (Layer(neurons=[], spike_train_length=self.spike_train_length))
            h_l.input_layer=False
            # weight = np.random.uniform(-1, 1, self.nr_inputs)

            for i in range(nr_of_neurons):
                weight_list = np.array([random.gauss(0, 1) for _ in range(nr_of_weights)])
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
        return Network(id = network_id, layers=[input_layer] + hidden_layers + [output_layer], batch_size=self.batch_size)


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

    # def get_population_prediction(self):
    #     """
    #     Get the prediction of the population
    #     :param answer: The correct answer
    #     :return: The prediction
    #     """
    #     predictions = []
    #     for network in self.networks:
    #         # p = network.get_prediction_score()
    #         predictions.append(network.get_prediction_score(self.batch_size)/self.batch_size)
    #     return predictions

    def evolve_population(self):
        #TODO MAYBE BUG HERE
        predictions = [] # List for storing predictions
        accuracy = [] # List for storing accuracys
        keeper = [] # List for storing survivors
        for network in self.networks:
            # Get predictions and accuracys
            prd, acc = network.get_prediction_score()
            predictions.append(prd)
            accuracy.append(acc)

        # Find index of best score in predictions list

        prd_inx = predictions.index(max(predictions))
        acc_inx = accuracy.index(max(accuracy))

        # If network with higiest predscore is not the same as the one with highest accuarcy
        # Keep both, if not, only copy it once
        if prd_inx != acc_inx:
            self.best_network = self.networks[prd_inx]
            self.best_acc_network = self.networks[acc_inx]
            # keeper.extend([self.best_network, self.best_acc_network])
            # self.networks.remove(self.best_network)
            # self.networks.remove(self.best_acc_network)
        else:
            self.best_network = self.networks[prd_inx]
            self.best_acc_network = self.best_network
            # keeper.append(self.best_network)
            # self.networks.remove(self.best_network)
        self.best_score_history.append(self.best_score)
        self.best_acc_history.append(self.best_acc_network.current_accuracy)
        self.best_score = self.best_network.current_prediction_score
        self.mutate_by_tournament(acc_inx)
        # tournament_winners = self.tournament_selection(self.tournament_size)
        # for network in tournament_winners:
        #     keeper.append(network)
        #     self.networks.remove(network)
        #
        #
        # # keeper.extend(tournament_winners)
        # self.mutate_population(keeper)
        # # self.best_network= keeper[0]
        # self.best_score_history.append(self.best_score)
        # self.best_acc_history.append(self.best_acc_network.current_accuracy)
        # self.best_score = self.best_network.current_prediction_score
        # self.networks.extend(keeper)
        # # print(f' From mutation we got the best {self.best_network} previouse best was {self.best_score_history[-1]}')
        # # self.plot_prediction()

    def plot_prediction(self):
        plt.plot(self.best_score_history+[self.best_score])
        plt.plot([0]+self.best_acc_history)
        plt.title("Prediction score over time")
        plt.ylabel("Prediction score")
        plt.xlabel("Epoch")
        plt.show()

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
                # to.append(network.current_prediction_score)
                # to.append(network.get_acc_x_prediction())
                to.append(network.current_accuracy)
            # to.append(pop.get_prediction_score(self.batch_size))
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
            network.set_genome(random.choice(pop).get_genome())
            network.mutate_network(self.mutation_rate)
        # self.networks.extend(pop)
        pass

    def mutate_by_tournament(self, best_acc):
        # best_genome = best_acc.get_genome()
        population = copy.deepcopy(self.networks)
        best_acc_n = self.networks.pop(best_acc)
        for i in range(len(self.networks)):
            net1 = random.choice(population)
            net2 = random.choice(population)
            if random.uniform(0, 1) > 0.2:
                net3 = random.choice(population)
                net4 = random.choice(population)
                if net1.current_accuracy > net2.current_accuracy:
                    if net3.current_accuracy > net4.current_accuracy:
                        self.networks[i].set_genome(self.crossover(net1.get_genome(), net3.get_genome()))
                    else:
                        self.networks[i].set_genome(self.crossover(net1.get_genome(), net4.get_genome()))
                else:
                    if net3.current_accuracy > net4.current_accuracy:
                        self.networks[i].set_genome(self.crossover(net2.get_genome(), net3.get_genome()))
                    else:
                        self.networks[i].set_genome(self.crossover(net2.get_genome(), net4.get_genome()))


            if net1.current_accuracy > net2.current_accuracy:
                g = net1.get_genome()
                self.genome_mutate(g, self.networks[i])
                # self.networks[i].set_genome(g)
                # self.networks[i].mutate_network(self.mutation_rate)
            else:
                g = net2.get_genome()
                self.genome_mutate(g, self.networks[i])
                # self.networks[i].set_genome(g)
                # self.networks[i].mutate_network(self.mutation_rate)
        self.networks.append(best_acc_n)

    def genome_mutate(self, genome, network):
        """
        Mutate the genome
        :param genome: The genome to mutate
        :return: The mutated genome
        """
        network.set_genome(genome)
        network.mutate_network(self.mutation_rate)

    def crossover(self, genome1, genome2):
        """
        Crossover the genomes
        :param genome1: The first genome
        :param genome2: The second genome
        :return: The crossovered genome
        """
        for i in range(len(genome1)):

            for u in range(len(genome1[i])):
                for x in range(len(genome1[i][u])):
                    if type(genome1[i][u][x]) == np.ndarray:
                        for y in range(len(genome1[i][u][x])):
                            if random.uniform(0, 1) > 0.5:
                                genome1[i][u][x][y] = genome2[i][u][x][y]






        return genome1


    def plot_best_network(self, image, ep):
        spike_train = self.best_acc_network.get_output()
        i = 0
        for train in spike_train:
            train = train + i
            plt.plot(train, label='Neuron {}'.format(i))
            i += 1
        plt.title(f'Spike train for the number: {image}, epoch: {ep}')
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        # plt.legend()
        plt.show()

    def plot_all_networks(self, image, ep):
        for network in self.networks:
            spike_train = network.get_output()
            i = 0
            for train in spike_train:
                train = train + i
                plt.plot(train, label='Neuron {}'.format(i))
                i += 1
            plt.title(f'Spike train for the number: {image}, epoch: {ep}')
            plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            # plt.legend()
            plt.show()
        pass

    def get_best_network(self):
        return self.best_network

    def load_network(self, network, nr_of_networks):
        self.best_network = network
        self.best_acc_network = network
        self.best_score = network.current_prediction_score
        self.best_score_history.append(self.best_score)
        self.best_acc_history.append(self.best_acc_network.current_accuracy)
        self.fill_population(nr_of_networks)

    def fill_population(self, nr_of_networks):
        gen = self.best_network.get_genome()
        for i in range(nr_of_networks):
            self.networks.append(Network(self.input_size, self.hidden_size, self.output_size, self.batch_size))
            self.networks[i].set_genome(gen)


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

