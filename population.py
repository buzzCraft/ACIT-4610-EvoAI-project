from net_classes import *
import random
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
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
        self.weight_mutate_rate = 0.1
        self.weight_mutate_rate_init = self.weight_mutate_rate

        ####
        self.genomes = []
        self.fitness = []
        self.best_network = None
        self.best_acc_network = None
        self.best_score = 0.0
        self.best_score_history = []
        self.best_acc_history = []
        self.best_genome = None
        self.f1_score_history = []
        self.log_loss_history = []

        self.helper_plot = [[],[],[]]

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

            for i in range(nr_of_neurons):
                weight_list = np.array([random.gauss(0, 1.5) for _ in range(nr_of_weights)])
                h_l.add_neuron(
                    Neuron(id=i, threshold=self.threshold, weight=weight_list, leakage=self.leakage, spike_train_length=self.spike_train_length))
            hidden_layers.append(h_l)
            h_l.update_output_array()

        output_layer = Layer(neurons=[], spike_train_length=self.spike_train_length)
        for i in range(self.nr_outputs):
            weight_list = np.array([random.gauss(0, 1.5) for _ in range(self.nr_hidden[-1])])
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

    def batch_reset(self):
        # Reset the scoring of the networks between each batch
        for network in self.networks:
            network.batch_reset()

    def get_population_output(self):
        # Get the output of the population
        output = []
        for network in self.networks:
            output.append(network.get_output())
        return output

    def get_best_scores(self):
        # Find best current_log_loss
        newlist = sorted(self.networks, key=lambda x: x.current_log_loss, reverse=False)
        best_log = newlist[0].current_log_loss
        newlist = sorted(self.networks, key=lambda x: x.current_f1_score, reverse=True)
        best_f1 = newlist[0].current_f1_score
        newlist = sorted(self.networks, key=lambda x: x.current_accuracy, reverse=True)
        best_acc = newlist[0].current_accuracy
        return best_log, best_f1, best_acc

    def evolve_population(self, crossover = True, mutation = True):

        # Get the fitness of the population
        for network in self.networks:
            network.calculate_prediction_score()

        l,f,a = self.get_best_scores()
        self.helper_plot[0].append(l)
        self.helper_plot[1].append(f)
        self.helper_plot[2].append(a)
        # self.plot_overall_best()
        # Sort the networks by their prediction score

        self.networks = sorted(self.networks, key=lambda x: x.current_f1_score, reverse=True)
        # self.networks = sorted(self.networks, key=lambda x: x.current_log_loss, reverse=False)
        # Get the best network and append score for plotting
        self.best_acc_network = self.networks[0]
        self.best_score = self.best_acc_network.current_prediction_score
        self.best_score_history.append(self.best_score)
        self.best_acc_history.append(self.best_acc_network.current_accuracy)
        self.f1_score_history.append(self.best_acc_network.current_f1_score)
        self.log_loss_history.append(self.best_acc_network.get_log_loss())


        # Get the best networks
        # TODO Legg til en sannsynlighet for å velge de beste

        best_networks = copy.deepcopy(self.networks[:3])
        # newlist = sorted(self.networks[3:], key=lambda x: x.current_log_loss, reverse=False)
        # best_networks.extend(copy.deepcopy(newlist[0:3]))
        newlist = sorted(self.networks[3:], key=lambda x: x.current_accuracy, reverse=True)
        best_networks.extend(copy.deepcopy(newlist[:3]))
        # Pick n random networks from self.networks
        # best_networks.extend(copy.deepcopy(random.sample(self.networks[3:], 3)))

        for network in best_networks:
            print(network.get_f1_score())

        if crossover:
            net_copy = copy.deepcopy(self.networks)
            for i in range(int(len(self.networks)/2)):

                if random.uniform(0, 1) > 0.2:


                    net1 = random.choice(net_copy)
                    net2 = random.choice(net_copy)
                    net3 = random.choice(net_copy)
                    net4 = random.choice(net_copy)
                    # Get the best of net1 and net2
                    if net1.current_f1_score < net2.current_f1_score:
                        net1 = net2
                    # Get the best of net3 and net4
                    if net3.current_accuracy < net4.current_accuracy:
                        net2 = net4
                    else:
                        net2 = net3

                    # Do crossover on net1 and net2
                    net1, net2 = self.crossover(net1.get_genome(), net2.get_genome())
                    self.networks[i].set_genome(net1)
                    self.networks[-i].set_genome(net2)
        if mutation:
            # Force evolution when network is stuck
            if len(self.best_score_history) > 5:
                # If stuck, add 0.1 to mutation rate
                if self.best_score == self.best_score_history[-2]:

                    self.weight_mutate_rate += np.random.normal(0, 0.2)
                    # if self.best_score == self.best_score_history[-5]:
                    #     # Remove 50% of the population
                    #     self.networks = self.networks[:int(len(self.networks)/2)]
                    #     # Add 50% new networks
                    #     self.create_population(int(len(self.networks)/2))
                    #     # Reset the population
                    #     self.reset_population()

                # If we got an improvement, go back to original mutation rate
                else:
                    self.weight_mutate_rate = self.weight_mutate_rate_init

            for net in self.networks:
                # if random.uniform(0, 1) < 0.5:
                net.mutate_network(self.mutation_rate, self.weight_mutate_rate)
            self.networks.remove(random.choice(self.networks))
            self.networks.append(best_networks[0])
            print(best_networks[0])







    # def tournament_selection(self, nr_of_tournaments):
    #     """
    #     Tournament selection
    #     :return: The survivors of the tournament
    #     """
    #     # First shuffle the population
    #
    #     # Create sublists of size nr_of_tournaments
    #     tournament_list = list(self.__generate_tournament_lists(nr_of_tournaments))
    #     survivors = []
    #     for pop in tournament_list:
    #
    #         to = []
    #         for network in pop:
    #             # to.append(network.current_prediction_score)
    #             # to.append(network.get_acc_x_prediction())
    #             to.append(network.current_accuracy)
    #         # to.append(pop.get_prediction_score(self.batch_size))
    #         survivors.append(pop[to.index(max(to))])
    #     return survivors

    # def __generate_tournament_lists(self, nr_of_tournaments):
    #     lst = self.networks
    #     # First shuffle the population
    #     random.shuffle(lst)
    #     # # Create sublists of size nr_of_tournaments
    #     subListLength = len(lst) // nr_of_tournaments
    #     for i in range(0, len(lst), subListLength):
    #         yield lst[i:i + subListLength]

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

    # def mutate_by_tournament(self, best_acc):
    #     # best_genome = best_acc.get_genome()
    #     population = copy.deepcopy(self.networks)
    #     _ = self.networks.pop(best_acc)
    #     best_acc_n = population.pop(best_acc)
    #     for i in range(int(len(self.networks)/2)):
    #         net1 = random.choice(population)
    #         net2 = random.choice(population)
    #         if random.uniform(0, 1) > 0.2:
    #             net3 = random.choice(population)
    #             net4 = random.choice(population)
    #             # Get the best of net1 and net2
    #             if net1.current_accuracy > net2.current_accuracy:
    #                 net1 = net1
    #             else:
    #                 net1 = net2
    #             # Get the best of net3 and net4
    #             if net3.current_accuracy > net4.current_accuracy:
    #                 net2 = net3
    #             else:
    #                 net2 = net4
    #
    #             # Do crossover on net1 and net2
    #             net1, net2 = self.crossover(net1.get_genome(), net2.get_genome())
    #             self.networks[i].set_genome(net1)
    #             self.networks[-i].set_genome(net2)
    #
    #
    #
    #     for net in self.networks:
    #         if random.uniform(0, 1) < 0.8:
    #             net.mutate_network(self.mutation_rate)
    #     self.networks.append(self.best_acc_network)
    def new_mutate(self):
        self.networks.sort()
        for net in self.networks:
            print(net)
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
        # for i in range(len(genome1)):
        for i in range(1,2):
            for u in range(len(genome1[i])):
                for x in range(len(genome1[i][u])):
                    if type(genome1[i][u][x]) == np.ndarray:
                        for y in range(len(genome1[i][u][x])):
                            if random.uniform(0, 1) > 0.5:
                                tmp = genome1[i][u][x][y]
                                genome1[i][u][x][y] = genome2[i][u][x][y]
                                genome2[i][u][x][y] = tmp
        return genome1, genome2


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
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):

        cm = confusion_matrix(self.best_acc_network.batch_target, self.best_acc_network.confusion_predicted)
        vmin = np.min(cm)
        vmax = np.max(cm)
        off_diag_mask = np.eye(*cm.shape, dtype=bool)

        fig = plt.figure()
        sns.heatmap(cm, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
        sns.heatmap(cm, annot=True, mask=off_diag_mask,vmin=vmin, vmax=vmax, cmap='Reds', cbar=False,
                    cbar_kws=dict(ticks=[]))
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    def plot_prediction(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Network with best prediction')
        ax1.plot(self.best_score_history, label='Best score')
        ax1.plot(self.best_acc_history, label='Best accuracy')
        ax1.plot(self.f1_score_history, label='F1 score')
        ax2.plot(self.log_loss_history, label='Log loss')
        ax1.legend()
        ax2.legend()
        plt.show()
    def plot_overall_best(self):
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Overall best scores across networks')
        ax1.plot(self.helper_plot[1], label='F1-score')
        ax1.plot(self.helper_plot[2], label='Accuracy')
        ax2.plot(self.helper_plot[0], label='Loss')
        ax1.legend()
        ax2.legend()
        plt.show()

    def plot_combo(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('Network with best prediction')
        # ax1.plot(self.best_score_history, label='Best score')

        ax1.plot(self.f1_score_history, label='F1 score')
        ax1.plot(self.best_acc_history, label='Best accuracy')
        ax2.plot(self.log_loss_history, label='Log loss')
        ax1.legend()
        ax2.legend()
        ax3.plot(self.helper_plot[1], label='F1-score')
        ax3.plot(self.helper_plot[2], label='Accuracy')
        ax4.plot(self.helper_plot[0], label='Loss')
        ax3.legend()
        ax4.legend()
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

