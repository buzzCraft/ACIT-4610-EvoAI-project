from matplotlib import pyplot as plt
import spikeGen
import population
import numpy as np
from tqdm import tqdm
import random
import sys
import pickle
import gzip

# Import the mist set
f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f, encoding='bytes')
f.close()
(train_X, train_y), (test_X, test_y) = data
# # Only keep images with y label in label_list
# label_list=[0,1,8]
# train_X = np.array([train_X[i] for i in range(len(train_X)) if train_y[i] in label_list])
# train_y = np.array([train_y[i] for i in range(len(train_y)) if train_y[i] in label_list])
# test_X = np.array([test_X[i] for i in range(len(test_X)) if test_y[i] in label_list])
# test_y = np.array([test_y[i] for i in range(len(test_y)) if test_y[i] in label_list])


def plot_spike_train(spike_train, title):
    i = 0
    for train in spike_train:
        train = train + i
        plt.plot(train, label='Neuron {}'.format(i))
        i += 1
    plt.title(title)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.show()


def pop_eve(spike_train_length, population, batch_size=10, resize=0, ran=False):
    # set current batch size
    population.set_batch_size(batch_size)
    for ep in range(10):
        print(f'\nEpoch {ep}')
        # Reset the population for a new batch
        population.batch_reset()
        for p in tqdm(range(batch_size)):
            # Reset spike train history for all networks
            population.reset_population()

            # Genereating a spike train for image p
            if ran:
                pick = random.randint(0, len(train_X) - 1)
            else:
                pick = p
            spikeTrain = np.array(spikeGen.rateCodingRand2D(train_X[pick], T=spike_train_length, resize=resize))

            # Update the networks with the spike train
            population.update_population(spikeTrain, train_y[pick])

        # Select and plot the best networks

        population.evolve_population()
        population.plot_best_network(train_y[pick], ep)
        population.plot_combo()







def save_network(network, filename="network", batch_size=10, spike_train_length=10):
    with open(filename + '.plk', 'wb') as f:
        pickle.dump(network, f)
    """
    Implement a textfile to go along with the pickle file
    Store the following information:
    - Number of inputs
    - Number of hidden neurons
    - Number of outputs
    - Best score
    - Number of epochs trained
    - Date / time
    """

    with open(filename + '.txt', 'w') as f:
        f.write(f'Network: {filename}\n')
        f.write(f'{network}\n')
        f.write(f'Trained on {batch_size} images\n')
        f.write(f'With a spike train of length {spike_train_length}\n')
    print("Network saved as " + filename + ".pkl")


def load_network(filename="network.pkl"):
    """
    Usage: network = load_network('network.pkl')
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    print(len(train_X))
    resize = 2

    # Get total pixels in an image
    nr_pix = int(train_X.shape[1] / resize * train_X.shape[2] / resize)

    spike_train_length = 50
    batch_size = 40
    pop = population.Population(nr_inputs=nr_pix, nr_hidden=[15], nr_outputs=10, size=15,
                                spike_train_length=spike_train_length, batch_size=batch_size, leakage=0.1,
                                threshold=1.5)
    pop.create_population()

    pop.weight_mutate_rate = 0.5
    loaded_pop = pop
    # loaded_pop = load_network('network_s_batch_100_ep_10_resized_5_5.plk')
    # network_s_batch_100_ep_10_resized_5_5.pkl
    for _ in range(10):
        loaded_pop.mutation_rate = 0.05
        pop_eve(spike_train_length, loaded_pop, batch_size=batch_size, resize=resize, ran=True)
        save_network(loaded_pop, filename=f"network_s2_batch_100_ep_10_resized_1_{_}", batch_size=batch_size,
                     spike_train_length=spike_train_length)

    net = loaded_pop.get_best_network()
    net.batch_reset()
    net.reset()
    size = len(test_X)
    net.batch_size = size
    for i in tqdm(range(size)):
        spikeTrain = np.array(spikeGen.rateCodingRand2D(test_X[i], T=spike_train_length, resize=resize))
        net.network_update(spikeTrain)
        net.get_prediction(test_y[i])

    print(
        f'The netowrk had a f1 score of {net.get_f1_score()}, a log loss of {net.get_log_loss()}, and an accuracy of {net.calculate_prediction_score()}')
    net.plot_confusion_matrix()