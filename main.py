from keras.datasets import mnist
from matplotlib import pyplot as plt
import pickle
import spikeGen
import population
import numpy as np
from tqdm import tqdm
import random



# Image set
(train_X, train_y), (test_X, test_y) = mnist.load_data()

def plot_spike_train(spike_train, title):
    i = 0
    for train in spike_train:
        train = train + i
        plt.plot(train, label='Neuron {}'.format(i))
        i += 1
    plt.title(title)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # plt.legend()
    plt.show()

def pop_eve(spike_train_length, population, batch_size=10, resize = 0, ran = False):
    population.set_batch_size(batch_size)
    for ep in tqdm(range(10)):
        print(f'\nEpoch {ep}')
        population.batch_reset()


        # Images to train on
        # for p in range(batch_size*ep, batch_size*ep+batch_size):
        for p in tqdm(range(batch_size)):
            # print(f'Image {p}')
            # Reset spike train history for all networks
            population.reset_population()

            # start = time.time()
            # TODO Generate all spike trains at once and store them in a text file ( not sure if needed )
            # Then read it in when needed
            # Genereating a spike train for image p
            if ran:
                pick = random.randint(0, len(train_X)-1)
            else:
                pick = p
            spikeTrain = np.array(spikeGen.rateCodingRand2D(train_X[pick], T=spike_train_length, resize=resize))
            # spikeTrain = np.array(spikeGen.img_setup(train_X[pick], T=spike_train_length, resize=resize))

            # Update the networks with the spike train
            population.update_population(spikeTrain, train_y[pick])
            # end = time.time()
            # delta = end - start
            # print(f"Image {p} took {delta} seconds to process")

        # Select the best networks
        population.evolve_population()
        population.plot_best_network(train_y[pick], ep)
        population.plot_prediction()
        # population.batch_reset()
        # population.plot_all_networks(train_y[p],ep)
        # print(population)


def save_network(network, filename="network", batch_size=10, spike_train_length=10):
    with open(filename+'.plk', 'wb') as f:
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

    with open(filename+'.txt', 'w') as f:
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
    import time
    resize = 2

    # Get total pixels in an image
    nr_pix = int(train_X.shape[1]/resize * train_X.shape[2]/resize)
    # nr_pix = int(train_X.shape[1] / resize)

    spike_train_length = 50

    # Nr_input neurons -> number of pixels in an image
    # pop = init(nr_input=nr_pix, nr_output=10, nr_hidden=10, number_of_networks=20, train_length=spike_train_length)
    # evolve3(spike_train_length)
    # initialize(nr_input=nr_pix, nr_hidden=20, nr_output=10, threshold=5, number_of_networks=20, leakage=0.05)
    # evolve2()

    # Bruker en ny måte å regne på.. Ikke implementert evolusjon enda
    # n = init2(nr_input=nr_pix, nr_hidden=20, nr_output=10, threshold=5, number_of_networks=20, leakage=0.05, train_length=spike_train_length)
    # evolve3(spike_train_length,n)
    batch_size = 25
    pop = population.Population(nr_inputs=nr_pix, nr_hidden=[20], nr_outputs=10, size=25,
                                spike_train_length=spike_train_length, batch_size=batch_size, leakage=0.1, threshold=1.2, tournament_size=5)
    pop.create_population()
    # pop.mutation_rate = 0.5
    # pop_eve(spike_train_length, pop, batch_size=batch_size, resize=resize, ran = True)
    #
    #
    # pop.plot_best_network(test_y[9],1)
    # # pop.mutation_rate = 0.7
    # # pop_eve(spike_train_length, pop, batch_size=batch_size, resize=resize, ran = True)
    # # pop.mutation_rate = 0.5
    # # pop_eve(spike_train_length, pop, batch_size=batch_size, resize=resize, ran = True)
    # save_network(pop, filename="network_s_l_100_batch_100_ep_10_resized_2", batch_size=batch_size, spike_train_length=spike_train_length)
    # loaded_pop = load_network(filename="network_s_batch_100_ep_10_resized_4_9.plk")
    pop.weight_mutate_rate=0.05
    loaded_pop = pop
    for _ in range(10):
        # loaded_pop = load_network(filename="network_s_l_100_batch_100_ep_10_resized_2.plk")
        loaded_pop.mutation_rate = 0.1
        pop_eve(spike_train_length, loaded_pop, batch_size=batch_size, resize=resize, ran = False)
        save_network(loaded_pop, filename=f"network_s_batch_100_ep_10_resized_5_{_}", batch_size=batch_size,
                     spike_train_length=spike_train_length)


# TODO Look into implementing this: https://stackoverflow.com/questions/43689829/how-to-use-multiprocess-in-python-on-a-class-object
# during network creation - Spread the networks across multiple cores


#TODO Look at auto encoder for MNIST
