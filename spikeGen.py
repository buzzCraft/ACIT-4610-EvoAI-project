import random
import numpy as np
from numpy import interp

# Code obtained from https://medium.com/@rmslick/neural-coding-generating-spike-trains-for-images-using-rate-coding-6bb61afef5d4


# Normalize the input image
def norm2D(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))



def rateCodingDeterministicPix2Spike(pVal, T = 100, freqHigh = 200, freqLow = 10):
    # Decide the length of the spike train
    T  # ms
    freqHigh  # Spike/sec
    freqLow # Spike/sec
    '''
        1. Calculated the low and high frequencies using the retinal firing rates (10,200)
    '''
    ffs = freqHigh * T * 1/1000 # full frequency state
    lfs = freqLow * T * 1/1000  # low frequency state
    '''
        2. Interpolate the input pixel intensity value using the retinal firing rates as points
    '''
    f_det = interp(pVal, [0,lfs], [1,ffs]) # deterministic frequency
    '''
        3. Generate the spike emission interval
    '''
    spike_emission_interval = int(T/f_det) # spike emission interval
    '''
        4. Use the interval to fill out a spike train of size T (100 in our case)
    '''
    spike_train = np.asarray([1 if i % spike_emission_interval == 0 else 0 for i in range(1,T+1)]) # Generate a spike at each spike_emission_interval count
    return (spike_train)

# Generate a spike train for the input image
def rateCodingRand2D(image, T = 100, freqHigh = 200, freqLow = 10, resize = 0):
    """
    :param image: 2D numpy array
    :param T: length of spike train
    :param freqHigh: high frequency
    :param freqLow: low frequency
    :param resize: resize the image to a smaller size 2 -> Reduce the size by 2
    :return: 2D numpy array
    """
    if resize:
        image = (image[::resize, ::resize])
    image = norm2D(image) # Normalize the image
    spikeTrain = []
    for row in image:
        for pix in row:
            spikeTrain.append(rateCodingDeterministicPix2Spike(pix, T, freqHigh, freqLow))
    return(np.asarray(spikeTrain))

def img_setup(image, T=100, resize=0):
    # Function to setup the image for the spike train
    # Spike for each row
    if resize:
        image = (image[::resize, ::resize])
    image = norm2D(image) # Normalize the image
    spikeTrain = []
    return np.asarray(row_spike(image, T))

def row_spike(image, T=100):
    bit_pr_pix = int(T/len(image[0]))
    spike_train = []
    for row in image:
        row_l = []
        for pix in row:
            for i in range(bit_pr_pix):
                if pix > random.uniform(0, 1):
                    row_l.append(1)
                else:
                    row_l.append(0)
        for i in range(len(row_l), T):
            row_l.append(0)
        spike_train.append(row_l)
    return spike_train

if __name__ == '__main__':
    from keras.datasets import mnist

    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    out = img_setup(train_X[0], resize=2)
    print(out)
