import random
import numpy as np
from numpy import interp



# Normalize the input image
def norm2D(image):
    return (image - np.min(image)) / (np.max(image) - np.min(image))



def rateCodingDeterministicPix2Spike(pVal, T = 100, freqHigh = 200, freqLow = 10):
    # Code obtained from https://medium.com/@rmslick/neural-coding-generating-spike-trains-for-images-using-rate-coding-6bb61afef5d4

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
    if resize > 1:
        imgesize = image.shape
        r = int(imgesize[0]/resize)
        image = nd_resize(image, new_shape=(r,r), operation='sum')
    image = norm2D(image) # Normalize the image
    spikeTrain = []
    for row in image:
        for pix in row:
            spikeTrain.append(rateCodingDeterministicPix2Spike(pix, T, freqHigh, freqLow))
    return(np.asarray(spikeTrain))

def nd_resize(ndarray, new_shape, operation='sum'):
    """
    Taken from https://stackoverflow.com/a/29042041/13484350
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,
                                                  ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1*(i+1))
    return ndarray



def row_spike(image, T=100):
    bit_pr_pix = int(T/len(image[0]))
    spike_train = []
    for row in image:
        row_l = []
        for pix in row:
            for i in range(bit_pr_pix):
                if pix > (1-(i/bit_pr_pix)):
                    row_l.append(1)
                else:
                    row_l.append(0)


        for i in range(len(row_l), T):
            row_l.append(0)
        spike_train.append(row_l)
    return spike_train

if __name__ == '__main__':
    import gzip
    import pickle
    f = gzip.open('mnist.pkl.gz', 'rb')
    data = pickle.load(f, encoding='bytes')
    f.close()
    (train_X, train_y), (test_X, test_y) = data
    out = img_setup(train_X[0], resize=2)
    print(out)
