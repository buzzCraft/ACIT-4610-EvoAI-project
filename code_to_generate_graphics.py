import gzip
import pickle
import random
from matplotlib import pyplot as plt

f = gzip.open('mnist.pkl.gz', 'rb')
data = pickle.load(f, encoding='bytes')
f.close()
(train_X, train_y), (test_X, test_y) = data
print('Train: X=%s, y=%s' % (train_X.shape, train_y.shape))
# print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
import spikeGen as sg
for i in range(3):
    # define subplot
    im = random.randint(0, len(train_X))
    plt.subplot(330 + 1 + i)
    # plot raw pixel data

    plt.imshow(sg.nd_resize(train_X[im], (28,28)), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 3+i)
    plt.imshow(sg.nd_resize(train_X[im], (14, 14)), cmap=plt.get_cmap('gray'))
    plt.subplot(330 + 1 + 6 + i)
    plt.imshow(sg.nd_resize(train_X[im], (7, 7)), cmap=plt.get_cmap('gray'))
# show the figure
plt.show()