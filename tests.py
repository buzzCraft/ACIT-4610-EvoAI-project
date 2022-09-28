from main import *
# TEST METHODS
def test_neuron():
    threshold=1.5
    n1 = Neuron(1, threshold=threshold, leakage=.05, weight=[1])
    n1.spike = 0.5
    response_list = []
    potential_list = []

    spike_list = [0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0]
    for element in spike_list:
        n1.update(element)
        potential_list.append(n1.potential)
        # potential_list.append(n1.get_potential())
        response_list.append(n1.get_spike())

    plt.plot(response_list, label='Spike')
    plt.plot(spike_list, label='Input')
    plt.plot(potential_list, label='Potential')
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.legend()
    plt.show()



def test_layer():
    inputLayer = Layer()
    for i in range(2):
        inputLayer.neurons.append(Neuron(id=i, threshold=.8, weight=[0.5], leakage=.01))
    # Creating hidden layer
    hiddenLayer = Layer()
    for i in range(1):
        hiddenLayer.neurons.append(Neuron(id=i, threshold=.6, weight=[.5], leakage=.05))

    response_list = []
    potential_list = []
    l1 = []
    l2 = []

    spike_list1 = [0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0]
    spike_list2 = [0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1]
    # spike_list2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(spike_list1)):
        inputLayer.neurons[0].update(spike_list1[i])
        inputLayer.neurons[1].update(spike_list2[i])
        l = inputLayer.neurons[0].get_spike()
        l1.append(l)
        u = inputLayer.neurons[1].get_spike()
        l2.append(u)
        potential_list.append(hiddenLayer.neurons[0].update( u+l ))
        # potential_list.append(n1.get_potential())
        response_list.append(hiddenLayer.neurons[0].get_spike())

    plt.plot(response_list, label='Spike')
    plt.plot(spike_list1, label='Input1')
    plt.plot(spike_list2, label='Input2')
    plt.plot(l1, label='1 pot')
    plt.plot(l2, label='2 pot')
    plt.plot(potential_list, label='Potential')
    plt.axhline(y=0.6, color='r', linestyle='-')
    plt.legend()
    plt.show()


def plot_spike_train(spike_train, title):
    i=0
    for train in spike_train:
        train = train + i
        plt.plot(train, label='Neuron {}'.format(i))
        i +=1
    plt.title(title)
    plt.legend()
    plt.show()

class Tester():
    def __init__(self, x):
        self.x = x


def linespace(start, stop, num):
    return [start + (stop-start)*i/(num-1) for i in range(num)]


if __name__ == '__main__':
    # import numpy as np
    # import copy
    # m = Tester(1)
    # n = copy.deepcopy(m)
    # n.x = 2
    # print(m.x)
    # print(n.x)


    # test_neuron()
    # one = np.array([[0, 0, 1, 0, 0, 1, 0, 1, 0, 1],[1,0,0,1,0,1,0,1,0,1]])
    # plot_spike_train(one, "test")
    #
    # one = np.array([0,0,0,1,0,1,0,1,0,1])
    # x1 = np.array([1,0,1,1,0,1,0,1,0,0])
    # x1 = x1 +1
    # y = np.linspace(1, 10, num=10)
    #
    # plt.title("Line graph")
    # plt.xlabel("X axis")
    # plt.ylabel("Y axis")
    # plt.plot(y,one,  color="red")
    # plt.plot(y,x1,  color="blue")
    # plt.show()

    space = 101
    start = 1
    stop =10
    print(linespace(start, stop, space))



