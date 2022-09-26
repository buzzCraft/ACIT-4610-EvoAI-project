from main import *
# TEST METHODS
def test_neuron():
    threshold=1.5
    n1 = Neuron(1, threshold=threshold, leakage=.1, weight=[.79])
    response_list = []
    potential_list = []

    spike_list = [0,0,1,1,0,0,1,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0]
    for element in spike_list:
        potential_list.append(n1.update(element))
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

if __name__ == '__main__':
    # test_neuron()
    # test_layer()
    #
    # # Spike train testing
    # spikeTrain = spikeGen.rateCodingRand2D(train_X[0], T=15)
    # print("done")
    import plotly.express as px


    # some numpy testing
    spike = 100 * np.random.random(100)
    arr = np.array([[1,0,0],[1,1,0],[0,1,9]])
    print(arr)
    num_rows, num_cols = arr.shape
    for i in range(num_cols):
        print(arr[:,i])
    # FUNKER IKEK!!
    one = [[0,0,0,1,0,1,0,1,0,1], [1,0,1,1,0,1,0,1,0,0]]
    y = np.linspace(1, 10, num=10)
    x = [w*s for w,s in zip(one[0], y)]
    x2 =  [w*s for w,s in zip(one[1], y)]
    one = [x,x2]
    lineoffsets1 = np.array([0, 1])

    linelengths1 = [0.5]

    fig, axs = plt.subplots()
    axs.eventplot(one,
                  lineoffsets=lineoffsets1,
                  linelengths=linelengths1)

    plt.eventplot(x)
    plt.show()