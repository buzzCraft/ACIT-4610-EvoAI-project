
<div
  <h3 align="center">Spiking Neural Net - with Evolution</h3>

  <p align="center">
    A project for the course ACIT 4610 - Evolutionary AI @ OsloMet

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


This project aims to use evolution to train a spiking neural network. The network is 
written from the bottom up, and is not based on any existing framework. This make
it easier to understand how the network works, and how it can be evolved, but it also
means that it has a lot of limitations. The network is not optimized for speed, and 
this is something that could be explored in the future.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Download the project and install requirements.txt with 
```python
pip install -r requirements.txt
```
The project is written in Python 3.11, and will
run on python 3.6 and above.






<!-- USAGE EXAMPLES -->
## Usage

To train a network, run the following code:
```python
    RESIZE = 2
    # Get total pixels in an image
    nr_pix = int(train_X.shape[1] / RESIZE * train_X.shape[2] / RESIZE)

    spike_train_length = 50
    batch_size = 200
    pop = population.Population(nr_inputs=nr_pix, nr_hidden=[15], nr_outputs=10, size=20,
                                spike_train_length=spike_train_length, batch_size=batch_size, leakage=0.1,
                                threshold=1.5)
    pop.create_population()

    pop.mutation_factor = 0.5
    pop.mutation_rate = 0.05
    pop_eve(spike_train_length, pop, batch_size=batch_size, resize=RESIZE, epochs = 10,  ran=True)
    save_network(pop, filename=f"network_s2_batch_100_ep_10_resized_1_{_}", batch_size=batch_size,
                 spike_train_length=spike_train_length)

```
To load and/or verify the network run the following code
````python
    pop = load_network(filename=f"FILENAME")
    net = pop.get_best_network()
    net.batch_reset()
    net.reset()
    size = len(test_X)
    net.batch_size = size
    for i in tqdm(range(size)):
        spikeTrain = np.array(spikeGen.rateCodingRand2D(test_X[i], T=spike_train_length, resize=RESIZE))
        net.network_update(spikeTrain)
        net.get_prediction(test_y[i])

    print(
        f'The netowrk had a f1 score of {net.get_f1_score()}, a log loss of {net.get_log_loss()}, and an accuracy of {net.calculate_prediction_score()}')
    net.plot_confusion_matrix()
````


<!-- ROADMAP -->
## Roadmap

- [x] Add a working POC
- [x] Add readme
- [x] Add saving and loading features
- [ ] Look into optimizing the network
- [ ] Rewrite so we can use backpropagation


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


