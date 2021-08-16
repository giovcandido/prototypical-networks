# vanilla-rw-protonets-project
Vanilla Prototypical Networks and PNs with Random Weights for image classification on Omniglot and mini-ImageNet. Made with Python3.

# How to use
First, you need to install the dependencies. You can achieve that by running: 
* sh install_deps.sh

With all deps properly installed, you're ready to go.

You can train two models: Prototypical Networks and Prototypical Networks with Random Weights.

And there are two available datasets: Omniglot and mini-ImageNet.

All you need to do is to run one of the following commands:
* sh exec_vanilla_omniglot.sh
* sh exec_vanilla_mini_imagenet.sh
* sh exec_random_weights_omniglot.sh
* sh exec_random_weights_mini_imagenet.sh

The directories with the prefix "results_" contain my trained models and the execution logs.

When you run one of the scripts above, the required dataset is automatically downloaded for you and the training begins. The results are be stored in a directory called "results". Bear in mind that you have to rename or delete the results directory before training again.

# About few-shot setup and splits
You can find the few-shot setup and other parameters in the config directory.

The splits and the implementation follow the procedure of [Prototypical Networks For Few-shot Learning](https://arxiv.org/abs/1703.05175).

The code was tested with Python3 in Debian 10.

# About results
The results obtained with this implementation are comparable to those obtained with the original one.

You can see it for yourself.

# Acknowledgements
This project was based on:
* [Cyprien Nielly](https://github.com/cnielly/prototypical-networks-omniglot) implementation of Prototypical Networks.

* The original implementation, which can be found in [Jake Snell's Github](https://github.com/jakesnell/prototypical-networks).

The idea of PNs can be originally found in [Prototypical Networks For Few-shot Learning](https://arxiv.org/abs/1703.05175).

It's worth mentioning that using weights in order to calculate the prototypes is an idea that can be found in [Improved prototypical networks for few-Shot learning](https://www.sciencedirect.com/science/article/abs/pii/S0167865520302610).
