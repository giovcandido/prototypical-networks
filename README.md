# Prototypical Networks Project

Vanilla and Prototypical Networks with Random Weights for image classification on Omniglot and mini-ImageNet.

It's made with Python3 and tested on Linux.

## Installation

Clone the repository or download the compressed source code. If you opted for the latter, you need to extract the source code to a desired directory.

In both cases, open the project directory in your terminal.

Now, install the requirements. You can achieve that by running:
```bash
pip3 install -r requirements.txt
```

In case you can't install the requirements as a user, run the following instead:
```bash
sudo pip3 install -r requirements.txt
```

With all dependencies properly installed, you're ready to go.

## Usage

You can train two models:
- Prototypical Networks;
- Prototypical Networks with Random Weights.

And there are two available datasets:
- Omniglot;
- mini-ImageNet.

All you need to do is to run one of the following commands:
```bash
sh exec_vanilla_omniglot.sh
```

```bash
sh exec_vanilla_mini_imagenet.sh
```

```bash
sh exec_random_weights_omniglot.sh
```

```bash
sh exec_random_weights_mini_imagenet.sh
```

When you run one of the scripts above, the required dataset is automatically downloaded for you and the training begins. The results are be stored in a directory called "results". Bear in mind that you have to rename or delete the results directory before training again.

## Few-Shot Setup

You can find the few-shot setup and other parameters in the config directory.

The splits and the implementation follow the procedure of [Prototypical Networks For Few-shot Learning](https://arxiv.org/abs/1703.05175).

## Results

The results obtained with this implementation are comparable to those obtained with the original one.

You can check my execution logs and trained models [here](https://drive.google.com/drive/folders/1O4RR72X0fOBeNdC-g23IwQSri4O1Sm87?usp=sharing).

## Acknowledgements

This project was based on:
- [Cyprien Nielly](https://github.com/cnielly/prototypical-networks-omniglot) implementation of Prototypical Networks.
- The original implementation, which can be found in [Jake Snell's Github](https://github.com/jakesnell/prototypical-networks).

The idea of PNs can be originally found in [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175).

It's worth mentioning that using weights in order to calculate the prototypes is an idea that can be found in [Improved Prototypical Networks for Few-Shot Learning](https://www.sciencedirect.com/science/article/abs/pii/S0167865520302610).
