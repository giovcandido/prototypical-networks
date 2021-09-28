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

You also need to install the protonets package with:
```bash
pip3 install -e .
```

You may need to install it with sudo:
```bash
sudo pip3 install -e .
```

After installing the requirements and the package, you're ready to go.

## Usage

You can train two models:
- Prototypical Networks;
- Prototypical Networks with Random Weights.

And there are two available datasets:
- Omniglot;
- mini-ImageNet.

First, you need to go to the __scripts__ directory.

Once you're in this directory, you need to download the datasets.

The dataset_downloader.py script takes a -d/--dataset argument. If you try to execute it without passing the required argument, you should expect to see the following message:
```bash
usage: dataset_downloader.py [-h] -d {all,omniglot,mini_imagenet}
dataset_downloader.py: error: the following arguments are required: -d/--dataset
```

Reading the output above we know that there are three possible choices: all, omniglot and mini_imagenet.

As an example, let's suppose we only want to download omniglot:
```bash
python3 dataset_downloader.py -d omniglot
```

After the download is complete, we can train a model on omniglot.

The training.py script takes two arguments: -m/--model and -d/--dataset. If you run it without passing the required arguments, you should expect to see the following message:
```bash
usage: training.py [-h] -m {vanilla,random_weights} -d {omniglot,mini_imagenet}
training.py: error: the following arguments are required: -m/--model, -d/--dataset
```

Reading the output above we know that both arguments have two possible values. For the first one, these values are: vanilla and random_weights. As for the latter, the values are: omniglot and mini_imagenet.

Since we have downloaded omniglot, let's run:
```bash
python3 training.py -m vanilla -d omniglot
```

After the training is complete, we can retrain by running:
```bash
python3 retraining.py
```

And after retraining, we can evaluate our model with:
```bash
python3 evaluation.py
```

The results are be stored in a directory called __results__.

__Bear in mind__ that you have to rename or delete the results directory before training another model.

The retraining and the evaluation scripts work with the model obtained when you first execute the training script.

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
