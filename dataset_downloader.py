import shutil
import os
import gdown
import argparse

import utils.new_os_functions as new_os

from utils.yaml_loader import load_yaml
from utils.arguments_parser import parse_dataset

# function to download omniglot dataset

def download_omniglot(parent_dir):
    omniglot_dir = os.path.join(parent_dir, 'omniglot')

    # create directory to store omniglot related files
    new_os.mkdir_if_not_exist(omniglot_dir)

    # make sure you have gdown: pip3 install gdown
    download_url = 'https://drive.google.com/u/0/uc?id=1ny3lCPETCLbcjSQHPc3aANUcNR82Aht5&export=download'
    output_file = os.path.join(omniglot_dir, 'omniglot.tar.gz')

    # download dataset
    gdown.download(download_url, output_file, quiet=False)

    # extract the images
    print('Unpacking dataset...')

    shutil.unpack_archive(output_file, omniglot_dir)

    # remove compressed file
    os.remove(output_file)


# function to download mini_imagenet dataset

def download_mini_imagenet(parent_dir):
    mini_imagenet_dir = os.path.join(parent_dir, 'mini_imagenet')

    # create directory to store mini_imagenet related files
    new_os.mkdir_if_not_exist(mini_imagenet_dir)

    # make sure you have gdown: pip3 install gdown
    download_url = 'https://drive.google.com/uc?id=1GUDPuoH3JfbGR078vsuF5UFHuJTGXGFb&export=download'
    output_file = os.path.join(mini_imagenet_dir, 'miniImageNet.tar.gz')

    # download dataset
    gdown.download(download_url, output_file, quiet=False)

    # extract the images
    print('Unpacking...')

    shutil.unpack_archive(output_file, mini_imagenet_dir)

    # remove compressed file
    os.remove(output_file)

    # rename pkl files
    new_os.rename_file(mini_imagenet_dir, 'mini-imagenet-cache-train.pkl', 'train.pkl')
    new_os.rename_file(mini_imagenet_dir, 'mini-imagenet-cache-val.pkl', 'valid.pkl')
    new_os.rename_file(mini_imagenet_dir, 'mini-imagenet-cache-test.pkl', 'test.pkl')



# here, we download the dataset

# let's capture the chosen dataset
args = parse_dataset()

dataset = args.dataset

# open config file to get data directory name
config_file = os.path.join('config', 'config.yaml')

directories = load_yaml(config_file)['directories']

# if not exists, create the data directory to store the datasets
data_dir = directories['data_dir']

new_os.mkdir_if_not_exist(data_dir)

# download the chosen dataset
if dataset == 'all':
    download_omniglot(data_dir)
    download_mini_imagenet(data_dir)
elif dataset == 'omniglot':
    download_omniglot(data_dir)
elif dataset == 'mini_imagenet':
    download_mini_imagenet(data_dir)