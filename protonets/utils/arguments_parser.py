import argparse

def parse_dataset():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-d', '--dataset', choices=['all', 'omniglot', 'mini_imagenet'], 
        help="choose the dataset", required=True)

    return parser.parse_args()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-m', '--model', choices=['vanilla', 'random_weights'],
        help="choose the model", required=True)

    parser.add_argument(
        '-d', '--dataset', choices=['omniglot', 'mini_imagenet'], 
        help="choose the dataset", required=True)

    return parser.parse_args()