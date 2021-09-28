from os import path
import yaml

def load_yaml(filename):
    with open(filename) as f:
        config = yaml.safe_load(f)

    return config
