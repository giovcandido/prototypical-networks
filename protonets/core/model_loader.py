from protonets.core.protonet_vanilla import load_protonet
from protonets.core.protonet_random_weights import load_protonet_random_weights

def load_model(model_to_load, *args):
    if model_to_load == 'vanilla':
        return load_protonet(*args)
    elif model_to_load == 'random_weights':
        return load_protonet_random_weights(*args)
