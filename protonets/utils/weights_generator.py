import numpy as np

def generate_random_weights(size):
    values = np.random.uniform(size=size)
    values_sum = sum(values)

    return np.array([(x / values_sum) for x in values])
