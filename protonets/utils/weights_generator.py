import numpy as np

def generate_random_weights(size):
    upper_bound = size * 10
    
    values_to_pick_from = list(range(1, upper_bound + 1))
    values_to_pick_from = np.array(values_to_pick_from)

    chosen_values = np.random.choice(values_to_pick_from, size, replace = False)
    chosen_values_sum = sum(chosen_values)

    return [(x / chosen_values_sum) for x in chosen_values]