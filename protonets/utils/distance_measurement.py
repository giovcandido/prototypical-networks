import torch

# function to calculate the euclidean distance between embeddings.

def euclidean_dist(x, y):
    elements_in_x = x.size(0)
    elements_in_y = y.size(0)

    dimension_elements = x.size(1)

    assert dimension_elements == y.size(1)

    x = x.unsqueeze(1).expand(elements_in_x , elements_in_y, dimension_elements)
    y = y.unsqueeze(0).expand(elements_in_x , elements_in_y, dimension_elements)

    distances = torch.pow(x - y, 2).sum(2)

    return distances