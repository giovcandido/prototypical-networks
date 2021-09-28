from warnings import filterwarnings
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from protonets.utils.distance_measurement import euclidean_dist

# ignore pytorch boring warnings
filterwarnings("ignore", category=UserWarning)

# the model below computes a class prototype by averaging the support embeddings of a class

# check if there is cuda available
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(dev)

    def set_forward_loss(self, episode_dict):
        # extract all images
        images = episode_dict['images'].to(dev)

        # get episode setup
        num_way = episode_dict['num_way'] # way
        num_shot = episode_dict['num_shot'] # shot
        num_query = episode_dict['num_query'] # number of query images

        # from each class, extract num_shot support images
        x_support = images[:, :num_shot] # lines are classes and columns are images

        # from each class, extract the remaining images as query images
        x_query = images[:, num_shot:] # lines are classes and columns are images

        # create indices from 0 to num_way-1 for classification
        target_inds = torch.arange(0, num_way).view(num_way, 1, 1)

        # replicate all indices num_query times (for each query image)
        target_inds = target_inds.expand(num_way, num_query, 1).long()

        # convert indices from Tensor to Variable
        target_inds = Variable(target_inds, requires_grad = False).to(dev)

        # transform x_support into a array in which all images are contiguous
        x_support = x_support.contiguous().view(
            num_way * num_shot, *x_support.size()[2:]) # no more lines and columns

        # transform x_query into a array in which all images are contiguous
        x_query = x_query.contiguous().view(
            num_way * num_query, *x_query.size()[2:]) # no more lines and columns

        # join all images into a single contiguous array
        x = torch.cat([x_support, x_query], 0)

        # encode all images
        z = self.encoder.forward(x) # embeddings

        # compute class prototypes
        z_dim = z.size(-1)
        z_proto = z[:(num_way * num_shot)].view(num_way, num_shot, z_dim).mean(1)

        # get the query embeddings
        z_query = z[(num_way * num_shot):]

        # compute distance between query embeddings and class prototypes
        dists = euclidean_dist(z_query, z_proto)

        # compute the log probabilities
        log_p_y = F.log_softmax(-dists, dim = 1).view(num_way, num_query, -1)

        # compute the loss
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

        # get the predicted labels for each query
        _, y_hat = log_p_y.max(2) # lines are classes and columns are query embeddings

        # compute the accuracy
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        # return output: loss, acc and predicted value
        return loss_val, {
            'loss': loss_val.item(), 'acc': acc_val.item(), 'y_hat': y_hat}


# function to load the model structure
def load_protonet(x_dim, hid_dim, z_dim):
    # define a convolutional block
    def conv_block(layer_input, layer_output):
        conv = nn.Sequential(
            nn.Conv2d(layer_input, layer_output, 3, padding=1),
            nn.BatchNorm2d(layer_output), nn.ReLU(),
            nn.MaxPool2d(2))

        return conv

    # create the encoder to the embeddings for the images
    # the encoder is made of four conv blocks
    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim), conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim), conv_block(hid_dim, z_dim), Flatten())

    return ProtoNet(encoder)
