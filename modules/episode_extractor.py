import numpy as np
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

# function to generate one random episode.

def extract_episode(img_set_x, img_set_y, num_way, num_shot, num_query):
    # get a list of all unique labels (no repetition)
    unique_labels = np.unique(img_set_y)

    # select num_way classes randomly without replacement
    chosen_labels = np.random.choice(unique_labels, num_way, replace = False)
    # number of examples per selected class (label)
    examples_per_label = num_shot + num_query

    # list to store the episode
    episode = []

    # iterate over all selected labels 
    for label_l in chosen_labels:
        # get all images with a certain label l
        images_with_label_l = img_set_x[img_set_y == label_l]

        # suffle images with label l
        shuffled_images = np.random.permutation(images_with_label_l)

        # chose examples_per_label images with label l
        chosen_images = shuffled_images[:examples_per_label]

        # add the chosen images to the episode
        episode.append(chosen_images)

    # turn python list into a numpy array
    episode = np.array(episode)

    # convert numpy array to tensor of floats
    episode = torch.from_numpy(episode).float()

    # reshape tensor (required)
    episode = episode.permute(0,1,4,2,3)

    # get the shape of the images
    img_dim = episode.shape[2:]
  
    # build a dict with info about the generated episode
    episode_dict = {
        'images': episode, 'num_way': num_way, 'num_shot': num_shot, 
        'num_query': num_query, 'img_dim': img_dim}

    return episode_dict


# function to display a grid representation of an episode.

def display_episode_images(episode_dict):
    # number of examples per class 
    examples_per_label = episode_dict['num_shot'] + episode_dict['num_query']

    # total number of images
    num_images = episode_dict['num_way'] * examples_per_label

    # select the images
    images = episode_dict['images'].view(num_images, *episode_dict['img_dim'])

    # create a grid with all the images
    grid_img = torchvision.utils.make_grid(images, nrow = examples_per_label)

    # reshape the tensor and convert to numpy array of integers 
    grid_img = grid_img.permute(1, 2, 0).numpy().astype(np.uint8)

    # chage image from BGR to RGB
    grid_img = cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB)

    # set a bigger size
    plt.figure("Episode Grid", figsize = (80, 8))

    # remove the axis
    plt.axis('off')

    # plot the grid image
    plt.imshow(grid_img)
    plt.show()