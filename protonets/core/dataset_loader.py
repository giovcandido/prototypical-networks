from tqdm import tqdm
import pickle
import cv2
import numpy as np

# function to load images and their labels from a pkl file.
def load_images(file_name):
    # get file content
    with open(file_name, 'rb') as f:
        info = pickle.load(f)

    img_data = info['image_data']
    class_dict = info['class_dict']

    # create arrays to store x and y of images
    images = [] # x
    labels = [] # y

    # loop over all images and store them
    loading_msg = 'Reading images from %s' % file_name

    # loop over all classes
    for item in tqdm(class_dict.items(), ascii=True, desc = loading_msg):
        # loop over all examples from the class
        for example_num in item[1]:
            # convert image to RGB color channels
            RGB_img = cv2.cvtColor(img_data[example_num], cv2.COLOR_BGR2RGB)

            # store image and corresponding label
            images.append(RGB_img)
            labels.append(item[0])

    # return set of images
    return np.array(images), np.array(labels)
