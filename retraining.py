from tqdm import trange
from os import path

import pickle
import json
import sys
import numpy as np
import torch
import torch.optim as optim

from protonets.core.episode_extractor import extract_episode
from protonets.core.dataset_loader import load_images
from protonets.core.model_loader import load_model

from protonets.utils.yaml_loader import load_yaml
from protonets.utils.arguments_parser import parse_arguments
from protonets.utils.log_creator import create_logger
from protonets.utils.time_measurement import measure_time

# function to retrain the model using both train and validation sets
def retrain(model, opt, retrain_data, logger):
    # load the saved model
    state_dict = torch.load(path.join(opt['results_dir'], 'best_model.pth'))
    model.load_state_dict(state_dict)

    # set Adam optimizer with an initial learning rate
    optimizer = optim.Adam(
      model.parameters(), lr = opt['learning_rate'])

    # schedule learning rate to be cut in half every 2000 episodes
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, opt['decay_every'], gamma = 0.5, last_epoch = -1)

    # set model to training mode
    model.train()

    # number of epochs so far
    epochs_so_far = 0

    # retrieve the epoch with best valid loss to determine max_epoch
    with open(path.join(opt['results_dir'], 'best_epoch.pkl'), 'rb') as f:
        best_epoch_num = pickle.load(f)['number']

    max_epoch = best_epoch_num + opt['patience']

    while epochs_so_far < max_epoch:
        epoch_loss = 0.0
        epoch_acc = 0.0

        logger.info('==> Epoch %d' % (epochs_so_far + 1))

        # do epoch_size classification tasks to evaluate the model
        for episode in trange(retrain_data['epoch_size']):
            # get the episode dict
            episode_dict = extract_episode(
                retrain_data['retrain_x'], retrain_data['retrain_y'],
                retrain_data['num_way'], retrain_data['num_shot'],
                retrain_data['num_query'])

            optimizer.zero_grad()

            # classify images and get the loss and the acc of the curr episode
            loss, output = model.set_forward_loss(episode_dict)

            # acumulate the loss and the acc
            epoch_loss += output['loss']
            epoch_acc += output['acc']

            # update the model parameters (weights and biases)
            loss.backward()
            optimizer.step()

        # average the loss and the acc to get the epoch loss and the acc
        epoch_loss = epoch_loss / retrain_data['epoch_size']
        epoch_acc = epoch_acc / retrain_data['epoch_size']

        # output the epoch loss and the epoch acc
        logger.info('Loss: %.4f / Acc: %.2f%%' % (epoch_loss, (epoch_acc * 100)))

        # increment the number of epochs
        epochs_so_far += 1

        # tell the scheduler to increment its counter
        scheduler.step()

    # save a model better than the previous one
    torch.save(
        model.state_dict(),
        path.join(opt['results_dir'], 'best_model.pth'))


# let's prepare everything and retrain the model
def main():
    # read the config file
    config = load_yaml(path.join('config', 'config.yaml'))

    # create a opt dict
    opt = {}

    opt.update(config['parameters'])
    opt.update(config['directories'])

    # recover the chosen model and the dataset
    with open(path.join(opt['results_dir'], 'info.json'), 'r', encoding='utf8') as f:
        info_dict = json.load(f)

    if not info_dict['trained']:
        print('Model seems not to be trained yet, train it first')

        sys.exit()

    if info_dict['retrained']:
        print('Model was already retrained')

        sys.exit()

    # load the desired model
    model = load_model(info_dict['model'], (3, 84, 84), 64, 64)

    # create retrain_data dict
    retrain_data = config[info_dict['dataset']]['train']

    # load train and valid set
    dataset_dir = path.join(opt['data_dir'], info_dict['dataset'])

    train_x, train_y = load_images(path.join(dataset_dir, 'train.pkl'))
    valid_x, valid_y = load_images(path.join(dataset_dir, 'valid.pkl'))

    # add sets to retrain_data
    retrain_data.update({
        'retrain_x': np.concatenate((train_x, valid_x)),
        'retrain_y': np.concatenate((train_y, valid_y))})

    # configure the logging instance
    retrain_logger = create_logger(opt['logging_dir'], 'retrain.log')

    # run retrain and compute the time taken
    time_taken = measure_time(retrain, model, opt, retrain_data, retrain_logger)

    # output the time taken to retrain
    retrain_logger.info('Time taken by the retraining: %s seconds' % str(time_taken))

    # record success, the chosen model and the dataset
    info_dict['retrained'] = True

    with open(path.join(opt['results_dir'], 'info.json'), 'w', encoding='utf8') as f:
        json.dump(info_dict, f)


if __name__ == '__main__':
    main()
