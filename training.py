from tqdm import trange
from os import path

import pickle
import json
import sys
import numpy as np
import torch
import torch.optim as optim

import utils.new_os_functions as new_os

from modules.episode_extractor import extract_episode
from modules.dataset_loader import load_images
from modules.model_loader import load_model

from utils.yaml_loader import load_yaml
from utils.arguments_parser import parse_arguments
from utils.log_creator import create_logger
from utils.time_measurement import measure_time

# function to train the model on the train set through many epochs

def train(model, opt, train_data, valid_data, logger):
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

    # train until early stopping says so
    # or until the max number of epochs is not achived 
    while epochs_so_far < opt['max_epoch'] and not opt['stop']:
        epoch_loss = 0.0
        epoch_acc = 0.0

        logger.info('==> Epoch %d' % (epochs_so_far + 1))

        logger.info('> Training')

        # do epoch_size classification tasks to train the model
        for episode in trange(train_data['epoch_size']):
            # get the episode dict
            episode_dict = extract_episode(
              train_data['train_x'], train_data['train_y'], train_data['num_way'], 
              train_data['num_shot'], train_data['num_query'])

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
        epoch_loss = epoch_loss / train_data['epoch_size']
        epoch_acc = epoch_acc / train_data['epoch_size']

        # output the epoch loss and the epoch acc
        logger.info('Loss: %.4f / Acc: %.2f%%' % (epoch_loss, (epoch_acc * 100)))

        # do one epoch of evaluation on the validation test
        evaluate_valid(model, opt, valid_data, epochs_so_far + 1, logger)

        # increment the number of epochs
        epochs_so_far += 1

        # tell the scheduler to increment its counter
        scheduler.step()

    # get dict with info about the best epoch
    best_epoch = opt['best_epoch']

    # at the end of the training, output the best loss and the best acc
    logger.info('Best loss: %.4f / Best Acc: %.2f%%' 
          % (best_epoch['loss'], (best_epoch['acc'] * 100)))

    # save dict with info about the best epoch
    with open(path.join(opt['results_dir'], 'best_epoch.pkl'), 'wb') as f:
        pickle.dump(best_epoch, f, pickle.HIGHEST_PROTOCOL)


# function to evaluate the model on the validation set

def evaluate_valid(model, opt, valid_data, curr_epoch, logger):
    # set model to evaluation mode
    model.eval()

    valid_loss = 0.0
    valid_acc = 0.0

    logger.info('> Validation')

    # do epoch_size classification tasks to evaluate the model
    for episode in trange(valid_data['epoch_size']):
        # get the episode dict
        episode_dict = extract_episode(
            valid_data['valid_x'], valid_data['valid_y'], valid_data['num_way'], 
            valid_data['num_shot'], valid_data['num_query'])

        # classify images and get the loss and the acc of the curr episode
        _, output = model.set_forward_loss(episode_dict)

        # acumulate the loss and the acc
        valid_loss += output['loss']
        valid_acc += output['acc']
  
    # average the loss and the acc to get the valid loss and the acc
    valid_loss = valid_loss / valid_data['epoch_size']
    valid_acc = valid_acc / valid_data['epoch_size']

    # output the valid loss and the valid acc
    logger.info('Loss: %.4f / Acc: %.2f%%' % (valid_loss, (valid_acc * 100)))

    # implement early stopping mechanism
    # check if valid_loss is the best so far
    if opt['best_epoch']['loss'] > valid_loss:
        # if true, save the respective train epoch
        opt['best_epoch']['number'] = curr_epoch

        # save the best loss and the respective acc
        opt['best_epoch']['loss'] = valid_loss
        opt['best_epoch']['acc'] = valid_acc

        # save the model with the best loss so far
        model_file = path.join(opt['results_dir'], 'best_model.pth')
        torch.save(model.state_dict(), model_file)

        logger.info('=> This is the best model so far! Saving...')

        # set wait to zero
        opt['wait'] = 0
    else:
        # if false, increment the wait
        opt['wait'] += 1

        # when the wait is bigger than the patience
        if opt['wait'] > opt['patience']:
            # the train has to stop
            opt['stop'] = True

            logger.info('Patience was exceeded... Stopping...')



# now, let's prepare everything and train the model

# get the chosen model and the dataset
args = parse_arguments()

# load the desired model
model = load_model(args.model, (3, 84, 84), 64, 64)

# read the config file
config = load_yaml(path.join('config', 'config.yaml'))

# create a opt dict 
opt = {}

opt.update(config['parameters'])
opt.update(config['directories'])

# create results dir with logging
results_dir_created = new_os.mkdir_if_not_exist(opt['results_dir'])

if not results_dir_created:
    print('There is already a results directory, you should delete it or rename it')
    
    sys.exit()
    
new_os.mkdir_if_not_exist(opt['logging_dir'])

# create a best_epoch entry 
best_epoch_dict = {
    'best_epoch': {
        'number': -1,
        'loss': np.inf,
        'acc': 0}}

# add best_epoch to opt
opt.update(best_epoch_dict)

# create train_data dict
train_data = config[args.dataset]['train']

# load train set
dataset_dir = path.join(opt['data_dir'], args.dataset)

train_x, train_y = load_images(path.join(dataset_dir, 'train.pkl'))

# add train set to train_data
train_data.update({
    'train_x': train_x, 
    'train_y': train_y})

# create valid_data dict
valid_data = config[args.dataset]['valid']

# load valid set
valid_x, valid_y = load_images(path.join(dataset_dir, 'valid.pkl'))

# add valid set to valid_data
valid_data.update({
    'valid_x': valid_x,
    'valid_y': valid_y})

# configure the logger instance
train_logger = create_logger(opt['logging_dir'], 'train.log')

# run train and compute the time taken
time_taken = measure_time(train, model, opt, train_data, valid_data, train_logger)

# check what the best_epoch was
best_epoch_file = path.join(opt['results_dir'], 'best_epoch.pkl')

with open(best_epoch_file, 'rb') as f:
    number = pickle.load(f)['number']

train_logger.info('Best epoch was the number: %i' % number)

# output the time taken to train
train_logger.info('Time taken by the training: %s seconds' % str(time_taken))

# record success, the chosen model and the dataset
info_dict = {
    'trained': True,
    'retrained': False,
    'model': args.model,
    'dataset': args.dataset}

with open(path.join(opt['results_dir'], 'info.json'), 'w', encoding='utf8') as f:
    json.dump(info_dict, f)