# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# Written by Ishrat Badami (badami.ishrat@gmail.com)
# ------------------------------------------------------------------------------
import argparse
import copy
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from config import config
from data_utils import RandomCrop, SegmentationDataset
from model import DualResNet_imagenet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_pre_trained_model_weights(path):
    cfg = config
    cfg.defrost()
    cfg.MODEL.PRETRAINED = path
    net = DualResNet_imagenet(cfg, pretrained=True)
    return net


def transfer_learning(model):
    """
    Modifies the last layers of the model from 19 class output to single class output
    :param model: DDRNet model pretrained on Cityscape dataset
    :return: model for single class
    """
    model.final_layer.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
    model.seghead_extra.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
    return model


def train_model(model, criterion, optimizer, data_loader, n_epochs=50000):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000

    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode

        running_loss = 0.0

        # Iterate over data.
        for data in data_loader:
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            # zero the parameter gradients
            for param in model.parameters():
                param.grad = None

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                preds = outputs[0]
                loss = criterion(preds, labels)

                # backward + optimize
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(training_dataset)

            print('training loss: {:.4f}'.format(epoch_loss))

            # deep copy the model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest val Acc: {:4f}'.format(best_loss))

    return best_model_wts


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train tabletop segmentation using pretrained DDRNet')
    parser.add_argument('--training_data', default="./dataset/train.lst", type=str, help='Path to training data file')
    parser.add_argument('--model', default="./pretrained_models/best_val_smaller.pth", type=str, 
                        help='path to pretrained model on cityscape')
    parser.add_argument('--model_save_path', default='./pretrained_models/final_state.pth', type=str,
                        help='output path for trained model on table top data')
    parser.add_argument('--epochs', default=50000, type=int, help='number of epochs')

    args = parser.parse_args()

    training_data = args.training_data
    path_to_model = args.model
    model_save_path = args.model_save_path
    num_epochs = args.epochs

    training_dataset = SegmentationDataset(dataset_list_file=training_data,
                                           transforms_op=transforms.Compose([
                                               RandomCrop((512)),
                                           ]))

    dataloader = DataLoader(training_dataset, shuffle=True, batch_size=2, drop_last=True)

    net = load_pre_trained_model_weights(path_to_model)
    net = transfer_learning(net)
    net = net.to(device=device)

    # The empirical weight of the class pos_weight is set to avoid model collapsing to always returning 1 for all its
    # pixels. 
    loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(0.3))
    adam_optimizer = torch.optim.Adam(params=net.parameters())  # default param works best
    
    # Fine tune model for table top semantic segmentation task
    trained_model_weights = train_model(net, loss, adam_optimizer, dataloader, n_epochs=num_epochs)
    
    # save model weights
    torch.save(trained_model_weights, os.path.join(model_save_path))

