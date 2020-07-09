"""
CNN Classifier class.
"""

import torch
from tqdm import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt

# relative imports
from cnn import CNN

class ClassifierCNN():
    def __init__(self, config):
        # get args
        self.conf = config

        # initialize logging to create new log file and log any level event
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', \
            filename='{}{}.log'.format(self.conf.ld, self.conf.name), filemode='w', \
            level=logging.DEBUG)

        # try to get gpu device, if not just use cpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # initialize classifier network
        self.net = CNN(
            in_chan=self.conf.nchan,
            out_dim=self.conf.nclass,
            out_act=None
        )

        # try to load pre-trained parameters
        try:
            self.net.load_state_dict(
                torch.load(self.conf.mf, map_location=self.device)
            )

            logging.info('Successfully loaded model parameters from \'{}\''.format(self.conf.mf))
        except:
            logging.info('Failed to load model parameters from \'{}\''.format(self.conf.mf))

        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # initialize optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.conf.lr)

        # move network to device
        self.net.to(self.device)

    def print_structure(self):
        print('[INFO] \'{}\' network structure \n{}'.format(self.conf.name, self.net))

    def compute_accuracy(self, data_loader):
        # counters for total number of samples in the dataloader and correct predictions
        total = 0
        correct = 0

        # iterate through batches
        for batch in data_loader:

            # parse batch and send to device
            sample_batch = batch[0].to(self.device)
            label_batch = batch[1].to(self.device)

            # get predicted outputs
            pred = torch.argmax(self.net(sample_batch), dim=1)

            # count where predicted == labels and add to correct count
            correct += torch.sum((pred == label_batch)).item()

            # add this batch size to total count
            total += sample_batch.shape[0]

        # compute accuracy on dataloader
        accuracy = 100.0 * (float(correct) / float(total))

        return accuracy

    def train(self, train_loader, test_loader, num_epochs):
        # iterate through epochs
        for e in range(num_epochs):

            # accumulator for loss over an epoch
            running_loss = 0.0

            # iterate through batches
            for i, batch in enumerate(train_loader):

                # parse batch and send to device
                sample_batch = batch[0].to(self.device)
                label_batch = batch[1].to(self.device)

                # compute output logits
                logits = self.net(sample_batch)

                # compute loss between logits and targets
                loss = self.loss_fn(logits, label_batch)

                # zero out gradients
                self.opt.zero_grad()

                # run backprop on loss
                loss.backward()

                # run optimizer step
                self.opt.step()

                # accumulate running loss
                running_loss += loss.item()

            # done with current epoch

            # compute average wasserstein distance over epoch
            epoch_avg_loss = running_loss / i

            # compute model accuracy on the testing set
            accuracy = self.compute_accuracy(test_loader)

            # log epoch stats info
            logging.info('| epoch: {:3} | training loss: {:6.2f} |' \
                ' validation accuracy: {:6.2f} |'.format(e+1, epoch_avg_loss, accuracy))

            # save current state of network
            torch.save(self.net.state_dict(), '{}{}_net.pt'.format(self.conf.ld, self.conf.name))

        # done with all epochs
