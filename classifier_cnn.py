"""
Wasserstein GAN class.
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

        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # initialize optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.conf.lr)

        # move network to device
        self.net.to(self.device)

    def print_structure(self):
        print('[INFO] \'{}\' network structure \n{}'.format(self.conf.name, self.net))

    def train(self, dataloader, num_epochs):
        # iterate through epochs
        for e in range(num_epochs):

            # accumulator for loss over an epoch
            running_loss = 0.0

            # iteate through batches
            for i, batch in enumerate(dataloader):

                # get images from batch
                sample_batch = batch[0].to(self.device)
                label_batch = batch[1].to(self.device)

                # compute output logits
                logits = self.net(sample_batch)

                # compute loss between logits and targets
                loss = self.loss_fn(logits, label_batch)

                # accumulate running loss
                running_loss += loss.item()

                if i % 100 == 99:
                    print(running_loss / i)

            # done with current epoch

            # compute average wasserstein distance over epoch
            epoch_avg_loss = running_loss / i

            # log epoch stats info
            logging.info('| epoch: {:3} | cross entropy loss: {:6.2f} \
                |'.format(e+1, epoch_avg_loss))

            # save current state of network
            torch.save(self.net.state_dict(), '{}{}_net.pt'.format(self.conf.ld, self.conf.name))

        # done with all epochs
