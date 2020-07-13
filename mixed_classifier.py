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

class MixedClassifier():
    def __init__(self, config):
        # get args
        self.conf = config

        # initialize logging to create new log file and log any level event
        logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', \
            filename='{}{}.log'.format(self.conf.ld, self.conf.name), filemode='a', \
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
                torch.load(self.conf.ld+self.conf.name+'.pt', map_location=self.device)
            )

            #logging.info('Successfully loaded model parameters from \'{}\''.format(self.conf.mf))
            print('Successfully loaded model parameters from \'{}\'' \
                .format(self.conf.ld+self.conf.name+'.pt'))
        except:
            #logging.info('Failed to load model parameters from \'{}\''.format(self.conf.mf))
            print('Failed to load model parameters from \'{}\'' \
                .format(self.conf.ld+self.conf.name+'.pt'))

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

                # grab another batch
                anotha_batch = iter(train_loader).next()

                # parse batch and send to device
                sample_batch = batch[0].to(self.device)
                label_batch = batch[1].to(self.device)
                anotha_sample_batch = anotha_batch[0].to(self.device)
                anotha_label_batch = anotha_batch[1].to(self.device)

                # get batch size
                bs = sample_batch.shape[0]

                # superimpose batches if they are the same size, otherwise we are at
                # the end of a batch so skip this iteration
                if (anotha_sample_batch.shape[0] == bs):
                    # mix samples from both batches
                    super_samples = sample_batch + anotha_sample_batch

                    # rescale everything so pixel vals still in [-1, 1]
                    new_min, new_max = -1., 1.
                    old_min, old_max = torch.min(super_samples), torch.max(super_samples)
                    super_samples = (((super_samples - old_min) / (old_max - old_min)) *  \
                        (new_max - new_min)) + new_min

                    # randomly grab a label from one of the label batches
                    select = torch.randint(low=0, high=2, size=(bs, 1))
                    not_select = 1 * (select == 0)
                    idx = torch.cat([select, not_select], dim=1).to(self.device)
                    both_labels = \
                        torch.cat([label_batch.view(bs, 1), anotha_label_batch.view(bs, 1)], dim=-1)
                    super_labels = \
                        torch.diagonal(torch.matmul(both_labels.float(), idx.T.float())).long()
                        
                else:
                    continue

                # compute output logits
                logits = self.net(super_samples)

                # compute loss between logits and targets
                loss = self.loss_fn(logits, super_labels)

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
            torch.save(self.net.state_dict(), '{}{}.pt'.format(self.conf.ld, self.conf.name))

        # done with all epochs
