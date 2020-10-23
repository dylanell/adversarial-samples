'''
Feature spread classifier model class. This model is trained with a dual
objective; one for standard classification and another to maximally spread
feature representations of different classes to make it more difficult to
confuse the model with small changes within inputs.
'''

import torch
from module.classifier import Classifier
import time
from torch.utils.tensorboard import SummaryWriter

from util.pytorch_utils import sillhouette_coefficient

class FeatureSpreadClassifier():
    def __init__(self, config):
        # training device - try to find a gpu, if not just use cpu
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using \'{}\' device'.format(self.device))

        # initialize model
        self.model = Classifier(
            config['input_dimensions'],
            config['output_dimension'],
            hid_act=config['hidden_activation'],
            norm=config['normalization'])

        # if model file provided, load pretrained params
        if config['model_file']:
            self.load(config['model_file'])

        # move the model to the training device
        self.model.to(self.device)

        self.config = config

    def load(self, model_file):
        self.model.load_state_dict(
            torch.load(model_file, map_location=self.device))
        print('[INFO]: loaded model from \'{}\''\
            .format(model_file))

    def logits(self, x):
        return self.model(x)

    def predict(self, x):
        return torch.argmax(self.model(x), dim=1)

    def train_epochs(self, train_loader, test_loader):
        # define cross entropy loss (requires logits as outputs)
        loss_fn = torch.nn.CrossEntropyLoss()

        # initialize an optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'])

        # initialize tensorboard writer
        writer = SummaryWriter('{}runs/{}/'.format(
            self.config['output_directory'], self.config['model_name']))

        print('[INFO]: training...')

        # train through all epochs
        for e in range(self.config['number_epochs']):
            # get epoch start time
            epoch_start = time.time()

            # reset accumulators
            train_epoch_loss = 0.0
            train_num_correct = 0
            test_epoch_loss = 0.0
            test_num_correct = 0
            epoch_s = 0.0

            # run through epoch of train data
            for i, batch in enumerate(train_loader):
                # parse batch and move to training device
                input_batch = batch['image'].to(self.device)
                label_batch = batch['label'].to(self.device)

                # compute output batch logits and predictions
                logits_batch = self.model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)

                # compute loss
                loss = loss_fn(logits_batch, label_batch)

                # accumulate loss
                train_epoch_loss += loss.item()

                # accumulate number correct
                train_num_correct += torch.sum(
                    (pred_batch == label_batch)
                ).item()

                # compute hidden layer activations
                hidden_batch = self.model.hidden(input_batch)

                # create cluster metrics regularizer for each hidden layer
                ss = []
                for reps in hidden_batch:
                    # NOTE: optionaly perform clustering here to compute
                    # estimated labels for unsupervised learning

                    s = sillhouette_coefficient(reps, label_batch)

                    ss.append(s.item())

                    # regularizer as langrangian pushing s _> 1
                    regularizer = (1 - s)**2

                    # add regularizer to loss
                    loss += (1. * regularizer)

                epoch_s += torch.mean(torch.Tensor(ss)).item()

                # zero out gradient attributes for all trainabe params
                optimizer.zero_grad()

                # compute gradients w.r.t loss (repopulate gradient
                # attribute for all trainable params)
                loss.backward()

                # update params with current gradients
                optimizer.step()

            # compute epoch average loss and accuracy metrics
            train_loss = train_epoch_loss / i
            train_acc = 100.0 * train_num_correct / self.config['number_train']

            # compute epoch average sillhouette coefficient
            avg_s = epoch_s / i

            # run through epoch of test data
            for i, batch in enumerate(test_loader):
                # parse batch and move to training device
                input_batch = batch['image'].to(self.device)
                label_batch = batch['label'].to(self.device)

                # compute output batch logits and predictions
                logits_batch = self.model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)

                # compute loss
                loss = loss_fn(logits_batch, label_batch)

                # accumulate loss
                test_epoch_loss += loss.item()

                # accumulate number correct
                test_num_correct += torch.sum(
                    (pred_batch == label_batch)
                ).item()

            # compute epoch average loss and accuracy metrics]
            test_loss = test_epoch_loss / i
            test_acc = 100.0 * test_num_correct / self.config['number_test']

            # compute epoch time
            epoch_time = time.time() - epoch_start

            # save model
            torch.save(self.model.state_dict(),'{}{}.pt'.format(
                self.config['output_directory'], self.config['model_name']))

            # add metrics to tensorboard
            writer.add_scalar('Batch Latent Sillhouette Coeff.', avg_s, e+1)
            writer.add_scalar('Loss/Train', train_loss, e+1)
            writer.add_scalar('Accuracy/Train', train_acc, e+1)
            writer.add_scalar('Loss/Test', test_loss, e+1)
            writer.add_scalar('Accuracy/Test', test_acc, e+1)

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, '\
                'Train Loss: {:.2f}, Train Accuracy: {:.2f}, '\
                'Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print(template.format(e+1, epoch_time, train_loss,
                train_acc, test_loss, test_acc))
