'''
Smooth classifier model class. This model is trained with a regularizer
that smooths model output as it is randomly perturbed.
'''

import torch
from module.classifier import Classifier
import time
from torch.utils.tensorboard import SummaryWriter

class SmoothClassifier():
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

        # initialize a random input distribution
        self.n_samples = 32
        mean = 0.0
        stdev = 0.25
        self.input_dist = torch.distributions.Normal(
            mean * torch.ones(
                config['batch_size'],
                self.n_samples,
                config['input_dimensions'][-1],
                config['input_dimensions'][0],
                config['input_dimensions'][1]),
            stdev * torch.ones(
                config['batch_size'],
                self.n_samples,
                config['input_dimensions'][-1],
                config['input_dimensions'][0],
                config['input_dimensions'][1])
        )

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
        # get number of samples in batch
        bs = x.shape[0]

        # add noise to input batch
        x = x.unsqueeze(1) + self.input_dist.sample()[:bs].to(self.device)

        # reshape input batch by stacking samples into batch dimension
        x = x.view((
            bs*self.n_samples,
            self.config['input_dimensions'][-1], self.config['input_dimensions'][0], self.config['input_dimensions'][1]))

        # compute output batch logits and predictions
        logits = self.model(x)
        pred = torch.argmax(logits, dim=1)

        # reshape predictions to unstack samples from batch dimension
        pred = pred.view((bs, self.n_samples))

        # take mode along sample dim to get final prediction
        pred = torch.mode(pred, dim=1)[0]

        return pred

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

            # run through epoch of train data
            for i, batch in enumerate(train_loader):
                # parse batch and move to training device
                input_batch = batch['image'].to(self.device)
                label_batch = batch['label'].to(self.device)

                # get number of samples in batch
                bs = input_batch.shape[0]

                # add noise to input batch
                input_batch = input_batch.unsqueeze(1) + \
                    self.input_dist.sample()[:bs].to(self.device)

                # reshape input batch by stacking samples into batch dimension
                input_batch = input_batch.view((
                    bs*self.n_samples,
                    self.config['input_dimensions'][-1], self.config['input_dimensions'][0], self.config['input_dimensions'][1]))

                # keep pixel values of batch in [-1, 1]
                new_min, new_max = -1., 1.
                old_min = torch.min(input_batch)
                old_max = torch.max(input_batch)
                input_batch = (((input_batch - old_min) / \
                    (old_max - old_min)) * \
                    (new_max - new_min)) + new_min

                # repeat and interleave label batch to repeat labels for each
                # samples stacked into batch dimension
                label_batch = label_batch.repeat_interleave(self.n_samples)

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

                # zero out gradient attributes for all trainabe params
                optimizer.zero_grad()

                # compute gradients w.r.t loss (repopulate gradient
                # attribute for all trainable params)
                loss.backward()

                # update params with current gradients
                optimizer.step()

            # compute epoch average loss and accuracy metrics
            train_loss = train_epoch_loss / i
            train_acc = 100.0 * train_num_correct / \
                (self.config['number_train'] * self.n_samples)

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

            # compute epoch average loss and accuracy metrics
            test_loss = test_epoch_loss / i
            test_acc = 100.0 * test_num_correct / self.config['number_test']

            # compute epoch time
            epoch_time = time.time() - epoch_start

            # save model
            torch.save(self.model.state_dict(),'{}{}.pt'.format(
                self.config['output_directory'], self.config['model_name']))

            # add metrics to tensorboard
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
