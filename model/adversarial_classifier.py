'''
Adversarial classifier model class. This model is trained with advsarial
samples.
'''

import torch
from module.classifier import Classifier
import time
from torch.utils.tensorboard import SummaryWriter

class AdversarialClassifier():
    def __init__(self, config):
        self.config = config

        # training device - try to find a gpu, if not just use cpu
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using \'{}\' device'.format(self.device))

        # initialize model
        self.model = Classifier(
            self.config['input_dimensions'], self.config['output_dimension'],
            hid_act=self.config['hidden_activation'],
            norm=config['normalization'])

        # if model file provided, load pretrained params
        if config['model_file']:
            self.model.load_state_dict(
                torch.load(config['model_file'], map_location=self.device))
            print('[INFO]: loaded model from \'{}\''\
                .format(config['model_file']))

        # initialize guide model (for making adversarial samples)
        self.guide_model = Classifier(
            self.config['input_dimensions'], self.config['output_dimension'],
            hid_act=self.config['hidden_activation'],
            norm=config['normalization'])

        # must be able to load guide model to proceed
        self.guide_model.load_state_dict(
            torch.load(config['guide_model_file'], map_location=self.device))
        print('[INFO]: loaded guide model from \'{}\''\
            .format(config['guide_model_file']))

        # define cross entropy loss (requires logits as outputs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # initialize an optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        # move the models to the training device
        self.model.to(self.device)
        self.guide_model.to(self.device)

        # initialize tensorboard writer
        self.writer = SummaryWriter(config['output_directory']+'runs/')

    def train_epochs(self, train_loader, test_loader):
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

                # require gradient for input data (need to do this to compute
                # the gradients for inputs dutring backward() call)
                input_batch.requires_grad = True

                # make adversarial samples from input batch
                adv_logits_batch = self.guide_model(input_batch)
                adv_loss = self.loss_fn(adv_logits_batch, label_batch)
                adv_loss.backward()
                adv_grads = input_batch.grad
                input_batch = input_batch + (0.2 * torch.sign(adv_grads))

                # keep pixel values in [-1, 1]
                new_min, new_max = -1., 1.
                old_min = torch.min(input_batch)
                old_max = torch.max(input_batch)
                input_batch = (((input_batch - old_min) / \
                    (old_max - old_min)) * \
                    (new_max - new_min)) + new_min

                # compute output batch logits and predictions
                logits_batch = self.model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)

                # compute loss
                loss = self.loss_fn(logits_batch, label_batch)

                # accumulate loss
                train_epoch_loss += loss.item()

                # accumulate number correct
                train_num_correct += torch.sum(
                    (pred_batch == label_batch)
                ).item()

                # zero out gradient attributes for all trainabe params
                self.optimizer.zero_grad()

                # compute gradients w.r.t loss (repopulate gradient
                # attribute for all trainable params)
                loss.backward()

                # update params with current gradients
                self.optimizer.step()

            # compute epoch average loss and accuracy metrics
            train_loss = train_epoch_loss / i
            train_acc = 100.0 * train_num_correct / self.config['number_train']

            # run through epoch of test data
            for i, batch in enumerate(test_loader):
                # parse batch and move to training device
                input_batch = batch['image'].to(self.device)
                label_batch = batch['label'].to(self.device)

                # compute output batch logits and predictions
                logits_batch = self.model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)

                # compute loss
                loss = self.loss_fn(logits_batch, label_batch)

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
            self.writer.add_scalar('Loss/Train', train_loss, e+1)
            self.writer.add_scalar('Accuracy/Train', train_acc, e+1)
            self.writer.add_scalar('Loss/Test', test_loss, e+1)
            self.writer.add_scalar('Accuracy/Test', test_acc, e+1)

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, '\
                'Train Loss: {:.2f}, Train Accuracy: {:.2f}, '\
                'Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print(template.format(e+1, epoch_time, train_loss,
                train_acc, test_loss, test_acc))
