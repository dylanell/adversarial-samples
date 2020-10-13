'''
Feature spread classifier model class. This model is trained with a dual
objective; one for standard classification and another to maximally spread
feature representations of different classes to make it more difficult to
confuse the model with small changes within inputs.
'''

import torch
from module.classifier import Classifier
import time

class FeatureSpreadClassifier():
    def __init__(self, config):
        self.config = config

        # training device - try to find a gpu, if not just use cpu
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print('[INFO]: using \'{}\' device'.format(self.device))

        # initialize model
        self.model = Classifier(
            self.config['input_dimensions'], self.config['output_dimension'])

        # if model file provided, load pretrained params
        if config['model_file']:
            self.model.load_state_dict(
                torch.load(config['model_file'], map_location=self.device))
            print('[INFO]: loaded model from \'{}\''\
                .format(config['model_file']))

        # define cross entropy loss (requires logits as outputs)
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # initialize an optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        # move the model to the training device
        self.model.to(self.device)

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

                # compute output batch logits and predictions
                logits_batch = self.model(input_batch)
                pred_batch = torch.argmax(logits_batch, dim=1)

                # compute loss
                loss = self.loss_fn(logits_batch, label_batch)

                # compute hidden layer activations
                hidden_batch = self.model.hidden(input_batch)

                # compute average class-wise feature representations for each
                # unique label in the current label batch
                class_reps = torch.cat([torch.mean(hidden_batch[-1][label_batch == i], dim=0, keepdim=True) for i in torch.unique(label_batch)], dim=0)

                # compute pairwise distances between each average class-wise
                # feature representation
                n = class_reps.shape[0]
                d = class_reps.shape[1]
                a = class_reps.unsqueeze(1).expand(n, n, d)
                b = class_reps.unsqueeze(0).expand(n, n, d)
                rep_dist = torch.pow(a - b, 2).sum(dim=2)

                # compute mean non-equivalent class paiwise distance
                avg_rep_dist = torch.mean(rep_dist[torch.triu(
                    torch.ones(class_reps.shape[0],
                    class_reps.shape[0]), diagonal=1) == 1])

                # computed weighted lagrangian for class-wise feature spread
                rep_loss = 0.001 * (avg_rep_dist - 1000.)**2

                # add class-wise feature spread regularizer to loss
                loss += rep_loss

                # zero out gradient attributes for all trainabe params
                self.optimizer.zero_grad()

                # compute gradients w.r.t loss (repopulate gradient attribute
                # for all trainable params)
                loss.backward()

                # update params with current gradients
                self.optimizer.step()

                # accumulate loss
                train_epoch_loss += loss.item()

                # accumulate number correct
                train_num_correct += torch.sum(
                    (pred_batch == label_batch)
                ).item()

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
            train_loss = train_epoch_loss / i
            train_acc = 100.0 * train_num_correct / self.config['number_train']
            test_loss = test_epoch_loss / i
            test_acc = 100.0 * test_num_correct / self.config['number_test']

            # compute epoch time
            epoch_time = time.time() - epoch_start

            # save model
            torch.save(self.model.state_dict(),'{}{}.pt'.format(
                self.config['output_directory'], self.config['model_name']))

            # print epoch metrics
            template = '[INFO]: Epoch {}, Epoch Time {:.2f}s, '\
                'Train Loss: {:.2f}, Train Accuracy: {:.2f}, '\
                'Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print(template.format(e+1, epoch_time, train_loss,
                train_acc, test_loss, test_acc))