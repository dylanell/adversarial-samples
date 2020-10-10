"""
Script to train a CNN classifier with PyTorch.
"""

import time
import yaml
import torch
from torchvision.transforms import transforms
import torch.nn.functional as F

from util.pytorch_utils import build_image_dataset
from util.data_utils import generate_df_from_image_dataset
from model.classifier import Classifier

def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # training device - try to find a gpu, if not just use cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('[INFO]: using \'{}\' device'.format(device))

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(config['dataset_directory'])

    # get number of classes in labels
    num_class = data_dict['train']['Label'].nunique()

    # build training dataloader
    train_set, train_loader = build_image_dataset(
        data_dict['train'],
        image_size=config['input_dimensions'][:-1],
        batch_size=config['batch_size'],
        num_workers=config['number_workers']
    )

    # build testing dataloader
    test_set, test_loader = build_image_dataset(
        data_dict['test'],
        image_size=config['input_dimensions'][:-1],
        batch_size=config['batch_size'],
        num_workers=config['number_workers']
    )

    # initialize the model
    model = Classifier(config['input_dimensions'], num_class)

    # define cross entropy loss (requires logits as outputs)
    loss_fn = torch.nn.CrossEntropyLoss()

    # initialize an optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate']
    )

    # move the model to the training device
    model.to(device)

    print('[INFO]: training...')

    # train through all epochs
    for e in range(config['number_epochs']):
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
            input_batch = batch['image'].to(device)
            label_batch = batch['label'].to(device)

            # compute output batch logits and predictions
            logits_batch = model(input_batch)
            pred_batch = torch.argmax(logits_batch, dim=1)

            # compute loss
            loss = loss_fn(logits_batch, label_batch)

            # zero out gradient attributes for all trainabe params
            optimizer.zero_grad()

            # compute gradients w.r.t loss (repopulate gradient attribute
            # for all trainable params)
            loss.backward()

            # update params with current gradients
            optimizer.step()

            # accumulate loss
            train_epoch_loss += loss.item()

            # accumulate number correct
            train_num_correct += torch.sum(
                (pred_batch == label_batch)
            ).item()

        # run through epoch of test data
        for i, batch in enumerate(test_loader):
            # parse batch and move to training device
            input_batch = batch['image'].to(device)
            label_batch = batch['label'].to(device)

            # compute output batch logits and predictions
            logits_batch = model(input_batch)
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
        train_loss = train_epoch_loss / i
        train_acc = 100.0 * train_num_correct / train_set.__len__()
        test_loss = test_epoch_loss / i
        test_acc = 100.0 * test_num_correct / test_set.__len__()

        # compute epoch time
        epoch_time = time.time() - epoch_start

        # print epoch metrics
        template = '[INFO]: Epoch {}, Epoch Time {:.2f}, Train Loss: {:.2f},'\
            ' Train Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
        print(template.format(e+1, epoch_time, train_loss,
            train_acc, test_loss, test_acc))

if __name__ == '__main__':
    main()
