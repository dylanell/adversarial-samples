"""
Generate adversarial MNIST datasets using dataloaders and a pretrained classifier.
"""

import argparse
import numpy as np

# relative imports
from classifier_cnn import Classifier
from dataloader_utils import make_mnist_dataloaders

def make_adversarial_dataset(classifier, dataloader, eps, out_dir=None):
    # run through all batches of the test loader
    for batch in dataloader:
        # parse batch
        batch_samples = batch[0]
        batch_labels = batch[1]

        # require gradient for input data (need to do this to compute the gradients for inputs dutring backward() call)
        batch_samples.requires_grad = True

        # compute model outputs
        outputs = classifier(batch_samples)

        # compute loss on current test batch
        loss = torch.nn.functional.cross_entropy(outputs, batch_labels)

        # compute gradients of loss on backward pass
        loss.backward()

        # get gradients of input data
        grads = batch_samples.grad

        # get sign of gradients
        signed_grads = torch.sign(grads)

        # perturb test samples using FGSM
        adv_batch_samples = batch_samples + ((eps) * signed_grads)

        # keep pixel vals within [-1, 1]
        new_min, new_max = -1., 1.
        old_min, old_max = torch.min(adv_batch_samples), torch.max(adv_batch_samples)
        adv_batch_samples = (((adv_batch_samples - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min

        # detach and convert samples to numpy
        adv_batch_samples = np.reshape(
            adv_batch_samples.detach().numpy(),
            [adv_batch_samples.shape[0], -1]
        )

        # detach and convert labels to numpy
        batch_labels = batch_labels.detach().numpy()

        # append current batch samples to samples file
        with open(out_dir+'samples.csv'.format(eps), 'ab+') as fp:
            np.savetxt(fp, adv_batch_samples)

        # append current batch labels to labels file
        with open(out_dir+'labels.csv', 'ab+') as fp:
            np.savetxt(fp, batch_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--nw', type=int, default=1, help='number of dataloader workers')
    parser.add_argument('--ne', type=int, default=20, help='number of training epochs')
    parser.add_argument('--ntrain', type=int, default=60000, help='number of training samples')
    parser.add_argument('--ntest', type=int, default=10000, help='number of testing samples')
    parser.add_argument('--nclass', type=int, default=10, help='number of classes')
    parser.add_argument('--nchan', type=int, default=1, help='sample channel dimension')
    parser.add_argument('--name', type=str, default='classifier', help='model name')
    parser.add_argument('--v', type=bool, default=False, help='verbose flag')
    parser.add_argument('--ld', type=str, default='/tmp/', help='log and other output directory')
    parser.add_argument('--dd', type=str, default='/tmp/mnist_data/', help='mnist data directory')
    parser.add_argument('--od', type=str, default='/tmp/', help='output data directory')
    args = parser.parse_args()

    # initialize model
    classifier = ClassifierCNN(args)

    # intialize MNIST dataloaders
    train_loader, test_loader = make_mnist_dataloaders(
        batch_size=args.bs,
        num_workers=args.nw,
        data_dir=args.dd,
        shuffle=False
    )

    # create a bunch of adversarial datasets from MNIST rraining data for a range of epsilon
    # strengths
    make_adversarial_dataset(
        classifier.net,
        train_loader,
        0.3,
        out_dir=args.od + 'train/'
    )

    # create a bunch of adversarial datasets from MNIST testing data for a range of epsilon
    # strengths
    make_adversarial_dataset(
        classifier.net,
        test_loader,
        0.3,
        out_dir=args.od + 'test/'
    )

if __name__ == '__main__':
    main()
