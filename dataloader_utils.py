"""
Pytorch Dataloader utils.
"""

import torch
import torchvision
from torchvision.transforms import transforms

class raw_pytorch_dataset(torch.utils.data.Dataset):
    """
    Description: Make a Pytorch dataloader from the provided samples and labels.
    """

    def __init__(self, samples, labels, transform=None):
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # get sample and label by idx
        pack = {0: self.samples[idx], 1: self.labels[idx]}

        # add transform
        if self.transform:
            pack = self.transform(pack)

        return pack


def make_mnist_dataloaders(batch_size=16, num_workers=1, data_dir='/tmp/mnist_data/'):
    """
    Description: Make a Pytorch dataloader for the MNIST training and testing sets that includes
        preprocessing transforms and batching.
    Args:
        - batch_size (int): size of each batch returned by the dataloader iterable object.
        - num_workers (int): number of threads to use for dataloader pre-fetching operations. This
            can usually be set to match the number of cpu cores available on your machine.
        - data_dir (str): directory to download MNIST data into.
    Returns:
        - train_loader (torch.utils.data.DataLoader): Python iteratable dataloader for MNIST
            training set.
        - test_loader (torch.utils.data.DataLoader): Python iteratable dataloader for MNIST
            testing set.
    """

    # define the transform chain to preprocess each sample as it is passed to a batch
    #   1. resize the sample (image) to 32x32 (h, w)
    #   2. convert resized sample to Pytorch tensor
    #   3. normalize sample values (pixel values) using mean 0.5 and stdev 0,5; [0, 255] -> [-1, 1]
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    )

    # load mnist training dataset with torchvision
    train_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    # load mnist testing dataset with torchvision
    test_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    # construct dataloader for the training set
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    # construct dataloader for the testing set
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, test_loader
