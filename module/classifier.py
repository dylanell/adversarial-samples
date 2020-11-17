"""
CNN classifier implemented in PyTorch.
"""

import torch

# torch activation functions
activation = {
    'relu': torch.nn.ReLU(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'tanh': torch.nn.Tanh(),
    'sigmoid': torch.nn.Sigmoid()
}


class Classifier(torch.nn.Module):
    # initialize and define all layers
    def __init__(self, image_dims, out_dim, hid_act='relu', norm=None):
        # run base class initializer
        super(Classifier, self).__init__()

        # define convolution layers
        self.conv_1 = torch.nn.Conv2d(
            image_dims[-1], 32, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(
            32, 64, 3, stride=2, padding=1)
        self.conv_3 = torch.nn.Conv2d(
            64, 128, 3, stride=2, padding=1)
        self.conv_4 = torch.nn.Conv2d(
            128, 256, 3, stride=2, padding=1)
        self.conv_5 = torch.nn.Conv2d(
            256, 512, 3, stride=2, padding=1)

        # define layer norm layers
        if norm == 'layer':
            self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
            self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
            self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
            self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
            self.norm_5 = torch.nn.LayerNorm([512, 2, 2])
        elif norm == 'batch':
            self.norm_1 = torch.nn.BatchNorm2d(32)
            self.norm_2 = torch.nn.BatchNorm2d(64)
            self.norm_3 = torch.nn.BatchNorm2d(128)
            self.norm_4 = torch.nn.BatchNorm2d(256)
            self.norm_5 = torch.nn.BatchNorm2d(512)
        else:
            self.norm_1 = torch.nn.Identity()
            self.norm_2 = torch.nn.Identity()
            self.norm_3 = torch.nn.Identity()
            self.norm_4 = torch.nn.Identity()
            self.norm_5 = torch.nn.Identity()

        # define fully connected layers
        self.fc_1 = torch.nn.Linear(512 * 2 * 2, out_dim)

        # define activation functions
        self.hid_act = activation[hid_act]

    # compute forward propagation of input x
    def forward(self, x):
        # compute output
        z_1 = self.hid_act(self.norm_1(self.conv_1(x)))
        z_2 = self.hid_act(self.norm_2(self.conv_2(z_1)))
        z_3 = self.hid_act(self.norm_3(self.conv_3(z_2)))
        z_4 = self.hid_act(self.norm_4(self.conv_4(z_3)))
        z_5 = self.hid_act(self.norm_5(self.conv_5(z_4)))
        z_5_flat = torch.flatten(z_5, start_dim=1)
        z_6 = self.fc_1(z_5_flat)
        return z_6

    # return all hidden feature representations
    def hidden(self, x):
        # compute output
        z_1 = self.hid_act(self.norm_1(self.conv_1(x)))
        z_2 = self.hid_act(self.norm_2(self.conv_2(z_1)))
        z_3 = self.hid_act(self.norm_3(self.conv_3(z_2)))
        z_4 = self.hid_act(self.norm_4(self.conv_4(z_3)))
        z_5 = self.hid_act(self.norm_5(self.conv_5(z_4)))
        z_1_flat = torch.flatten(z_1, start_dim=1)
        z_2_flat = torch.flatten(z_2, start_dim=1)
        z_3_flat = torch.flatten(z_3, start_dim=1)
        z_4_flat = torch.flatten(z_4, start_dim=1)
        z_5_flat = torch.flatten(z_5, start_dim=1)
        return [z_1_flat, z_2_flat, z_3_flat, z_4_flat, z_5_flat]
