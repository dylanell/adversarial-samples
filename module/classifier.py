'''
CNN classifier implemented in PyTorch.
'''

import torch
import torch.nn.functional as F

class Classifier(torch.nn.Module):
    # initialize and define all layers
    def __init__(self, image_dims, out_dim):
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
        self.norm_1 = torch.nn.LayerNorm([32, 32, 32])
        self.norm_2 = torch.nn.LayerNorm([64, 16, 16])
        self.norm_3 = torch.nn.LayerNorm([128, 8, 8])
        self.norm_4 = torch.nn.LayerNorm([256, 4, 4])
        self.norm_5 = torch.nn.LayerNorm([512, 2, 2])

        # define fully connected layers
        self.fc_1 = torch.nn.Linear(512*2*2, out_dim)

    # compute forward propagation of input x
    def forward(self, x):
        # compute output
        z_1 = self.norm_1(F.relu(self.conv_1(x)))
        z_2 = self.norm_2(F.relu(self.conv_2(z_1)))
        z_3 = self.norm_3(F.relu(self.conv_3(z_2)))
        z_4 = self.norm_4(F.relu(self.conv_4(z_3)))
        z_5 = self.norm_5(F.relu(self.conv_5(z_4)))
        z_5_flat = torch.flatten(z_5, start_dim=1)
        z_6 = self.fc_1(z_5_flat)
        return z_6

    # return all hiddeen feature representations
    def hidden(self, x):
        # compute output
        z_1 = self.norm_1(F.relu(self.conv_1(x)))
        z_2 = self.norm_2(F.relu(self.conv_2(z_1)))
        z_3 = self.norm_3(F.relu(self.conv_3(z_2)))
        z_4 = self.norm_4(F.relu(self.conv_4(z_3)))
        z_5 = self.norm_5(F.relu(self.conv_5(z_4)))
        z_1_flat = torch.flatten(z_1, start_dim=1)
        z_2_flat = torch.flatten(z_2, start_dim=1)
        z_3_flat = torch.flatten(z_3, start_dim=1)
        z_4_flat = torch.flatten(z_4, start_dim=1)
        z_5_flat = torch.flatten(z_5, start_dim=1)
        return [z_1_flat, z_2_flat, z_3_flat, z_4_flat, z_5_flat]
