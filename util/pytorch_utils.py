"""
Pytorch Dataloader utils.
"""

import torch
from torchvision.transforms import transforms

from util.pytorch_datasets import ImageDataset


# Given a dataframe of the form [img_paths, labels], construct a TensorFlow
# dataset object and perform all of the standard image dataset processing
# functions (resizing, standardization, etc.).
def build_image_dataset(
        dataframe, image_size=(32, 32), batch_size=64, num_workers=1):
    # define the transform chain to process each sample
    # as it is passed to a batch
    #   1. resize the sample (image) to 32x32 (h, w)
    #   2. convert resized sample to Pytorch tensor
    #   3. normalize sample values (pixel values) using
    #      mean 0.5 and stdev 0,5; [0, 255] -> [0, 1]
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # create dataset
    dataset = ImageDataset(dataframe, transform=transform)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return dataset, dataloader


def silhouette_coefficient(x, y):
    """
    Computing the Silhouette Coefficient of hidden
    representations.
    """

    # get unique labels present in this batch
    unique_labels, unique_counts = torch.unique(
        y, return_counts=True)

    # remove unique labels where we only have one class sample
    unique_labels = unique_labels[unique_counts > 1]

    '''
    a: The mean distance between a sample and all other
    points in the same class. Calling this 'mean_intra_class_dist'.
    '''

    # construct pairwise euclidean distance matrix of
    # latent representations
    n = x.shape[0]
    d = x.shape[1]
    a = x.unsqueeze(1).expand(n, n, d)
    b = x.unsqueeze(0).expand(n, n, d)
    x_dist_matrix = torch.pow(a - b, 2).sum(dim=2)
    # print(x_dist_matrix)

    # extract pairwise distances only for members of the
    # same class for each unique class in the label batch
    # and only if there are more than 1 instance of a class
    class_wise_x_dist_matrices = [x_dist_matrix[y == l][:, y == l]
                                  for j, l in enumerate(unique_labels)]
    # print(class_wise_x_dist_matrices[0])

    # grab only the upper triangular values of each
    # class-wise representation distance matrix to compute
    # the mean distance between all different samples of
    # the same class
    mean_class_wise_rep_dists = torch.cat([
        torch.mean(class_wise_x_dist_matrices[j][torch.triu(torch.ones(
            class_wise_x_dist_matrices[j].shape[0],
            class_wise_x_dist_matrices[j].shape[0]),
            diagonal=1) == 1]).expand(1)
        for j, l in enumerate(unique_labels)], dim=0)

    # compute the total mean class-wise representation
    # distance across all classes
    mean_intra_class_dist = torch.mean(mean_class_wise_rep_dists)

    '''
    b: The mean distance between a sample and all other points in
    the next nearest cluster. Calling this 'mean_inter_class_dist'.
    '''

    # compute average class-wise feature representations for each
    # unique label in the current label batch (class clusters)
    class_clusters = torch.cat([torch.mean(x[y == l], dim=0, keepdim=True)
                                for j, l in enumerate(unique_labels)], dim=0)
    # print(class_clusters)

    # compute pairwise distances between each class cluster
    n = class_clusters.shape[0]
    d = class_clusters.shape[1]
    a = class_clusters.unsqueeze(1).expand(n, n, d)
    b = class_clusters.unsqueeze(0).expand(n, n, d)
    class_cluster_dist_matrix = torch.pow(a - b, 2).sum(dim=2)
    # print(class_cluster_dist_matrix)

    # get nearest clusters
    nearest_cluster = torch.argsort(class_cluster_dist_matrix, dim=1)[:, 1]
    # print(nearest_cluster)

    # get distance between each point in cluster i, to each point
    # in cluster nearest to i, and compute mean of these distances
    # for each cluster, i.
    inter_dists = []
    for j, l in enumerate(unique_labels):
        test_points = x[y == l]
        nearest_points = x[y == unique_labels[nearest_cluster[j]]]
        n = test_points.shape[0]
        m = nearest_points.shape[0]
        a = test_points.unsqueeze(-1).repeat(1, 1, m)
        b = torch.transpose(
            nearest_points.unsqueeze(-1).repeat(1, 1, n), dim0=2, dim1=0)
        test_nearest_dist_matrix = torch.pow(a - b, 2).sum(dim=1)
        inter_dists.append(torch.mean(test_nearest_dist_matrix).expand(1))

    # compute mean of mean nearest inter class distances for each
    # cluster, i
    mean_inter_class_dist = torch.mean(torch.cat(inter_dists, dim=0))
    # print(mean_inter_class_dist)

    '''
    s: Compute the Silhouette Coefficient of latent space
    representations given class assignments from labels.
    '''

    # silhouette coeff
    s = (mean_inter_class_dist - mean_intra_class_dist) \
        / torch.max(mean_intra_class_dist, mean_inter_class_dist)

    return s
