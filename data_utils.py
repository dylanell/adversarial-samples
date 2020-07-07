"""
General data handling utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

def compute_pytorch_model_accuracy(model, data_loader):
    # counters for total number of samples in the dataloader and correct predictions
    total = 0
    correct = 0

    # iterate through batches
    for batch in data_loader:

        # parse batch and send to device
        sample_batch = batch[0]
        label_batch = batch[1]

        # get predicted outputs
        pred = torch.argmax(model(sample_batch), dim=1)

        # count where predicted == labels and add to correct count
        correct += torch.sum((pred == label_batch)).item()

        # add this batch size to total count
        total += sample_batch.shape[0]

    # compute accuracy on dataloader
    accuracy = 100.0 * (float(correct) / float(total))

    return accuracy

def inputs_with_outputs(inputs, labels, output_probs, output_preds, figsize=(8, 10)):

    # remap inputs to [0 - 255] and convetr to uint 8
    new_min, new_max = 0, 255
    min, max = np.min(inputs), np.max(inputs)
    inputs = (((inputs - min) / (max - min)) * (new_max - new_min)) + new_min
    inputs = inputs.astype(np.uint8)

    # count number of rows
    num_rows = inputs.shape[0]

    # make a figure to hold multiple plots
    fig, axs = plt.subplots(num_rows, 2, figsize=figsize)

    for i in range(inputs.shape[0]):

        # save current image to file
        img = inputs[i, :, :, 0]

        #axs[i, 0].axis('off')
        #axs[i, 1].axis('off')

        # add image trace
        axs[i, 0].imshow(img)
        axs[i, 0].set_title('Target: {}, Prediction: {}'.format(labels[i], output_preds[i]))
        axs[i, 0].axis('off')

        axs[i, 1].bar(x=range(len(output_probs[i])), height=output_probs[i])
        axs[i, 1].set_title('Prediction Probabilities')
        axs[i, 1].set_xticks(range(len(output_probs[i])))
        axs[i, 1].grid('major')
        axs[i, 1].set_ylim(0., 1.1)

    fig.tight_layout()

    return fig
