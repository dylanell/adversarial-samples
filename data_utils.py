"""
General data handling utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

def make_adversarial_datasets(classifier, dataloader, eps_vals):
    # dictionary to hold adversarial datasets
    adv_datasets = {eps: [] for eps in eps_vals}

    # list to hold labels
    labels = []

    # run through all batches of the test loader
    for batch in dataloader:
        # parse batch
        batch_samples = batch[0]
        batch_labels = batch[1]

        # append labels
        labels.append(batch_labels.detach().numpy())

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

        # range of adversarial perturbution strengths
        for eps in eps_vals:

            # perturb test samples using FGSM
            adv_batch_samples = batch_samples + ((eps) * signed_grads)

            # keep pixel vals within [-1, 1]
            new_min, new_max = -1., 1.
            old_min, old_max = torch.min(adv_batch_samples), torch.max(adv_batch_samples)
            adv_batch_samples = (((adv_batch_samples - old_min) / (old_max - old_min)) * (new_max - new_min)) + new_min

            # append these adversarial examples to the dictionary
            adv_datasets[eps].append(adv_batch_samples.detach().numpy())

    # concatenate all lists in dataset dictionary to numpy arrays
    adv_datasets = dict(map(lambda x: (x[0], np.concatenate(x[1], axis=0)), adv_datasets.items()))

    # concatenate labels to numpy array
    labels = np.concatenate(labels, axis=0)

    return adv_datasets, labels

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
