"""
General data handling utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def inputs_with_outputs(inputs, labels, output_probs, output_preds):

    # remap inputs to [0 - 255] and convetr to uint 8
    new_min, new_max = 0, 255
    min, max = np.min(inputs), np.max(inputs)
    inputs = (((inputs - min) / (max - min)) * (new_max - new_min)) + new_min
    inputs = inputs.astype(np.uint8)

    # count number of rows
    num_rows = inputs.shape[0]

    # make a figure to hold multiple plots
    fig, axs = plt.subplots(num_rows, 2, figsize=(8, 20))

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

    fig.tight_layout()

    return fig
