'''
General data handling utilities.
'''

import pandas as pd
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt

def generate_df_from_image_dataset(path):
    # read train/test label files to dataframe
    train_df = pd.read_csv('{}train_labels.csv'.format(path))
    test_df = pd.read_csv('{}test_labels.csv'.format(path))

    # convert filename column to absolute paths
    train_df['Filename'] = train_df['Filename'] \
        .map(lambda x: '{}train/{}'.format(path, x))
    test_df['Filename'] = test_df['Filename'] \
        .map(lambda x: '{}test/{}'.format(path, x)).to_list()

    # package data to dictionary
    data_dict = {'train': train_df, 'test': test_df}

    return data_dict

def make_gif_from_numbered_images(wildcard_str, dest_path='/tmp/nice.gif'):
    '''
    Description: Given a wildcard string to match all image files within a directory (e.g.
        '/tmp/mymodel*.png'), construct a gif from all files matching this wildcard. Requires
        image file names end with an order number (e.g. 'image_<order_number>.png') and gif is constructed in ascending order by order number. Saves gif to /tmp directory by default.
    Args:
        - wildcard_str (string): wildcard string to match all image files to make gif.
        - dest_path (string): path and filename to save gif.
    Returns:
        - None
    '''

    # get all images that match wildcard string
    img_files = glob.glob(wildcard_str)

    # sort filenames by numbers
    img_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))

    # create list if image objects
    img_list = [imageio.imread(img) for img in img_files]

    # write image list to gif
    imageio.mimwrite(dest_path, img_list, fps=200)

def tile_images(imgs):
    '''
    Description: Given a batch of images (must be a perfect square number of samples e.g. 64),
        organize them into a larger tiled image.
    Args:
        - image (4D numpy array): batch of 3D images.
    Returns:
        - tile_imgs (3D numpy array): single image of tiled image batch.
    '''

    # scale pixel values to [0, 255] and cast to unsigned 8-bit integers (common image datatype)
    min_data, max_data = [float(np.min(imgs)), float(np.max(imgs))]
    min_scale, max_scale = [0., 255.]
    imgs = ((max_scale - min_scale) * (imgs - min_data) / (max_data - min_data)) + min_scale
    imgs = imgs.astype(np.uint8)

    # tile images to larger image
    n_dim, h_dim, w_dim, d_dim = imgs.shape
    b_h, b_w = int(np.sqrt(n_dim)), int(np.sqrt(n_dim))
    tile_imgs = np.zeros((b_h*h_dim, b_w*w_dim, d_dim), dtype=np.uint8)
    for t_idx in range(n_dim):
        n_idx = w_dim * (t_idx % b_w)
        m_idx = h_dim * (t_idx // b_h)
        tile_imgs[n_idx:(n_idx+h_dim), m_idx:(m_idx+w_dim), :] = imgs[t_idx]

    return tile_imgs

def graph_inputs_with_predictions(
    inputs, labels, output_probs, output_preds, figsize=(8, 10)):

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
        axs[i, 1].set_xlabel('Class')
        axs[i, 1].set_ylabel('Probability')

    fig.tight_layout()

    return fig
