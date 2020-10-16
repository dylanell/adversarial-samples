'''
Script to train a CNN classifier with PyTorch.
'''

import yaml
import torch
from torchvision.transforms import transforms

from util.pytorch_utils import build_image_dataset
from util.data_utils import generate_df_from_image_dataset
from model.vanilla_classifier import VanillaClassifier
from model.adversarial_classifier import AdversarialClassifier
from model.smooth_classifier import SmoothClassifier
from model.feature_spread_classifier import FeatureSpreadClassifier

def main():
    # parse configuration file
    with open('config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # generate filenames/labels df from image data directory
    data_dict = generate_df_from_image_dataset(
        config['dataset_directory'])

    # add number of classes in labels to config
    config['output_dimension'] = data_dict['train']['Label'].nunique()

    # if training as adversarial, use the first 20000 training samples,
    # otherwise use the last 20000 training samples
    if config['adversary']:
        train_df = data_dict['train'].iloc[
            :int(config['number_train']/2), :]
    else:
        train_df = data_dict['train'].iloc[
            int(config['number_train']/2):, :]

    # add number of samples to config
    config['number_train'] = len(train_df)
    config['number_test'] = len(data_dict['test'])

    # build training dataloader
    train_set, train_loader = build_image_dataset(
        train_df,
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
    if config['model_type'] == 'vanilla_classifier':
        model = VanillaClassifier(config)
    elif config['model_type'] == 'adversarial_classifier':
        model = AdversarialClassifier(config)
    elif config['model_type'] == 'smooth_classifier':
        model = SmoothClassifier(config)
    elif config['model_type'] == 'feature_spread_classifier':
        model = FeatureSpreadClassifier(config)
    else:
        print('[ERROR]: unknown model type \'{}\''\
            .format(config['model_type']))
        exit()

    # train the model
    model.train_epochs(train_loader, test_loader)

if __name__ == '__main__':
    main()
