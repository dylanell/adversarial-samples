# advserarial-samples
Exploration of adversarial samples generated for a CNN classifier.

### Environment:

- Python 3.7.4

### Python Packages:

- jupyterlab
- pytorch
- torchvision
- imageio
- pandas
- pyyaml
- mayplotlib
- plotly

### Image Dataset Format:

This project assumes you have the CIFAR-10 dataset pre-configured locally on your machine in the format described below. My [dataset-helpers](https://github.com/dylanell/dataset-helpers) Github project also contains tools that perform this local configuration automatically within the `cifar` directory of the project.

The CIFAR-10 dataset contains images of several different animals and objects including dogs, cats, automobiles, trucks, etc. Full information about the dataset, including download links can be found on the [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html). Using the helper scripts from the [dataset-helpers](https://github.com/dylanell/dataset-helpers) project, we organize the dataset into training/testing directories of raw image files (`png` or `jpg`) accompanied by a `csv` file listing one-to-one correspondences between the image file names and their label. This "generic image dataset format" is summarized by the directory tree structure below.

```
dataset_directory/
|__ train_labels.csv
|__ test_labels.csv
|__ train/
|   |__ train_image_01.png
|   |__ train_image_02.png
|   |__ ...
|__ test/
|   |__ test_image_01.png
|   |__ test_image_02.png
|   |__ ...   
```

Each labels `csv` file has the format:

```
Filename, Label
train_image_01.png, 4
train_image_02.png, 7
...
```

If you would like to re-use the code here to work with other image datasets, just format any new image dataset to follow the outline above and be sure to edit corresponding hyperparameters in the `config.yaml` file.

### Training:

Training options and hyperparameters are pulled from the `config.yaml` file and can be changed by editing the file contents. The `train.py` scripts only several specific values for the `model_type` variable in `coinfig.yaml` coorrespoinding to the type of classifier model you would like to train. Additionally, when the `adversary` variable is `False`, the training script uses the second half of the training data split as trainin data and when the `adversary` variable is `True`, it uses the first half of the training data split.

Train a classifier by running:

```
$ python train.py
```

### Jupyter Notebook:

This project is accompanied by a Jupyter notebook that explains some of the theory behind generating adversarial samples using the Fast Gradient Sign Method (FGSM) as well as some analysis of how several models perform on adversarial samples.

Run the following command to start the Jupyter notebook server in your browser:

```
$ jupyter-notebook notebook.ipynb
```

### References:
