from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from dog_breed_model import load_breed_names

PATH_TO_TRAIN_SET = 'dogImages/train'
PATH_TO_VALIDATION_SET = 'dogImages/valid'
PATH_TO_TEST_SET = 'dogImages/test'
PATH_TO_BOTTLENECK_FEATURES = 'bottleneck_features/DogResnet50Data.npz'

def load_dataset(path):
    """Loads a dataset
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def get_all_datasets():
    """Loads the train, validation and test sets. The targets are loaded from the origin files.
    The precomputed ResNet50 values are used for training to make the training process faster.
    :returns: A list of tuples of the following structure:
    [(train, train_targets), (valid, valid_targets), (test, test_targets)]
    """
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset(PATH_TO_TRAIN_SET)
    valid_files, valid_targets = load_dataset(PATH_TO_VALIDATION_SET)
    test_files, test_targets = load_dataset(PATH_TO_TEST_SET)

    # load list of dog names
    breed_names = load_breed_names()

    # Instead of using the images, the precomputed outputs of the images from ResNet are used.
    # Obtain bottleneck features.
    bottleneck_features = np.load(PATH_TO_BOTTLENECK_FEATURES)
    train_Resnet50 = bottleneck_features['train']
    valid_Resnet50 = bottleneck_features['valid']
    test_Resnet50 = bottleneck_features['test']

    # print statistics about the dataset
    print('There are %d total dog categories.' % len(breed_names))
    print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('There are %d training dog images.' % len(train_files))
    print('There are %d validation dog images.' % len(valid_files))
    print('There are %d test dog images.'% len(test_files))

    return [(train_Resnet50, train_targets), (valid_Resnet50, valid_targets), (test_Resnet50, test_targets)]