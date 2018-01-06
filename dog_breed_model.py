from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras.models import Sequential
from keras.preprocessing import image
import numpy as np
import pickle

PATH_TO_MODEL_WEIGHTS = 'saved_models/weights.best.Resnet50Dogs.hdf5'

def extract_Resnet50(tensor):
    """Uses pretrained Resnet weights to calculate the feature vector before the first FC layer
    :param tensor: an image tensor of shape 224x224x3
    :returns: a feature vector from the last identity block before the FC layers.
    """
    return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

def dog_breed_model():
    """Defines the last layers of the model.
    :returns: a keras sequential model.
    """
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    model.add(Dropout(0.2))
    model.add(Dense(133, activation='softmax'))
    return model

def path_to_tensor(img_path):
    """Creates a tensor out of an image path.
    :returns: An image tensor of shape 224x224x3
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def load_breed_names():
    """Reads the dog breeds from the pickle file.
    :returns: the list of dog breeds
    """
    with open('breeds.pkl', 'rb') as file:
        breed_names = pickle.load(file)
    return breed_names