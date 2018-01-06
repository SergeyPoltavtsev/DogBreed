# Project overview
The project is a solution to the [Convolutional Neural Networks (CNN) task](https://github.com/udacity/dog-project) in the Udacity Deep Learning Nanodegree (DLND). The aim of the project is to predict the dog breed by a provided image. Moreover, if an image of a human is supplied, the algorithm will identify the resembling dog breed. 

# Architecture
The main focus was to use the transfer learning technique to get a suitable model fast. The pre-trained on the "ImageNet" dataset [ResNet50](https://keras.io/applications/#resnet50) model was used as the starting model. The last fully connected (FC) layers were sliced off and new fully connected were added and actually trained.

# Obtain the dataset (for training and testing)
- Obtain the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). Unzip and place the folder at location ./dogImages
- Obtain the [ResNet50 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) for the dog dataset and place them under ./bottleneck_features

# Dependencies
Please find the required dependencies under the [original project instructions](https://github.com/udacity/dog-project#instructions)

# Scripts
- `train.py` - trains the FC layers of the model from scratch.
- `test.py` - test the model and outputs the accuracy tested in the test set
- `guess_breed.py` - predicts dog breed or resembled dog breed.