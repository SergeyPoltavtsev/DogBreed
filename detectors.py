import cv2
import numpy as np
from dog_breed_model import path_to_tensor
from keras.applications.resnet50 import ResNet50, preprocess_input

def face_detector(img_path):
    """Detects if a human face is presented in the image.
    :returns: "True" if face is detected in image stored at img_path
    """
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def ResNet50_predict_labels(img_path):
    """Runs the forward path in the ResNet50 model and outputs the index of the best class
    :returns: the index of the most probable class.
    """
    # the ResNet50 model is used for dog identification.
    ResNet50_model = ResNet50(weights='imagenet')
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    """Detects if there is a dog in the image
    :returns: True of False depending whether there is a dog in the image
    """
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 