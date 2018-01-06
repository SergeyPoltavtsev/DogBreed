import numpy as np
import sys, getopt
from detectors import face_detector, dog_detector
from dog_breed_model import extract_Resnet50, dog_breed_model, path_to_tensor, load_breed_names
from dog_breed_model import PATH_TO_MODEL_WEIGHTS

def pretty_output(breed, is_human):
    """Constructs a pretty output strings
    """
    if(is_human):
        print("The human in the picture looks like " + breed)
    else:
        print("The dog breed is " + breed)

def breed_service(img_path):
    """Checks if there is a dog of a human face in the picture. 
    If so runs the model forward path in order to predict dog breed in case of dog 
    or resembling breed in case of human face.
    """
    # The dog detector is the first one as it is more accurate.
    if(dog_detector(img_path)):
        predicted_breed = predict_breed(img_path)
        pretty_output(predicted_breed, False)
    elif(face_detector(img_path)):
        resembling_breed = predict_breed(img_path)
        pretty_output(resembling_breed, True)
    else:
        print("There is not a human or a dog in the image")

def predict_breed(img_path):
    """Predicts the bog breed based on the image
    ":returns: A dog breed
    """
    breed_names = load_breed_names();

    Resnet50_model = dog_breed_model()
    # loading the model weights
    Resnet50_model.load_weights(PATH_TO_MODEL_WEIGHTS)
    # create a tensor out of the image
    img_tensor = path_to_tensor(img_path)
    # get the feature vector from the Resnet50
    resnet50_features = extract_Resnet50(img_tensor)
    # feed the extracted features into the retrained FC layers.
    predicted_vector = Resnet50_model.predict(resnet50_features)
    return breed_names[np.argmax(predicted_vector)]

def usage():
    print('guess_breed.py -i <path_to_image>')

if __name__ == '__main__':
    image_path = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hi:",["image="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-i", "--image"):
            image_path = arg
    if not image_path:
        usage()
    else:
        print('image path is ', image_path)
        breed_service(image_path)
