from dog_breed_model import dog_breed_model
from dataset import get_all_datasets
import numpy as np
from dog_breed_model import PATH_TO_MODEL_WEIGHTS

def test():
    """Tests the model
    """
    dataset = get_all_datasets()
    test_set, test_targets = dataset[2]

    Resnet50_breed_model = dog_breed_model()
    Resnet50_breed_model.load_weights(PATH_TO_MODEL_WEIGHTS)
    Resnet50_predictions = [np.argmax(Resnet50_breed_model.predict(np.expand_dims(feature, axis=0))) for feature in test_set]

    # report test accuracy
    test_accuracy = 100*np.sum(np.array(Resnet50_predictions)==np.argmax(test_targets, axis=1))/len(Resnet50_predictions)
    print('Test accuracy: %.4f%%' % test_accuracy)

if __name__ == '__main__':
    test()