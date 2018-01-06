from dog_breed_model import dog_breed_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys, getopt
from dataset import get_all_datasets
from dog_breed_model import PATH_TO_MODEL_WEIGHTS

def train():
    """Trains the model
    """
    dataset = get_all_datasets()
    train_set, train_targets = dataset[0]
    valid_set, valid_targets = dataset[1]

    Resnet50_breed_model = dog_breed_model()
    Resnet50_breed_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=PATH_TO_MODEL_WEIGHTS, verbose=1, save_best_only=True)

    Resnet50_breed_model.fit(train_set, train_targets, 
            validation_data=(valid_set, valid_targets),
            epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

if __name__ == '__main__':
    train()
