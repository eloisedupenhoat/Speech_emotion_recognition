##### OBJ - MODEL1: initialize, compile, train and evaluate #####
'''PENSER A SAUVEGARDER LE MODELE + METTRE F1 METRIQUE + REGARDER BALANCING + CORRIGER 9 COLONNES'''

##### LIBRARIES #####
import numpy as np
import pandas as pd
import os

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.metrics import Precision, Recall

# Import all variables from our params.py + functions from b_data...
from params import *
from b_data import *
from a_utils import *

# Grid iteration
from itertools import izip

###################################################################
#########              DEFINITION OF X AND y              #########
###################################################################
def X_value(dictionnary): # Return an np.array list of tensors (images)
    X_list = [] # Create empty list
    for value in dictionnary.values(): # Iterate through each value of the dictionnary (each tensor)
        X_list.append(value) # Append each tensor to a list
    return np.array(X_list) # Return an np.array list of tensors (images)

def y_value(dictionnary): # Returns a one-encoded emotion array (8 classes = 8 possible emotions)
    y_list = [] # Create empty list
    for key in dictionnary: # Iterate through each key of the dictionnary (each file name)
        key_emotion = emotion(key) - 1 # Slices the emotion value (from 0 to 7 instead of 1 to 8) for each key (each file)
        y_list.append(key_emotion) # Adds that values to the list
    y_list = to_categorical(y_list, num_classes=8) # One-hot encodes the emotion value (8 classes = 8 possible emotions)
    return np.array(y_list)

###################################################################
#########                MODEL DEFINITION                 #########
###################################################################
def initialize_model_baseline(shape) -> Model:
    """Initialize the Neural Network with random weights"""
    model = Sequential() # Initialize a linear stack of layers
    model.add(layers.Input(shape=(shape[1], shape[2], shape[3]))) # Input the correct shape (64, 64, 3)
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu')) # Captures local patterns
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu')) # Captures more abtract patterns
    model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu')) # Captures even more abtract patterns
    model.add(layers.Flatten()) # Converts the 3D tensor into a 1D vector (flattens)
    model.add(layers.Dense(8, activation='softmax')) # Final classification layer (8 emotions = 8 neurons)
    print("✅ Model initialized")
    return model

def compile_model_baseline(model: Model, learning_rate=0.0005) -> Model:
    """Compile the Neural Network"""
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy','recall', 'precision'])
    print("✅ Model compiled")
    return model

def train_model_baseline(model: Model,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size=int(BATCH_SIZE),
                        patience=5,
                        validation_data=None, # overrides validation_split
                        validation_split=0.3,
                        epochs=100) -> tuple[Model, dict]:
    """Fit the model and return a tuple (fitted_model, history)"""
    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                        restore_best_weights=True,
                        verbose=1)
    history = model.fit(X,y,
                        validation_data=validation_data,
                        validation_split=validation_split,
                        epochs=epochs,
                        batch_size=int(BATCH_SIZE),
                        callbacks=[es],
                        verbose=1)
    print(f"✅ Model trained on {len(X)} rows")
    #print(f"Métriques : {history.history}")
    return model, history

def evaluate_model_baseline(model: Model,X: np.ndarray, y: np.ndarray,
                            batch_size=int(BATCH_SIZE)) -> tuple[Model, dict]:
    """Evaluate trained model performance on the dataset"""
    metrics = model.evaluate(x=X, y=y, batch_size=int(BATCH_SIZE),verbose=1,
                             #callbacks=None,
                             return_dict=True)
    loss = metrics["loss"]
    #print(metrics)
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    print(f"✅ Model evaluated, accuracy: {accuracy}, recall: {recall}, precision: {precision}")
    return metrics

if __name__ == '__main__':
    dictionnary=load_prepoc_data() # This is a dictionnary
    # Keys = file names (e.g: Spectograms_64p_RGB_60db/Actor_01/03-01-01-01-01-01-01)
    # Values = tensor that represents each image (shape = (64,64,3))

    # Parameters
    resolution = 64
    decibels = 60
    channels = "RGB"
    model_string = "CNN"
    lr = [0.0005, 0.005, 0.05]
    batch_size = [16, 32, 64, 128]
    patience = [3, 5, 10]
    validation_split = 0.30
    epochs = 20
    version = "v1"
    for a_row,b_row in izip(lr, batch_size, patience):
        for a_item, b_item in izip(a_row,b_row):
            if a_item.isWhatever:
                b_item.doSomething()

    # X and Y
    X = X_value(dictionnary)
    y = y_value(dictionnary)

    # Model Training
    model = initialize_model_baseline(X.shape)
    model = compile_model_baseline(model, lr)
    model, history = train_model_baseline(model,X, y, batch_size=batch_size, patience=patience, validation_split=validation_split, epochs=epochs)
    metrics = evaluate_model_baseline(model, X, y)

    # Save Model
    model_name = f"Spec_{64}p_RGB_{decibels}db_{model_string}_lr{lr}_bs{batch_size}_pat{patience}_vs{validation_split}_{epochs}ep_{version}"
    model.save(f"models/{model_name}.keras")
