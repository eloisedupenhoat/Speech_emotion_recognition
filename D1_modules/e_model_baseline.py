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


###################################################################
#########              DEFINITION OF X AND y              #########
###################################################################
def X_value(dictionnary):
    X_list = []
    for value in dictionnary.values():
        X_list.append(value)
    return np.array(X_list)

def y_value(dictionnary):
    y_list = []
    for key in dictionnary:
        key_emotion = emotion(key)
        y_list.append(key_emotion)
    y_list = to_categorical(y_list, num_classes=9) # Va de 0 à 8 (ald 1 à 8)
    return np.array(y_list)

###################################################################
#########                MODEL DEFINITION                 #########
###################################################################
def initialize_model_baseline() -> Model:
    """Initialize the Neural Network with random weights"""
    model = Sequential()
    model.add(layers.Input(shape=(64, 64, 3))) # Attention : mettre une variable pour les 64 et le 3
    model.add(layers.Conv2D(50, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(25, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Conv2D(10, kernel_size=(3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(9, activation='softmax')) # Attention : maj ac le nombre d'émotions
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
                        patience=2,
                        validation_data=None, # overrides validation_split
                        validation_split=0.3) -> tuple[Model, dict]:
    """Fit the model and return a tuple (fitted_model, history)"""
    es = EarlyStopping(monitor="val_loss",
                       patience=patience,
                        restore_best_weights=True,
                        verbose=1)
    history = model.fit(X,y,
                        validation_data=validation_data,
                        validation_split=validation_split,
                        epochs=1,
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

    dictionnary=load_prepoc_data()
    X = X_value(dictionnary)
    y = y_value(dictionnary)
    # print(X.shape)
    # print(y.shape)

    model = initialize_model_baseline()
    compile_model_baseline(model)
    model, history = train_model_baseline(model,X,y)
    metrics = evaluate_model_baseline(model,X, y)
