##### OBJ - MODEL1: initialize, compile, train and evaluate #####

##### LIBRARIES #####
import numpy as np
import pandas as pd
import os

from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping

# Import all variables from our params.py
from params import *

def initialize_model_baseline(input_shape: tuple) -> Model:
    """Initialize the Neural Network with random weights"""
    model = Sequential()
    model.add(Input(shape=(10,)))
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dense(25, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    print("✅ Model initialized")
    return model

def compile_model_baseline(model: Model, learning_rate=0.0005) -> Model:
    """Compile the Neural Network"""
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    print("✅ Model compiled")
    return model

def train_model_baseline(model: Model,
                        X: np.ndarray,
                        y: np.ndarray,
                        batch_size=BATCH_SIZE,
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
                        batch_size=BATCH_SIZE,
                        callbacks=[es],
                        verbose=1)
    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")
    return model, history

def evaluate_model_baseline(model: Model,X: np.ndarray, y: np.ndarray,
                            batch_size=BATCH_SIZE) -> tuple[Model, dict]:
    """Evaluate trained model performance on the dataset"""
    metrics = model.evaluate(x=X, y=y, batch_size=BATCH_SIZE,verbose=1,
                             #callbacks=None,
                             return_dict=True)
    loss = metrics["loss"]
    mae = metrics["mae"]
    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")
    return metrics

if __name__ == '__main__':

    X='image'
    y='numero_extrait_du_titre'
    model = initialize_model_baseline()
    compile_model_baseline(model)
    model, history = train_model_baseline(model,X,y)
    metrics = evaluate_model_baseline(model,X, y)
