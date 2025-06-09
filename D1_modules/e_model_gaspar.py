##### OBJ - MODEL1: initialize, compile, train and evaluate #####
'''PENSER A SAUVEGARDER LE MODELE + METTRE F1 METRIQUE + REGARDER BALANCING + CORRIGER 9 COLONNES'''

##### LIBRARIES #####
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import re

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
from itertools import product

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
def initialize_model_baseline(shape, dropout_ratio=0.2) -> Model:
    """Initialize the Neural Network with random weights"""
    model = Sequential() # Initialize a linear stack of layers

    # Input layer
    model.add(layers.Input(shape=(shape[1], shape[2], shape[3]))) # Input the correct shape (64, 64, 3)

    # Block 1
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')) # Captures local patterns
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_ratio))

    # Block 2
    model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same')) # Captures more abtract patterns
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_ratio))

    # Block 3
    model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')) # Captures more abtract patterns
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(dropout_ratio))

    # Classification Layer
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

# train model
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
                        batch_size=batch_size,
                        callbacks=[es],
                        verbose=1)
    print(f"✅ Model trained on {len(X)} rows")
    #print(f"Métriques : {history.history}")
    return model, history

def evaluate_model_baseline(model: Model,X: np.ndarray, y: np.ndarray,
                            batch_size=int(BATCH_SIZE)) -> tuple[Model, dict]:
    """Evaluate trained model performance on the dataset"""
    metrics = model.evaluate(x=X, y=y, batch_size=batch_size,verbose=1,
                             #callbacks=None,
                             return_dict=True)
    loss = metrics["loss"]
    #print(metrics)
    accuracy = metrics["accuracy"]
    precision = metrics["precision"]
    recall = metrics["recall"]
    print(f"✅ Model evaluated, accuracy: {accuracy}, recall: {recall}, precision: {precision}")
    return metrics

def plot_and_save_losses(history, test_loss, model_name):
    # Create a folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Plot losses
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.axhline(test_loss, color='red', linestyle='--', label='Test Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the figure
    plot_path = f"models/{model_name}_loss.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📉 Loss plot saved to {plot_path}")

def save_model_with_name(model, length, width, decibels, model_type, lr, bs, pat, val_split, epochs, version="v1", dropout_ratio=0.2):
    """
    Saves a Keras model with a standardized filename based on training params.

    Returns:
        model_name (str): Name used to save the model, useful for logging or plotting
    """
    model_name = f"Spec_{length}*{width}p_RGB_{decibels}db_{model_type}_lr{lr}_bs{bs}_pat{pat}_drop{dropout_ratio}_vs{int(val_split*100)}_{epochs}ep_{version}"
    save_path = f"models/{model_name}.keras"
    os.makedirs("models", exist_ok=True)
    model.save(save_path)
    print(f"💾 Model saved to {save_path}")
    return model_name

def log_metrics_csv(model_name, metrics):
    path = "models/metrics_log_3.csv"
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model_name'] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        row = {'model_name': model_name}
        row.update(metrics)
        writer.writerow(row)

def get_actor_id(file_path: str) -> int:
    match = re.search(r'Actor_(\d{2})', file_path)
    if not match:
        raise ValueError(f"Actor ID not found in path: {file_path}")
    return int(match.group(1))

if __name__ == '__main__':
    # Parameters
    l = 64
    w = 32
    decibels = 60
    channels = "L"
    model_string = "CNN"
    lr = [0.00005, 0.0001, 0.0005]
    batch_size = [32, 64]
    patience = [5, 10, 20]
    dropout_ratio = [0.2, 0.3, 0.4]
    validation_split = 0.15
    epochs = 1
    version = "v3"

    dictionnary=load_prepoc_data(color_mode = channels) # This is a dictionnary
    # Keys = file names (e.g: Spectograms_64p_RGB_60db/Actor_01/03-01-01-01-01-01-01)
    # Values = tensor that represents each image (shape = (64,32,3))

    for lr_, bs_, pat_, dr_ in product(lr, batch_size, patience, dropout_ratio):
        print(f"🚀 Training with lr={lr_}, bs={bs_}, patience={pat_}, dropout_ratio={dr_}")

        # X and Y
        TRAIN_ACTORS = set(range(1, 17))  # Actor IDs 1–20
        TEST_ACTORS = set(range(17, 20))  # Actor IDs 21–24

        train_dict = {k: v for k, v in dictionnary.items() if get_actor_id(k) in TRAIN_ACTORS}
        # Loop through each key (k) and value (v) in the dictionnary
        # The keys are the file names & the values are the tensors that represent each image
        # We create a new dictionnary with the same keys and values
        # Only for the IDs that are in the train actors list
        test_dict = {k: v for k, v in dictionnary.items() if get_actor_id(k) in TEST_ACTORS}

        X_train = X_value(train_dict)
        y_train = y_value(train_dict)

        # 🔀 Shuffle training data
        from sklearn.utils import shuffle
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        X_test  = X_value(test_dict)
        y_test  = y_value(test_dict)

        # Model Training
        model = initialize_model_baseline(X_train.shape, dr_)
        model = compile_model_baseline(model, lr_)
        model, history = train_model_baseline(model,X_train, y_train, batch_size=bs_, patience=pat_, validation_split=validation_split, epochs=epochs)
        metrics = evaluate_model_baseline(model, X_test, y_test)
        test_loss = metrics["loss"]

        # Save & Plot Model
        model_name = model_name = save_model_with_name(
        model=model,
        length = l,
        width = w,
        decibels=decibels,
        model_type=model_string,
        lr=lr_,
        bs=bs_,
        pat=pat_,
        val_split=validation_split,
        epochs=epochs,
        version=version,
        dropout_ratio=dr_
    )
        plot_and_save_losses(history, test_loss, model_name)
        log_metrics_csv(model_name, metrics)
