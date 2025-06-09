# D1_modules/e_model_mfcc1.py

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def focal_loss(gamma=2., alpha=.25):
    def focal(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        ce = tf.keras.backend.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        return tf.reduce_mean(alpha * tf.pow(1 - pt, gamma) * ce)
    return focal

def build_compile_model(n_classes, MAX_LEN=200, n_feats=137):
    """
    Construit et compile un modèle CNN1D + BiLSTM.
    - n_classes  : nombre de classes de sortie (taille du softmax).
    - MAX_LEN    : dimension temporelle (200).
    - n_feats    : nombre de features (ici 137).
    """
    from tensorflow.keras.layers import (
        Input, Conv1D, BatchNormalization, Dropout,
        Bidirectional, LSTM, GlobalAveragePooling1D, Dense
    )

    inputs = Input(shape=(MAX_LEN, n_feats))

    # Bloc Conv1D 1
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)    
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Bloc Conv1D 2
    x = Conv1D(64, 3, padding='same', activation='relu')(x)         
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Bloc Conv1D 3
    x = Conv1D(96, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # BiLSTM
    x = Bidirectional(LSTM(64, return_sequences=True))(x)           
    x = GlobalAveragePooling1D()(x)

    # Dense final
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),  
        loss=focal_loss(gamma=2., alpha=0.75),
        metrics=['accuracy']
    )
    return model

def train_and_evaluate(X_train, y_train_cat, X_val, y_val_cat, X_test, y_test, le, class_weights_dict):
    """
    Entraîne le modèle, sauvegarde les meilleurs poids, puis affiche classification_report + heatmap.
    - X_train, y_train_cat  : numpy arrays (train).
    - X_val, y_val_cat      : numpy arrays (validation).
    - X_test, y_test        : (test).
    - le                    : LabelEncoder (pour récupérer le nom de chaque classe).
    - class_weights_dict    : dictionnaire pour gérer le déséquilibre des classes.
    """
    # 1) Build / compile
    n_classes = y_train_cat.shape[1]
    model = build_compile_model(n_classes=n_classes)

    # 2) Callbacks
    cb = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True, 
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=8, 
            factor=0.5, 
            min_lr=1e-6, 
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            './best_model.h5', 
            monitor='val_accuracy', 
            save_best_only=True, 
            mode='max'
        )
    ]

    # 3) Entraînement
    history = model.fit(
        X_train.transpose(0, 2, 1),  # (samples, time=200, features=137)
        y_train_cat,
        validation_data=(X_val.transpose(0, 2, 1), y_val_cat),
        epochs=80,
        batch_size=16,
        class_weight=class_weights_dict,
        callbacks=cb
    )

    # 4) Évaluation sur le test set
    model.load_weights('./best_model.h5')
    y_pred = model.predict(X_test.transpose(0, 2, 1), verbose=0).argmax(axis=1)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title("Matrice de confusion après retraining")
    plt.show()

    return history, model
