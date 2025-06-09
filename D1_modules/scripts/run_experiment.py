# scripts/run_experiment.py

import numpy as np
from D1_modules.d_preproc_mfcc import (
    load_and_split_data,
    build_XY_1D_balanced,
    extract_features
)
from D1_modules.e_model_mfcc1 import train_and_evaluate, build_compile_model
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# 1. Chargement + split
train_files, train_labels, val_files, val_labels, test_files, test_labels, le, n_classes = load_and_split_data()

# 2. Construction des jeux de données (avec oversampling sur liste de labels)
#    Récupérer l’index des classes à sur-échantillonner (par ex. '01', '03', '06')
index_01 = np.where(le.classes_ == '01')[0][0]
index_03 = np.where(le.classes_ == '03')[0][0]
index_06 = np.where(le.classes_ == '06')[0][0]

# 2.a) Train set avec oversampling sur [01, 03, 06]
X_train, y_train = build_XY_1D_balanced(
    train_files,
    train_labels,
    augmentations=[
        lambda y, sr: y,
        lambda y, sr: add_noise(y, 0.006),
        lambda y, sr: time_shift(y),
        lambda y, sr: pitch_shift(y, sr, steps=random.uniform(-2, 2)),
        lambda y, sr: stretch(y, rate=random.uniform(0.85, 1.15)),
    ],
    labels_to_oversample=[index_01, index_03, index_06]
)

# 2.b) Jeu de validation (aucun oversampling, augmentation identique à “raw”)
X_val, y_val = build_XY_1D_balanced(
    val_files,
    val_labels,
    augmentations=[lambda y, sr: y],
    labels_to_oversample=None
)

# 2.c) Jeu de test (idem que validation)
X_test, y_test = build_XY_1D_balanced(
    test_files,
    test_labels,
    augmentations=[lambda y, sr: y],
    labels_to_oversample=None
)

# 3. Normalisation 1D (moyenne/écart‐type calculés sur X_train)
mu    = X_train.mean(axis=(0, 2), keepdims=True)
sigma = X_train.std(axis=(0, 2), keepdims=True) + 1e-8

def norm_1D(X):
    return (X - mu) / sigma

X_train = norm_1D(X_train)
X_val   = norm_1D(X_val)
X_test  = norm_1D(X_test)

# 4. Encodage one-hot + class weights
y_train_cat = to_categorical(y_train, num_classes=n_classes)
y_val_cat   = to_categorical(y_val,   num_classes=n_classes)
y_test_cat  = to_categorical(y_test,  num_classes=n_classes)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

# 5. Lancer l’entraînement et l’évaluation
history, model = train_and_evaluate(
    X_train, y_train_cat,
    X_val,   y_val_cat,
    X_test,  y_test,
    le,
    class_weights_dict
)

# Optionnel : on peut sauvegarder history.history dans un .json ou tracer des courbes supplémentaires.
