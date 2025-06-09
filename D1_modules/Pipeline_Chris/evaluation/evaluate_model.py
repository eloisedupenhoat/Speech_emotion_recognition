# evaluation/evaluate_model.py

import numpy as np
import tensorflow as tf

from data.gcs_handler import download_npy_array

def evaluate():
    """
    Charge X_val.npy, y_val.npy (et X_test, y_test) depuis GCS,
    charge le modèle best_model.h5 et calcule les metrics.
    """
    print("[INFO] Téléchargement de X_val.npy, y_val.npy depuis GCS…")
    X_val = download_npy_array("Features/X_val.npy")
    y_val = download_npy_array("Features/y_val.npy")
    print(f"[INFO] X_val.shape = {X_val.shape}, y_val.shape = {y_val.shape}")

    print("[INFO] Téléchargement de X_test.npy, y_test.npy depuis GCS…")
    X_test = download_npy_array("Features/X_test.npy")
    y_test = download_npy_array("Features/y_test.npy")
    print(f"[INFO] X_test.shape = {X_test.shape}, y_test.shape = {y_test.shape}")

    print("[INFO] Chargement du modèle saved : best_model.h5")
    model = tf.keras.models.load_model("best_model_optuna.h5")

    # Évaluation sur validation
    print("[INFO] Évaluation sur le jeu de validation…")
    val_results = model.evaluate(X_val, y_val, verbose=0)
    print(f"  → Validation Loss = {val_results[0]:.4f}, Accuracy = {val_results[1]:.4f}")

    # Évaluation sur test
    print("[INFO] Évaluation sur le jeu de test…")
    test_results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  → Test Loss = {test_results[0]:.4f}, Accuracy = {test_results[1]:.4f}")

    # (Optionnel) Confusion Matrix, classification report, etc.
    preds = np.argmax(model.predict(X_test), axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n[RESULTS] Classification Report (test set) :")
    print(classification_report(y_test, preds))
    print("[RESULTS] Confusion Matrix (test set) :")
    print(confusion_matrix(y_test, preds))
