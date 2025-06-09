# scripts/transposer_npy.py

import sys
import os

# --- 1) On ajoute la racine du projet au PYTHONPATH pour trouver config.py et data.gcs_handler ---
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)


# --- 2) Import nécessaire une fois le path corrigé ---
import numpy as np
from data.gcs_handler import download_npy_array, upload_npy_array
from config import MAX_LEN, FEATURE_DIM, FEATURES_PREFIX




def transpose_and_overwrite_on_gcs(mode: str):
    """
    - Télécharge X_{mode}.npy depuis le bucket GCS (Features/X_{mode}.npy)
    - Transpose (N, 137, 200) → (N, 200, 20)
    - Vérifie que la forme est (N, MAX_LEN, FEATURE_DIM)
    - Ré-uploade la version transposée au même emplacement GCS
    """
    blob_X = f"{FEATURES_PREFIX}X_{mode}.npy"
    print(f"[INFO] Tentative de téléchargement de '{blob_X}' depuis GCS…")

    try:
        X = download_npy_array(blob_X)
    except Exception as e:
        print(f"[WARN] Impossible de télécharger '{blob_X}' depuis GCS : {e}")
        return

    print(f"[INFO] Forme avant transpose : {X.shape}")
    # On suppose initialement que X.shape == (N, 137, 200)
    # Transpose : (N, 137, 200) → (N, 200, 137)
    X_t = X.transpose(0, 2, 1)
    print(f"[INFO] Forme après transpose : {X_t.shape}")

    # Vérifier la forme attendue (N, MAX_LEN, FEATURE_DIM)
    n, t, f = X_t.shape
    if t != MAX_LEN or f != FEATURE_DIM:
        raise ValueError(
            f"Après transpose, on obtient X_{mode}.shape = {X_t.shape}, "
            f"mais on attend (N, {MAX_LEN}, {FEATURE_DIM})"
        )

    # Ré-uploader la version transposée sur GCS
    print(f"[INFO] Ré-upload de '{blob_X}' sur GCS en version transposée…")
    upload_npy_array(X_t, blob_X)
    print(f"[INFO] '{blob_X}' a été écrasé avec la forme correcte (N, {MAX_LEN}, {FEATURE_DIM}).\n")


if __name__ == "__main__":
    # On traite train, val et test dans l’ordre
    for mode in ["train", "val", "test"]:
        transpose_and_overwrite_on_gcs(mode)

    print("[INFO] Transposition terminée pour tous les splits.")
