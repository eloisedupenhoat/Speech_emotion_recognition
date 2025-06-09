# preprocessing/preprocess_pipeline.py

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

from data.gcs_handler import list_gcs_files, download_audio_as_fileobj, upload_npy_array
from config import RAW_AUDIO_PREFIX, FEATURES_PREFIX, SR
from preprocessing.feature_extraction import extract_features
from preprocessing.augmentations import add_noise, time_shift, pitch_shift, stretch

from config import SR, N_FFT, HOP_LENGTH, N_MELS, FEATURE_DIM, MAX_LEN

# ---------------------------------------------
# FONCTIONS DE VISUALISATION (facultatives)
# ---------------------------------------------

def plot_label_distribution(y, mode):
    """
    Trace un histogramme montrant le nombre d’exemples par classe pour le dataset `mode`.
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    plt.figure()
    plt.bar(unique_labels, counts)
    plt.xlabel("Classe")
    plt.ylabel("Nombre d'exemples")
    plt.title(f"Distribution des classes pour '{mode}'")
    plt.show()


def plot_spectrogram_from_audio(buffer):
    """
    Reçoit un BytesIO contenant un fichier .wav, extrait un spectrogramme et l'affiche.
    """
    buffer.seek(0)
    y, sr = librosa.load(buffer, sr=None)
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256))
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    plt.figure()
    librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogramme (dB)")
    plt.show()


# ---------------------------------------------
# CONSTANTES
# ---------------------------------------------

# Dimension spatiale des features (ex. 137 coefficients MFCC ou similaire)
FEATURE_DIM = 137


# ---------------------------------------------
# FONCTIONS DE LISTING / LABEL / ACTEUR
# ---------------------------------------------

def extract_label_from_name(filename):
    """
    Extrait l’étiquette (label émotion) depuis le nom de fichier.
    Exemple : "03-happy-07.wav" → "happy"
    Vous pouvez adapter le parsing selon votre format de nommage.
    """
    base = os.path.basename(filename)
    parts = base.split('-')
    if len(parts) >= 2:
        return parts[1]  # ex. "happy"
    return "unknown"


def extract_actor_from_name(blob_name):
    """
    Extrait l’acteur (Actor_XX) d’un blob GCS comme "Raw/Actor_01/fichier.wav".
    """
    parts = blob_name.split('/')
    if len(parts) >= 2:
        return parts[1]  # ex. "Actor_01"
    return None


def build_file_label_list(prefix):
    """
    Liste tous les fichiers audio sur GCS sous le préfixe donné (ex. "Raw/train/"),
    et renvoie trois listes parallèles :
      - blobs  : e.g. ["Raw/train/file001.wav", "Raw/train/file002.wav", …]
      - labels : étiquette extraite via extract_label_from_name
      - actors : acteur extrait via extract_actor_from_name
    """
    blobs = list_gcs_files(prefix)
    print(f"[INFO] Nombre de fichiers audio trouvés sous '{prefix}': {len(blobs)}")

    labels = []
    actors = []
    valid_blobs = []

    for blob in blobs:
        if not blob.lower().endswith('.wav'):
            continue

        label = extract_label_from_name(blob)
        actor = extract_actor_from_name(blob)
        if actor is None or label == "unknown":
            # On ignore tout fichier dont on ne peut pas déterminer acteur ou label
            continue

        valid_blobs.append(blob)
        labels.append(label)
        actors.append(actor)

    print(f"[DEBUG] Après filtrage (.wav et parsing) : {len(valid_blobs)} fichiers valides")
    return valid_blobs, labels, actors


# ---------------------------------------------
# FONCTION PRINCIPALE : BUILD DATASET
# ---------------------------------------------

def build_dataset(mode):
    """
    Construit X_{mode}.npy et y_{mode}.npy pour le mode donné ("train", "val", "test").
    - Charge les blobs sous f"{RAW_AUDIO_PREFIX}{mode}/".
    - Extrait les features, transpose pour obtenir (frames, features).
    - Pad ou coupe pour avoir exactement (MAX_LEN, FEATURE_DIM).
    - Empile en tableau NumPy de forme (N, MAX_LEN, FEATURE_DIM).
    - Sauvegarde localement dans Features/ puis upload sur GCS.
    """
    assert mode in ("train", "val", "test"), "Mode invalide : choisir parmi 'train', 'val', 'test'."

    prefix = f"{RAW_AUDIO_PREFIX}{mode}/"
    print(f"\n[INFO] Début de build_dataset('{mode}') avec prefix = '{prefix}'")

    all_files, all_labels, all_actors = build_file_label_list(prefix)
    if len(all_files) == 0:
        print(f"[WARNING] Aucun fichier trouvé sous '{prefix}'. Le dataset '{mode}' ne sera pas créé.")
        return

    print(f"[DEBUG] {len(all_files)} fichiers listés, {len(set(all_labels))} classes différentes")

    # Afficher distribution brute des labels avant extraction
    label_counts = Counter(all_labels)
    print(f"[DEBUG] Distribution brute des labels avant extraction pour '{mode}' : {label_counts}")

    X_list = []
    y_list = []

    for idx, (blob_name, lab) in enumerate(zip(all_files, all_labels), start=1):
        print(f"  • Traitement {idx}/{len(all_files)} : {blob_name}")

        # 1) Télécharger le .wav dans un BytesIO
        buffer = download_audio_as_fileobj(blob_name)

        # 2) Extraction des features (ex. MFCC)
        feats = extract_features(buffer)
        # On suppose ici : feats.shape == (FEATURE_DIM, T_i)
        print(f"    - Forme brute extraite par extract_features : {feats.shape}")

        # 3) Transpose pour passer à (T_i, FEATURE_DIM)
        feats = feats.T
        T_i, F_i = feats.shape
        print(f"    - Après transpose (frames, features)  : {feats.shape}")
        #    → F_i doit être égal à FEATURE_DIM

        # 4) Pad ou coupe pour obtenir (MAX_LEN, FEATURE_DIM)
        if T_i < MAX_LEN:
            pad = np.zeros((MAX_LEN - T_i, FEATURE_DIM), dtype=feats.dtype)
            feats_padded = np.vstack([feats, pad])  # → (MAX_LEN, FEATURE_DIM)
            print(f"    - Padding à {MAX_LEN} frames → {feats_padded.shape}")
        else:
            feats_padded = feats[:MAX_LEN, :]       # → (MAX_LEN, FEATURE_DIM)
            print(f"    - Coupure à {MAX_LEN} frames → {feats_padded.shape}")

        # 5) Stocker dans les listes finales
        X_list.append(feats_padded)
        y_list.append(lab)

        # 6) Affichage pédagogique (spectrogramme du 1er fichier)
        if idx == 1:
            print(f"[PLOT] Spectrogramme du premier fichier : {blob_name}")
            plot_spectrogram_from_audio(buffer)

    # 7) Conversion en NumPy arrays
    X = np.stack(X_list, axis=0)   # shape = (N_exemples, MAX_LEN, FEATURE_DIM)
    y = np.array(y_list, dtype=np.int32)
    print(f"[INFO] Jeu '{mode}' : {X.shape[0]} exemples, X.shape = {X.shape}, y.shape = {y.shape}")

    # 8) Distribution finale des labels
    dist_final = Counter(y.tolist())
    print(f"[INFO] Distribution finale des labels pour '{mode}' : {dist_final}")

    # 9) Sauvegarde locale
    os.makedirs(FEATURES_PREFIX, exist_ok=True)
    local_X = os.path.join(FEATURES_PREFIX, f"X_{mode}.npy")
    local_y = os.path.join(FEATURES_PREFIX, f"y_{mode}.npy")
    np.save(local_X, X)
    np.save(local_y, y)
    print(f"[INFO] Sauvegardé localement : {local_X} ({os.path.getsize(local_X)//1024} Ko), {local_y}")

    # 10) Upload sur GCS
    upload_npy_array(X, f"{FEATURES_PREFIX}X_{mode}.npy")
    upload_npy_array(y, f"{FEATURES_PREFIX}y_{mode}.npy")
    print(f"[INFO] Jeu '{mode}' construit et uploadé sur GCS sous '{FEATURES_PREFIX}'.")


# ---------------------------------------------
# EXEMPLE D’UTILISATION (décommenter pour tester localement)
# ---------------------------------------------

# if __name__ == "__main__":
#     build_dataset("train")
#     build_dataset("val")
#     build_dataset("test")
