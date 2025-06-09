# preprocessing.py

import os
import random
import numpy as np
import librosa

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from google.cloud import storage

import config

# -------------------------------------------------
# 1. CLIENT GCS & UTILITAIRES
# -------------------------------------------------
def get_gcs_client():
    """
    Retourne un client GCS (authentifié via GOOGLE_APPLICATION_CREDENTIALS).
    """
    return storage.Client()

def list_audio_files_from_bucket(bucket_name: str, prefix: str = "") -> list[str]:
    """
    Liste tous les blobs .wav dans le bucket bucket_name sous le préfixe prefix.
    Ex : prefix="Raw/" pour lister Raw/xxx.wav.
    Retourne une liste de chemins GCS relatifs (ex : 'Raw/03-01-05-01-01-01-01.wav').
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    files = []
    for blob in blobs:
        if blob.name.lower().endswith(".wav"):
            files.append(blob.name)
    return files

def download_wav_from_gcs(bucket_name: str, blob_name: str, local_dir: str) -> str:
    """
    Télécharge un blob GCS (blob_name) depuis bucket_name dans local_dir.
    Si le fichier existe déjà localement, il n’est pas retéléchargé.
    Renvoie le chemin local complet vers le fichier téléchargé.
    """
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(blob_name))
    if os.path.exists(local_path):
        return local_path

    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path

# -------------------------------------------------
# 2. EXTRACTION DES FEATURES 1D
# -------------------------------------------------
def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extrait un vecteur 1D de features (prosodie enrichie).
    Retourne un array shape=(276, MAX_LEN).
    """
    # 1) MFCC statiques + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=config.N_MFCC, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (20, T)
    mfcc_delta  = librosa.feature.delta(mfcc, order=1)   # (20, T)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)   # (20, T)
    mfcc_stack = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (60, T)

    # 2) Log-Mel + Δ + ΔΔ
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (64, T)
    S_db    = librosa.power_to_db(S)                     # (64, T)
    S_db_d1 = librosa.feature.delta(S_db, order=1)       # (64, T)
    S_db_d2 = librosa.feature.delta(S_db, order=2)       # (64, T)
    mel_stack = np.vstack([S_db, S_db_d1, S_db_d2])      # (192, T)

    # 3) ZCR, Chroma, Tonnetz
    zcr     = librosa.feature.zero_crossing_rate(y=y, hop_length=config.HOP_LENGTH)  # (1, T)
    chroma  = librosa.feature.chroma_stft(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (12, T)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)  # (6, T')

    # 4) Spectral features
    rms       = librosa.feature.rms(y=y, frame_length=config.N_FFT, hop_length=config.HOP_LENGTH)  # (1, T)
    centroid  = librosa.feature.spectral_centroid(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (1, T)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (1, T)
    rolloff   = librosa.feature.spectral_rolloff(
        y=y, sr=sr, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH
    )  # (1, T)

    # 5) Fundamental frequency (f0) via PYIN
    try:
        f0, _, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr,
            frame_length=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )  # f0: (T,)
        f0 = np.nan_to_num(f0).reshape(1, -1)  # (1, T)
    except Exception:
        T_frames = mfcc.shape[1]
        f0 = np.zeros((1, T_frames))

    # 6) Align / pad sur T_max
    feats = [mfcc_stack, mel_stack, zcr, chroma, tonnetz, rms, centroid, bandwidth, rolloff, f0]
    T_max = max(f.shape[1] for f in feats)
    aligned = []
    for f in feats:
        if f.shape[1] < T_max:
            f_pad = np.pad(f, ((0, 0), (0, T_max - f.shape[1])), mode='constant')
        else:
            f_pad = f[:, :T_max]
        aligned.append(f_pad)

    stacked = np.vstack(aligned)  # (276, T_max)

    # 7) Padding/Cropping final sur MAX_LEN
    if stacked.shape[1] < config.MAX_LEN:
        stacked = np.pad(stacked, ((0, 0), (0, config.MAX_LEN - stacked.shape[1])), mode='constant')
    else:
        stacked = stacked[:, :config.MAX_LEN]

    return stacked  # (276, MAX_LEN)


# -------------------------------------------------
# 3. FONCTION AU NIVEAU MODULE pour traiter UN fichier
# -------------------------------------------------
def process_one_file(args):
    """
    Doit être un top-level function pour être picklable.
    args = (path, label)
    Retourne ([X_loc], [y_loc]) pour ce fichier.
    """
    path, label = args
    X_loc, y_loc = [], []

    # Télécharger le .wav depuis GCS
    local_wav = download_wav_from_gcs(config.GCS_BUCKET_NAME, path, config.LOCAL_AUDIO_DIR)

    # Charger l’audio
    y_audio, sr = librosa.load(local_wav, sr=config.SR)

    # Comptage pour oversampling
    # NOTE : on passe 'counts' et 'max_count' via variables fermées au moment d'appeler executor.map
    global _counts_global, _max_count_global, _labels_to_oversample_global

    counts = _counts_global
    max_count = _max_count_global
    labels_to_oversample = _labels_to_oversample_global

    # Déterminer n_repeat selon l’oversampling
    if label in labels_to_oversample:
        if label == _index_01_global or label == _index_08_global:
            n_repeat = max(1, (max_count // counts[label]) * 2)  # oversampling ×2 pour 01 & 08
        else:
            n_repeat = max(1, max_count // counts[label])       # oversampling ×1 pour 03 & 06
    else:
        n_repeat = 1

    # Appliquer n_repeat fois l’ensemble des augmentations si augment=True
    for _ in range(n_repeat):
        if _augment_global:
            for aug in _AUGS_global:
                y_aug = aug(y_audio, sr)
                feats = extract_features(y_aug, sr)  # (276, MAX_LEN)
                X_loc.append(feats.astype(np.float32))
                y_loc.append(label)
        else:
            feats = extract_features(y_audio, sr)
            X_loc.append(feats.astype(np.float32))
            y_loc.append(label)

    return X_loc, y_loc


# -------------------------------------------------
# 4. FONCTION PRINCIPALE DE BUILD avec cache
# -------------------------------------------------
def build_XY_with_cache(
    file_list: list[str],
    label_list: list[int],
    cache_X_path: str,
    cache_y_path: str,
    augment: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    - Si cache existant (cache_X_path, cache_y_path), charge et renvoie (X, y).
    - Sinon, calcule X, y en parallèle avec ProcessPoolExecutor, sauvegarde puis renvoie.
    """
    if os.path.exists(cache_X_path) and os.path.exists(cache_y_path):
        X = np.load(cache_X_path)
        y = np.load(cache_y_path)
        print(f"→ Chargé depuis cache : {cache_X_path}, {cache_y_path}")
        return X, y

    print(f"→ Pas de cache trouvé pour {cache_X_path}. Extraction en cours…")

    # Préparer les variables globales pour process_one_file
    counts = Counter(label_list)
    max_count = max(counts.values())

    # Exemples : classes à sur-échantillonner {01, 03, 06, 08}
    # Adaptez ces indices si vos codes ne sont pas exactement 0="01", 2="03", etc.
    index_01 = np.where(np.array(sorted(set(label_list))) == sorted(set(label_list))[0])[0][0]
    index_03 = index_01 + 2
    index_06 = index_01 + 5
    index_08 = index_01 + 7

    labels_to_oversample = [index_01, index_03, index_06, index_08] if augment else []

    # Variables « globales » pour que process_one_file y ait accès
    global _counts_global, _max_count_global, _labels_to_oversample_global
    global _index_01_global, _index_08_global, _augment_global, _AUGS_global

    _counts_global = counts
    _max_count_global = max_count
    _labels_to_oversample_global = labels_to_oversample
    _index_01_global = index_01
    _index_08_global = index_08
    _augment_global = augment
    _AUGS_global = [
        lambda y, sr: y,
        lambda y, sr: np.clip(y + 0.01 * np.random.randn(len(y)), -1.0, 1.0),
        lambda y, sr: np.roll(y, int(len(y) * random.uniform(-0.2, 0.2))),
        lambda y, sr: librosa.effects.pitch_shift(y=y, sr=sr, n_steps=random.uniform(-4, 4)),
        lambda y, sr: librosa.effects.time_stretch(y=y, rate=random.uniform(0.75, 1.25)),
        lambda y, sr: librosa.effects.preemphasis(y, coef=0.97),
    ]

    X_list, y_list = [], []
    args_list = [(file_list[i], label_list[i]) for i in range(len(file_list))]

    # Exécution parallèle : process_one_file doit être défini au niveau module
    with ProcessPoolExecutor(max_workers=config.N_WORKERS) as executor:
        for X_loc, y_loc in executor.map(process_one_file, args_list):
            X_list.extend(X_loc)
            y_list.extend(y_loc)

    X = np.stack(X_list)
    y = np.array(y_list)

    # Sauvegarder sur disque
    np.save(cache_X_path, X)
    np.save(cache_y_path, y)
    print(f"→ Sauvegardé en cache : {cache_X_path}, {cache_y_path}")
    return X, y
