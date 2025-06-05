# D1_modules/d_preproc_mfcc.py

import os
import random
import glob
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from collections import Counter

# =========================
# 1. PARAMÈTRES GLOBAUX
# =========================
SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
N_MFCC = 20
MAX_LEN = 200

# Remplacer ce chemin par le dossier où sont stockés vos .wav
ROOT_AUDIO_DIR = "/Users/greenwaymusic/code/Chrisgrd/archive/"  

# =========================
# 2. UTILS D’AUGMENTATION
# =========================
def add_noise(y, factor=0.004):
    return np.clip(y + factor * np.random.randn(len(y)), -1.0, 1.0)

def time_shift(y, max_ratio=0.2):
    shift = int(len(y) * random.uniform(-max_ratio, max_ratio))
    return np.roll(y, shift)

def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)

def stretch(y, rate=0.9):
    return librosa.effects.time_stretch(y=y, rate=rate)

def spec_augment(melspec, num_mask=2, freq_masking_max=8, time_masking_max=16):
    m = melspec.copy()
    for _ in range(num_mask):
        f = random.randint(0, freq_masking_max)
        t = random.randint(0, time_masking_max)
        f0 = random.randint(0, m.shape[0] - f)
        t0 = random.randint(0, m.shape[1] - t)
        m[f0 : f0 + f, :] = 0
        m[:, t0 : t0 + t] = 0
    return m

# =========================
# 3. EXTRACTION DE FEATURES
# =========================
def extract_features(y, sr):
    # 1) MFCC statiques + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mfcc_delta  = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stack = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (60, T)

    # 2) Log-Mel + SpecAugment
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db    = librosa.power_to_db(S)
    S_db_aug = spec_augment(S_db)  # (64, T)

    # 3) ZCR, Chroma
    zcr    = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)  # (1, T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)  # (12, T)

    # 4) Concatener en (137, T_max)
    feats = [mfcc_stack, S_db_aug, zcr, chroma]
    T_max = max(f.shape[1] for f in feats)
    aligned = []
    for f in feats:
        if f.shape[1] < T_max:
            f_pad = np.pad(f, ((0, 0), (0, T_max - f.shape[1])), mode='constant')
        else:
            f_pad = f[:, :T_max]
        aligned.append(f_pad)
    stacked = np.vstack(aligned)  # (137, T_max)

    # 5) Padding/Cropping final sur MAX_LEN (=200)
    if stacked.shape[1] < MAX_LEN:
        stacked = np.pad(stacked, ((0, 0), (0, MAX_LEN - stacked.shape[1])), mode='constant')
    else:
        stacked = stacked[:, :MAX_LEN]

    return stacked  # (137, 200)

# =========================
# 4. CHARGEMENT + SPLIT SPEAKER-INDEPENDANT
# =========================
def load_and_split_data():
    all_files, all_labels, all_actors = [], [], []
    for root, dirs, files in os.walk(ROOT_AUDIO_DIR):
        for f in files:
            if f.endswith('.wav'):
                path     = os.path.join(root, f)
                actor_id = f.split('-')[-1].split('.')[0]
                emotion  = f.split('-')[2]
                all_files.append(path)
                all_labels.append(emotion)
                all_actors.append(actor_id)

    le = LabelEncoder()
    y = le.fit_transform(all_labels)
    n_classes = len(le.classes_)

    # Speaker‐independent split
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_idx, test_idx = next(gss.split(all_files, groups=all_actors))
    train_files   = [all_files[i]  for i in train_idx]
    test_files    = [all_files[i]  for i in test_idx]
    train_labels  = [y[i]          for i in train_idx]
    test_labels   = [y[i]          for i in test_idx]
    train_actors  = [all_actors[i] for i in train_idx]
    test_actors   = [all_actors[i] for i in test_idx]

    # Validation split (20 % du train)
    gss2 = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    subtrain_idx, val_idx = next(gss2.split(train_files, groups=train_actors))
    val_files   = [train_files[i] for i in val_idx]
    val_labels  = [train_labels[i] for i in val_idx]
    train_files = [train_files[i] for i in subtrain_idx]
    train_labels= [train_labels[i] for i in subtrain_idx]

    return (train_files, train_labels,
            val_files, val_labels,
            test_files, test_labels,
            le, n_classes)

# =========================
# 5. BUILD_XY_1D_BALANCED (OVERSAMPLING)
# =========================
AUGS = [
    lambda y, sr: y,
    lambda y, sr: add_noise(y, 0.006),
    lambda y, sr: time_shift(y),
    lambda y, sr: pitch_shift(y, sr, steps=random.uniform(-2, 2)),
    lambda y, sr: stretch(y, rate=random.uniform(0.85, 1.15)),
]

def build_XY_1D_balanced(file_list, label_list, augmentations=AUGS, labels_to_oversample=None):
    """
    - labels_to_oversample : liste d’indices de classes à sur-échantillonner.
    - Chaque échantillon dont la label ∈ labels_to_oversample sera répété (max_count // counts[label]) fois.
    """
    X, y_out = [], []
    counts = Counter(label_list)
    max_count = max(counts.values())

    for path, label in zip(file_list, label_list):
        y_audio, sr = librosa.load(path, sr=SR)

        if labels_to_oversample and label in labels_to_oversample:
            n_repeat = max(1, max_count // counts[label])
        else:
            n_repeat = 1

        for _ in range(n_repeat):
            for aug in augmentations:
                y_aug = aug(y_audio, sr)
                feats = extract_features(y_aug, sr)
                X.append(feats.astype(np.float32))
                y_out.append(label)

    return np.stack(X), np.array(y_out)
