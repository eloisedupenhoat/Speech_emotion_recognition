# ===== fichier: preprocessing/feature_extraction.py =====
import numpy as np
import librosa
from config import SR, N_FFT, HOP_LENGTH, N_MELS, FEATURE_DIM, MAX_LEN
from preprocessing.augmentations import spec_augment

def extract_features(y, sr):
    """Extrait les features MFCC (avec Δ et ΔΔ), log-Mel, ZCR, Chroma et les aligne sur MAX_LEN"""
    # 1) MFCC statiques + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=FEATURE_DIM,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )  # shape (20, T)
    mfcc_delta  = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc_stack = np.vstack([mfcc, mfcc_delta, mfcc_delta2])  # (60, T)

    # 2) Log-Mel + SpecAugment
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )  # (64, T)
    S_db    = librosa.power_to_db(S)
    S_db_aug = spec_augment(S_db)  # (64, T)

    # 3) ZCR, Chroma
    zcr    = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)  # (1, T)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)  # (12, T)

    # 4) Concaténation
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

    # 5) Padding/Cropping final sur MAX_LEN
    if stacked.shape[1] < MAX_LEN:
        stacked = np.pad(stacked, ((0, 0), (0, MAX_LEN - stacked.shape[1])), mode='constant')
    else:
        stacked = stacked[:, :MAX_LEN]

    return stacked  # (137, MAX_LEN)

