# ===== fichier: preprocessing/augmentations.py =====
import numpy as np
import random
import librosa

def add_noise(y, factor=0.004):
    return np.clip(y + factor * np.random.randn(len(y)), -1.0, 1.0)

def time_shift(y, max_ratio=0.2):
    shift = int(len(y) * random.uniform(-max_ratio, max_ratio))
    return np.roll(y, shift)

def pitch_shift(y, sr, steps=2):
    return librosa.effects.pitch_shift(
        y=y,
        sr=sr,
        n_steps=steps
    )

def stretch(y, rate=0.9):
    return librosa.effects.time_stretch(
        y=y,
        rate=rate
    )

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

