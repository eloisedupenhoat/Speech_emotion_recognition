# main.py

import os

os.system("python data/download_gcp_wavs.py")
os.system("python features/extract_wav2vec2_embeddings.py")
os.system("python training/train_mlp_on_embeddings.py")
os.system("python evaluation/report.py")


if os.system("python data/download_gcp_wavs.py") != 0:
    raise RuntimeError("Échec étape download_gcp_wavs.py")

if os.system("python features/extract_wav2vec2_embeddings.py") != 0:
    raise RuntimeError("Échec étape extract_wav2vec2_embeddings.py")

# etc.
