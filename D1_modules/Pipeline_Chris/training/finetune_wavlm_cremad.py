#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fine-tune WavLM-base sur CREMA-D stocké dans un bucket GCS.

Prérequis :
  export GOOGLE_APPLICATION_CREDENTIALS=/chemin/vers/ta_clé.json
"""

import os
from pathlib import Path

import gcsfs
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    HubertForSequenceClassification,
    WavLMForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ========= PARAMÈTRES =========
GCP_PROJECT   = "speech-emotion-1976"
BUCKET_AUDIO  = "speech-emotion-bucket/CremaD_Raw"
CSV_PATH      = Path(
    "D1_modules/Pipeline_Chris/data/cremad_labels.csv"
)
OUTPUT_DIR    = Path("./wavlm_cremad_finetuned")
SAMPLE_RATE   = 16_000
NUM_LABELS    = 6
EPOCHS        = 50
BATCH_SIZE    = 32
LR            = 2e-5
# ===============================

# ---- GCS filesystem (une seule instance) ----
# GCS_TOKEN = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
# fs = gcsfs.GCSFileSystem(project=GCP_PROJECT, token=GCS_TOKEN)
fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)

# ---- Dataset brut ----
class CremadGCPSpeech(Dataset):
    def __init__(self, csv_path: Path, bucket: str, sample_rate: int = 16000):
        self.meta        = pd.read_csv(csv_path)
        self.bucket      = bucket.rstrip("/")
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row      = self.meta.iloc[idx]
        gcs_uri  = f"{self.bucket}/{row.file}"
        label    = int(row.label)

        with fs.open(gcs_uri, "rb") as f:
            wav, sr = torchaudio.load(f)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.mean(dim=0)                   # mono (Tensor 1-D)
        return {"waveform": wav, "label": label}

# ---- Collate : convertit en numpy AVANT l’extractor ----
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def collate_fn(batch):
    wavs   = [item["waveform"].numpy() for item in batch]  # <- conversion
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    feats = feature_extractor(
        wavs,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    feats["labels"] = labels
    return feats

# ---- Modèle ----
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=NUM_LABELS,
    problem_type="single_label_classification",
)

# ---- Trainer ----
train_ds = CremadGCPSpeech(CSV_PATH, BUCKET_AUDIO, SAMPLE_RATE)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    save_strategy="epoch",
    eval_strategy="no",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    data_collator=collate_fn,
)

# ---- Fine-tuning 🚀 ----
trainer.train()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print(f"✅ Modèle fine-tuné sauvegardé dans : {OUTPUT_DIR.resolve()}")
