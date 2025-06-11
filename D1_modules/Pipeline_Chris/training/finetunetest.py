#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import gcsfs
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ========= PARAMÈTRES =========
GCP_PROJECT   = "speech-emotion-1976"
BUCKET_AUDIO  = "speech-emotion-bucket/CremaD_Raw"
CSV_PATH      = Path("D1_modules/Pipeline_Chris/data/cremad_labels.csv")
OUTPUT_DIR    = Path("./wav2vec2_cremad_finetuned")
SAMPLE_RATE   = 1000
NUM_LABELS    = 6
EPOCHS        = 5
BATCH_SIZE    = 8
LR            = 2e-5
# ===============================

fs = gcsfs.GCSFileSystem(project=GCP_PROJECT)

class CremadGCPSpeech(Dataset):
    def __init__(self, df, bucket: str, sample_rate: int = 1000):
        self.meta = df.reset_index(drop=True)
        self.bucket = bucket.rstrip("/")
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        gcs_uri = f"{self.bucket}/{row.file}"
        label = int(row.label)

        with fs.open(gcs_uri, "rb") as f:
            wav, sr = torchaudio.load(f)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        wav = wav.mean(dim=0)  # mono
        return {"waveform": wav, "label": label}

# ---- Chargement + split train/val ----
meta_full = pd.read_csv(CSV_PATH).sample(n=1000, random_state=42)
train_df, val_df = train_test_split(meta_full, test_size=0.1, stratify=meta_full["label"], random_state=42)

train_ds = CremadGCPSpeech(train_df, BUCKET_AUDIO, SAMPLE_RATE)
val_ds   = CremadGCPSpeech(val_df, BUCKET_AUDIO, SAMPLE_RATE)

# ---- Feature extractor ----
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

def collate_fn(batch):
    wavs = [item["waveform"].numpy() for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    feats = feature_extractor(
        wavs,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    feats["labels"] = labels
    return feats

# ---- compute_metrics ----
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }

# ---- Modèle ----
model = Wav2Vec2ForSequenceClassification.from_pretrained(
    "facebook/wav2vec2-base",
    num_labels=NUM_LABELS,
    problem_type="single_label_classification",
)

# ---- Training config ----
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    gradient_checkpointing=True,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to=[],
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# ---- Entraînement 🚀 ----
trainer.train()

# ---- Sauvegarde ----
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print(f"✅ Modèle fine-tuné sauvegardé dans : {OUTPUT_DIR.resolve()}")
