#!/usr/bin/env python
"""train_va_mae.py — Phase‑1 valence–arousal regression + masked‑patch MAE

Finetunes the Phase‑0‑adapted **WavLM‑XXL** backbone on continuous Valence
& Arousal (VA) labels using:
  • **Std‑Pooling** (mean + std concat) of frame embeddings.
  • Dual‐head VA regressor (2 units, tanh) with combined L1+L2 loss.
  • **Masked Patch MAE** self‑supervision (25 % time‑frame masking).

Datasets supported (automatic detection by filename stem):
  • MSP‑Podcast  → csv with columns: file, valence, arousal
  • IEMOCAP impro → csv same schema
  • SEMAINE      → csv same schema

Example usage (single GPU):
  python train_va_mae.py \
      --data_csv data/va_meta.csv \
      --wav_root data/wavs \
      --init_ckpt models/wavlm_xxl_ssl/p0_final.pt \
      --output_dir models/wavlm_xxl_va \
      --epochs 8 --batch_size 3 \
      --lr 2e-5 --specaug --use_lora

Author: SER‑Project – Phase‑1

"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking
from transformers import (
    WavLMModel,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)
from accelerate import Accelerator
from peft import LoraConfig, TaskType, get_peft_model

hf_logging.set_verbosity_error()

###############################################################################
# Dataset                                                                     #
###############################################################################
class VADataset(Dataset):
    """Loads (wav, valence, arousal) triplets from a CSV metadata file."""

    def __init__(
        self,
        csv_path: str | Path,
        wav_root: str | Path,
        segment_seconds: float = 5.0,
        sample_rate: int = 16000,
    ):
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.records: List[Tuple[Path, float, float]] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append((Path(wav_root) / row["file"], float(row["valence"]), float(row["arousal"])) )
        if not self.records:
            raise RuntimeError("Empty VA dataset!")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        wav_path, valence, arousal = self.records[idx]
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)
        # Random crop/pad
        if waveform.numel() > self.segment_samples:
            start = torch.randint(0, waveform.numel() - self.segment_samples, (1,)).item()
            waveform = waveform[start : start + self.segment_samples]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, self.segment_samples - waveform.numel()))
        return waveform, torch.tensor([valence, arousal], dtype=torch.float32)


###############################################################################
# SpecAugment / MAE helpers                                                   #
###############################################################################
class MaskedMAE(nn.Module):
    """Masked patch MAE (time masking) auxiliary loss."""

    def __init__(self, hidden_size: int, mask_ratio: float = 0.25):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.loss_fn = nn.L1Loss()

    def forward(self, hidden_states):
        # hidden_states: (B, T, H)
        B, T, H = hidden_states.size()
        device = hidden_states.device
        mask = torch.rand(B, T, device=device) < self.mask_ratio  # True=mask
        masked_tokens = hidden_states[mask]
        targets = hidden_states.detach()[mask]
        if masked_tokens.numel() == 0:
            return torch.tensor(0.0, device=device)
        recon = self.proj(masked_tokens)
        return self.loss_fn(recon, targets)


class TimeFreqMask(nn.Module):
    """SpecAugment for log‑mel features (used as data augmentation)."""

    def __init__(self):
        super().__init__()
        self.time_mask = TimeMasking(time_mask_param=80)
        self.freq_mask = FrequencyMasking(freq_mask_param=27)

    def forward(self, mel):
        return self.freq_mask(self.time_mask(mel))


###############################################################################
# Model wrapper                                                               #
###############################################################################
class SERBackbone(nn.Module):
    def __init__(
        self,
        wavlm_ckpt: str | Path,
        freeze_layers: float = 0.5,
        use_lora: bool = False,
    ):
        super().__init__()
        kwargs = {"trust_remote_code": True, "gradient_checkpointing": True, "low_cpu_mem_usage": True}
        self.encoder = WavLMModel.from_pretrained(wavlm_ckpt, **kwargs)
        # Optionally freeze bottom X %
        if freeze_layers > 0:
            total = len(self.encoder.encoder.layers)
            freeze_until = int(total * freeze_layers)
            for idx, layer in enumerate(self.encoder.encoder.layers):
                if idx < freeze_until:
                    for p in layer.parameters():
                        p.requires_grad = False
        if use_lora:
            lora_cfg = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj"])
            self.encoder = get_peft_model(self.encoder, lora_cfg)
        self.pool = StdPooling()
        self.head_va = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * 2, 256), nn.GELU(), nn.Linear(256, 2), nn.Tanh()
        )
        self.mae = MaskedMAE(self.encoder.config.hidden_size)

    def forward(self, input_values, attention_mask):
        outs = self.encoder(input_values, attention_mask=attention_mask).last_hidden_state
        pooled = self.pool(outs, attention_mask)
        va_pred = self.head_va(pooled)
        mae_loss = self.mae(outs)
        return va_pred, mae_loss


class StdPooling(nn.Module):
    def forward(self, hidden, mask):
        # hidden: (B, T, H) ; mask: (B, T)
        lengths = mask.sum(1).clamp(min=1).unsqueeze(-1)  # (B,1)
        mean = (hidden * mask.unsqueeze(-1)).sum(1) / lengths
        var = ((hidden ** 2) * mask.unsqueeze(-1)).sum(1) / lengths - mean ** 2
        std = torch.sqrt(torch.clamp(var, min=1e-12))
        return torch.cat([mean, std], dim=-1)  # (B, 2H)


###############################################################################
# Training                                                                    #
###############################################################################

def train(args):
    accelerator = Accelerator(log_with="tensorboard", project_dir=args.output_dir)
    device = accelerator.device

    dataset = VADataset(args.data_csv, args.wav_root, segment_seconds=args.segment_seconds)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = SERBackbone(args.init_ckpt, freeze_layers=args.freeze_layers, use_lora=args.use_lora)

    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_update, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * len(loader) * args.epochs),
        num_training_steps=len(loader) * args.epochs,
    )

    criterion = torch.nn.SmoothL1Loss(beta=0.2)  # robust to outliers

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for wave, labels in loader:
            wave = wave.to(device)
            mask = (wave != 0).int()
            labels = labels.to(device)
            preds, mae_loss = model(wave, mask)
            loss_va = criterion(preds, labels)
            loss = loss_va + args.lambda_mae * mae_loss
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if accelerator.is_local_main_process and global_step % args.log_steps == 0:
                accelerator.log({"loss_total": loss.item(), "loss_va": loss_va.item(), "loss_mae": mae_loss.item()}, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            ckpt = Path(args.output_dir) / f"checkpoint_epoch{epoch+1}.pt"
            accelerator.save_state(str(ckpt))
            accelerator.print(f"Epoch {epoch+1} ckpt saved → {ckpt}")

    if accelerator.is_main_process:
        accelerator.save_state(str(Path(args.output_dir) / "va_final.pt"))
        accelerator.print("Phase‑1 training complete (VA regression).")


###############################################################################
# Arg‑parse                                                                    #
###############################################################################

def _parse_args():
    p = argparse.ArgumentParser(description="Phase‑1 VA regression with masked MAE")
    p.add_argument("--specaug", action="store_true")
    p.add_argument("--data_csv", required=True, help="CSV file with wav paths + VA labels")
    p.add_argument("--wav_root", required=True, help="Root directory of wav files")
    p.add_argument("--init_ckpt", required=True, help="Path to Phase‑0 WavLM checkpoint")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--segment_seconds", type=float, default=5.0)
    p.add_argument("--batch_size", type=int, default=3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--freeze_layers", type=float, default=0.5, help="Fraction of bottom layers to freeze (0‑1)")
    p.add_argument("--lambda_mae", type=float, default=0.3, help="Weight of MAE auxiliary loss")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--log_steps", type=int, default=25)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "train_va_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    train(args)
