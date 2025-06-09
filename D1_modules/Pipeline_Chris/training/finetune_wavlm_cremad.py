import os
import gcsfs
import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMForSequenceClassification, WavLMFeatureExtractor, TrainingArguments, Trainer

GCP_PROJECT = 'Speech-Emotion-1976'
BUCKET_AUDIO = 'speech-emotion-bucket/Raw/crema-d/AudioWAV'
CSV_PATH = '/Users/greenwaymusic/code/eloisedupenhoat/Speech_emotion_recognition/D1_modules/Pipeline_Chris/data/cremad_labels.csv'
OUTPUT_DIR = './wavlm_cremad_finetuned'
NUM_LABELS = 6  # Neutral, Happy, Sad, Angry, Fearful, Disgust
SAMPLE_RATE = 16000

# 1. Dataset
class CremadGCPDataset(Dataset):
    def __init__(self, csv_path, bucket_path, project, sample_rate=16000):
        self.data = pd.read_csv(csv_path)
        self.bucket = bucket_path
        self.fs = gcsfs.GCSFileSystem(project=project)
        self.sample_rate = sample_rate
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = f'{self.bucket}/{row.file}'
        with self.fs.open(file_path, 'rb') as f:
            wav, sr = torchaudio.load(f)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0)  # mono
        return {'input_values': wav, 'labels': int(row.label)}

# 2. Collate fn
def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = torch.tensor([item['labels'] for item in batch], dtype=torch.long)
    # Pad sequences
    input_values = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    attention_mask = (input_values != 0).int()
    return {'input_values': input_values, 'attention_mask': attention_mask, 'labels': labels}

# 3. Modèle + Feature Extractor
feature_extractor = WavLMFeatureExtractor.from_pretrained('microsoft/wavlm-base')
model = WavLMForSequenceClassification.from_pretrained(
    'microsoft/wavlm-base',
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)

# 4. Adapter dataset pour Huggingface Trainer
class CremadHFDataset(Dataset):
    def __init__(self, cremaddataset):
        self.ds = cremaddataset
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, idx):
        d = self.ds[idx]
        # Normalise [-1,1]
        inputs = feature_extractor(d['input_values'].numpy(), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
        out = {k: v.squeeze(0) for k, v in inputs.items()}
        out['labels'] = d['labels']
        return out

train_dataset = CremadGCPDataset(CSV_PATH, BUCKET_AUDIO, GCP_PROJECT)
hf_dataset = CremadHFDataset(train_dataset)

# 5. Training arguments (à ajuster selon ta VRAM !)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,  # Augmente/diminue selon ta VRAM
    learning_rate=2e-5,
    save_strategy="epoch",
    evaluation_strategy="no",
    logging_steps=20,
    fp16=torch.cuda.is_available(),
    remove_unused_columns=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=hf_dataset,
    tokenizer=feature_extractor,
    data_collator=collate_fn,
)

trainer.train()
model.save_pretrained(OUTPUT_DIR)
feature_extractor.save_pretrained(OUTPUT_DIR)
print(f"Modèle fine-tuné sauvegardé dans {OUTPUT_DIR}")
