import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import numpy as np

MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
id2label = model.config.id2label

def predict_emotion_wav(path):
    audio, sr = sf.read(path)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(proba))
        pred_label = id2label[pred_idx]
    print(f"Label: {pred_label} | Probas: {np.round(proba,3)}")

# Teste avec un vrai fichier du dataset d'origine
predict_emotion_wav("test.wav")
