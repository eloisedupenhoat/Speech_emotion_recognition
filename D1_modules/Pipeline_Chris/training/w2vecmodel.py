import torch
import sounddevice as sd
import numpy as np
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
SAMPLING_RATE = 16000
WINDOW_DURATION = 2.0  # en secondes

print("Chargement du modèle et du feature extractor...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
model.eval()
id2label = model.config.id2label

def predict_emotion(audio_np):
    inputs = feature_extractor(audio_np, sampling_rate=SAMPLING_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(proba))
    return id2label[pred_idx], proba

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_np = indata.flatten()
    emotion, proba = predict_emotion(audio_np)
    print(f"Émotion détectée : {emotion} | Proba: {np.round(proba, 3)}")

print(f"--- SER Live ({MODEL_ID}) ---")
print(f"Parle dans le micro ({SAMPLING_RATE}Hz, {WINDOW_DURATION}s) | Ctrl+C pour stop")
try:
    with sd.InputStream(
        channels=1,
        samplerate=SAMPLING_RATE,
        callback=callback,
        blocksize=int(SAMPLING_RATE * WINDOW_DURATION)
    ):
        while True:
            sd.sleep(1000)
except KeyboardInterrupt:
    print("Arrêt.")
