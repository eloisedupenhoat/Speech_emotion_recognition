from google.cloud import storage
import io
import soundfile as sf
import numpy as np
import torch
from data.download_gcp_wavs import extract_embedding_from_bytes, get_label_from_filename


# --- INITIALISATION DU MODELE wav2vec2 ---
from transformers import Wav2Vec2Processor, Wav2Vec2Model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# --- Connexion GCP ---
client = storage.Client.from_service_account_json("/Users/greenwaymusic/code/eloisedupenhoat/Speech_emotion_recognition/D1_modules/Pipeline Chris/credentials/gcp_key.json")
bucket = client.bucket("speech_emotion_recognition")

# --- Parcours des fichiers du bucket (exemple : tous les .wav d’un dossier) ---
embeddings = []
labels = []
for blob in bucket.list_blobs(prefix="Raw/"):
    if not blob.name.endswith('.wav'):
        continue
    print(f"Traitement : {blob.name}")
    audio_bytes = blob.download_as_bytes()
    emb = extract_embedding_from_bytes(audio_bytes, processor, model)
    label = get_label_from_filename(blob.name)
    embeddings.append(emb)
    labels.append(label)

embeddings = np.stack(embeddings)
labels = np.array(labels)
np.save("embeddings.npy", embeddings)
np.save("labels.npy", labels)
