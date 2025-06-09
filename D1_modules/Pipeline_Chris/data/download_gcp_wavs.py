import os
import io
import numpy as np
from google.cloud import storage
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# === CONFIGURATION ===
BUCKET_NAME = "speech-emotion-bucket"
GCP_PREFIX = "Raw/"  # Tous les fichiers .wav dans ce dossier (et sous-dossiers)
GCP_CREDENTIALS_PATH = "./credentials/gcp_key.json"  # Adapter si besoin

# === FONCTIONS GCP ===

def list_gcp_wav_blobs(bucket_name, prefix, credentials_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    # Récupère tous les blobs (fichiers) .wav sous le préfixe donné
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".wav")]
    return blobs

def download_blob_as_bytes(blob):
    return blob.download_as_bytes()

# === LABELING FONCTION ===
def get_label_from_filename(filename):
    """Adapte ce parsing selon ton dataset (ici : 03-01-01-01-01-01-01.wav → label=3ᵉ champ)."""
    parts = os.path.basename(filename).split('-')
    label_id = int(parts[2])  # ADAPTE CETTE LIGNE SI BESOIN !
    return label_id

# === WAV2VEC2 EMBEDDINGS ===
def extract_embedding_from_bytes(audio_bytes, processor, model):
    with io.BytesIO(audio_bytes) as f:
        audio_input, sr = sf.read(f)
        if sr != 16000:
            import librosa
            audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
        # Gère mono/multi-channels
        if len(audio_input.shape) > 1:
            audio_input = np.mean(audio_input, axis=1)
        input_values = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_values
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
            embedding = hidden_states.mean(dim=1).squeeze().numpy()
        return embedding

# === MAIN PIPELINE ===
def main():
    print("Listing blobs GCP...")
    blobs = list_gcp_wav_blobs(BUCKET_NAME, GCP_PREFIX, GCP_CREDENTIALS_PATH)
    print(f"Trouvé {len(blobs)} fichiers wav dans le bucket.")

    print("Chargement du modèle Wav2Vec2...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model.eval()
    embeddings = []
    labels = []

    for idx, blob in enumerate(blobs):
        try:
            audio_bytes = download_blob_as_bytes(blob)
            emb = extract_embedding_from_bytes(audio_bytes, processor, model)
            label = get_label_from_filename(blob.name)
            embeddings.append(emb)
            labels.append(label)
            if idx % 25 == 0:
                print(f"Progression: {idx}/{len(blobs)} fichiers traités…")
        except Exception as e:
            print(f"Erreur sur {blob.name} : {e}")

    X = np.stack(embeddings)
    y = np.array(labels)
    print(f"\nShape X: {X.shape}, y: {y.shape}")

    # === TRAIN/TEST & MLP ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=30, verbose=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()


