##### OBJ: functions to import and export data (locally or in gcloud) #####

##### LIBRARIES #####
import pandas as pd
from io import BytesIO
from google.cloud import storage
import librosa
import os
from params import *

##### RAW DATA #####
def load_raw_data(): #fonction pour chargé la donné brut

    PREFIX = "Raw/"  # on veut tout ce qui est dans Raw

    # Connexion à GCS
    client = storage.Client()
    blobs = client.list_blobs(BUCKET_NAME, prefix=PREFIX)

    # Dictionnaire pour stocker les fichiers audio
    raw_data = {}

    # Itération sur tous les fichiers .wav
    for blob in blobs:
        if blob.name.endswith(".wav"):
            # “Téléchargement” en bytes du fichier
            bytes = blob.download_as_bytes(raw_download=True)
            binary = BytesIO(bytes)
            # Lecture de l’audio
            signal, sr = librosa.load(binary, sr=None)
            # Stockage en mémoire
            raw_data[blob.name] = {
                "signal": signal,
                "sampling_rate": sr
            }
            print(f"{blob.name} → {len(signal)} échantillons à {sr} Hz")
    print(f"\n:white_check_mark: {len(raw_data)} fichiers audio chargés dans `raw_data`.")

    return raw_data

##### PREPOCESS DATA #####

def load_prepoc_data(DATA_PREPROC): # pour load un dataset d'image

    # Crée un client GCS
    client = storage.Client.from_service_account_json("chemin/vers/credentials.json")
    bucket = client.bucket(BUCKET_NAME)

    # Liste les blobs dans le dossier
    blobs = bucket.list_blobs(prefix=DATA_PREPROC)

    # Stocke les images en mémoire
    images = {}

    for blob in blobs:
        if blob.name.endswith(".jpg"):
            byte_data = blob.download_as_bytes()
            images[blob.name] = byte_data
            print(f"Chargé en mémoire : {blob.name}")

    return images

def download():
    pass

def upload():
    pass

if __name__ == '__main__': #

    row_data = load_raw_data()
    print(row_data)
    print(len(row_data))
