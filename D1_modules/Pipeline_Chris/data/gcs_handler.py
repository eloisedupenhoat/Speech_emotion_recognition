# data/gcs_handler.py

import os
import io
import numpy as np
from google.cloud import storage
from google.oauth2 import service_account

# ─── 1) Chemin absolu vers votre JSON de credentials ──────────────────────────────
# Ajustez ce chemin si nécessaire, pour pointer vers gcp_key.json dans le dossier credentials/
KEY_PATH = os.path.join(
    os.path.dirname(__file__),  # …/Pipeline Chris/data
    '..',                        # …/Pipeline Chris
    'credentials',               # nom du dossier contenant gcp_key.json
    'gcp_key.json'               # nom exact du fichier JSON
)
KEY_PATH = os.path.abspath(KEY_PATH)

# ─── 2) Charger explicitement les credentials depuis ce JSON ─────────────────────
credentials = service_account.Credentials.from_service_account_file(KEY_PATH)

# ─── 3) Créer un client GCS en passant ces credentials ───────────────────────────
client = storage.Client(credentials=credentials, project=credentials.project_id)

# ─── 4) Nom exact de votre bucket GCS (vérifiez dans la console Storage) ──────────
BUCKET_NAME = "speech-emotion-bucket"

# Affichage de debug (supprimez ce print une fois validé) :
print(">> KEY_PATH:", KEY_PATH, "Exists?", os.path.isfile(KEY_PATH))


def upload_npy_array(arr, destination_blob_name):
    """
    Sérialise le numpy array `arr` dans un buffer, puis l’envoie sur GCS
    en utilisant un chunk_size de 5 Mo (le SDK passera automatiquement en mode resumable).
    """
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_blob_name)

    # Forcer un chunk_size de 5 Mo pour activer le mode « resumable »
    blob.chunk_size = 5 * 1024 * 1024  # 5 Mo

    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)

    # On supprime l’argument `resumable=True`, le SDK gère le découpage depuis chunk_size
    blob.upload_from_file(
        buffer,
        content_type="application/octet-stream",
        timeout=300
    )


def list_gcs_files(prefix=""):
    """
    Retourne la liste des noms d’objets dans le bucket dont le chemin commence par `prefix`.
    """
    bucket = client.bucket(BUCKET_NAME)
    return [blob.name for blob in bucket.list_blobs(prefix=prefix)]


def download_audio_as_fileobj(blob_name):
    """
    Télécharge un fichier audio (ou tout autre blob) depuis GCS dans un BytesIO et renvoie ce buffer.
    Utile pour charger un flux audio directement en mémoire.
    """
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)
    file_obj = io.BytesIO()
    blob.download_to_file(file_obj)
    file_obj.seek(0)
    return file_obj


def download_npy_array(blob_name):
    """
    Télécharge un fichier .npy depuis le bucket GCS dans un BytesIO,
    puis le charge en tant que numpy array et le retourne.
    Usage :
        arr = download_npy_array("features/X_train.npy")
    """
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(blob_name)

    buffer = io.BytesIO()
    blob.download_to_file(buffer)
    buffer.seek(0)
    arr = np.load(buffer)
    return arr
