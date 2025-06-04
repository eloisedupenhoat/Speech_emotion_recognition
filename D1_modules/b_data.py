##### OBJ: functions to import and export data (locally or in gcloud) #####

#########################################################################
#####################       LIBRARIES      ##############################
#########################################################################

import pandas as pd
from io import BytesIO
from google.cloud import storage
import librosa
import os
from params import *
from PIL import Image
import io
import tensorflow as tf
import pickle

#########################################################################
######################       RAW DATA      ##############################
#########################################################################

## Fonction pour chargé la donné brut ##
def load_raw_data(): # Load raw data

    # Check if file already exists
    path = 'load_raw_data.pkl'
    if os.path.isfile(path):
        with open('filename.pickle', 'rb') as handle:
            raw_data = pickle.load(handle)

        return raw_data

    # Within the bucket, this is the folder we will be working on
    # We take all the data from Raw/
    PREFIX = "Raw/"

    # Connecting too Google Cloud Storage
    # Creates a client object to interact with GCS
    client = storage.Client()

    # The list of files (blobs) inside the bucket
    # Stores those files inside a variable blobs
    blobs = client.list_blobs(BUCKET_NAME, prefix=PREFIX)

    # Initialize a raw_data dictionary
    raw_data = {}

    # Iterate through each file i(blob) nside the bucket
    print("Starting for loop")
    for blob in blobs:
        if blob.name.endswith(".wav"):
            print(f"Downloading file: {blob.name}")
            # Download each audio file
            bytes = blob.download_as_bytes(raw_download=True)
            binary = BytesIO(bytes)

            # Load each audio file with librosa
            # Signal is a NumPy array with the audio data in waveform format
            signal, sr = librosa.load(binary, sr=None)

            # For each audio file, store the signal and the sampling rate in a raw_data variable
            raw_data[blob.name] = {
                "signal": signal,
                "sampling_rate": sr
            }

    with open(path, 'wb') as handle:
        pickle.dump(raw_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return raw_data

## Fonction pour uploadé la donné préprocessé ##
def gcloud_upload():
    pass

#########################################################################
#####################     PREPOCESS DATA    #############################
#########################################################################

## Fonction pour charger la donnée préprocessé ##
def load_prepoc_data():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=DATA_PREPROC)

    data = {}
    count = 0

    for blob in blobs:
        if blob.name.endswith(".jpg"):
            byte_data = blob.download_as_bytes()

            # 🔁 Convertir les bytes en image PIL, puis en array numpy, puis en tensor normalisé entre 0 et 1
            img = Image.open(io.BytesIO(byte_data)).convert(COLOR_MODE)
            img_array = np.array(img)
            img_tensor = tf.convert_to_tensor(img_array / 255.0, dtype=tf.float32)  #

            data[blob.name] = img_tensor
            count += 1
            print(f"Chargé en mémoire : {blob.name}")

            if count >= SAMPLE_SIZE:
                break

    return data




if __name__ == '__main__':

    images = load_prepoc_data()
    print(images)
    print(len(images))
