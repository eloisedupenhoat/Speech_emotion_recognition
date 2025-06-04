##### OBJ: 1. Convert .wav files into spectrograms/MFCC #####
#####      2. Feature engineering                       #####

########## LIBRARIES ##########
from io import BytesIO
from google.cloud import storage
import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from params import *
from D1_modules.b_data import load_raw_data

####### SPECTOGRAM PREPROCESSING #######

def scale_waveform_data(y): # Scales the waveform so the maximum value is 1
    y = y / np.abs(y).max()
    return y

def remove_silence(y): # Silence is defined by a decibel threshold (60 db)
    y, _ = librosa.effects.trim(y)
    return y

def compute_spectogram(y, sr): # It's a 2d array (rows = frequency ; columns = time ; values = intensity (brightness))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max) # Convert the spectogram from raw energy (power) into decibel scale (log scale)
    return S_dB

def convert_to_image(S_dB): # Convert the standardized spectogram into an image file
    fig = plt.figure(figsize=(4, 4), dpi=100) # Create matlotplib figure
    plt.imshow(S_dB, cmap="magma", origin="lower", aspect="auto") # Display image
    plt.axis("off") # Remove axis
    buf = BytesIO() # Create temporary memory (buf)
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0, dpi=100) # Save image
    plt.close() # Close figure to free memory
    buf.seek(0) # Close temporary memory (buf)
    return buf

def resize_image(buf, lenght, witdh, channel): # Resize image in RGB & using the LANCZOS algorithm
    image = Image.open(buf).convert(f"{channel}")
    image = image.resize((lenght, witdh), resample=Image.Resampling.LANCZOS)
    resized_buf = BytesIO() # Save image back to buffer for cloud upload
    image.save(resized_buf, format="JPEG")
    resized_buf.seek(0)
    return resized_buf

def upload_image_in_GCP(resized_buf, file, BUCKET_NAME, resolution, color):
    client = storage.Client()
    output_path = file.replace("Raw/",f"Spectrograms_{resolution}p_{color}_{60}db/") # Save the spectogram images into a new cloud folder
    output_path = output_path.replace(".wav", ".jpg") # Change the files from audio to images
    bucket = client.bucket(BUCKET_NAME) # Get the bucket where we want to store our data
    image_blob = bucket.blob(output_path) # Create a new blob (a new file) for each image
    image_blob.upload_from_file(resized_buf, content_type="image/jpeg")  # Upload the image in each blob (in the cloud)

####### SPECTOGRAM PREPROCESSING #######

def convert_to_spectogram_images(raw_data, length = 64, width = 64, channel = "RGB", BUCKET_NAME = BUCKET_NAME):
    for filename, file in raw_data.items():
        print(f"Iterating file: {filename}")
        signal = file["signal"]
        sr = file["sampling_rate"]
        signal = scale_waveform_data(signal)
        signal = remove_silence(signal)
        spectogram = compute_spectogram(signal, sr)
        buf = convert_to_image(spectogram)
        image = resize_image(buf, length, width, channel)
        upload_image_in_GCP(image, filename, BUCKET_NAME, length, channel)

####### INIT #######

if __name__ == '__main__':
    raw_data = load_raw_data()
    convert_to_spectogram_images(raw_data)
