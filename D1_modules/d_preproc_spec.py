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
import random
from collections import defaultdict

####### WAVEFORM PREPROCESSING #######

def scale_waveform_data(y): # Normalize waveform amplitude betqeen -1 and 1
    y = y / np.abs(y).max()
    return y

def trim_silence(y): # Removes leading / trailing silence using a decibel threshold (60 db)
    y, _ = librosa.effects.trim(y)
    return y

def add_noise(y, noise_level=0.005): # Add Gaussian noise to simulate background variability
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def speed_change(y, rate=1.1): # Change playback speed (e.g: 1.1 speeds it up by 10%)
    try: # If librosa doesn't return an error apply this function
        return librosa.effects.time_stretch(y, rate)
    except librosa.util.exceptions.ParameterError: # Else: return the original sound (y)
        return y

def pitch_shift(y, sr, n_steps=2): # Shifts pitch (up/down in semitones, e.g: ±2 steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

####### SPECTOGRAM PREPROCESSING #######

def compute_spectogram(y, sr): # It's a 2d array (rows = frequency ; columns = time ; values = intensity (brightness))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max) # Convert the spectogram from raw energy (power) into decibel scale (log scale)
    return S_dB

####### SPECTOGRAM IMAGE PREPROCESSING #######

def get_actor_id(filename): # Returns the actor ID from the filename
    return int(filename.split("-")[-1].split(".")[0])

def get_emotion_id(filename): # Returns the emotion ID from the filename
    return int(filename.split("-")[2])

def convert_to_spectogram_image(S_dB): # Convert the standardized spectogram into an image file
    fig = plt.figure(figsize=(4, 4), dpi=100) # Create matlotplib figure
    plt.imshow(S_dB, cmap="gray", origin="lower", aspect="auto") # Display image
    plt.axis("off") # Remove axis
    buf = BytesIO() # Create temporary memory (buf)
    plt.savefig(buf, format="jpg", bbox_inches="tight", pad_inches=0, dpi=100) # Save image
    plt.close() # Close figure to free memory
    buf.seek(0) # Close temporary memory (buf)
    return buf # Buf is a memory buffer that stores the spectogram image

def resize_image(buf, lenght, witdh, channel): # Resize image in RGB & using the LANCZOS algorithm
    image = Image.open(buf).convert(f"{channel}")
    image = image.resize((lenght, witdh), resample=Image.Resampling.LANCZOS)
    resized_buf = BytesIO() # Save image back to buffer for cloud upload
    image.save(resized_buf, format="JPEG")
    resized_buf.seek(0)
    return resized_buf # Stores the resized spectogram image

def upload_image_in_GCP(resized_buf, file, BUCKET_NAME, resolution, color):
    client = storage.Client()
    output_path = file.replace("Raw/",f"Spectrograms_{resolution}p_{color}_{60}db/") # Save the spectogram images into a new cloud folder
    output_path = output_path.replace(".wav", ".jpg") # Change the files from audio to images
    bucket = client.bucket(BUCKET_NAME) # Get the bucket where we want to store our data
    image_blob = bucket.blob(output_path) # Create a new blob (a new file) for each image
    image_blob.upload_from_file(resized_buf, content_type="image/jpeg")  # Upload the image in each blob (in the cloud)

####### FULL PREPROCESSING #######

TRAIN_ACTORS = set(range(1, 22))   # Actors 1–21 → training
TEST_ACTORS = set(range(22, 25))   # Actors 22–24 → testing

def convert_to_spectogram_images(raw_data, resolutions, length=64, width=64, channel="L", BUCKET_NAME=BUCKET_NAME, augment=True):
    # Separate training files
    training_files = [(filename, file) for filename, file in raw_data.items() if get_actor_id(filename) in TRAIN_ACTORS]

    # Group by emotion
    emotion_groups = defaultdict(list)
    for filename, file in training_files:
        emotion_id = get_emotion_id(filename)
        emotion_groups[emotion_id].append((filename, file))

    # Balance emotion 1 to reach 192 files (96 natural + 96 augmented)
    emotion_1_files = emotion_groups[1][:]
    random.shuffle(emotion_1_files)
    base_count = len(emotion_1_files)
    needed = 96 - (base_count // 2)  # Only need 96 synthetic files in pass 1
    first_pass_aug_map = {}

    aug_types = ["noise", "speed", "pitch"] * ((needed // 3) + 1)
    aug_types = aug_types[:needed]
    random.shuffle(aug_types)

    for (filename, _), aug in zip(emotion_1_files, aug_types):
        first_pass_aug_map[filename] = aug

    # Assign second-pass augmentation types for all training files while avoiding duplicates for emotion 1
    second_pass_files = [f for f in training_files if f[0] in first_pass_aug_map or get_emotion_id(f[0]) != 1]
    random.shuffle(second_pass_files)
    second_aug_types = ["noise", "speed", "pitch"] * ((len(second_pass_files) // 3) + 1)
    second_aug_types = second_aug_types[:len(second_pass_files)]
    second_pass_aug_map = {}

    i = 0
    for filename, _ in second_pass_files:
        prev = first_pass_aug_map.get(filename)
        while second_aug_types[i] == prev:
            i += 1
        second_pass_aug_map[filename] = second_aug_types[i]
        i += 1

    # Loop through all files with an index (idx), so we can selectively apply augmentation
    for idx, (filename, file) in enumerate(raw_data.items()):
        print(f"Iterating file: {filename}")
        actor_id = get_actor_id(filename)
        emotion_id = get_emotion_id(filename)

        signal = file["signal"]
        sr = file["sampling_rate"]

        # Step 1: Preprocessing - normalize and trim silence
        signal_clean = scale_waveform_data(signal)
        signal_clean = trim_silence(signal_clean)

        # Step 2: Convert and upload original (clean) spectrogram
        spectrogram = compute_spectogram(signal_clean, sr)
        buf = convert_to_spectogram_image(spectrogram)
        image = resize_image(buf, length, width, channel)
        clean_filename = filename.replace(".wav", f"_trimmed.jpg")
        upload_image_in_GCP(image, clean_filename, BUCKET_NAME, length, channel)

        # Step 3a: Apply first augmentation for Emotion 1 balancing
        if augment and filename in first_pass_aug_map:
            aug_type = first_pass_aug_map[filename]
            aug_signal = signal_clean.copy()

            if aug_type == "noise":
                noise_level = round(np.random.uniform(0.003, 0.01), 4)
                aug_signal = add_noise(aug_signal, noise_level=noise_level)
                suffix = f"noise_{noise_level}"
            elif aug_type == "speed":
                rate = round(np.random.uniform(0.9, 1.2), 2)
                aug_signal = speed_change(aug_signal, rate=rate)
                suffix = f"speed_{rate}"
            elif aug_type == "pitch":
                n_steps = np.random.choice([-2, -1, 1, 2])
                aug_signal = pitch_shift(aug_signal, sr, n_steps=n_steps)
                suffix = f"pitch_{n_steps:+d}"

            for (length, width) in resolutions:
                aug_spec = compute_spectogram(aug_signal, sr)
                aug_buf = convert_to_spectogram_image(aug_spec)
                aug_image = resize_image(aug_buf, length, width, channel)
                aug_filename = filename.replace(".wav", f"_trimmed_aug1_{suffix}.jpg")
                upload_image_in_GCP(aug_image, aug_filename, BUCKET_NAME, f"{length}x{width}", channel)

        # Step 3b: Apply second-pass augmentation for all training files (including emotion 1 again)
        if augment and actor_id in TRAIN_ACTORS:
            aug_type = second_pass_aug_map.get(filename)
            aug_signal = signal_clean.copy()

            if aug_type == "noise":
                noise_level = round(np.random.uniform(0.003, 0.01), 4)
                aug_signal = add_noise(aug_signal, noise_level=noise_level)
                suffix = f"noise_{noise_level}"
            elif aug_type == "speed":
                rate = round(np.random.uniform(0.9, 1.2), 2)
                aug_signal = speed_change(aug_signal, rate=rate)
                suffix = f"speed_{rate}"
            elif aug_type == "pitch":
                n_steps = np.random.choice([-2, -1, 1, 2])
                aug_signal = pitch_shift(aug_signal, sr, n_steps=n_steps)
                suffix = f"pitch_{n_steps:+d}"

            for (length, width) in resolutions:
                aug_spec = compute_spectogram(aug_signal, sr)
                aug_buf = convert_to_spectogram_image(aug_spec)
                aug_image = resize_image(aug_buf, length, width, channel)
                aug_filename = filename.replace(".wav", f"_trimmed_aug2_{suffix}.jpg")
                upload_image_in_GCP(aug_image, aug_filename, BUCKET_NAME, f"{length}x{width}", channel)

####### INIT #######

if __name__ == '__main__':
    raw_data = load_raw_data()
    resolutions = [(64, 64), (128, 64)]
    convert_to_spectogram_images(raw_data, resolutions=resolutions, augment=True)
