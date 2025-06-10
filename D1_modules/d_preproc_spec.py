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

def scale_waveform_data(y): # Applies a min max to scale the waveform data between -1 and 1 (amplitude)
    y = y / np.abs(y).max()
    return y

def trim_silence(y): # Removes leading / trailing silence using a decibel threshold (60 db)
    y, _ = librosa.effects.trim(y)
    return y

def add_noise(y, noise_level=0.005): # Add gaussian noise to simulate background variability (0.005 is the standard)
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def speed_change(y, sr, rate=1.1): # Change the speed by resampling
    # Resample the sampling rate (# data points per second) in order to mimic speed change
    return librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))

def pitch_shift(y, sr, n_steps=2): # Shifts pitch (up/down in semitones, e.g: ±2 steps) (2 steps is the standard)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

####### SPECTOGRAM PREPROCESSING #######

def compute_spectogram(y, sr): # It's a 2d array (rows = frequency ; columns = time ; values = intensity (brightness))
    S = librosa.feature.melspectrogram(y=y, sr=sr) # Convert waveform data to spectogram
    S_dB = librosa.power_to_db(S, ref=np.max) # Convert the spectogram from raw energy (power) into decibel scale (log scale) (so that it works for humans)
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

def resize_image(buf, lenght, width, channel): # Resize image in RGB & using the LANCZOS algorithm
    image = Image.open(buf).convert(f"{channel}")
    image = image.resize((lenght, width), resample=Image.Resampling.LANCZOS)
    resized_buf = BytesIO() # Save image back to buffer for cloud upload
    image.save(resized_buf, format="JPEG")
    resized_buf.seek(0)
    return resized_buf # Stores the resized spectogram image in a memory buffer

def upload_image_in_GCP(resized_buf, file, BUCKET_NAME, resolution, color):
    client = storage.Client()
    output_path = file.replace("Raw/",f"Spectrograms_{resolution}p_{color}_{60}db/") # Save the spectogram images into a new cloud folder
    output_path = output_path.replace(".wav", ".jpg") # Change the files from audio to images
    bucket = client.bucket(BUCKET_NAME) # Get the bucket where we want to store our data
    image_blob = bucket.blob(output_path) # Create a new blob (a new file) for each image
    image_blob.upload_from_file(resized_buf, content_type="image/jpeg")  # Upload the image in each blob (in the cloud)

####### FULL PREPROCESSING #######

TRAIN_ACTORS = set(range(1, 21)) # Defines training set (Actors 1–20)
TEST_ACTORS = set(range(21, 25)) # Defines test set (Actors 21–24)

def convert_to_spectogram_images(raw_data, resolutions, length=64, width=64, channel="L", BUCKET_NAME=BUCKET_NAME, augment=True):
    # If the actor is part of the train actors set:
    # Iterates over the filename and the file (waveform signal + sampling rate) for each file in the raw_data dictionnary
    # And returns a list of tuples (each tuple has the filename and the file)
    training_files = [(filename, file) for filename, file in raw_data.items() if get_actor_id(filename) in TRAIN_ACTORS]

    # Create a dictionary with each filename + file for each emotion (basically groups emotions in a dictionary)
    emotion_groups = defaultdict(list) # Creates a dictionarry where each key is an emotion ID, and each value is the list of files
    for filename, file in training_files: # Iterates through each filename and file in the training_files list
        emotion_id = get_emotion_id(filename) # Extracts the emotion ID for each filename
        emotion_groups[emotion_id].append((filename, file)) # For each emotion ID within the emotion_groups dictionary, it appends the filename and the file

    # Balance emotion 1 to reach 192 files (96 natural + 96 augmented)
    emotion_1_files = emotion_groups[1][:] # Creates a copy and gets all files that are labeled as emotion 1
    random.shuffle(emotion_1_files) # Shuffles though each file within emotion 1

    # Split 96 files into 3 parts (for data augmentation)
    noise_files = emotion_1_files[:32]
    speed_files = emotion_1_files[32:64]
    pitch_files = emotion_1_files[64:96]

    # Map 1st pass augmentations
    first_pass_aug_map = {} # Initializes an empty dictionary that will hold the filename and the augmentation applied to it

    for filename, _ in noise_files:
        first_pass_aug_map[filename] = "noise"
        # Assigns the value "noise" to match the key (file) for all the files where we are going to apply a noise augmenation

    for filename, _ in speed_files:
        first_pass_aug_map[filename] = "speed"
        # Assigns the value "speed" to match the key (file) for all the files where we are going to apply a speed augmenation

    for filename, _ in pitch_files:
        first_pass_aug_map[filename] = "pitch"
        # Assigns the value "pitch" to match the key (file) for all the files we are going to apply a pitch augmenation

    # Creates a list that includes all files from emotion 1 (that were augmented) as well as the other files (other emotions)
    # Returns a list of tuples (filename + file)
    second_pass_files = [f for f in training_files if f[0] in first_pass_aug_map or get_emotion_id(f[0]) != 1]

    # Modifies the list in order to randomly assign one augmenation per random file
    random.shuffle(second_pass_files)
    # Number of files we want to augment in the second pass
    num_files = len(second_pass_files)
    # Base list of augmentation types to cycle through
    base_aug_types = ["noise", "speed", "pitch"]
    # Number of full sets of 3 augmentations we need to cover most of the files + 1
    num_repeats = (num_files // len(base_aug_types)) + 1
    # Increases the list to make sure we assign one augmenation per file later on
    repeated_aug_list = base_aug_types * num_repeats

    # Takes the augmenation type list and trims it to make sure there's no indexing errors.
    second_aug_types = repeated_aug_list[:num_files]

    second_pass_aug_map = {}
    # Initialize a new dictionnary that maps the second-pass augmentations for each file
    i = 0 # Initialize a counter i that tracks the indexes
    for filename, _ in second_pass_files:
        # Iterates through each file name in the second_pass_files list
        prev = first_pass_aug_map.get(filename)
        # Checks which files have already been augmented (emotion 1) in the first pass and assigns it to a variable prev
        try:
            while second_aug_types[i] == prev:
                i += 1
                # If the loop goes through a file that has already been agumented:
                # It skips that augmentation type (+1) and assigns it another augmentation
        except IndexError:
            # If no different augmentation is found, restart from 0
            i = 0
        second_pass_aug_map[filename] = second_aug_types[i]
        # Finally, it assigns an augmentatio type (value) to each key (filename)
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
                aug_signal = speed_change(aug_signal, sr, rate=rate)
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
                aug_signal = speed_change(aug_signal, sr, rate=rate)
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
    resolutions = [(64, 64), (128, 64), (64, 32)]
    convert_to_spectogram_images(raw_data, resolutions=resolutions, augment=True)
