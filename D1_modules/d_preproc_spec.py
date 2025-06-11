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
import wave
import sys
import pickle

from D1_modules.a_utils import *

###########################################################################
##########       PREPROC 0 ON .WAV - NORMALIZE & TRIM NOISE      ##########
###########################################################################

def scale_waveform_data(y): # Applies a min max to scale the waveform data between -1 and 1 (amplitude)
    y = y / np.abs(y).max()
    return y

def trim_silence(y): # Removes leading / trailing silence using a decibel threshold (60 db)
    y, _ = librosa.effects.trim(y)
    return y

###########################################################################
##########        PREPROC 1 ON .WAV - DATA AUGMENTATION          ##########
###########################################################################

############################ INITIAL FUNCTIONS ############################

def add_noise(y, noise_level=0.005): # Add gaussian noise to simulate background variability (0.005 is the standard)
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def speed_change(y, sr, rate=1.1): # Change the speed by resampling
    # Resample the sampling rate (# data points per second) in order to mimic speed change
    return librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))

def pitch_shift(y, sr, n_steps=2): # Shifts pitch (up/down in semitones, e.g: ±2 steps) (2 steps is the standard)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

######################## APPLY DATA AUGMENTATION ##########################

def apply_noise_to_dict(dico):
    noisy_dict = {}
    for filename, value in dico.items():
        noisy_value = add_noise(value)
        noisy_dict[filename] = noisy_value
    return noisy_dict

def apply_speed_to_dict(dico, sr):
    speed_dict = {}
    for filename, value in dico.items():
        speed_value = speed_change(value, sr)
        speed_dict[filename] = speed_value
    return speed_dict

def apply_pitch_to_dict(dico, sr):
    pitch_dict = {}
    for filename, value in dico.items():
        pitch_value = pitch_shift(value, sr)
        pitch_dict[filename] = pitch_value
    return pitch_dict

###########################################################################
##########         PREPROC RAVDESS DATASET - BY EMOTION          ##########
###########################################################################

########################## PRE-DATA AUGENTATION ###########################

# Création d'un dico global {filename:file}
def ravdess_dico_name_file(raw_data):
    # Récupérer le sample rate du premier fichier (ils sont tous identiques)
    first_file = next(iter(raw_data.values()))
    sr = first_file['sampling_rate']
    # Extraire les données audio
    audio_data = {}
    for filename, file_data in raw_data.items():
        clean_filename = filename.split('/')[-1].replace('.wav','')
        audio_data[clean_filename] = file_data['signal']
    return audio_data, sr

# Dictionnaire ne contenant que les fichiers avec une émotion définie
def filter_dict_specific_emotion(dico, the_emotion):
    # Créer un dictionnaire pour stocker les entrées filtrées
    filtered_dict = {}
    # Parcourir chaque entrée du dictionnaire
    for key, value in dico.items():
        if emotion(key) == the_emotion:
            filtered_dict[key] = value
    return filtered_dict

# Division en 4 des dictionnaires (à ne pas faire sur l'émotion 1)
def divide_dico_into_four(dico):
    items = list(dico.items())   # dico = {'a': 1, 'b': 2} --> [('a', 1), ('b', 2)]
    total_items = len(items)
    # Calculer les tailles des quatre parties
    base_size = total_items // 4
    remainder = total_items % 4
    # Répartir le reste sur les premiers dictionnaires
    sizes = [base_size, base_size, base_size, base_size]
    for i in range(remainder):
        sizes[i] += 1
    # Découper la liste d'items en 4 parties
    idx1 = sizes[0]
    idx2 = idx1 + sizes[1]
    idx3 = idx2 + sizes[2]
    # Créer les dictionnaires
    dict1 = dict(items[:idx1])
    dict2 = dict(items[idx1:idx2])
    dict3 = dict(items[idx2:idx3])
    dict4 = dict(items[idx3:])
    return dict1, dict2, dict3, dict4

######################### POST-DATA AUGENTATION ##########################

# Par émotion, créer une liste avec les 3 dictionnaires
def list_dict_emotion(dict2, dict3, dict4):
    list_dict = [dict2, dict3, dict4]
    return list_dict

# Concaténation des dictionnaires augmentés, par émotion
def concatenate_augmented_dico(*dico): # Entrer la liste list_dict de list_dict_emotion
    augmented_dict = {}
    suffixes = ['_noise', '_speed', '_pitch']
    for i, dict_to_add in enumerate(dico):
        suffix = suffixes[i] if i < len(suffixes) else f'_aug{i}'
        for filename, value in dict_to_add.items():
            new_filename = filename + suffix
            augmented_dict[new_filename] = value
    return augmented_dict

def concatenate_emotion_dico(original_dict, augmented_dict):
    global_dict = {}
    # Ajouter les données originales avec suffixe '_original'
    for filename, value in original_dict.items():
        new_filename = filename + '_original'
        global_dict[new_filename] = value
    # Ajouter les données augmentées
    global_dict.update(augmented_dict)
    return global_dict

def concatenate_global_dico(*global_dicts): # A refaire sans la partie préfixe
    meta_dict = {}
    for global_dict in global_dicts:
        meta_dict.update(global_dict)
    return meta_dict

def shuffle_dico(dico):
    items = list(dico.items())
    random.shuffle(items)
    return dict(items)

###########################################################################
##########       PREPROC 2 - TRANSFORM .WAV INTO NP.ARRAY        ##########
###########################################################################

########################## GENERATE SPECTROGRAM ###########################

def compute_spectogram(y, sr): # It's a 2d array (rows = frequency ; columns = time ; values = intensity (brightness))
    S = librosa.feature.melspectrogram(y=y, sr=sr) # Convert waveform data to spectogram
    S_dB = librosa.power_to_db(S, ref=np.max) # Convert the spectogram from raw energy (power) into decibel scale (log scale) (so that it works for humans)
    return S_dB

####################### FROM SPECTROGRAM TO IMAGE #########################

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

###########################################################################
##########           PREPROC 3 - USE PREPROCESSED DATA           ##########
###########################################################################

####################### UPLOAD DATA IN GCP / API  #########################

def upload_image_in_GCP(resized_buf, file, BUCKET_NAME, resolution, color):
    client = storage.Client()
    output_path = file.replace("Raw/",f"Spectrograms_{resolution}p_{color}_{60}db/") # Save the spectogram images into a new cloud folder
    output_path = output_path.replace(".wav", ".jpg") # Change the files from audio to images
    bucket = client.bucket(BUCKET_NAME) # Get the bucket where we want to store our data
    image_blob = bucket.blob(output_path) # Create a new blob (a new file) for each image
    image_blob.upload_from_file(resized_buf, content_type="image/jpeg")  # Upload the image in each blob (in the cloud)

def upload_image_in_api(): # Déjà fait dans d_api.py
    pass

####################### AUGMENT DATA ON TRAIN SET #########################

TRAIN_ACTORS = set(range(1, 21)) # Defines training set (Actors 1–20)
TEST_ACTORS = set(range(21, 25)) # Defines test set (Actors 21–24)

# Puis, virer les données synthétiques des acteurs définis en test



####### OLD #######
"""
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
"""

####### INIT #######

if __name__ == '__main__':
    #raw_data = load_raw_data()
    #resolutions = [(64, 64), (128, 64), (64, 32)]
    #convert_to_spectogram_images(raw_data, resolutions=resolutions, augment=True)

    # Create emotion dictionnaries
    '''Dico global'''
    initial_dico, sr = ravdess_dico_name_file(raw_data)
    '''Dico par émotion'''
    dico_1 = filter_dict_specific_emotion(initial_dico, 1)
    dico_2 = filter_dict_specific_emotion(initial_dico, 2)
    dico_3 = filter_dict_specific_emotion(initial_dico, 3)
    dico_4 = filter_dict_specific_emotion(initial_dico, 4)
    dico_5 = filter_dict_specific_emotion(initial_dico, 5)
    dico_6 = filter_dict_specific_emotion(initial_dico, 6)
    dico_7 = filter_dict_specific_emotion(initial_dico, 7)
    dico_8 = filter_dict_specific_emotion(initial_dico, 8)

    # Division en 4 des dictionnaires (à ne pas faire sur l'émotion 1)
    dico_2a, dico_2b, dico_2c, dico_2d = divide_dico_into_four(dico_2)
    dico_3a, dico_3b, dico_3c, dico_3d = divide_dico_into_four(dico_3)
    dico_4a, dico_4b, dico_4c, dico_4d = divide_dico_into_four(dico_4)
    dico_5a, dico_5b, dico_5c, dico_5d = divide_dico_into_four(dico_5)
    dico_6a, dico_6b, dico_6c, dico_6d = divide_dico_into_four(dico_6)
    dico_7a, dico_7b, dico_7c, dico_7d = divide_dico_into_four(dico_7)
    dico_8a, dico_8b, dico_8c, dico_8d = divide_dico_into_four(dico_8)

    # Data augmentation sur l'émotion 1
    dico_1_noise = apply_noise_to_dict(dico_1)
    dico_1_speed = apply_speed_to_dict(dico_1, sr)
    dico_1_pitch = apply_pitch_to_dict(dico_1, sr)

    # Data augmentation sur les autres émotions
    dico_2_noise = apply_noise_to_dict(dico_2b)
    dico_2_speed = apply_speed_to_dict(dico_2c, sr)
    dico_2_pitch = apply_pitch_to_dict(dico_2d, sr)

    # Data augmentation sur les autres émotions
    dico_3_noise = apply_noise_to_dict(dico_3b)
    dico_3_speed = apply_speed_to_dict(dico_3c, sr)
    dico_3_pitch = apply_pitch_to_dict(dico_3d, sr)

    # Data augmentation sur les autres émotions
    dico_4_noise = apply_noise_to_dict(dico_4b)
    dico_4_speed = apply_speed_to_dict(dico_4c, sr)
    dico_4_pitch = apply_pitch_to_dict(dico_4d, sr)

    # Data augmentation sur les autres émotions
    dico_5_noise = apply_noise_to_dict(dico_5b)
    dico_5_speed = apply_speed_to_dict(dico_5c, sr)
    dico_5_pitch = apply_pitch_to_dict(dico_5d, sr)

    # Data augmentation sur les autres émotions
    dico_6_noise = apply_noise_to_dict(dico_6b)
    dico_6_speed = apply_speed_to_dict(dico_6c, sr)
    dico_6_pitch = apply_pitch_to_dict(dico_6d, sr)

    # Data augmentation sur les autres émotions
    dico_7_noise = apply_noise_to_dict(dico_7b)
    dico_7_speed = apply_speed_to_dict(dico_7c, sr)
    dico_7_pitch = apply_pitch_to_dict(dico_7d, sr)

    # Data augmentation sur les autres émotions
    dico_8_noise = apply_noise_to_dict(dico_8b)
    dico_8_speed = apply_speed_to_dict(dico_8c, sr)
    dico_8_pitch = apply_pitch_to_dict(dico_8d, sr)

    # Create emotion dictionnaries
    '''Dico global'''
    initial_dico, sr = ravdess_dico_name_file(raw_data)
    '''Dico par émotion'''
    dico_1 = filter_dict_specific_emotion(initial_dico, 1)
    dico_2 = filter_dict_specific_emotion(initial_dico, 2)
    dico_3 = filter_dict_specific_emotion(initial_dico, 3)
    dico_4 = filter_dict_specific_emotion(initial_dico, 4)
    dico_5 = filter_dict_specific_emotion(initial_dico, 5)
    dico_6 = filter_dict_specific_emotion(initial_dico, 6)
    dico_7 = filter_dict_specific_emotion(initial_dico, 7)
    dico_8 = filter_dict_specific_emotion(initial_dico, 8)

    # Division en 4 des dictionnaires (à ne pas faire sur l'émotion 1)
    dico_2a, dico_2b, dico_2c, dico_2d = divide_dico_into_four(dico_2)
    dico_3a, dico_3b, dico_3c, dico_3d = divide_dico_into_four(dico_3)
    dico_4a, dico_4b, dico_4c, dico_4d = divide_dico_into_four(dico_4)
    dico_5a, dico_5b, dico_5c, dico_5d = divide_dico_into_four(dico_5)
    dico_6a, dico_6b, dico_6c, dico_6d = divide_dico_into_four(dico_6)
    dico_7a, dico_7b, dico_7c, dico_7d = divide_dico_into_four(dico_7)
    dico_8a, dico_8b, dico_8c, dico_8d = divide_dico_into_four(dico_8)

    # Data augmentation sur l'émotion 1
    dico_1_noise = apply_noise_to_dict(dico_1)
    dico_1_speed = apply_speed_to_dict(dico_1, sr)
    dico_1_pitch = apply_pitch_to_dict(dico_1, sr)
    dico_1_noise['03-01-01-01-01-01-01']

    # Data augmentation sur les autres émotions
    dico_2_noise = apply_noise_to_dict(dico_2b)
    dico_2_speed = apply_speed_to_dict(dico_2c, sr)
    dico_2_pitch = apply_pitch_to_dict(dico_2d, sr)

    # Data augmentation sur les autres émotions
    dico_3_noise = apply_noise_to_dict(dico_3b)
    dico_3_speed = apply_speed_to_dict(dico_3c, sr)
    dico_3_pitch = apply_pitch_to_dict(dico_3d, sr)

    # Data augmentation sur les autres émotions
    dico_4_noise = apply_noise_to_dict(dico_4b)
    dico_4_speed = apply_speed_to_dict(dico_4c, sr)
    dico_4_pitch = apply_pitch_to_dict(dico_4d, sr)

    # Data augmentation sur les autres émotions
    dico_5_noise = apply_noise_to_dict(dico_5b)
    dico_5_speed = apply_speed_to_dict(dico_5c, sr)
    dico_5_pitch = apply_pitch_to_dict(dico_5d, sr)

    # Data augmentation sur les autres émotions
    dico_6_noise = apply_noise_to_dict(dico_6b)
    dico_6_speed = apply_speed_to_dict(dico_6c, sr)
    dico_6_pitch = apply_pitch_to_dict(dico_6d, sr)

    # Data augmentation sur les autres émotions
    dico_7_noise = apply_noise_to_dict(dico_7b)
    dico_7_speed = apply_speed_to_dict(dico_7c, sr)
    dico_7_pitch = apply_pitch_to_dict(dico_7d, sr)

    # Data augmentation sur les autres émotions
    dico_8_noise = apply_noise_to_dict(dico_8b)
    dico_8_speed = apply_speed_to_dict(dico_8c, sr)
    dico_8_pitch = apply_pitch_to_dict(dico_8d, sr)

    # Par émotion, créer une liste avec les 3 dictionnaires
    pre_list_1 = list_dict_emotion(dico_1_noise, dico_1_speed, dico_1_pitch)
    pre_list_2 = list_dict_emotion(dico_2_noise, dico_2_speed, dico_2_pitch)
    pre_list_3 = list_dict_emotion(dico_3_noise, dico_3_speed, dico_3_pitch)
    pre_list_4 = list_dict_emotion(dico_4_noise, dico_4_speed, dico_4_pitch)
    pre_list_5 = list_dict_emotion(dico_5_noise, dico_5_speed, dico_5_pitch)
    pre_list_6 = list_dict_emotion(dico_6_noise, dico_6_speed, dico_6_pitch)
    pre_list_7 = list_dict_emotion(dico_7_noise, dico_7_speed, dico_7_pitch)
    pre_list_8 = list_dict_emotion(dico_8_noise, dico_8_speed, dico_8_pitch)

    # Concaténation des dictionnaires augmentés, par émotion
    dico_1_pre_augm = concatenate_augmented_dico(*pre_list_1) # * pour dépaqueter la liste
    dico_2_pre_augm = concatenate_augmented_dico(*pre_list_2)
    dico_3_pre_augm = concatenate_augmented_dico(*pre_list_3)
    dico_4_pre_augm = concatenate_augmented_dico(*pre_list_4)
    dico_5_pre_augm = concatenate_augmented_dico(*pre_list_5)
    dico_6_pre_augm = concatenate_augmented_dico(*pre_list_6)
    dico_7_pre_augm = concatenate_augmented_dico(*pre_list_7)
    dico_8_pre_augm = concatenate_augmented_dico(*pre_list_8)

    dico_1_augm = concatenate_emotion_dico(dico_1, dico_1_pre_augm)

    dico_2_augm = concatenate_emotion_dico(dico_2a, dico_2_pre_augm)
    dico_3_augm = concatenate_emotion_dico(dico_3a, dico_3_pre_augm)
    dico_4_augm = concatenate_emotion_dico(dico_4a, dico_4_pre_augm)
    dico_5_augm = concatenate_emotion_dico(dico_5a, dico_5_pre_augm)
    dico_6_augm = concatenate_emotion_dico(dico_6a, dico_6_pre_augm)
    dico_7_augm = concatenate_emotion_dico(dico_7a, dico_7_pre_augm)
    dico_8_augm = concatenate_emotion_dico(dico_8a, dico_8_pre_augm)

    global_dicts = [dico_1_augm, dico_2_augm, dico_3_augm, dico_4_augm,
                    dico_5_augm, dico_6_augm, dico_7_augm, dico_8_augm]

    meta_dict = concatenate_global_dico(*global_dicts)

    final_meta_dict = shuffle_dico(meta_dict)


