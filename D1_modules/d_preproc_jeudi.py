# ============================================================================
# SCRIPT 1: PREPROCESSING RAVDESS avec Data Augmentation pour Équilibrage
# À exécuter sur GCP ou en local
# ============================================================================

import os
import numpy as np
import pickle
import io
import re
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
from google.cloud import storage
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Configuration
BUCKET_NAME = "speech-emotion-bucket"
PREPROC_FOLDER = "raw_data_06_12_spect"  # Dossier spécifique pour ce preprocessing
DATASET_NAME = "ravdess"

# Paramètres spectrogrammes (vos paramètres recommandés)
SPEC_PARAMS = {
    'n_mels': 128,          # Résolution fréquentielle
    'hop_length': 512,      # Chevauchement
    'n_fft': 2048,          # Taille FFT
    'win_length': 2048,     # Taille fenêtre
    'fmin': 20,             # Fréquence min (Hz)
    'fmax': 8000,           # Fréquence max (Hz)
    'top_db': 80            # Range dynamique (dB)
}

# Paramètres augmentation
AUG_PARAMS = {
    'noise_level': 0.005,
    'speed_rate': 1.1,
    'pitch_steps': 2
}

# ============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ============================================================================

def load_raw_data(base_dir='../raw_data'):
    """Charge les données audio depuis un dossier local"""
    raw_data = {}

    print(f"[INFO] Chargement des données depuis {base_dir}...")

    for dirpath, _, filenames in os.walk(base_dir):
        for filename in filenames:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(dirpath, filename)
                try:
                    y, sr = librosa.load(full_path, sr=None)  # sr=None conserve la fréquence native
                    raw_data[full_path] = {
                        "signal": y,
                        "sampling_rate": sr
                    }
                except Exception as e:
                    print(f"[ERREUR] Impossible de charger {full_path} : {e}")

    print(f"[INFO] {len(raw_data)} fichiers audio chargés depuis {base_dir}")
    return raw_data

# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_actor_id(file_path: str) -> int:
    """Extrait l'ID de l'acteur depuis le chemin"""
    match = re.search(r'Actor_(\d{2})', file_path)
    if not match:
        raise ValueError(f"Actor ID not found in path: {file_path}")
    return int(match.group(1))

def emotion(name):
    """Extrait le code émotion depuis le nom de fichier"""
    filename = name.split('/')[-1]
    filename_no_ext = filename.replace('.wav', '')
    parts = filename_no_ext.split('-')
    if len(parts) < 3:
        raise ValueError(f"Format de fichier invalide: {filename}")
    emotion_code = int(parts[2])
    return emotion_code

def create_dynamic_model_name(spec_params, aug_params, sample_count, model_params):
    """Crée un nom de modèle dynamique basé sur les paramètres réels"""
    n_mels = spec_params['n_mels']
    top_db = spec_params['top_db']
    lr_str = str(model_params['learning_rate']).replace('.', '')
    batch_size = model_params['batch_size']
    patience = model_params['patience']
    epochs = model_params['epochs']
    validation_split = int(model_params.get('validation_split', 0) * 100)

    model_name = f"Spec-{n_mels}p-Gray-{top_db}db_sample{sample_count}_cnn-lr{lr_str}-bs{batch_size}-pat{patience}-vs{validation_split}-{epochs}ep"

    return model_name

# ============================================================================
# FONCTIONS GCP
# ============================================================================

def upload_to_gcp(data, filename, bucket_name, folder_name):
    """Upload des données vers GCP dans un dossier spécifique - Version optimisée"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{folder_name}/{filename}")

    if filename.endswith('.npz'):
        # Upload direct en mémoire pour les fichiers NPZ
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **data)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

    elif filename.endswith('.pkl'):
        # Upload direct en mémoire pour les fichiers pickle
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type='application/octet-stream')

    else:
        raise ValueError(f"Type de fichier non supporté: {filename}")

    print(f"[INFO] ✅ {filename} uploadé vers GCP: gs://{bucket_name}/{folder_name}/{filename}")

# ============================================================================
# FONCTIONS D'AUGMENTATION
# ============================================================================

def add_noise(y, noise_level=0.005):
    """Ajoute du bruit gaussien"""
    noise = np.random.normal(0, noise_level, y.shape)
    return y + noise

def speed_change(y, sr, rate=1.1):
    """Change la vitesse de lecture"""
    return librosa.resample(y, orig_sr=sr, target_sr=int(sr * rate))

def pitch_shift(y, sr, n_steps=2):
    """Décale la hauteur tonale"""
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def quadruple_emotion_1_with_augmentation(emotion_1_dict, sr):
    """Quadruple l'émotion 1 avec augmentation"""
    print(f"[INFO] Quadruplage de l'émotion 1 (neutre): {len(emotion_1_dict)} → {len(emotion_1_dict) * 4}")

    quadrupled_dict = {}
    for filename, y in emotion_1_dict.items():
        quadrupled_dict[filename + '_original'] = y
        quadrupled_dict[filename + '_noise'] = add_noise(y, AUG_PARAMS['noise_level'])
        quadrupled_dict[filename + '_speed'] = speed_change(y, sr, AUG_PARAMS['speed_rate'])
        quadrupled_dict[filename + '_pitch'] = pitch_shift(y, sr, AUG_PARAMS['pitch_steps'])

    return quadrupled_dict

def double_emotion_with_mixed_augmentation(emotion_dict, sr):
    """Double une émotion avec augmentation mixte (1/3 noise, 1/3 speed, 1/3 pitch)"""
    items = list(emotion_dict.items())
    total_items = len(items)

    # Diviser en 3 groupes pour les 3 types d'augmentation
    third = total_items // 3
    remainder = total_items % 3

    # Répartir le reste sur les premiers groupes
    sizes = [third, third, third]
    for i in range(remainder):
        sizes[i] += 1

    idx1 = sizes[0]
    idx2 = idx1 + sizes[1]

    noise_items = items[:idx1]
    speed_items = items[idx1:idx2]
    pitch_items = items[idx2:]

    final_dict = {}

    # Ajouter les originaux
    for filename, y in emotion_dict.items():
        final_dict[filename + '_original'] = y

    # Ajouter les versions augmentées
    # 1/3 avec bruit
    for filename, y in noise_items:
        final_dict[filename + '_noise'] = add_noise(y, AUG_PARAMS['noise_level'])

    # 1/3 avec changement de vitesse
    for filename, y in speed_items:
        final_dict[filename + '_speed'] = speed_change(y, sr, AUG_PARAMS['speed_rate'])

    # 1/3 avec pitch shift
    for filename, y in pitch_items:
        final_dict[filename + '_pitch'] = pitch_shift(y, sr, AUG_PARAMS['pitch_steps'])

    return final_dict

# ============================================================================
# FONCTIONS DE PREPROCESSING RAVDESS
# ============================================================================

def ravdess_dico_name_file(raw_data):
    """Convertit les données brutes en dictionnaire {filename: signal}"""
    first_file = next(iter(raw_data.values()))
    sr = first_file['sampling_rate']

    audio_data = {}
    for filepath, file_data in raw_data.items():
        # Extraire le nom du fichier depuis le chemin complet
        clean_filename = os.path.basename(filepath).replace('.wav', '')
        audio_data[clean_filename] = file_data['signal']

    print(f"[INFO] ✅ Conversion terminée: {len(audio_data)} échantillons à {sr} Hz")
    return audio_data, sr

def filter_dict_specific_emotion(dico, the_emotion):
    """Filtre le dictionnaire pour une émotion spécifique"""
    filtered_dict = {}
    for key, value in dico.items():
        try:
            if emotion(key) == the_emotion:
                filtered_dict[key] = value
        except ValueError as e:
            print(f"[WARNING] Impossible d'extraire l'émotion de {key}: {e}")
            continue
    return filtered_dict

def shuffle_dico(dico):
    """Mélange aléatoirement un dictionnaire"""
    items = list(dico.items())
    random.shuffle(items)
    return dict(items)

def concatenate_global_dico(*global_dicts):
    """Concatène plusieurs dictionnaires"""
    meta_dict = {}
    for global_dict in global_dicts:
        meta_dict.update(global_dict)
    return meta_dict

def apply_balanced_augmentation(audio_data, sr):
    """Applique l'augmentation équilibrée selon votre stratégie"""
    print("\n[INFO] 🎯 Application de l'augmentation équilibrée...")

    # Séparer par émotion
    emotion_dicts = {}
    for emotion_code in range(1, 9):  # Émotions 1-8
        emotion_dicts[emotion_code] = filter_dict_specific_emotion(audio_data, emotion_code)
        print(f"  - Émotion {emotion_code}: {len(emotion_dicts[emotion_code])} échantillons")

    # Traitement spécial pour l'émotion 1 (neutre) - QUADRUPLER
    print(f"\n[INFO] Traitement émotion 1 (neutre) - quadruplage...")
    emotion_1_augmented = quadruple_emotion_1_with_augmentation(emotion_dicts[1], sr)
    print(f"  ✅ Émotion 1: {len(emotion_dicts[1])} → {len(emotion_1_augmented)} (x4)")

    # Traitement des autres émotions (2-8) - DOUBLER avec augmentation mixte
    print(f"\n[INFO] Traitement émotions 2-8 - doublement avec augmentation mixte...")
    other_emotions_augmented = {}
    for emotion_code in range(2, 9):
        emotion_dict = emotion_dicts[emotion_code]
        if len(emotion_dict) > 0:
            # Doubler avec augmentation mixte (1/3 noise, 1/3 speed, 1/3 pitch)
            augmented = double_emotion_with_mixed_augmentation(emotion_dict, sr)
            other_emotions_augmented[emotion_code] = augmented
            print(f"  ✅ Émotion {emotion_code}: {len(emotion_dict)} → {len(augmented)} (x2)")
        else:
            print(f"  ⚠️ Émotion {emotion_code}: {len(emotion_dict)} (aucun échantillon)")

    # Concaténer tout
    all_augmented_dicts = [emotion_1_augmented] + list(other_emotions_augmented.values())
    final_audio_data = concatenate_global_dico(*all_augmented_dicts)

    # Mélanger pour éviter les biais d'ordre
    final_audio_data = shuffle_dico(final_audio_data)

    print(f"\n[INFO] ✅ Augmentation terminée: {len(audio_data)} → {len(final_audio_data)} échantillons")

    # Vérification finale des proportions
    print(f"\n[INFO] 📊 Vérification de l'augmentation:")
    for emotion_code in range(1, 9):
        original_count = len(emotion_dicts[emotion_code])
        augmented_emotion_data = filter_dict_specific_emotion_from_augmented(final_audio_data, emotion_code)
        augmented_count = len(augmented_emotion_data)
        multiplier = augmented_count / original_count if original_count > 0 else 0
        print(f"  - Émotion {emotion_code}: {original_count} → {augmented_count} (x{multiplier:.1f})")

    return final_audio_data

def filter_dict_specific_emotion_from_augmented(dico, the_emotion):
    """Filtre le dictionnaire augmenté pour une émotion spécifique"""
    filtered_dict = {}
    for key, value in dico.items():
        try:
            # Extraire le nom original (avant les suffixes d'augmentation)
            original_key = key.split('_')[0]
            if emotion(original_key) == the_emotion:
                filtered_dict[key] = value
        except ValueError:
            continue
    return filtered_dict

def create_mel_spectrogram(y, sr, spec_params):
    """Crée un spectrogramme mel-scale"""
    # Spectrogramme mel
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=spec_params['n_mels'],
        hop_length=spec_params['hop_length'],
        n_fft=spec_params['n_fft'],
        win_length=spec_params['win_length'],
        fmin=spec_params['fmin'],
        fmax=spec_params['fmax']
    )

    # Conversion en dB
    mel_spec_db = librosa.power_to_db(mel_spec, top_db=spec_params['top_db'])

    # Normalisation [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    return mel_spec_norm

def split_by_actors(data_dict, train_actors=list(range(1, 21)), test_actors=list(range(21, 25))):
    """Sépare les données par acteurs"""
    train_data = {}
    test_data = {}

    for filename, data in data_dict.items():
        # Extraire l'ID de l'acteur depuis le nom de fichier original
        original_filename = filename.split('_')[0]  # Retire les suffixes d'augmentation
        try:
            # Format RAVDESS: XX-XX-XX-XX-XX-XX-XX.wav où le 7ème champ est l'acteur
            parts = original_filename.split('-')
            if len(parts) >= 7:
                actor_id = int(parts[6])
            else:
                # Fallback: chercher pattern dans le filename
                match = re.search(r'(\d{2})$', parts[-1])  # Dernier nombre à 2 chiffres
                if match:
                    actor_id = int(match.group(1))
                else:
                    print(f"[WARNING] Impossible d'extraire l'ID acteur de {filename}")
                    continue
        except (IndexError, ValueError) as e:
            print(f"[WARNING] Erreur extraction acteur pour {filename}: {e}")
            continue

        if actor_id in train_actors:
            train_data[filename] = data
        elif actor_id in test_actors:
            test_data[filename] = data

    return train_data, test_data

# ============================================================================
# FONCTION PRINCIPALE DE PREPROCESSING
# ============================================================================

def main_preprocessing(base_dir='../raw_data'):
    """
    Fonction principale de preprocessing

    Args:
        base_dir (str): Chemin vers le dossier contenant les données RAVDESS
    """

    print("=" * 70)
    print("DÉBUT DU PREPROCESSING RAVDESS AVEC AUGMENTATION ÉQUILIBRÉE")
    print("=" * 70)

    # 1. Chargement du dataset depuis le dossier local
    print(f"\n1. Chargement du dataset depuis {base_dir}...")

    raw_data = load_raw_data(base_dir)
    if not raw_data:
        print("[ERROR] Aucune donnée chargée depuis le dossier local")
        return None

    print(f"[INFO] ✅ Dataset chargé: {len(raw_data)} fichiers audio")

    # 2. Conversion en dictionnaire audio
    print("\n2. Conversion en dictionnaire audio...")
    audio_data, sr = ravdess_dico_name_file(raw_data)

    # 3. Split train/test AVANT augmentation (important pour le test)
    print("\n3. Séparation train/test par acteurs...")
    train_audio, test_audio = split_by_actors(audio_data)
    print(f"[INFO] ✅ Split terminé:")
    print(f"  - Train (acteurs 1-20): {len(train_audio)} échantillons")
    print(f"  - Test (acteurs 21-24): {len(test_audio)} échantillons")

    # 4. Augmentation UNIQUEMENT sur les données d'entraînement
    print("\n4. Augmentation des données d'entraînement...")
    train_audio_augmented = apply_balanced_augmentation(train_audio, sr)
    print(f"[INFO] ✅ Augmentation train terminée: {len(train_audio)} → {len(train_audio_augmented)}")

    # 5. Génération des spectrogrammes
    print("\n5. Génération des spectrogrammes...")
    print(f"[INFO] Paramètres: {SPEC_PARAMS}")

    # Train spectrograms (avec augmentation)
    X_train = []
    y_train = []
    train_paths = []

    print("  - Génération spectrogrammes train...")
    for i, (filename, signal) in enumerate(train_audio_augmented.items()):
        if i % 200 == 0:
            print(f"    Traitement: {i}/{len(train_audio_augmented)}")

        try:
            # Spectrogramme
            spec = create_mel_spectrogram(signal, sr, SPEC_PARAMS)
            X_train.append(spec)

            # Label (émotion)
            original_filename = filename.split('_')[0]  # Retire suffixes augmentation
            emotion_code = emotion(original_filename)
            y_train.append(emotion_code - 1)  # 0-indexé pour Keras

            train_paths.append(filename)
        except Exception as e:
            print(f"[WARNING] Erreur traitement {filename}: {e}")
            continue

    # Test spectrograms (sans augmentation)
    X_test = []
    y_test = []
    test_paths = []

    print("  - Génération spectrogrammes test...")
    for i, (filename, signal) in enumerate(test_audio.items()):
        if i % 50 == 0:
            print(f"    Traitement: {i}/{len(test_audio)}")

        try:
            # Spectrogramme
            spec = create_mel_spectrogram(signal, sr, SPEC_PARAMS)
            X_test.append(spec)

            # Label
            emotion_code = emotion(filename)
            y_test.append(emotion_code - 1)

            test_paths.append(filename)
        except Exception as e:
            print(f"[WARNING] Erreur traitement {filename}: {e}")
            continue

    # Conversion en arrays NumPy
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Ajouter dimension channel pour CNN (H, W, C)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Conversion en categorical pour Keras
    y_train_cat = to_categorical(y_train, num_classes=8)
    y_test_cat = to_categorical(y_test, num_classes=8)

    print(f"[INFO] ✅ Spectrogrammes générés:")
    print(f"  - X_train: {X_train.shape}")
    print(f"  - X_test: {X_test.shape}")
    print(f"  - y_train: {y_train_cat.shape}")
    print(f"  - y_test: {y_test_cat.shape}")

    # 6. Statistiques des classes
    print("\n6. Analyse de l'équilibrage des classes:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    emotions_mapping = {
        0: "Neutre", 1: "Calme", 2: "Heureux", 3: "Triste",
        4: "Colère", 5: "Peur", 6: "Dégoût", 7: "Surprise"
    }

    print("  Distribution train (après augmentation):")
    for emotion_idx, count in zip(unique_train, counts_train):
        emotion_name = emotions_mapping[emotion_idx]
        print(f"    {emotion_name} (classe {emotion_idx}): {count} échantillons")

    print("  Distribution test (originale):")
    for emotion_idx, count in zip(unique_test, counts_test):
        emotion_name = emotions_mapping[emotion_idx]
        print(f"    {emotion_name} (classe {emotion_idx}): {count} échantillons")

    # 7. Préparation des métadonnées
    metadata = {
        'spec_params': SPEC_PARAMS,
        'aug_params': AUG_PARAMS,
        'num_classes': 8,
        'emotions_mapping': emotions_mapping,
        'train_actors': list(range(1, 21)),
        'test_actors': list(range(21, 25)),
        'sample_rate': sr,
        'augmentation_applied': True,
        'train_distribution': dict(zip(unique_train, counts_train)),
        'test_distribution': dict(zip(unique_test, counts_test)),
        'preprocessing_date': datetime.now().isoformat(),
        'total_train_samples': len(X_train),
        'total_test_samples': len(X_test),
        'preprocessing_folder': PREPROC_FOLDER,
        'base_dir': base_dir
    }

    # 8. Sauvegarde sur GCP
    print(f"\n8. Sauvegarde vers GCP dans le dossier '{PREPROC_FOLDER}'...")

    # Données principales
    data_to_save = {
        'X_train': X_train,
        'y_train': y_train_cat,
        'X_test': X_test,
        'y_test': y_test_cat,
        'train_paths': np.array(train_paths),
        'test_paths': np.array(test_paths)
    }

    try:
        upload_to_gcp(data_to_save, 'ravdess_spectrograms.npz', BUCKET_NAME, PREPROC_FOLDER)
        upload_to_gcp(metadata, 'metadata.pkl', BUCKET_NAME, PREPROC_FOLDER)

        print(f"[INFO] ✅ Sauvegarde terminée!")
        print(f"  - Données: gs://{BUCKET_NAME}/{PREPROC_FOLDER}/ravdess_spectrograms.npz")
        print(f"  - Métadonnées: gs://{BUCKET_NAME}/{PREPROC_FOLDER}/metadata.pkl")

    except Exception as e:
        print(f"[ERROR] Échec sauvegarde GCP: {e}")
        print("[INFO] Sauvegarde locale...")
        np.savez_compressed('ravdess_spectrograms.npz', **data_to_save)
        with open('metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("[INFO] ✅ Sauvegarde locale terminée")

    # 9. Génération du nom de modèle pour la suite
    model_params = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'patience': 3,
        'epochs': 5,
        'validation_split': 0
    }

    model_name = create_dynamic_model_name(SPEC_PARAMS, AUG_PARAMS, len(X_train), model_params)
    print(f"\n[INFO] 🎯 Nom suggéré pour le modèle: {model_name}")

    print(f"\n✅ PREPROCESSING TERMINÉ!")
    print(f"📊 Résumé:")
    print(f"  - Dossier de données: {base_dir}")
    print(f"  - Échantillons train: {len(X_train)} (avec augmentation)")
    print(f"  - Échantillons test: {len(X_test)} (sans augmentation)")
    print(f"  - Classes équilibrées: ✅")
    print(f"  - Données sauvées dans: gs://{BUCKET_NAME}/{PREPROC_FOLDER}/")
    print(f"  - Prêt pour l'entraînement!")

    return X_train, y_train_cat, X_test, y_test_cat, metadata

# ============================================================================
# FONCTION D'USAGE SIMPLIFIÉE
# ============================================================================

def preprocess_ravdess_data(base_dir='../raw_data'):
    """Interface simplifiée pour preprocessing des données RAVDESS"""
    return main_preprocessing(base_dir=base_dir)

# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    # Preprocessing depuis le dossier local
    # Changez le chemin selon votre structure de dossiers
    X_train, y_train, X_test, y_test, metadata = preprocess_ravdess_data('../raw_data')
