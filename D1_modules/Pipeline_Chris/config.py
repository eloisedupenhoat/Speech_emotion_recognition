# 📁 config.py

SR = 16000
N_FFT = 512
HOP_LENGTH = 256
N_MELS = 64
#N_MFCC = 20



MAX_LEN = 200
FEATURE_DIM  = 137  

BUCKET_NAME = "speech-emotion-bucket"
RAW_AUDIO_PREFIX = "Raw/"
FEATURES_PREFIX = "Features/"

N_CLASSES = 8  # à ajuster après LabelEncoder

# Path local pour la clé
GCP_CREDENTIALS_PATH = "./Users/greenwaymusic/code/eloisedupenhoat/Speech_emotion_recognition/D1_modules/Pipeline Chris/credentials"
