import os
import numpy as np

from dotenv import load_dotenv
# load_dotenv('/home/eloisedupenhoat/code/eloisedupenhoat/Speech_emotion_recognition/.env')
load_dotenv()

##################  VARIABLES  ##################
DATA_SIZE = os.environ.get("DATA_SIZE")
#CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
LOCATION = os.environ.get("LOCATION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
BATCH_SIZE = os.environ.get("BATCH_SIZE")
GCP_REGION = os.environ.get("GCP_REGION")
INSTANCE = os.environ.get("INSTANCE")
PORT = os.environ.get("PORT")
GAR_IMAGE = os.environ.get("GAR_IMAGE")
GAR_MEMORY = os.environ.get("GAR_MEMORY")
PREFIX = os.environ.get("PREFIX")

BUCKET_NAME = os.environ.get("BUCKET_NAME")
DATA_PREPROC = os.environ.get("DATA_PREPROC")
COLOR_MODE = os.environ.get("COLOR_MODE")
SAMPLE_SIZE = os.environ.get("SAMPLE_SIZE", 100)

##################  CONSTANTS  #####################
