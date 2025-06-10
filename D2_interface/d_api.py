import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import librosa
from tensorflow import keras # EXPLIQUER
from keras import Model, Sequential # EXPLIQUER
from keras.utils import to_categorical
from io import BytesIO
from google.cloud import storage
import io
from io import BytesIO

# J'indique que l'on travaille sur le dossier source du dossier dans lequel on se trouve
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from D1_modules.b_data import load_model
from D1_modules.a_utils import *
from D1_modules.d_preproc_spec import *

'''
DEPUIS MISE A JOUR : IMPOSSIBLE DE TELECHARGER UN .WAV DEPUIS FAST API !!!!!
TBD : mettre à jour le modèle que l'API doit aller chercher sur gcp
Mettre à jour le RGB and co
'''
###########################################################################
##########         A REMPLIR AVANT LE LANCEMENT DE L'API         ##########
###########################################################################

model = "Spec-64p-RGB-60db_sample1000_cnn-lr0_0005-bs32-pat2-vs30-50ep.keras"
# Les length, width, channel doivent ê cohérents ac les inputs du modèle choisi :
length = 64
width = 64
channel = "RGB"

###########################################################################
##########                       PREPARATION                     ##########
###########################################################################
class Message(BaseModel):
    message: str
    description: str | None = None

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

###########################################################################
##########       AU LANCEMENT DE L'API - CHARGER LE MODELE       ##########
###########################################################################

'''
Sur le terminal : se positionner dans speech_emotion_recognition.
Puis, exécuter : uvicorn D2_interface.d_api:app --reload
Pourquoi le faire mainteant ? Car le modèle prend du temps à charger et
qu'on veut faire une démo en live (dc éviter d'attendre que le modèle charge
en public)
'''

app.state.model = load_model(model)


###########################################################################
##########            EN LIVE - OUTPUT - PREDICTION              ##########
###########################################################################

'''
En phase de dvt : Postman pour tester, avec <url du terminal>/uploadfile/
Ou encore plus facile : http://127.0.0.1:8000/docs (avec http://127.0.0.1:8000 le code fourni par le terminal)
'''

@app.post('/predict/')
async def predict(my_file: UploadFile = File(...)):

    # Upload file
    filename = my_file.filename
    contents = await my_file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)
    file = "temp.wav"

    # Data preprocessing
    signal, sr = librosa.load(file, sr=None) # None ok car pas de data augmentation ici
    signal = scale_waveform_data(signal)
    signal = trim_silence(signal)
    spectogram = compute_spectogram(signal, sr)
    buf = convert_to_spectogram_image(spectogram)

    # Les length, width, channel doivent ê cohérents ac les inputs du modèle choisi :
    #length = 64
    #width = 64
    #channel = "RGB"

    image_in_byte = resize_image(buf, length, width, channel)
    img = Image.open(image_in_byte).convert(channel)
    img_array = np.array(img)
    if channel == 'L':
        img_array = np.expand_dims(img_array, axis=2)

    # X and y
    X = np.expand_dims(img_array, axis=0)
    emotion_code = emotion(my_file.filename)
    #y_true = to_categorical([emotion_code - 1], num_classes=8)
    y_pred = app.state.model.predict(X)

    # Answer
    max_indices = np.argmax(y_pred, axis=1)
    emotion_code_pred = max_indices[0] + 1
    emotion_pred_written = decodeur_emotion(emotion_code_pred)
    # emotion_true_written = decodeur_emotion(emotion_code)

    # return {
    #     "emotion": emotion_pred_written,
    #     "true_emotion": emotion_true_written
    # }
    return {
        "emotion": emotion_pred_written
    }
