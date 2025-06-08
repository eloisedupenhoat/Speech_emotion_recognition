import pandas as pd
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
from tensorflow import keras # EXPLIQUER
from keras import Model, Sequential # EXPLIQUER
from io import BytesIO
from google.cloud import storage

from D1_modules.b_data import load_model
from D1_modules.a_utils import *
from D1_modules.d_preproc_spec import *
from D1_modules.e_model_baseline import X_value

# J'indique que l'on travaille sur le dossier source du dossier dans lequel on se trouve
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

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
Puis, exécuter : uvicorn D2_interface.d_api.api.fast:app --reload
Pourquoi le faire mainteant ? Car le modèle prend du temps à charger et
qu'on veut faire une démo en live (dc éviter d'attendre que le modèle charge
en public)
'''

model = "Spec-64p-RGB-60db_sample1000_cnn-lr0_0005-bs32-pat2-vs30-50ep.keras"
app.state.model = load_model(model)


###########################################################################
##########            EN LIVE - OUTPUT - PREDICTION              ##########
###########################################################################

'''
En phase de dvt : Postman pour tester, avec <url du terminal>/uploadfile/
Ou encore plus facile : http://127.0.0.1:8000/docs (avec http://127.0.0.1:8000 le code fourni par le terminal)
'''

@app.post('/predict/')
async def predict(my_file: UploadFile):

    # Upload file
    filename = my_file.filename
    contents = await my_file.read()
    with open("temp.wav", "wb") as f:
        f.write(contents)
    file = "temp.wav"

    # Data preprocessing
    X_pred = X_value(file) # Bug ici
    '''
    File "/home/eloisedupenhoat/code/eloisedupenhoat/Speech_emotion_recognition/D1_modules/e_model_baseline.py", line 26, in X_value
    for value in dictionnary.values():
    AttributeError: 'str' object has no attribute 'values'
    '''
    X_pred_processed = convert_to_spectogram_images(X_pred) # Changer et mettre le bon nom de fonction
    y_pred = app.state.model.predict(X_pred_processed)
    emotion_ref = float(y_pred[0][0]) # Voir si je récupère bien ce que je veux
    emotion = decodeur_emotion(emotion_ref)
    return {'Test':emotion_ref,'Your emotion is':emotion}
