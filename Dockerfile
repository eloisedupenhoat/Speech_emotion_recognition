#directive pour créer une image à partir d'une image déjà existante
FROM python:3.12-slim

WORKDIR /prod

#directive pour copier mon code vers le container
COPY D1_modules/ D1_modules/
COPY D2_interface/ D2_interface/
COPY requirements_prod.txt requirements.txt

#instalation des package dans mon container
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY params.py params.py
COPY models/ models/


#directive COMMAND : creation d'un serveur Uvicorn
EXPOSE $PORT
CMD uvicorn D2_interface.d_api:app --reload --host 0.0.0.0 --port $PORT
#la variable port est celle du .env en local, alors que c'est google qui va la definir dans le cloud
