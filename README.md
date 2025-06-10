# 🎤 Speech Emotion Recognition

Projet de prédiction des émotions humaines à partir de la voix.  
Réalisé dans le cadre d’une formation IA / Data Science.

---

## 🧠 Objectif du projet

L'objectif est de détecter l’émotion exprimée par un locuteur à partir d’un enregistrement audio.  
Le pipeline inclut : prétraitement, extraction de features audio, modélisation, interface Streamlit.

---

## 🚀 Fonctionnalités

- 🔊 Extraction automatique des caractéristiques audio (MFCC, spectrogrammes, etc.)
- 🧪 Modèles entraînés avec Keras & PyTorch (classification)
- 📊 Analyse de performance des modèles
- 🖥️ Interface utilisateur via Streamlit pour tester une prédiction en live
- 🐳 Intégration Docker pour déploiement reproductible

---

## 🛠️ Stack technique

- **Langages** : Python
- **Librairies principales** :
  - Audio : `librosa`, `soundfile`
  - Machine learning : `scikit-learn`, `keras`, `torch`
  - Interface : `streamlit`
  - Utilitaires : `numpy`, `pandas`, `matplotlib`, `seaborn`


---


## 📦 Installation

Cloner le repo et installer les dépendances :

`git clone https://github.com/ton-user/Speech_emotion_recognition.git` 
`cd Speech_emotion_recognition`
`pip install -r requirements.txt`

_ _  _

## 📁 Structure du projet


`.`
`├── D1_modules/         # Traitement, extraction de features, modèles`
`├── D2_interface/       # Interface utilisateur (Streamlit)`
`├── hooks/              # Pre-commit et outils dev`
`├── params.py           # Paramètres globaux du projet`
`├── requirements.txt    # Dépendances pour dev`
`├── requirements_prod.txt`
`├── Dockerfile          # Image Docker pour déploiement`
`── install-hooks.sh    # Script pour activer les hooks Git`
`└── README.md           # Ce fichier`



## 👥 Auteurs

Projet réalisé par :

eloisedupenhoat

gaspardasilveira

Chrisgrd

FRmathieu13


## 📄 Licence
Ce projet est sous licence MIT.




