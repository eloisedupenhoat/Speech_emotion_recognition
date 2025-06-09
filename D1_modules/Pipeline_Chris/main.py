# main.py

from preprocessing.preprocess_pipeline import build_dataset
from evaluation.evaluate_model import evaluate
from training.train_model import train


if __name__ == "__main__":
    # Construire les datasets (train, val, test) et les uploader sur GCS
    #build_dataset("train")
    #build_dataset("val")
    #build_dataset("test")

    # Lancer l’entraînement du modèle
    train()
    #evaluate()
