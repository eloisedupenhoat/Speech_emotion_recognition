import numpy as np
from tensorflow.keras.layers import Input, Masking, Conv1D, BatchNormalization, Activation, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import class_weight

from data.gcs_handler import download_npy_array

MAX_LEN = 200
FEATURE_DIM = 137
NUM_CLASSES = 8
SEED = 42

BEST_PARAMS = {
    "n_conv_blocks": 2,
    "conv_filters_block1": 64,
    "conv_filters_block2": 128,
    "conv_kernel_size": 4,
    "lstm_units": 64,
    "dropout_rate": 0.2,
    "l2_reg": 0.000801,
    "learning_rate": 0.00056,
    "batch_size": 16
}



def train():
    print("[INFO] Chargement des données GCS...")
    X_train = download_npy_array("Features/X_train.npy")
    y_train = download_npy_array("Features/y_train.npy")
    X_val   = download_npy_array("Features/X_val.npy")
    y_val   = download_npy_array("Features/y_val.npy")

    print(f"Train: {X_train.shape}, {y_train.shape} | Val: {X_val.shape}, {y_val.shape}")

    classes = np.unique(y_train)
    weights = class_weight.compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    print(f"[INFO] class_weight = {class_weight_dict}")

    model = build_final_model(BEST_PARAMS)
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    lr_reduce  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=BEST_PARAMS["batch_size"],
        callbacks=[early_stop, lr_reduce],
        class_weight=class_weight_dict,
        verbose=2
    )

    model.save("best_model_optuna.h5")
    print("[INFO] Modèle sauvegardé sous best_model_optuna.h5")

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

    # 1. Courbes d'apprentissage (loss et accuracy)
    def plot_learning_curves(history):
        plt.figure(figsize=(14,5))
        # LOSS
        plt.subplot(1,2,1)
        plt.plot(history.history["loss"], label="Train Loss")
        plt.plot(history.history["val_loss"], label="Val Loss")
        plt.title("Courbe de Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        # ACCURACY
        plt.subplot(1,2,2)
        plt.plot(history.history["accuracy"], label="Train Acc")
        plt.plot(history.history["val_accuracy"], label="Val Acc")
        plt.title("Courbe d'Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig("learning_curves.png")
        plt.show()

    plot_learning_curves(history)

    print("\n[INFO] Évaluation sur le split validation...")
    y_val_pred_proba = model.predict(X_val)
    y_val_pred = np.argmax(y_val_pred_proba, axis=1)

    print("[INFO] Classification Report (validation) :")
    print(classification_report(y_val, y_val_pred, digits=4))

    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix (Validation Set)")
    plt.savefig("confusion_matrix_val.png")
    plt.show()

# Ce qui suit est optionnel, mais c'est une bonne pratique :
if __name__ == "__main__":
    train()
