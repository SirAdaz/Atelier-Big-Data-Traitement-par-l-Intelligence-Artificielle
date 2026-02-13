"""
Entraînement d'un réseau de neurones (MLP) pour la reconnaissance
des chiffres à partir du dossier archive (numbers.csv + images).
Utilise scikit-learn pour éviter les problèmes de DLL TensorFlow sur Windows.
"""
import os
from typing import cast

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Chemins
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive" / "numbers"
CSV_PATH = BASE_DIR / "archive" / "numbers.csv"
MODEL_PATH = BASE_DIR / "model_chiffres.joblib"

# Hyperparamètres
IMG_SIZE = 28
VALIDATION_SPLIT = 0.20
MAX_SAMPLES = None  # Limite pour la mémoire (None = tout utiliser)


def charger_et_preparer_donnees():
    """Charge le CSV et prépare les chemins des images et labels."""
    df = pd.read_csv(CSV_PATH)
    df["path"] = df["file"].apply(lambda f: ARCHIVE_DIR / f)
    df = df[df["path"].apply(os.path.exists)]
    if MAX_SAMPLES:
        df = df.sample(n=min(MAX_SAMPLES, len(df)), random_state=42)
    result = cast(pd.DataFrame, df[["path", "label"]].copy())
    return result.reset_index(drop=True)


def charger_image(path):
    """Charge une image, la redimensionne en 28x28 et la normalise."""
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.ravel()  # 784 valeurs pour le MLP


def charger_echantillons(df):
    """Charge toutes les images et labels en mémoire (pour scikit-learn)."""
    n = len(df)
    x = np.zeros((n, IMG_SIZE * IMG_SIZE), dtype=np.float64)
    y = df["label"].values.astype(np.int32)
    for i in range(n):
        if (i + 1) % 5000 == 0:
            print(f"  Chargement : {i + 1}/{n}")
        x[i] = charger_image(df.iloc[i]["path"])
    return x, y


def main():
    print("Chargement des métadonnées depuis", CSV_PATH)
    df = charger_et_preparer_donnees()
    print(f"Nombre d'échantillons : {len(df)}")

    df_train, df_val = train_test_split(
        df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df["label"]
    )
    print(f"Entraînement : {len(df_train)}, Validation : {len(df_val)}")

    print("Chargement des images d'entraînement...")
    x_train, y_train = charger_echantillons(df_train)
    print("Chargement des images de validation...")
    x_val, y_val = charger_echantillons(df_val)

    print("Construction du réseau de neurones (MLP)...")
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate="adaptive",
        max_iter=50,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    print("Entraînement en cours...")
    model.fit(x_train, y_train)

    score = model.score(x_val, y_val)
    print(f"Précision sur la validation : {score:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Modèle enregistré : {MODEL_PATH}")


if __name__ == "__main__":
    main()
