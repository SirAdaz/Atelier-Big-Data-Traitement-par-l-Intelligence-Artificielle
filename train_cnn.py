"""
Entraînement d'un réseau de neurones (MLP) pour la reconnaissance
des chiffres et des lettres à partir de archive + Handwritten letters.
Utilise scikit-learn pour éviter les problèmes de DLL TensorFlow sur Windows.
"""
import json
import os
from typing import cast

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Chemins
BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive" / "numbers"
CSV_PATH = BASE_DIR / "archive" / "numbers.csv"
LETTERS_DIR = BASE_DIR / "Handwritten letters.v1i.folder" / "train"
MODEL_PATH = BASE_DIR / "model_chiffres_lettres.joblib"
CLASSES_PATH = BASE_DIR / "model_classes.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"
CONFUSION_MATRIX_CHIFFRES_PATH = BASE_DIR / "confusion_matrix_chiffres.png"
CONFUSION_MATRIX_LETTRES_PATH = BASE_DIR / "confusion_matrix_lettres.png"

# Ordre des classes : 0-9, A-Z, a-z (62 classes)
CLASS_NAMES = (
    [str(i) for i in range(10)]
    + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
)
LABEL_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

# Normaliser les noms de dossiers lettres (aa -> a, bb -> b, ...)
def _normalize_folder(name: str) -> str:
    if len(name) == 2 and name[0].lower() == name[1].lower():
        return name[0].lower()
    return name


# Hyperparamètres
IMG_SIZE = 28
VALIDATION_SPLIT = 0.20
MAX_SAMPLES_ARCHIVE = 15000
MAX_SAMPLES_LETTERS = 15000


def charger_image(path):
    """Charge une image (PNG ou JPG), redimensionne en 28x28, normalise."""
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.ravel()


def charger_donnees_archive():
    """Charge les chiffres depuis archive/numbers.csv."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=["path", "label_id"])
    df = pd.read_csv(CSV_PATH)
    df["path"] = df["file"].apply(lambda f: ARCHIVE_DIR / f)
    df = df[df["path"].apply(os.path.exists)]
    if MAX_SAMPLES_ARCHIVE:
        df = df.sample(n=min(MAX_SAMPLES_ARCHIVE, len(df)), random_state=42)
    df["label_id"] = df["label"].astype(int)
    result = cast(pd.DataFrame, df[["path", "label_id"]].copy())
    return result.reset_index(drop=True)


def charger_donnees_lettres():
    """Charge les lettres (et chiffres) depuis Handwritten letters.v1i.folder/train."""
    if not LETTERS_DIR.exists():
        return pd.DataFrame(columns=["path", "label_id"])
    rows = []
    for folder in sorted(LETTERS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        raw_label = folder.name
        norm = _normalize_folder(raw_label)
        if norm not in LABEL_TO_ID:
            continue
        label_id = LABEL_TO_ID[norm]
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for f in folder.glob(ext):
                rows.append({"path": f, "label_id": label_id})
    if not rows:
        return pd.DataFrame(columns=["path", "label_id"])
    df = pd.DataFrame(rows)
    if MAX_SAMPLES_LETTERS and len(df) > MAX_SAMPLES_LETTERS:
        df = df.sample(n=MAX_SAMPLES_LETTERS, random_state=42)
    return df.reset_index(drop=True)


def charger_et_preparer_donnees():
    """Fusionne archive (chiffres) et lettres."""
    df_arch = charger_donnees_archive()
    df_lettres = charger_donnees_lettres()
    if df_arch.empty and df_lettres.empty:
        raise FileNotFoundError("Aucune donnée trouvée (archive ou Handwritten letters).")
    df = pd.concat([df_arch, df_lettres], ignore_index=True)
    return df


def charger_echantillons(df):
    """Charge toutes les images et labels en mémoire."""
    n = len(df)
    x = np.zeros((n, IMG_SIZE * IMG_SIZE), dtype=np.float64)
    y = df["label_id"].values.astype(np.int32)
    for i in range(n):
        if (i + 1) % 5000 == 0:
            print(f"  Chargement : {i + 1}/{n}")
        x[i] = charger_image(df.iloc[i]["path"])
    return x, y


def main():
    print("Chargement des métadonnées (archive + lettres)...")
    df = charger_et_preparer_donnees()
    print(f"Nombre d'échantillons : {len(df)}")
    print(f"Classes : {len(CLASS_NAMES)} (0-9, A-Z, a-z)")

    # Chiffres seuls (0-9) -> 45 itérations ; chiffres + lettres (62 classes) -> plus d'itérations
    uniquement_chiffres = df["label_id"].max() < 10
    max_iter = 45 if uniquement_chiffres else 80
    print(f"Mode : {'chiffres seuls' if uniquement_chiffres else 'chiffres + lettres'} (max_iter={max_iter})")

    df_train, df_val = train_test_split(
        df, test_size=VALIDATION_SPLIT, random_state=42, stratify=df["label_id"]
    )
    print(f"Entraînement : {len(df_train)}, Validation : {len(df_val)}")

    print("Chargement des images d'entraînement...")
    x_train, y_train = charger_echantillons(df_train)
    print("Chargement des images de validation...")
    x_val, y_val = charger_echantillons(df_val)

    print("Construction du réseau de neurones (MLP)...")
    model = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,  # type: ignore[arg-type]
        learning_rate="adaptive",
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )

    print("Entraînement en cours...")
    model.fit(x_train, y_train)

    score = model.score(x_val, y_val)
    print(f"Précision sur la validation : {score:.4f}")

    joblib.dump(model, MODEL_PATH)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, ensure_ascii=False)
    print(f"Modèle enregistré : {MODEL_PATH}")
    print(f"Classes enregistrées : {CLASSES_PATH}")

    # Matrices de confusion
    y_pred = model.predict(x_val)

    if uniquement_chiffres:
        # Chiffres seuls : une seule matrice lisible
        _, ax = plt.subplots(figsize=(8, 7))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred, ax=ax, colorbar=True, values_format="d",
            display_labels=CLASS_NAMES[:10],
        )
        ax.set_title("Matrice de confusion – Chiffres (validation)")
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Matrice de confusion enregistrée : {CONFUSION_MATRIX_PATH}")
    else:
        # Chiffres + lettres : une matrice par bloc pour rester lisible
        # 1) Chiffres : vrais chiffres (0-9) vs prédit (0-9 ou "lettre")
        mask_chiffres = y_val < 10
        if np.any(mask_chiffres):
            y_val_c = y_val[mask_chiffres]
            y_pred_c = np.where(y_pred[mask_chiffres] < 10, y_pred[mask_chiffres], 10)
            labels_chiffres = [str(i) for i in range(10)] + ["lettre"]
            _, ax = plt.subplots(figsize=(8, 7))
            ConfusionMatrixDisplay.from_predictions(
                y_val_c, y_pred_c, ax=ax, colorbar=True, values_format="d",
                display_labels=labels_chiffres,
            )
            ax.set_title("Matrice de confusion – Chiffres (validation)\n(col. « lettre » = prédit comme une lettre)")
            plt.tight_layout()
            plt.savefig(CONFUSION_MATRIX_CHIFFRES_PATH, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Matrice confusion chiffres : {CONFUSION_MATRIX_CHIFFRES_PATH}")

        # 2) Lettres : vrais lettres (10-61) vs prédit (10-61)
        mask_lettres = y_val >= 10
        if np.any(mask_lettres):
            y_val_l = y_val[mask_lettres] - 10  # 0-51
            y_pred_l = np.clip(y_pred[mask_lettres], 10, 61) - 10
            labels_lettres = CLASS_NAMES[10:]
            _, ax = plt.subplots(figsize=(14, 12))
            ConfusionMatrixDisplay.from_predictions(
                y_val_l, y_pred_l, ax=ax, colorbar=True, values_format="d",
                display_labels=labels_lettres,
            )
            ax.set_title("Matrice de confusion – Lettres A-Z, a-z (validation)")
            ax.tick_params(axis="both", labelsize=7)
            plt.tight_layout()
            plt.savefig(CONFUSION_MATRIX_LETTRES_PATH, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Matrice confusion lettres : {CONFUSION_MATRIX_LETTRES_PATH}")

        # 3) Matrice globale 62x62 (compacte, pour référence)
        _, ax = plt.subplots(figsize=(22, 20))
        ConfusionMatrixDisplay.from_predictions(
            y_val, y_pred, ax=ax, colorbar=True, values_format="d",
            display_labels=CLASS_NAMES,
        )
        ax.set_title("Matrice de confusion – Toutes classes (validation)")
        ax.tick_params(axis="both", labelsize=5)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"Matrice de confusion globale : {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
