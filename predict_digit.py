"""
Prédiction du chiffre ou de la lettre sur une image avec le réseau de neurones entraîné.
Usage : python predict_digit.py <chemin_image.png>
        python predict_digit.py <dossier_images>
"""
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_chiffres_lettres.joblib"
CLASSES_PATH = BASE_DIR / "model_classes.json"
IMG_SIZE = 28


def charger_image(path):
    """Charge une image (PNG ou JPG), redimensionne en 28x28, normalise, aplatie."""
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.ravel().reshape(1, -1)


def predicteur(chemin, model, class_names):
    """Retourne le caractère prédit (0-9, A-Z, a-z), les probabilités et l'index."""
    x = charger_image(chemin)
    proba = model.predict_proba(x)[0]
    idx = int(model.predict(x)[0])
    caractere = class_names[idx] if idx < len(class_names) else str(idx)
    return caractere, proba, idx


def main():
    if len(sys.argv) < 2:
        print("Usage : python predict_digit.py <image.png|.jpg> ou <dossier>")
        sys.exit(1)

    cible = Path(sys.argv[1])
    if not MODEL_PATH.exists():
        print(f"Modèle introuvable : {MODEL_PATH}")
        print("Lancez d'abord : python train_cnn.py")
        sys.exit(1)
    if not CLASSES_PATH.exists():
        print(f"Fichier des classes introuvable : {CLASSES_PATH}")
        sys.exit(1)

    with open(CLASSES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = joblib.load(MODEL_PATH)

    if cible.is_file():
        chemins = [cible]
    elif cible.is_dir():
        chemins = []
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            chemins.extend(cible.rglob(ext))
        chemins = sorted(set(chemins))[:50]
        if not chemins:
            print("Aucune image .png ou .jpg dans le dossier.")
            sys.exit(1)
    else:
        print("Fichier ou dossier introuvable.")
        sys.exit(1)

    print(f"Réseau de neurones chargé depuis {MODEL_PATH}\n")
    for p in chemins:
        try:
            caractere, proba, idx = predicteur(p, model, class_names)
            conf = proba[idx]
            print(f"  {p.name} -> prédit : '{caractere}' (confiance : {conf:.2%})")
        except Exception as e:
            print(f"  {p.name} -> erreur : {e}")


if __name__ == "__main__":
    main()
