"""
Prédiction du chiffre sur une image à l'aide du réseau de neurones entraîné.
Usage : python predict_digit.py <chemin_image.png>
        python predict_digit.py <dossier_images>
"""
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_chiffres.joblib"
IMG_SIZE = 28


def charger_image(path):
    """Charge une image, redimensionne en 28x28, normalise, aplatie."""
    img = Image.open(path).convert("L")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float64) / 255.0
    return arr.ravel().reshape(1, -1)


def predicteur(chemin):
    """Retourne le chiffre prédit (0-9) et les probabilités."""
    model = joblib.load(MODEL_PATH)
    X = charger_image(chemin)
    proba = model.predict_proba(X)[0]
    chiffre = int(model.predict(X)[0])
    return chiffre, proba


def main():
    if len(sys.argv) < 2:
        print("Usage : python predict_digit.py <image.png> ou <dossier>")
        sys.exit(1)

    cible = Path(sys.argv[1])
    if not MODEL_PATH.exists():
        print(f"Modèle introuvable : {MODEL_PATH}")
        print("Lancez d'abord : python train_cnn.py")
        sys.exit(1)

    if cible.is_file():
        chemins = [cible]
    elif cible.is_dir():
        chemins = sorted(cible.rglob("*.png"))[:50]
        if not chemins:
            print("Aucune image .png dans le dossier.")
            sys.exit(1)
    else:
        print("Fichier ou dossier introuvable.")
        sys.exit(1)

    print(f"Réseau de neurones chargé depuis {MODEL_PATH}\n")
    for p in chemins:
        try:
            chiffre, proba = predicteur(p)
            conf = proba[chiffre]
            print(f"  {p.name} -> chiffre prédit : {chiffre} (confiance : {conf:.2%})")
        except Exception as e:
            print(f"  {p.name} -> erreur : {e}")


if __name__ == "__main__":
    main()
