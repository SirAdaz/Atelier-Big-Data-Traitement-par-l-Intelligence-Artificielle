import os
import glob
from PIL import Image, ImageDraw, ImageFont

# --- Paramètres ---
BASE_OUTPUT_DIR = "./images_lettres"   # Changer le chemin du Dossier racine
IMG_SIZE = (80, 80)                  # (largeur, hauteur) en pixels
BACKGROUND_COLOR = (255, 255, 255)   # Blanc
TEXT_COLOR = (0, 0, 0)               # Noir

LETTERS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]       # Lettres à générer
NUM_IMAGES = 2                      # Nombre d'images par lettre/police
N_FONTS = 200                         # Nombre max de polices différentes

# Dossier des polices Windows
WINDOWS_FONTS_DIR = r"C:\Windows\Fonts"

def load_system_fonts(max_fonts=10, size=50):
    font_files = glob.glob(os.path.join(WINDOWS_FONTS_DIR, "*.ttf"))
    fonts = []

    for path in font_files[:max_fonts]:
        try:
            fonts.append(ImageFont.truetype(path, size=size))
        except Exception:
            # On ignore les polices qui posent problème
            continue

    return fonts

def main():
    # 1. Créer le dossier racine
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # 2. Charger automatiquement les polices système
    fonts = load_system_fonts(max_fonts=N_FONTS, size=50)

    # Si aucune police trouvée, utiliser la police par défaut
    if not fonts:
        print("Aucune police chargée, utilisation de la police par défaut.")
        fonts.append(ImageFont.load_default())

    # 3. Générer les images
    for letter in LETTERS:
        # On met A et a dans le même dossier "A", B et b dans "B", etc.
        letter_folder_name = letter.upper()
        letter_output_dir = os.path.join(BASE_OUTPUT_DIR, letter_folder_name)

        # Créer le sous-dossier pour cette lettre
        os.makedirs(letter_output_dir, exist_ok=True)

        for idx, font in enumerate(fonts):
            for n in range(NUM_IMAGES):
                img = Image.new("RGB", IMG_SIZE, BACKGROUND_COLOR)
                draw = ImageDraw.Draw(img)

                # Calcul de la taille du texte (pour centrage)
                bbox = draw.textbbox((0, 0), letter, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                x = (IMG_SIZE[0] - text_width) // 2
                y = (IMG_SIZE[1] - text_height) // 2

                draw.text((x, y), letter, fill=TEXT_COLOR, font=font)

                # Nom de fichier, ex : A_font0_0.png, a_font0_1.png, ...
                filename = f"{letter}_font{idx}_{n}.png"
                filepath = os.path.join(letter_output_dir, filename)
                img.save(filepath)

                print(f"Image sauvegardée : {filepath}")

if __name__ == "__main__":
    main()