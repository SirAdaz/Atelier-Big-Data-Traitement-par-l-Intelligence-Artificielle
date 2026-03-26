import os
import random
from PIL import Image
import tempfile, shutil

"""
numbers  -> category (chars74k_png/mnist_png) -> typeOfImg (BadImage/Fnt/GoodImg/Hnd) -> whichNumber (SampleX) -> img
train    -> whichSymbol (letter or nb) -> img
"""

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
NUMBERS_CSV = os.path.join(os.getcwd(), "numbers.csv")

def get_mask_images():
    return len(set(f for f in os.listdir(os.path.join(os.getcwd(), "static")) if f.startswith("noise_mask")))

def generate_noise_image(intensity: float):
    mask = Image.effect_noise((128,128), intensity).convert("RGB").save(os.path.join(os.getcwd(),"static/noise_mask.png"),"PNG")

def create_noise(intensity: float, size: tuple, mode: str) -> Image:
    return Image.effect_noise(size, intensity).convert(mode)


def noisyfy_image(path: str, file: str, intensity: float, opacity: float, mask_image) -> str:
    """
    Blends an image with noise and saves it in the same folder with a "noised_" prefix.
    Returns the new filename.
    """
    with Image.open(os.path.join(path, file)) as img:
        img = img.convert("RGB")

        if mask_image:
            noise = Image.open(os.path.join(os.getcwd(), "static", "noise_mask.png"))
        else :
            noise = create_noise(intensity, img.size, img.mode)
        noise = noise.resize(img.size).convert("RGB")
        noised = Image.blend(img, noise, opacity)
        name = "noised_" + os.path.splitext(file)[0] + ".png"
        noised.save(os.path.join(path, name), "PNG")
    return name


def is_valid_image(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in VALID_EXTENSIONS


def walk_numbers(base: str):
    """Yields (origin, group, label, label_path, img) for each valid non-noised image under numbers/."""
    numbers_path = os.path.join(base, "numbers")
    for origin in os.listdir(numbers_path):
        origin_path = os.path.join(numbers_path, origin)
        if not os.path.isdir(origin_path):
            continue
        print(origin)
        for group in os.listdir(origin_path):
            group_path = os.path.join(origin_path, group)
            if not os.path.isdir(group_path):
                continue
            print(group)
            for label in os.listdir(group_path):
                label_path = os.path.join(group_path, label)
                if not os.path.isdir(label_path):
                    continue
                print(label)
                for img in os.listdir(label_path):
                    yield origin, group, label, label_path, img


def walk_train(base: str):
    """Yields (sample_path, img) for each valid non-noised image under train/."""
    train_path = os.path.join(base, "train")
    for sample in os.listdir(train_path):
        sample_path = os.path.join(train_path, sample)
        if not os.path.isdir(sample_path):
            continue
        print(sample)
        for img in os.listdir(sample_path):
            yield sample_path, img


def make_some_noise(selected: set = None):
    print("GENERATING noise for folders :", selected)
    base = os.getcwd()
    new_lines = []
    mask_image = get_mask_images()

    for origin, group, label, label_path, img in walk_numbers(base):
        full = origin+"/"+group+"/"+label
        if selected and full not in selected:
            continue
        if img.startswith("noised_") or not is_valid_image(img):
            continue
        if random.random() >= 0.5:
            name = noisyfy_image(label_path, img, random.randint(10, 90), 0.45, mask_image)
            relative_path = os.path.relpath(os.path.join(label_path, name), base)
            new_lines.append(f"{origin},{group},{label},{relative_path}\n")

    with open(NUMBERS_CSV, "r") as f:
        lines = f.readlines()
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    if new_lines:
        with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
            tmp.writelines(line for line in lines if "noised_" not in line)
            tmp.writelines(new_lines)
        shutil.move(tmp.name, NUMBERS_CSV)

    for sample_path, img in walk_train(base):
        folder_name = os.path.basename(sample_path)
        if selected and folder_name not in selected:
            continue
        if img.startswith("noised_") or not is_valid_image(img):
            continue
        if random.random() >= 0.5:
            noisyfy_image(sample_path, img, random.randint(10, 90), 0.45, mask_image)

def silence(mode: str):

    if mode == "numbers":
        return clean_numbers_folder()
    elif mode == "csv":
        return clean_csv()
    elif mode == "train":
        return clean_train_folder()

def clean_numbers_folder():
    print("CLEANING NUMBERS FOLDER")
    base = os.getcwd()
    removed = 0
    for origin, group, label, label_path, img in walk_numbers(base):
        if not img.startswith("noised_"):
            continue
        full_path = os.path.join(label_path, img)
        os.remove(full_path)
        removed += 1
    return removed

def clean_train_folder():
    print("CLEANING TRAIN FOLDER")
    base = os.getcwd()
    removed = 0
    for sample_path, img in walk_train(base):
        if img.startswith("noised_"):
            os.remove(os.path.join(sample_path, img))
            removed += 1
    return removed

def clean_csv():
    print("CLEANING NUMBERS.CSV")

    with open(NUMBERS_CSV, "r") as f:
        lines = f.readlines()
    with tempfile.NamedTemporaryFile("w", delete=False) as tmp:
        kept = [line for line in lines if "noised_" not in line]
        removed = len(lines) - len(kept)
        tmp.writelines(kept)
    shutil.move(tmp.name, NUMBERS_CSV)
    return removed