import os.path
from flask import Flask, render_template, request, jsonify
import base64
import io
import os
from PIL import Image
import predict_digit
import threading
import json
import joblib

import image_handler
import train_cnn
from data_type.history_class import History

# flask --app interface run --debug
app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():

    train_history = []
    with open("dashboard_data/precision_history.csv", "r") as file :
        file.readline() # skip the header
        line = file.readline().strip()
        while line:
            train_history.append(History(line))
            line = file.readline()
        latest = train_history[-1] if train_history else None
        latest_diff = round(float(latest.precision) - float((train_history[-2].precision if train_history else 0)),2)

    path = os.path.join(os.getcwd(), "static")
    mask_images = [f for f in os.listdir(path) if f.startswith("noise_mask")]

    return render_template('dashboard.html', history=train_history, latest=latest, latest_diff=latest_diff, mask_images=mask_images)

@app.route("/regenerate-mask", methods=["POST"])
def regen_mask():
    intensity = request.json.get("intensity")

    image_handler.generate_noise_image(float(intensity))

    # find updated mask files
    path = os.path.join(os.getcwd(), "static")
    mask_images = [f for f in os.listdir(path) if f.startswith("noise_mask")]

    return jsonify({
        "success": True,
        "images": mask_images
    })

@app.route("/make_some_noise", methods=["POST"])
def msn():
    try:
        data = request.json or {}
        selected = data.get("selected", None)  # None means all folders
        image_handler.make_some_noise(selected)
        return jsonify({
            "success": True,
            "message": "Noisy images have been created."
        })
    except Exception as e:
        app.logger.error(e)
        return jsonify({"success": False, "message": str(e)})

@app.route("/silence", methods=["POST"])
def shush():
    mode = ""
    try:
        mode = request.json.get("mode")
        print(mode)
        nb = image_handler.silence(mode)
        return jsonify({
            "success": True,
            "message": mode+" has been cleaned of "+str(nb)+" lines"
        })
    except Exception as e:
        app.logger.error(e, exc_info=True)
        return jsonify({"success": False, "message": "Something went wrong while cleaning "+mode+" : " + str(e)})

@app.route("/get-folders", methods=["GET"])
def get_folders():
    base = os.getcwd()
    numbers = {}
    train = []

    numbers_path = os.path.join(base, "numbers")
    for origin in os.listdir(numbers_path):
        origin_path = os.path.join(numbers_path, origin)
        if not os.path.isdir(origin_path):
            continue
        for group in os.listdir(origin_path):
            group_path = os.path.join(origin_path, group)
            if not os.path.isdir(group_path):
                continue
            key = f"{origin}/{group}"
            numbers[key] = []
            for label in os.listdir(group_path):
                if os.path.isdir(os.path.join(group_path, label)):
                    numbers[key].append(label)

    train_path = os.path.join(base, "train")
    for sample in os.listdir(train_path):
        if os.path.isdir(os.path.join(train_path, sample)):
            train.append(sample)

    return jsonify({"numbers": numbers, "train": train})

@app.route("/train", methods=["POST"])
def train_model():
    def run():
        try:
            train_cnn.main()
        except Exception as e:
            app.logger.error(e, exc_info=True)

    thread = threading.Thread(target=run)
    thread.daemon = True
    thread.start()
    return jsonify({"success": True, "message": "Training started"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        image_b64 = data.get("image")
        filename = data.get("filename", "fichier_prediction.png")

        # Charger le modèle et les classes
        model_path = os.path.join(os.getcwd(), "model_chiffres_lettres.joblib")
        classes_path = os.path.join(os.getcwd(), "model_classes.json")

        if not os.path.exists(model_path):
            return jsonify({"success": False, "message": "No trained model found, please train first."})

        model = joblib.load(model_path)
        with open(classes_path, encoding="utf-8") as f:
            class_names = json.load(f)

        # Décoder et sauvegarder
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        ext = os.path.splitext(filename)[1] or ".png"
        save_path = os.path.join(os.getcwd(), "fichier_prediction" + ext)
        img.save(save_path)

        # Appliquer le masque de bruit
        mask_path = os.path.join(os.getcwd(), "static", "noise_mask.png")
        if not os.path.exists(mask_path):
            image_handler.generate_noise_image(50.0)
        noise = Image.open(mask_path).resize(img.size).convert("RGB")
        blended = Image.blend(img.convert("RGB"), noise, 0.45)
        blended.save(save_path)

        # Prédiction
        caractere, proba, idx = predict_digit.predicteur(save_path, model, class_names)

        return jsonify({
            "success": True,
            "prediction": caractere,
            "confidence": float(proba[idx])
        })

    except Exception as e:
        app.logger.error(e, exc_info=True)
        return jsonify({"success": False, "message": str(e)})
