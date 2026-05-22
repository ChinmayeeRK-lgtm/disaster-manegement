import os
from pathlib import Path

from flask import Flask, request, jsonify
from flask_cors import CORS

import torch
import joblib

from PIL import Image

from torchvision import transforms, models

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

mobilenet = models.mobilenet_v2()

mobilenet.classifier[1] = torch.nn.Linear(
    1280,
    5
)

mobilenet.load_state_dict(
    torch.load(MODEL_DIR / "mobilenet.pth", map_location="cpu")
)

mobilenet.eval()

classes = [
    "earthquake_major",
    "earthquake_minor",
    "flood_high",
    "flood_low",
    "flood_medium"
]

yolo_model = YOLO(
    str(MODEL_DIR / "yolo_best.pt")
)

rf_model = joblib.load(
    MODEL_DIR / "random_forest.pkl"
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict-flood", methods=["POST"])
def predict_flood():

    return jsonify({

        "risk": "High",

        "damage": "Severe",

        "priority": "Immediate"

    })

@app.route("/predict", methods=["POST"])

def predict():

    if "image" not in request.files:
        return jsonify({"error": "image is required"}), 400

    if "water_level" not in request.form:
        return jsonify({"error": "water_level is required"}), 400

    file = request.files["image"]

    image = Image.open(file.stream).convert("RGB")

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        output = mobilenet(img)

        predicted = torch.argmax(output).item()

    disaster_class = classes[predicted]

    yolo_results = yolo_model.predict(
        image
    )

    detected_objects = []

    for result in yolo_results:

        for box in result.boxes:

            cls = int(box.cls)

            detected_objects.append(
                result.names[cls]
            )

    try:
        water_level = int(
            request.form["water_level"]
        )
    except ValueError:
        return jsonify({"error": "water_level must be a number"}), 400

    rf_input = [[
        0,
        water_level,
        90,
        200,
        1,
        95
    ]]

    rf_prediction = rf_model.predict(
        rf_input
    )

    return jsonify({

        "disaster_class": disaster_class,

        "detected_objects":
            detected_objects,

        "risk_prediction":
            int(rf_prediction[0])

    })

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
