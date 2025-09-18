from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)
model = load_model("model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = Image.open(file).convert("L").resize((28, 28))
    arr = np.array(img) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    pred = model.predict(arr)[0][0]
    result = int(pred > 0.5)
    return jsonify({"prediction": result, "probability": float(pred)})

if __name__ == "__main__":
    app.run(debug=True)