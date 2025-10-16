from flask import Flask, request, jsonify
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

# Cargar el modelo entrenado
model = keras.models.load_model("modelo.h5")

# Clases en el mismo orden del entrenamiento
class_names = ['with_mask', 'mask_weared_incorrect', 'without_mask']

app = Flask(__name__)

@app.route("/")
def home():
    return "Hola"

@app.route("/predict", methods=["POST"])
def predict():
    # Verificar que se envió una imagen
    if "file" not in request.files:
        return jsonify({"error": "No se envió ninguna imagen"}), 400

    file = request.files["file"]
    img = Image.open(io.BytesIO(file.read())).resize((300, 300))

    # Convertir a array y normalizar
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Predicción
    prediction = model.predict(x)
    predicted_class = np.argmax(prediction, axis=1)[0]
    label = class_names[predicted_class]

    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)