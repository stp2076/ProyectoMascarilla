from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io, base64

app = Flask(__name__)

# Carga del modelo
model = tf.keras.models.load_model("model.h5")

# Diccionario de etiquetas
class_names = ['with_mask', 'mask_weared_incorrect', 'without_mask']

@app.route('/')
def home():
    return "Servidor activo ✅"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recibir datos base64
        data = request.data.decode('utf-8')
        image_data = base64.b64decode(data)
        img = Image.open(io.BytesIO(image_data))
        
        # Preprocesar imagen
        img = img.resize((300, 300))  # el tamaño que usaste en el modelo
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # normalización si tu modelo la usa

        # Predicción
        pred = model.predict(x)
        result = np.argmax(pred)
        label = class_names[result]

        return jsonify({'prediction': int(result), 'label': label})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)