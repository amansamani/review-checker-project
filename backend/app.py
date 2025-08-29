from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model and tokenizer
model_path = os.path.join(BASE_DIR, "review.keras")
model = tf.keras.models.load_model(model_path)

tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    threshold = data.get("threshold", 0.6)

    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=MAXLEN)

    score = model.predict(padded)[0][0]
    sentiment = "Positive" if score >= threshold else "Negative"

    return jsonify({
        "review": review,
        "score": float(score),
        "threshold_used": threshold,
        "sentiment": sentiment
    })

# Do NOT include app.run() on Render
# Render will use gunicorn: gunicorn app:app --bind 0.0.0.0:$PORT
