from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS
app = Flask(__name__,static_folder="../frontend", template_folder="../frontend")

CORS(app)
# 🔹 Load model + tokenizer
model = tf.keras.models.load_model("review.keras")
with open("tokenizer.pkl", "rb") as f:
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

if __name__ == "__main__":
    app.run(debug=True)
