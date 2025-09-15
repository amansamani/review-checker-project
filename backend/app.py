import os
# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Disable oneDNN logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask_cors import CORS

app = Flask(__name__, static_folder="../frontend", template_folder="../frontend")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
model_path = os.path.join(BASE_DIR, "review.keras")
tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")

# Lazy load model
model = None  

# Load tokenizer at startup
with open(tokenizer_path, "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        print("Loading model for the first time... ✅")
        model = tf.keras.models.load_model(model_path)

    data = request.get_json()
    review = data.get("review", "")

    # Convert review to sequence
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=MAXLEN)

    # Model prediction
    score = model.predict(padded)[0][0]

    # Sentiment categories with emojis
    if 0.8 <= score <= 1.0:
        sentiment = "Excellent Review 😍"
    elif 0.6 <= score < 0.8:
        sentiment = "Good Review 🙂"
    elif 0.4 <= score < 0.6:
        sentiment = "Not So Good Review 😐"
    elif 0.2 <= score < 0.4:
        sentiment = "Bad Review 😕"
    else:
        sentiment = "Worst Review 😡"

    return jsonify({
        "review": review,
        "score": float(score),
        "sentiment": sentiment
    })

# Do NOT include app.run() on Render
# Render will use: gunicorn app:app --bind 0.0.0.0:$PORT
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)