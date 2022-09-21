import model
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

model.get_data()
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(data['question'])
    return jsonify({'prediction': prediction})