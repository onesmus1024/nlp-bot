import model
from flask import Flask, request, jsonify
import tensorflow as tf

model.get_data()
app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(data['question'])
    return jsonify({'prediction': prediction})