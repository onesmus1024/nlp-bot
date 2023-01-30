from operator import mod
import model
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

model.get_data()
# model.create_model()
# model.plot_graphs()
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    if data['question'] == 'traiN':
        model.create_model()
        return jsonify({'prediction': 'Model trained successfully'})
    prediction = model.predict(data['question'])
    return jsonify({'prediction': prediction})