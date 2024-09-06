from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('weights.pkl', 'rb') as mfile:
    model = pickle.load(mfile)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    weight = data['weight']
    distance = data['distance']
    loading_d = data['loading_meters']
    features = np.array([[weight, loading_d, distance]])
    results = model.predict(features)
    return jsonify({'predictions': float(results[0])})


if __name__ == '__main__':
    app.run(debug=True)
