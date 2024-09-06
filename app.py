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
    features = np.array(data['features']).reshape(1, -1)
    results = model.predict(features)
    return jsonify({'predictions': results.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
