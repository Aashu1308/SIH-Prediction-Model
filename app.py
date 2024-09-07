from flask import Flask, request, jsonify
import pickle
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

app = Flask(__name__)

with open('weights.pkl', 'rb') as mfile:
    model = pickle.load(mfile)


def calculate_distance(origin, destination):
    loc = Nominatim(user_agent="GetLoc")
    origin = loc.geocode(origin)
    o_coords = (origin.latitude, origin.longitude)
    destination = loc.geocode(destination)
    d_coords = (destination.latitude, destination.longitude)
    return round((geodesic(o_coords, d_coords).km), 3)


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    print("Received data:", data)
    weight = float(data['weight'])
    origin = data['origin']
    destination = data['destination']
    distance = calculate_distance(origin, destination)
    loading_d = 0.000195  # data['loading_meters']
    features = np.array([[weight, loading_d, distance]])
    results = model.predict(features)
    return jsonify({'predictions': float(results[0])})


if __name__ == '__main__':
    app.run(debug=True, port=5001)
