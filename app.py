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
    origin_location = loc.geocode(origin)
    if origin_location is None:
        raise AttributeError(f"Origin '{origin}' could not be found.")
    destination_location = loc.geocode(destination)
    if destination_location is None:
        raise AttributeError(f"Destination '{destination}' could not be found.")
    o_coords = (origin_location.latitude, origin_location.longitude)
    d_coords = (destination_location.latitude, destination_location.longitude)
    return round((geodesic(o_coords, d_coords).km), 3)


@app.errorhandler(400)
def handle_400_error(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        weight = float(data['weight'])
        origin = data['origin']
        destination = data['destination']
        distance = calculate_distance(origin, destination)
        loading_d = 0.000195  # data['loading_meters']
        features = np.array([[weight, loading_d, distance]])
        results = model.predict(features)
        return jsonify({'predictions': round(float(results[0]), 3)})
    except (ValueError, AttributeError) as e:
        return handle_400_error(e)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
