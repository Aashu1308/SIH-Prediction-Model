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
    origin_location = loc.geocode(origin, timeout=10)
    if origin_location is None:
        raise AttributeError(f"Origin '{origin}' could not be found.")
    destination_location = loc.geocode(destination, timeout=10)
    if destination_location is None:
        raise AttributeError(f"Destination '{destination}' could not be found.")
    o_coords = (origin_location.latitude, origin_location.longitude)
    d_coords = (destination_location.latitude, destination_location.longitude)
    return round((geodesic(o_coords, d_coords).km), 3)


@app.errorhandler(400)
def handle_400_error(error):
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(500)
def handle_500_error(error):
    return jsonify({'error': 'Internal Server Error', 'message': str(error)}), 500


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
    except Exception as e:
        return handle_500_error(e)


if __name__ == '__main__':
    app.run()
