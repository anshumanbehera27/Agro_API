from flask import Flask, jsonify, request, session
import pickle 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import os 
import secrets

app = Flask(__name__)

if 'SECRET_KEY' in os.environ:
    app.secret_key = os.environ['SECRET_KEY']
else:
    app.secret_key = secrets.token_hex(16)

# Define the path to the models directory
models_dir = os.path.join(app.root_path, 'Models')

# Load machine learning models
Model_crop_path = os.path.join(models_dir, 'NBClassifier.pkl')
Model_fertilizer_path = os.path.join(models_dir, 'RandomForest.pkl')
Model_location_path = os.path.join(models_dir, 'DecisionTree.pkl')

Model_crop = pickle.load(open(Model_crop_path, 'rb'))
Model_fertilizer = pickle.load(open(Model_fertilizer_path, 'rb'))
Model_location = pickle.load(open(Model_location_path, 'rb'))

# Define encoding functions
def encode_soil_types(original_soil_types, new_soil_type):
    soil_type_label_encoder = LabelEncoder().fit(original_soil_types)
    encoded_soil_type = soil_type_label_encoder.transform([new_soil_type])
    return encoded_soil_type[0]

def encode_crop_types(original_crop_types, new_crop_type):
    crop_type_label_encoder = LabelEncoder().fit(original_crop_types)
    encoded_crop_type = crop_type_label_encoder.transform([new_crop_type])
    return encoded_crop_type[0]

encoded_fertilizers = {
    0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'
}

def get_fertilizer_name(encoded_value):
    return encoded_fertilizers.get(encoded_value, 'Unknown')


def get_main_crop(state_name):
    crops_mapping = {
        "Andhra Pradesh": "Rice",
        "Arunachal Pradesh": "Oranges",
        "Assam": "Tea",
        "Bihar": "Rice",
        "Chhattisgarh": "Rice",
        "Goa": "Coconut",
        "Gujarat": "Cotton",
        "Haryana": "Wheat",
        "Himachal Pradesh": "Apple",
        "Jharkhand": "Rice",
        "Karnataka": "Sugarcane",
        "Kerala": "Rubber",
        "Madhya Pradesh": "Wheat",
        "Maharashtra": "Sugarcane",
        "Manipur": "Rice",
        "Meghalaya": "Maize",
        "Mizoram": "Maize",
        "Nagaland": "Maize",
        "Odisha": "Rice",
        "Punjab": "Wheat",
        "Rajasthan": "Wheat",
        "Sikkim": "Maize",
        "Tamil Nadu": "Rice",
        "Telangana": "Rice",
        "Tripura": "Rice",
        "Uttar Pradesh": "Sugarcane",
        "Uttarakhand": "Rice",
        "West Bengal": "Rice"
    }
    
    state_name = state_name.title()
    
    return crops_mapping.get(state_name, "Crop data not available for this state.")

# Define root route
@app.route('/')
def index():
    return "Welcome to the Bharat Agro API. Use the endpoints to get predictions."

# Define API routes
@app.route('/predictcrop', methods=['POST'])
def predictcrop():
    try:
        data = request.get_json()  # Ensure that data is parsed as JSON
        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['Ph'])
        rainfall = float(data['Rainfall'])

        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        prediction = Model_crop.predict(features)
        crop = prediction[0]

        result = {"message": "{} is the best crop to be cultivated there.".format(crop)}
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": "Missing form field - {}".format(e)}), 400
    except ValueError as e:
        return jsonify({"error": "Invalid form data - {}".format(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predictfertilizer', methods=['POST'])
def predictfertilizer():
    try:
        data = request.get_json()  # Ensure that data is parsed as JSON
        Temperature = float(data['Temperature'])
        Humidity = float(data['Humidity'])
        Soil_Moisture = float(data['SoilMoisture'])
        SoilType = data['soil_type']
        Crop_type = data['crop_type']
        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])

        known_soil_types = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
        known_crop_types = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

        encoded_soil_type = encode_soil_types(known_soil_types, SoilType)
        encoded_crop_type = encode_crop_types(known_crop_types, Crop_type)

        features = np.array([[Temperature, Humidity, Soil_Moisture, encoded_soil_type, encoded_crop_type, N, P, K]])
        prediction = Model_fertilizer.predict(features)

        P_fertilizer = get_fertilizer_name(prediction[0])
        result = {"message": "{} is the best fertilizer to use in the field.".format(P_fertilizer)}
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": "Missing form field - {}".format(e)}), 400
    except ValueError as e:
        return jsonify({"error": "Invalid form data - {}".format(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/cropLocation', methods=['POST'])
def find_crop_production():
    try:
        data = request.get_json()  # Ensure that data is parsed as JSON
        state_name = data['StateName']
        district_name = data['DistrictName']
        season = data['Season']

        prodution = get_main_crop(state_name)
        result = {"message": "{} is the best crop to be cultivated there.".format(prodution)}
        return jsonify(result)
    except KeyError as e:
        return jsonify({"error": "Missing form field - {}".format(e)}), 400
    except ValueError as e:
        return jsonify({"error": "Invalid form data - {}".format(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
