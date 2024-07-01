from flask import Flask, request, jsonify, session
import pickle 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
import os 
import secrets

app = Flask(__name__)

app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(16))

# Define the path to the models directory
models_dir = os.path.join(app.root_path, 'Models')

# Load machine learning models
Model_crop_path = os.path.join(models_dir, 'NBClassifier.pkl')
Model_fertilizer_path = os.path.join(models_dir, 'RandomForest.pkl')

# Load machine learning models
Model_crop = pickle.load(open(Model_crop_path, 'rb'))
Model_fertilizer = pickle.load(open(Model_fertilizer_path, 'rb'))

# Define routes
@app.route('/')
def index():
    return "Hello, welcome to the API!"

@app.route('/predictcrop', methods=['POST'])
def predictcrop():
    try:
        data = request.get_json()  # Parse JSON data from the request
        if data is None:
            raise ValueError("No JSON data received")

        # Get form data
        N = float(data['Nitrogen'])
        P = float(data['Phosphorus'])
        K = float(data['Potassium'])
        temp = float(data['Temperature'])
        humidity = float(data['Humidity'])
        ph = float(data['pH'])
        rainfall = float(data['Rainfall'])

        # Perform prediction
        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        prediction = Model_crop.predict(features)
        crop = prediction[0]

        # Format the result message
        result = "{} is the best crop to be cultivated there.".format(crop)

        # Store the result in session
        session['result'] = result

        # Return the prediction as JSON response
        return jsonify({"crop_prediction": result})
    except KeyError as e:
        error_message = f"Error: Missing or incorrect form field - {str(e)}"
        return jsonify({"error": error_message}), 400
    except ValueError as e:
        error_message = f"Error: Invalid form data - {str(e)}"
        return jsonify({"error": error_message}), 400
    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify({"error": error_message}), 500


## Encode the values 
# Define encoding functions outside the route function
def encode_soil_types(original_soil_types, new_soil_type):
    soil_type_label_encoder = LabelEncoder().fit(original_soil_types)
    encoded_soil_type = soil_type_label_encoder.transform([new_soil_type])
    return encoded_soil_type[0]

def encode_crop_types(original_crop_types, new_crop_type):
    crop_type_label_encoder = LabelEncoder().fit(original_crop_types)
    encoded_crop_type = crop_type_label_encoder.transform([new_crop_type])
    return encoded_crop_type[0]

## Encoder Fertilizer name 
encoded_fertilizers = {
    0: '10-26-26', 1: '14-35-14', 2: '17-17-17', 3: '20-20', 4: '28-28', 5: 'DAP', 6: 'Urea'
}
def get_fertilizer_name(encoded_value):
    return encoded_fertilizers.get(encoded_value, 'Unknown')

@app.route('/predictfertilizer', methods=['POST'])
def predictfertilizer():
    try:
        # Get form data
        Temperature = float(request.form['Temperature'])
        Humidity = float(request.form['Humidity'])
        Soil_Moisture = float(request.form['SoilMoisture'])
        SoilType = request.form['soil_type']
        Crop_type = request.form['crop_type']
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorous'])
        K = float(request.form['Potassium'])
        
        # Encode Soil Type and Crop Type
        encoded_soil_type = encode_soil_types(['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'], SoilType)
        encoded_crop_type = encode_crop_types(['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy', 'Barley',
                                               'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts'], Crop_type)

        # Perform prediction
        features = np.array([[Temperature, Humidity, Soil_Moisture, encoded_soil_type, encoded_crop_type, N, P, K]])
        prediction = Model_fertilizer.predict(features)

        P_fertilizer = get_fertilizer_name(prediction[0])

        result = "{} is the best fertilizer to use in the field.".format(P_fertilizer)

        # Store the result in session
        session['result'] = result

        return  jsonify({"fertilizer_prediction": result})
    except KeyError as e:
        return jsonify({"error": f"Missing form field - {e}"}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid form data - {e}"})


if __name__ == '__main__':
    app.run(debug=True)









