# Bharat Agro API

Welcome to the Bharat Agro API. This API provides predictions for crop selection, fertilizer recommendation, and crop production based on location. 

### Models
- NBClassifier.pkl - Model for crop prediction.
- RandomForest.pkl - Model for fertilizer prediction.
- DecisionTree.pkl - Model for crop production based on location.


## Base URL

The base URL for all API endpoints will be provided by **Render** once deployed.
```
https://agro-api-xnfo.onrender.com
```
### 1. Predict Crop

Predict the best crop to cultivate based on various soil and environmental parameters.

**URL:** `https://agro-api-xnfo.onrender.com/predictcrop`

**Method:** `POST`

**Request Body:**

```json
{
    "Nitrogen": float,
    "Phosphorus": float,
    "Potassium": float,
    "Temperature": float,
    "Humidity": float,
    "Ph": float,
    "Rainfall": float
}
```
**Success Response:**
```json
{
    "message": "<crop> is the best crop to be cultivated there."
}

```
**Error Response**
```json
{
    "error": "Missing form field - <field_name>"
}
{
    "error": "Invalid form data - <error_details>"
}
{
    "error": "<exception_details>"
}
```
### 2. Predict Fertilizer
Predict the best fertilizer to use based on various soil and crop parameters.

```
URL: /predictfertilizer

Method: POST
```
**Request Body:**
```json
{
    "Temperature": float,
    "Humidity": float,
    "SoilMoisture": float,
    "soil_type": "Sandy" | "Loamy" | "Black" | "Red" | "Clayey",
    "crop_type": "Maize" | "Sugarcane" | "Cotton" | "Tobacco" | "Paddy" | "Barley" | "Wheat" | "Millets" | "Oil seeds" | "Pulses" | "Ground Nuts",
    "Nitrogen": float,
    "Phosphorus": float,
    "Potassium": float
}
```
**Success Response:**
```json
{
  "message": "<fertilizer> is the best fertilizer to use in the field."

}
```
**Error Response**
```json
{
 "error": "Missing form field - <field_name>"

}
{
"error": "Invalid form data - <error_details>"

}
{
    "error": "<exception_details>"
}
```

### Directory Structure
<Agro-api>/
│
├── app.py             # Main application file
├── Models/            # Directory containing the machine learning models
│   ├── NBClassifier.pkl
│   ├── RandomForest.pkl
│   └── DecisionTree.pkl
├── requirements.txt   # List of dependencies
└── README.md          # This file


### Dependencies
- falsk 
- numpy 
- scikit-learn 

### Running Locally
**1. Install the required packages:**
```
pip install -r requirements.txt

```
**2. Set the environment variable for the secret key:**
```
export SECRET_KEY=<your-secret-key>

```
**3.Run the Flask application:** 
```
python app.py

```








