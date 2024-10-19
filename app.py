from flask import Flask, request, jsonify
import joblib  # For loading the model
import numpy as np
from flask_cors import CORS

# Load the pre-trained model
model = joblib.load('Prediction_model.joblib')

# Initialize Flask app
app = Flask(__name__)
CORS(app) 

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.json
        print("Hii")
        # Extract the input values from the request
        input_features = [
            data['bhk'], 
            data['size'], 
            data['area_type'], 
            data['pincode'], 
            data['furnishing'], 
            data['tenant_type'], 
            data['bathrooms']
        ]

        # Convert to numpy array and reshape for the model
        input_array = np.array(input_features).reshape(1, -1)

        # Perform the prediction
        prediction = model.predict(input_array)[0]
        # Ensure the prediction is a float value, not a numpy type
        prediction = float(prediction)
        
        # Return the prediction result as JSON
        return jsonify({'prediction': round(prediction, 2)})

    except Exception as e:
        # If something goes wrong, return the error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the app on port 5001
    app.run(host='0.0.0.0', port=5000, debug=True)
