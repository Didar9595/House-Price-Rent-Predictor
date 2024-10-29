from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS

# Set environment variables to suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the pre-trained model (ensure the file path is correct)
try:
    model = joblib.load('Prediction_Model.joblib')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)

        # Validate the input
        if not all(value is not None for value in data.values()):
            return jsonify({'error': 'Missing input values'}), 400

        # Prepare input features for prediction
        input_features = np.array([
            float(data['bhk']),
            float(data['size']),
            float(data['area_type']),
            float(data['pincode']),
            float(data['furnishing']),
            float(data['tenant_type']),
            float(data['bathrooms'])
        ]).reshape(1, -1)

        # Perform prediction
        prediction = model.predict(input_features)
        return jsonify({'prediction': round(float(prediction[0]), 2)})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Set debug=False for production
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
