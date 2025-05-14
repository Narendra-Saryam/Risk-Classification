from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Re-create encoder with same categories as training
encoder = OneHotEncoder(sparse_output=False)
encoder.fit(pd.DataFrame({
    "liver_function": ['Mild', 'Moderate', 'Severe'],
    "kidney_function": ['Mild', 'Moderate', 'Severe'],
    "lung_function": ['Mild', 'Moderate', 'Severe'],
    "addiction_dependence": ['No', 'Yes', 'No']
}))

@app.route('/')
def home():
    return "Clinical Risk Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert input into DataFrame
        input_df = pd.DataFrame([data])

        # Validate required fields
        required_fields = ['age', 'gender', 'weight', 'height', 'alcohol_consumption',
                         'tobacco_chewing', 'smoking', 'duration', 'liver_function',
                         'kidney_function', 'lung_function', 'addiction_dependence',
                         'cancer', 'diabetes', 'hypertension']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        # One-hot encode categorical fields
        categorical_features = ['liver_function', 'kidney_function', 'lung_function', 'addiction_dependence']
        encoded = encoder.transform(input_df[categorical_features])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))

        # Final feature set
        final_features = pd.concat([
            input_df[['age', 'gender', 'weight', 'height',
                      'alcohol_consumption', 'tobacco_chewing',
                      'smoking', 'duration']],
            encoded_df,
            input_df[['cancer', 'diabetes', 'hypertension']]
        ], axis=1)

        # Predict
        prediction = model.predict(final_features)[0]
        return jsonify({'risk_prediction': int(prediction)})

    except ValueError as ve:
        return jsonify({'error': f'Invalid data format: {str(ve)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
