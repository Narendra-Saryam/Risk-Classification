from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
from model import ClinicalGNN  # Ensure model.py has this class

app = Flask(__name__)
CORS(app)

# Constants
RISK_CLASSES = {0: 'Normal', 1: 'Low', 2: 'High'}

# Load the trained GNN model
model = ClinicalGNN(in_dim=15, hid_dim=128, out_dim=3)  # Adjust `in_dim` if needed
model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
model.eval()

@app.route('/')
def home():
    return "GNN Clinical Risk Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data received'}), 400

        # Required fields (match input structure)
        required_fields = [
            'age', 'gender', 'weight', 'height',
            'alcohol_consumption', 'alcohol_duration',
            'tobacco_chewing', 'tobacco_duration',
            'smoking', 'smoking_duration',
            'addiction_dependence',
            'liver_function', 'kidney_function', 'lung_function',
            'cancer', 'diabetes', 'hypertension'
        ]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {", ".join(missing)}'}), 400

        # Map categorical inputs to numeric
        severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}
        yesno_map = {'No': 0, 'Yes': 1}

        features = [
            float(data['age']),
            float(data['alcohol_consumption']),
            float(data['alcohol_duration']),
            float(data['tobacco_chewing']),
            float(data['tobacco_duration']),
            float(data['smoking']),
            float(data['smoking_duration']),
            yesno_map.get(data['addiction_dependence'], 0),
            severity_map.get(data['liver_function'], 0),
            severity_map.get(data['kidney_function'], 0),
            severity_map.get(data['lung_function'], 0),
            yesno_map.get(data['cancer'], 0),
            yesno_map.get(data['diabetes'], 0),
            yesno_map.get(data['hypertension'], 0),
            int(data['gender'])  # Assuming gender is 0/1 already
        ]

        # Create graph data object for single patient (no edge_index needed)
        x = torch.tensor([features], dtype=torch.float)
        edge_index = torch.tensor([[], []], dtype=torch.long)  # No edges for single node
        graph = Data(x=x, edge_index=edge_index)

        with torch.no_grad():
            output = model(graph)
            probs = F.softmax(output, dim=1).numpy()[0]
            predicted = int(np.argmax(probs))

        return jsonify({
            'risk_level': RISK_CLASSES[predicted],
            'probabilities': {
                'Normal': round(probs[0] * 100, 2),
                'Low': round(probs[1] * 100, 2),
                'High': round(probs[2] * 100, 2)
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
