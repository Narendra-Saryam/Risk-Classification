from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# --- Define the ClinicalGNN class ---
class ClinicalGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.clinical_norm = nn.BatchNorm1d(hid_dim)
        self.classifier = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(0.7)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = self.clinical_norm(x)
        x = F.elu(self.conv2(x, edge_index))
        return F.log_softmax(self.classifier(x), dim=1)

# --- Load the model and scaler ---
with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = ClinicalGNN(saved['input_dim'], 128, 3)
model.load_state_dict(saved['model_state_dict'])
model.eval()

scaler = saved['scaler']

@app.route('/')
def home():
    return "Clinical GNN API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Required features used during training
        edge_features = [
            'age', 'gender', 'height', 'weight',
            'alcohol_consumption_per_day_in_liter', 'alcohol_duration',
            'tobacco_chewing_per_day_in_gram', 'tobacco_duration',
            'smoking_per_day', 'smoking_duration',
            'addiction_dependence',
            'liver_function', 'kidney_function', 'lung_function',
            'cancer', 'diabetes', 'hypertension'
        ]

        # Check for missing fields
        missing_fields = [field for field in edge_features if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400

        # Create DataFrame and scale
        input_df = pd.DataFrame([data])
        input_scaled = scaler.transform(input_df[edge_features])
        node_features = torch.tensor(input_scaled, dtype=torch.float)

        # Dummy edge index for single node prediction
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        graph = Data(x=node_features, edge_index=edge_index)

        with torch.no_grad():
            out = model(graph)
            prediction = torch.argmax(out, dim=1).item()

        return jsonify({'risk_prediction': prediction})

    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
