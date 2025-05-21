import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import pickle

full_df = pd.read_csv('patient_dataset.csv')
# Preprocess for PyTorch Geometric
scaler = StandardScaler()
edge_features = [
    'age', 'alcohol_consumption_per_day_in_liter', 'alcohol_duration',
    'tobacco_chewing_per_day_in_gram', 'tobacco_duration',
    'smoking_per_day', 'smoking_duration', 'addiction_dependence',
    'liver_function', 'kidney_function', 'lung_function',
    'cancer', 'diabetes', 'hypertension'
]
edge_matrix = scaler.fit_transform(full_df[edge_features])

# Create graph edges
knn = NearestNeighbors(n_neighbors=8, metric='cosine').fit(edge_matrix)
distances, indices = knn.kneighbors()

# Get the number of patients from the DataFrame
num_patients = len(full_df) # Added this line to define num_patients

edges = []
for i in range(num_patients):
    for j in indices[i]:
        if i != j and (j, i) not in edges:
            edges.append((i, j))

# Convert to PyTorch tensors
edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
labels = torch.tensor(full_df['risk'].values, dtype=torch.long)
node_features = torch.tensor(full_df.drop(columns=['patient_id', 'risk']).values, dtype=torch.float)

# Graph data
data = Data(x=node_features, edge_index=edge_index, y=labels)

# ... (rest of the code)

class ClinicalGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, hid_dim)
        self.clinical_norm = nn.BatchNorm1d(hid_dim)  # Added medical normalization
        self.classifier = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(0.7)  # Increased for medical data

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))  # Changed to ELU for medical data
        x = self.clinical_norm(x)
        x = F.elu(self.conv2(x, edge_index))
        return F.log_softmax(self.classifier(x), dim=1)

model = ClinicalGNN(data.num_node_features, 128, 3)  # Increased hidden size

train_losses = []
val_accuracies = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-3)  # AdamW for medical data
criterion = nn.CrossEntropyLoss()

# Clinical data split (70/30)
train_mask = torch.zeros(num_patients, dtype=torch.bool)
train_mask[:280] = True  # 70% of 400
val_mask = torch.zeros(num_patients, dtype=torch.bool)
val_mask[280:] = True

# Training loop with early stopping
best_acc = 0
for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())

    # Validation every 10 epochs
    if epoch % 10 == 0:
        model.eval()
        with torch.no_grad():
            _, preds = out.max(dim=1)
            acc = (preds[val_mask] == data.y[val_mask]).sum().item() / val_mask.sum().item()
            val_accuracies.append(acc)

            # Early stopping
            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), 'best_model.pt')

# Load best model
model.load_state_dict(torch.load('best_model.pt'))

# Calculate additional metrics
accuracy = (preds[val_mask] == data.y[val_mask]).sum().item() / val_mask.sum().item()
print(f"\nFinal Validation Accuracy: {accuracy:.4f}")

# Save model and metadata using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump({
        'model_state_dict': model.state_dict(),
        'input_dim': data.num_node_features,
        'scaler': scaler  # Optional: Include if you need to transform future inputs
    }, f)

print("Model saved as 'model.pkl'")