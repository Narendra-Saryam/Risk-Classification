import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import pickle
from faker import Faker

fake = Faker()

def generate_clinical_dataset(num_patients):
    np.random.seed(42)
    num_patients = (num_patients // 3) * 3  # Ensure divisible by 3
    cluster_sizes = [num_patients // 3] * 3

    # Base features
    node_data = {
        "patient_id": np.arange(num_patients),
        "name": [fake.name() for _ in range(num_patients)],
        "age": np.clip(np.random.normal(45, 15, num_patients), 18, 100).astype(int),
        "gender": np.random.choice([0, 1], num_patients, p=[0.55, 0.45]),
        "weight": np.clip(np.random.normal(70, 15, num_patients), 40, 150),
        "height": np.clip(np.random.normal(170, 10, num_patients), 150, 200),
        "alcohol_consumption": np.concatenate([
            np.random.randint(0, 3, cluster_sizes[0]),
            np.random.randint(3, 6, cluster_sizes[1]),
            np.random.randint(6, 9, cluster_sizes[2])
        ]),
        "tobacco_chewing": np.concatenate([
            np.random.randint(0, 3, cluster_sizes[0]),
            np.random.randint(3, 6, cluster_sizes[1]),
            np.random.randint(6, 9, cluster_sizes[2])
        ]),
        "smoking": np.concatenate([
            np.random.randint(0, 3, cluster_sizes[0]),
            np.random.randint(3, 6, cluster_sizes[1]),
            np.random.randint(6, 9, cluster_sizes[2])
        ]),
        "duration": np.concatenate([
            np.random.randint(0, 5, cluster_sizes[0]),
            np.random.randint(5, 12, cluster_sizes[1]),
            np.random.randint(12, 19, cluster_sizes[2])
        ]),
        "risk": np.repeat([0, 1, 2], cluster_sizes)
    }

    medical_data = {
        "liver_function": np.repeat(['Mild', 'Moderate', 'Severe'], cluster_sizes),
        "kidney_function": np.repeat(['Mild', 'Moderate', 'Severe'], cluster_sizes),
        "lung_function": np.repeat(['Mild', 'Moderate', 'Severe'], cluster_sizes),
        "addiction_dependence": np.repeat(['No', 'No', 'Yes'], cluster_sizes),
        "cancer": np.repeat([0, 0, 1], cluster_sizes),
        "diabetes": np.repeat([0, 1, 1], cluster_sizes),
        "hypertension": np.repeat([0, 1, 1], cluster_sizes)
    }

    full_df = pd.DataFrame({**node_data, **medical_data})
    return full_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Generate dataset
full_df = generate_clinical_dataset(600)

# One-hot encoding for categorical columns
encoder = OneHotEncoder(sparse_output=False)  # Changed 'sparse' to 'sparse_output'
categorical_features = ['liver_function', 'kidney_function', 'lung_function', 'addiction_dependence']
encoded_features = encoder.fit_transform(full_df[categorical_features])
encoded_df = pd.DataFrame(encoded_features,
                          columns=encoder.get_feature_names_out(categorical_features))

# Prepare final features
X = pd.concat([
    full_df[['age', 'gender', 'weight', 'height', 'alcohol_consumption',
             'tobacco_chewing', 'smoking', 'duration']],
    encoded_df,
    full_df[['cancer', 'diabetes', 'hypertension']]
], axis=1)

y = full_df['risk']

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
rf.fit(X_train, y_train)

# Save the model as a pickle file
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model saved as 'random_forest_model.pkl'")
