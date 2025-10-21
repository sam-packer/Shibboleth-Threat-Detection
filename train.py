import torch
import torch.nn as nn
import torch.optim as optim
from model import RBAModel
import json
import pandas as pd
import os
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import numpy as np


# --- 1. DATA LOADING & PREPARATION ---

def load_and_prepare_data(zip_filepath, sample_size=50000):
    """
    Loads the RBA dataset, creates a composite 'IsHighRisk' target,
    and prepares a large, balanced dataset for training.
    """
    if not os.path.exists(zip_filepath):
        print(f"--- ERROR: Missing required files ---")
        print(f"- Dataset not found: {zip_filepath}")
        print("Please download the dataset from https://www.kaggle.com/datasets/dasgroup/rba-dataset")
        print(f"and place it in this directory with the name '{zip_filepath}'.")
        exit()

    print(f"Loading data from '{zip_filepath}'...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as z:
            with z.open('rba-dataset.csv') as f:
                df = pd.read_csv(f, low_memory=False)
    except Exception as e:
        print(f"Error reading the zip file: {e}")
        exit()

    print("Actual columns found in CSV:", df.columns.tolist())

    column_mapping = {
        'User ID': 'username', 'IP Address': 'client_ip_address',
        'Is Account Takeover': 'IsAccountTakeover', 'Is Attack IP': 'IsAttackIP',
        'Country': 'Country', 'City': 'City', 'ASN': 'ASN', 'Device Type': 'DeviceType',
        'OS Name and Version': 'OS', 'Browser Name and Version': 'Browser',
        'Login Timestamp': 'LoginTimestamp'
    }
    df.rename(columns=column_mapping, inplace=True)

    print("Starting feature engineering...")
    df = df[list(column_mapping.values())]

    df['IsAccountTakeover'] = df['IsAccountTakeover'].astype(bool)
    df['IsAttackIP'] = df['IsAttackIP'].astype(bool)
    df['IsHighRisk'] = ((df['IsAccountTakeover']) | (df['IsAttackIP'])).astype(int)

    df['LoginTimestamp'] = pd.to_datetime(df['LoginTimestamp'], errors='coerce')
    df.dropna(subset=['LoginTimestamp'], inplace=True)
    df['LoginHour'] = df['LoginTimestamp'].dt.hour
    df['DayOfWeek'] = df['LoginTimestamp'].dt.dayofweek

    df['ASN'] = df['ASN'].astype(str)

    feature_cols_to_clean = ['username', 'Country', 'City', 'OS', 'Browser', 'DeviceType', 'ASN']
    for col in feature_cols_to_clean:
        if df[col].dtype == 'object':
            df[col].fillna('Unknown', inplace=True)
        else:
            df[col].fillna(0, inplace=True)

    high_risk_df = df[df['IsHighRisk'] == 1]
    normal_df = df[df['IsHighRisk'] == 0]
    num_high_risk_samples = min(len(high_risk_df), sample_size // 3)
    num_normal_samples = min(len(normal_df), sample_size - num_high_risk_samples)

    actual_high_risk = high_risk_df.sample(n=num_high_risk_samples, random_state=42)
    actual_normal = normal_df.sample(n=num_normal_samples, random_state=42)
    balanced_df = pd.concat([actual_high_risk, actual_normal]).sample(frac=1, random_state=42)

    print(f"Using a balanced dataset of {len(balanced_df)} records.")
    print(f"Malicious samples: {len(actual_high_risk)}, Normal samples: {len(actual_normal)}")

    # --- FIX: Handle high-cardinality for ASN ---
    print(f"Original number of unique ASNs: {balanced_df['ASN'].nunique()}")
    asn_counts = balanced_df['ASN'].value_counts()
    # Keep the top 1000 most frequent ASNs, plus 'Unknown'
    top_n_asns = 1000
    top_asns = asn_counts.nlargest(top_n_asns).index.tolist()
    if 'Unknown' not in top_asns:
        top_asns.append('Unknown')

    # Group rare ASNs into an 'ASN_OTHER' category
    balanced_df['ASN'] = balanced_df['ASN'].where(balanced_df['ASN'].isin(top_asns), 'ASN_OTHER')
    print(f"Number of unique ASNs after grouping: {balanced_df['ASN'].nunique()}")

    return balanced_df


# --- 2. TRAINING LOOP ---
def train_model(dataset_filename):
    df = load_and_prepare_data(dataset_filename)

    categorical_cols = ['username', 'Country', 'City', 'OS', 'Browser', 'DeviceType', 'ASN']
    numerical_cols = ['LoginHour', 'DayOfWeek']

    feature_maps = {}
    for col in categorical_cols:
        # Build mappings from the now-reduced feature set
        unique_vals = sorted(df[col].unique().tolist())
        if 'UNK' not in unique_vals: unique_vals.insert(0, 'UNK')
        feature_maps[f'{col}_to_id'] = {val: i for i, val in enumerate(unique_vals)}

    for col in categorical_cols:
        mapping = feature_maps[f'{col}_to_id']
        df[col] = df[col].map(mapping).fillna(0)  # Default to 'UNK' ID

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    val_df[numerical_cols] = scaler.transform(val_df[numerical_cols])

    X_cat_train = torch.LongTensor(train_df[categorical_cols].values)
    X_num_train = torch.FloatTensor(train_df[numerical_cols].values)
    y_train = torch.FloatTensor(train_df['IsHighRisk'].values).unsqueeze(1)

    X_cat_val = torch.LongTensor(val_df[categorical_cols].values)
    X_num_val = torch.FloatTensor(val_df[numerical_cols].values)
    y_val = torch.FloatTensor(val_df['IsHighRisk'].values).unsqueeze(1)

    vocab_sizes = {col: len(feature_maps[f'{col}_to_id']) for col in categorical_cols}
    embedding_dims = {
        'username': 50, 'Country': 20, 'City': 30,
        'OS': 20, 'Browser': 20, 'DeviceType': 10, 'ASN': 25
    }

    model = RBAModel(vocab_sizes, embedding_dims, num_numerical_feats=len(numerical_cols))

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    epochs = 15
    batch_size = 256
    best_val_loss = np.inf
    best_model_state = None

    print("\nStarting model training...")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_cat_train.size(0))
        for i in range(0, X_cat_train.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_cat, batch_num, batch_labels = X_cat_train[indices], X_num_train[indices], y_train[indices]

            optimizer.zero_grad()
            outputs = model(batch_cat, batch_num)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_cat_val, X_num_val)
            val_loss = criterion(val_outputs, y_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                print(f"  ** New best model saved at epoch {epoch + 1} with Val Loss: {val_loss.item():.4f} **")

            preds = (val_outputs > 0.5).float()
            y_val_np = y_val.cpu().numpy()
            preds_np = preds.cpu().numpy()

            precision, recall, f1, _ = precision_recall_fscore_support(y_val_np, preds_np, average='binary',
                                                                       zero_division=0)
            auroc = roc_auc_score(y_val_np, val_outputs.cpu().numpy())

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Val Acc: {(preds == y_val).float().mean():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
                  f'F1: {f1:.4f}, AUROC: {auroc:.4f}')

    print("\nTraining complete. Saving best performing model and artifacts...")
    torch.save(best_model_state, 'rba_model.pth')

    with open('feature_maps.json', 'w') as f:
        json.dump(feature_maps, f, indent=2)

    joblib.dump(scaler, 'scaler.joblib')

    print("Artifacts saved successfully.")


if __name__ == '__main__':
    train_model('archive.zip')

