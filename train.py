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
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, \
    average_precision_score
import matplotlib.pyplot as plt
import numpy as np


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
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(0)

    high_risk_df = df[df['IsHighRisk'] == 1]
    normal_df = df[df['IsHighRisk'] == 0]
    num_high_risk_samples = min(len(high_risk_df), sample_size // 3)
    num_normal_samples = min(len(normal_df), sample_size - num_high_risk_samples)

    actual_high_risk = high_risk_df.sample(n=num_high_risk_samples, random_state=42)
    actual_normal = normal_df.sample(n=num_normal_samples, random_state=42)
    balanced_df = pd.concat([actual_high_risk, actual_normal]).sample(frac=1, random_state=42)

    print(f"Using a balanced dataset of {len(balanced_df)} records.")
    print(f"Malicious samples: {len(actual_high_risk)}, Normal samples: {len(actual_normal)}")

    print(f"Original number of unique ASNs: {balanced_df['ASN'].nunique()}")
    asn_counts = balanced_df['ASN'].value_counts()
    top_n_asns = 1000
    top_asns = asn_counts.nlargest(top_n_asns).index.tolist()
    if 'Unknown' not in top_asns:
        top_asns.append('Unknown')

    balanced_df['ASN'] = balanced_df['ASN'].where(balanced_df['ASN'].isin(top_asns), 'ASN_OTHER')
    print(f"Number of unique ASNs after grouping: {balanced_df['ASN'].nunique()}")

    return balanced_df


def train_model(dataset_filename):
    df = load_and_prepare_data(dataset_filename)

    categorical_cols = ['username', 'Country', 'City', 'OS', 'Browser', 'DeviceType', 'ASN']
    numerical_cols = ['LoginHour', 'DayOfWeek']

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['IsHighRisk']  # Ensure ratio is maintained
    )

    feature_maps = {}
    for col in categorical_cols:
        unique_vals = sorted(train_df[col].unique().tolist())
        if 'UNK' not in unique_vals:
            unique_vals.insert(0, 'UNK')
        feature_maps[f'{col}_to_id'] = {val: i for i, val in enumerate(unique_vals)}

    for col in categorical_cols:
        mapping = feature_maps[f'{col}_to_id']
        train_df[col] = train_df[col].apply(lambda x: mapping.get(x, 0))
        val_df[col] = val_df[col].apply(lambda x: mapping.get(x, 0))

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

    num_positives = (y_train == 1).sum()
    num_negatives = (y_train == 0).sum()
    pos_weight_value = num_negatives.float() / num_positives.float()

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=2
    )

    epochs = 50
    batch_size = 256
    best_val_loss = np.inf
    best_model_state = None
    best_epoch_num = 0

    epochs_no_improve = 0
    patience = 4

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
            val_outputs = model(X_cat_val, X_num_val)  # Raw logits
            val_loss = criterion(val_outputs, y_val)

            scheduler.step(val_loss)

            val_probs = torch.sigmoid(val_outputs)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                epochs_no_improve = 0
                best_epoch_num = epoch + 1
                print(f"  ** New best model saved at epoch {epoch + 1} with Val Loss: {val_loss.item():.4f} **")
            else:
                epochs_no_improve += 1

            preds = (val_probs > 0.5).float()
            y_val_np = y_val.cpu().numpy()
            preds_np = preds.cpu().numpy()
            probs_np = val_probs.cpu().numpy()

            precision, recall, f1, _ = precision_recall_fscore_support(y_val_np, preds_np, average='binary',
                                                                       zero_division=0)
            auroc = roc_auc_score(y_val_np, probs_np)

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, '
                  f'Val Acc: {(preds == y_val).float().mean():.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, '
                  f'F1: {f1:.4f}, AUROC: {auroc:.4f}')

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}!")
            break

    print(f"\nTraining complete. Saving best model from epoch {epoch + 1 - epochs_no_improve}...")
    torch.save(best_model_state, 'rba_model.pth')

    print("Generating Precision-Recall curve for the best model...")

    # Load the best model state
    model.load_state_dict(best_model_state)
    model.eval()

    with torch.no_grad():
        val_outputs = model(X_cat_val, X_num_val)
        val_probs = torch.sigmoid(val_outputs)
        y_val_np = y_val.cpu().numpy()
        probs_np = val_probs.cpu().numpy()

    # Calculate PR data
    precision_points, recall_points, _ = precision_recall_curve(y_val_np, probs_np)
    ap_score = average_precision_score(y_val_np, probs_np)

    # Plot the curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_points, precision_points, color='blue', label=f'AP = {ap_score:.4f}')
    plt.title(f'Precision-Recall Curve (Best Model - Epoch {best_epoch_num})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_recall_curve.png')
    print("Precision-Recall curve saved to 'precision_recall_curve.png'")

    with open('feature_maps.json', 'w') as f:
        json.dump(feature_maps, f, indent=2)

    joblib.dump(scaler, 'scaler.joblib')

    print("Artifacts saved successfully.")


if __name__ == '__main__':
    train_model('archive.zip')
