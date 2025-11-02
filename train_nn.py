import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import joblib

from ensembler import ensemble_threat_score
from model import SimpleRBAModel
from nn_helper import get_best_device

# Load connection info
load_dotenv()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)
_device = get_best_device()

FEATURE_COLUMNS = [
    "focus_changes", "blur_events", "click_count", "key_count",
    "avg_key_delay_ms", "pointer_distance_px", "pointer_event_count",
    "scroll_distance_px", "scroll_event_count", "dom_ready_ms",
    "time_to_first_key_ms", "time_to_first_click_ms", "idle_time_total_ms",
    "input_focus_count", "paste_events", "resize_events",
    "tz_offset_min", "device_memory_gb", "hardware_concurrency",
    "screen_width_px", "screen_height_px", "pixel_ratio", "color_depth",
    "touch_support", "webauthn_supported"
]


def load_training_data(limit: int = 10000):
    """
    Fetch training data from rba_login_event.
    """
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, nn_score, human_verified
        FROM rba_login_event
        WHERE nn_score >= 0.0
        ORDER BY random()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})

        # Ensure human_verified is boolean and fill NaNs
        if 'human_verified' in df.columns:
            df['human_verified'] = df['human_verified'].fillna(False).astype(bool)
        else:
            print("Warning: 'human_verified' column not found. Defaulting to False.")
            df['human_verified'] = False

    print(f"Loaded {len(df)} rows.")
    return df


def preprocess_training_data(df):
    # Handle device_memory_gb imputations
    if "device_memory_gb" in df.columns and "platform" in df.columns:
        missing_mask = df["device_memory_gb"].isna()

        win_mask = missing_mask & df["platform"].str.contains("Win", case=False, na=False)
        if win_mask.any():
            win_mean = df.loc[df["platform"].str.contains("Win", case=False, na=False), "device_memory_gb"].mean()
            df.loc[win_mask, "device_memory_gb"] = win_mean
            print(f"Imputed {win_mask.sum()} Windows rows with mean {win_mean:.2f} GB")

        iphone_mask = missing_mask & df["platform"].str.contains("iPhone", case=False, na=False)
        if iphone_mask.any():
            iphone_mode = df.loc[df["platform"].str.contains("iPhone", case=False, na=False), "device_memory_gb"].mode()
            if not iphone_mode.empty:
                df.loc[iphone_mask, "device_memory_gb"] = iphone_mode.iloc[0]
                print(f"Imputed {iphone_mask.sum()} iPhone rows with mode {iphone_mode.iloc[0]:.2f} GB")

        remaining = df["device_memory_gb"].isna().sum()
        if remaining > 0:
            median_val = df["device_memory_gb"].median()
            df["device_memory_gb"].fillna(median_val, inplace=True)
            print(f"⚙️ Imputed {remaining} remaining rows with median {median_val:.2f} GB")

    for col in FEATURE_COLUMNS:
        if df[col].isna().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Encode usernames into numeric IDs for the embedding layer
    unique_users = df["username"].unique()
    user_to_id = {u: i for i, u in enumerate(unique_users)}
    df["user_id"] = df["username"].map(user_to_id)

    # Prepare feature/target tensors
    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["nn_score"].astype(np.float32)
    user_ids = df["user_id"].astype(np.int64)  # embedding indices must be int64

    df['is_true_threat'] = ((df['nn_score'] == 1.0) & (df['human_verified'] == True)).astype(int)
    true_labels = df["is_true_threat"].values

    # Split sets
    X_train, X_val, \
        y_train, y_val, \
        user_train, user_val, \
        true_labels_train, true_labels_val = train_test_split(
        X, y, user_ids, true_labels, test_size=0.5, random_state=41, stratify=true_labels
    )

    # Normalize numeric features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    user_train_tensor = torch.tensor(user_train.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    user_val_tensor = torch.tensor(user_val.values, dtype=torch.long)

    num_users = len(unique_users)
    print(f"Preprocessing complete:")
    print(f"\tTrain: {len(X_train)} rows | Val: {len(X_val)} rows | {X_train.shape[1]} features | {num_users} users")

    return (
        X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor, user_train_tensor, user_val_tensor, true_labels_val,
        scaler, num_users
    )


def train_neural_network(
        X_train, X_val, y_train, y_val, user_train, user_val, num_users,
        num_epochs=500, lr=1e-3, patience=5, checkpoint_path="best_rba_model.pt"
):
    """
    Train the personalized RBA model with user embeddings and early stopping.
    Automatically saves the best model checkpoint (lowest validation loss).
    """
    input_dim = X_train.shape[1]

    embed_dim = int(min(max(np.log2(num_users), 4), 64))
    print(f"Using embedding dimension = {embed_dim} for {num_users} users.")

    model = SimpleRBAModel(input_dim=input_dim, num_users=num_users, embed_dim=embed_dim).to(_device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = X_train.to(_device)
    y_train = y_train.to(_device)
    user_train = user_train.to(_device)
    X_val = X_val.to(_device)
    y_val = y_val.to(_device)
    user_val = user_val.to(_device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train, user_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val, user_val)
            val_loss = criterion(val_outputs, y_val)

        if val_loss.item() < best_val_loss - 1e-6:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved at epoch {epoch + 1} (val_loss={best_val_loss:.5f})")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} "
                  f"| Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs "
                  f"(best val loss = {best_val_loss:.5f}).")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    print(f"Training complete. Best model restored from {checkpoint_path}.")
    return model


def find_best_threshold(model, X_val, y_val, user_val, ip_flags=None):
    model.eval()
    with torch.no_grad():
        X_val_dev = X_val.to(_device)
        user_val_dev = user_val.to(_device)
        nn_preds = model(X_val_dev, user_val_dev).cpu().numpy().flatten()

    # default: all IPs clean
    if ip_flags is None:
        ip_flags = np.zeros_like(nn_preds)

    # apply your real ensemble logic
    ensemble_preds = np.array([
        ensemble_threat_score(nn, ip)
        for nn, ip in zip(nn_preds, ip_flags)
    ])

    # sweep thresholds to find best F1 (or other metric)
    thresholds = np.linspace(0.0, 1.0, 201)
    f1_scores = []

    for t in thresholds:
        preds_bin = (ensemble_preds >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds_bin))

    best_idx = int(np.argmax(f1_scores))
    best_t = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"✅ Best threshold = {best_t:.3f} (F1 = {best_f1:.3f})")
    return best_t


if __name__ == "__main__":
    df = load_training_data()
    (X_train, X_val, y_train, y_val, user_train, user_val, true_labels_val, scaler,
     num_users) = preprocess_training_data(df)

    model = train_neural_network(X_train, X_val, y_train, y_val, user_train, user_val, num_users)
    joblib.dump(scaler, "rba_scaler.pkl")
    print("Saved model as rba_model.pt and scaler as rba_scaler.pkl")
    best_threshold = find_best_threshold(model, X_val, true_labels_val, user_val)
    print(f"Recommended threshold for Shibboleth plugin: {best_threshold:.3f}")
