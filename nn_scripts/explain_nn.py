import logging
import os
import random
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from nn_scripts.model import SimpleRBAModel
from nn_scripts.ensembler import ensemble_threat_score
from helpers.globals import select_device, resolve_path, cfg

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
FEATURE_COLUMNS = cfg("data.feature_columns")
LIMIT = cfg("data.limit")

MODEL_PATH = resolve_path("NN_MODEL_PATH", os.path.join(cfg("model.output_dir"), cfg("model.checkpoint")))
SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.scaler")))
USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.user_map")))
PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH", os.path.join(cfg("preprocessing.output_dir"),
                                                                      cfg("preprocessing.artifacts.preprocessor")))

_device = select_device()


def load_threshold():
    try:
        threshold_path = resolve_path(
            "NN_THRESHOLD_PATH",
            os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.threshold"))
        )
        with open(threshold_path, "r") as f:
            return float(f.read().strip())
    except Exception as e:
        logging.warning(f"Failed to read threshold file: {e}. Using a random value because I hate you.")
        return random.random()


THRESHOLD = load_threshold()


def load_data():
    engine = create_engine(POSTGRES_CONNECTION_STRING)
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, nn_score, platform,
               human_verified, impossible_travel
        FROM {cfg("data.table")}
        WHERE nn_score >= 0.0
        ORDER BY random()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": LIMIT})

    # Recreate targets
    df["human_verified"] = df.get("human_verified", False).fillna(False).astype(bool)
    df["impossible_travel"] = df.get("impossible_travel", False).fillna(False).astype(bool)

    df["is_true_threat"] = (
            (df["nn_score"] == 1.0) & (df["human_verified"] == True)
            | (df["impossible_travel"] == True)
    ).astype(int)

    return df


def get_predictions(model, X_tensor, user_tensor):
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(_device), user_tensor.to(_device)).cpu().numpy().flatten()
    return preds


def main():
    print("Loading neural network artifacts...")
    scaler = joblib.load(SCALER_PATH)
    user_to_id = joblib.load(USER_MAP_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # Load Data
    df = load_data()
    print(f"Loaded {len(df)} rows for explanation.")

    # Preprocess
    df_transformed = preprocessor.transform_df(df.copy(), FEATURE_COLUMNS)
    df_transformed["user_id"] = df["username"].map(user_to_id).fillna(0).astype(int)

    X_raw = df_transformed[FEATURE_COLUMNS].values.astype(np.float32)
    y_true = df["is_true_threat"].values
    user_ids = df_transformed["user_id"].values

    X_scaled = scaler.transform(X_raw)

    # Load Model
    num_users = len(user_to_id)
    input_dim = len(FEATURE_COLUMNS)
    raw_dim = np.log2(num_users) if cfg("model.embed_dim_scale") == "log2" else cfg("model.embed_dim_scale")
    embed_dim = int(min(max(raw_dim, cfg("model.min_embed_dim")), cfg("model.max_embed_dim")))

    model = SimpleRBAModel(input_dim=input_dim, num_users=num_users, embed_dim=embed_dim).to(_device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    print("Model loaded successfully.")

    # Tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    user_tensor = torch.tensor(user_ids, dtype=torch.long)
    y_tensor = torch.tensor(y_true, dtype=torch.float32).unsqueeze(1).to(_device)

    # Baseline Loss
    criterion = nn.BCELoss()
    with torch.no_grad():
        baseline_loss = criterion(model(X_tensor.to(_device), user_tensor.to(_device)), y_tensor).item()

    print(f"Baseline Loss: {baseline_loss:.5f}")

    feature_importance = {}
    for i, col_name in enumerate(FEATURE_COLUMNS):
        original_col = X_scaled[:, i].copy()
        np.random.shuffle(X_scaled[:, i])

        X_permuted = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            permuted_loss = criterion(model(X_permuted.to(_device), user_tensor.to(_device)), y_tensor).item()

        importance = permuted_loss - baseline_loss
        feature_importance[col_name] = importance
        X_scaled[:, i] = original_col

    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    keys = [x[0] for x in sorted_features]
    values = [x[1] for x in sorted_features]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Feature Importance
    ax1.barh(keys[:15][::-1], values[:15][::-1], color='skyblue')
    ax1.set_xlabel('Increase in Loss (Importance)')
    ax1.set_title('Feature Importance (Permutation)')

    # Confusion Matrix
    print(f"Generating Confusion Matrix (Threshold: {THRESHOLD})...")
    raw_preds = get_predictions(model, X_tensor, user_tensor)
    ensemble_preds = np.array([ensemble_threat_score(p, 0) for p in raw_preds])
    binary_preds = (ensemble_preds >= THRESHOLD).astype(int)

    cm = confusion_matrix(y_true, binary_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Threat"])

    # Plot directly onto the second axis
    disp.plot(cmap="Blues", ax=ax2, colorbar=True)
    ax2.set_title(f"Confusion Matrix (Threshold: {THRESHOLD})")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
