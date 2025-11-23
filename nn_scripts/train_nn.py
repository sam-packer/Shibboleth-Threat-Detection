import logging
import os
from contextlib import nullcontext
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import joblib

from nn_scripts.ensembler import ensemble_threat_score
from nn_scripts.feature_preprocessor import FeaturePreprocessor
from helpers.globals import select_device, resolve_path, cfg
from nn_scripts.model import SimpleRBAModel

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

_device = select_device()

FEATURE_COLUMNS = cfg("data.feature_columns")
TEST_SIZE = cfg("data.test_size")
VAL_SIZE = cfg("data.validation_size")
RANDOM_STATE = cfg("data.random_state")
LIMIT = cfg("data.limit")

EVAL_CFG = cfg("evaluation.threshold_sweep")

# Paths
MODEL_PATH = resolve_path("NN_MODEL_PATH", os.path.join(cfg("model.output_dir"), cfg("model.checkpoint")))
SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.scaler")))
USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.user_map")))
PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH", os.path.join(cfg("preprocessing.output_dir"),
                                                                      cfg("preprocessing.artifacts.preprocessor")))

# MLFlow Config
ENABLE_MLFLOW = cfg("mlflow.enable")
MLFLOW_TRACKING_URI = cfg("mlflow.tracking_uri")
MLFLOW_REGISTRY_URI = cfg("mlflow.registry_uri")
MLFLOW_EXPERIMENT_PATH = cfg("mlflow.experiment_path")
UC_CATALOG = cfg("mlflow.uc_catalog")
UC_SCHEMA = cfg("mlflow.uc_schema")
UC_MODEL_NAME = cfg("mlflow.uc_model_name")
FULL_UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

if ENABLE_MLFLOW and not MLFLOW_EXPERIMENT_PATH:
    raise RuntimeError("ENABLE_MLFLOW is true, but MLFLOW_EXPERIMENT_PATH is not set.")

if not ENABLE_MLFLOW:
    print("MLFlow is disabled. Files will be saved locally only. Consider setting up MLFlow for reproducibility.")
    mlflow.log_param = lambda *args, **kwargs: None
    mlflow.log_metric = lambda *args, **kwargs: None
    mlflow.log_artifacts = lambda *args, **kwargs: None
    mlflow.set_experiment = lambda *args, **kwargs: None


def load_training_data(limit: int):
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, nn_score, platform,
               human_verified, impossible_travel
        FROM {cfg("data.table")}
        WHERE nn_score >= 0.0
        ORDER BY random()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
        df["human_verified"] = df.get("human_verified", False).fillna(False).astype(bool)
        df["impossible_travel"] = df.get("impossible_travel", False).fillna(False).astype(bool)
        mlflow.log_param("source_query", str(query))
        mlflow.log_metric("n_rows", len(df))
    print(f"Loaded {len(df)} rows.")
    return df


def preprocess_training_data(df):
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(df, FEATURE_COLUMNS)
    df = preprocessor.transform_df(df, FEATURE_COLUMNS)

    # Encode usernames
    unique_users = df["username"].unique()
    user_to_id = {u: i for i, u in enumerate(unique_users)}
    df["user_id"] = df["username"].map(user_to_id)

    # Ground truth
    df["is_true_threat"] = (
            (df["nn_score"] == 1.0) & (df["human_verified"] == True)
            | (df["impossible_travel"] == True)
    ).astype(int)

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df["is_true_threat"].astype(np.float32)
    user_ids = df["user_id"].astype(np.int64)
    true_labels = df["is_true_threat"].values

    # Split 1: Separate the holdout test set (model doesn't see this)
    X_temp, X_test, y_temp, y_test, user_temp, user_test, true_temp, true_labels_test = train_test_split(
        X, y, user_ids, true_labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=true_labels
    )

    # Split 2: Separate train from validation (Model sees val for early stopping)
    X_train, X_val, y_train, y_val, user_train, user_val, true_labels_train, true_labels_val = train_test_split(
        X_temp, y_temp, user_temp, true_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=true_temp
    )

    print(f"Split Sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Normalize numeric features
    scaler = StandardScaler()

    # Fit on train
    X_train_scaled = scaler.fit_transform(X_train.values)

    # Apply that same math to validation and test sets
    X_val_scaled = scaler.transform(X_val.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Convert to tensors
    train_tensors = (
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(user_train.values, dtype=torch.long)
    )

    val_tensors = (
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(user_val.values, dtype=torch.long),
        true_labels_val  # Keep raw labels for F1 calc
    )

    test_tensors = (
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1),
        torch.tensor(user_test.values, dtype=torch.long),
        true_labels_test  # Keep raw labels for F1 calc
    )

    return (
        train_tensors, val_tensors, test_tensors,
        scaler, len(unique_users), user_to_id, preprocessor
    )


def train_neural_network(train_tensors, val_tensors, num_users, checkpoint_path):
    X_train, y_train, user_train = train_tensors
    X_val, y_val, user_val, _ = val_tensors  # Ignore true_labels for training loop

    raw = np.log2(num_users) if cfg("model.embed_dim_scale") == "log2" else cfg("model.embed_dim_scale")
    embed_dim = int(min(max(raw, cfg("model.min_embed_dim")), cfg("model.max_embed_dim")))

    min_delta = float(cfg("training.min_delta"))
    num_epochs = int(cfg("training.num_epochs"))
    learning_rate = float(cfg("training.learning_rate"))
    patience = int(cfg("training.patience"))

    mlflow.log_param("embed_dim", embed_dim)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("patience", patience)

    model = SimpleRBAModel(input_dim=X_train.shape[1], num_users=num_users, embed_dim=embed_dim).to(_device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train, y_train, user_train = X_train.to(_device), y_train.to(_device), user_train.to(_device)
    X_val, y_val, user_val = X_val.to(_device), y_val.to(_device), user_val.to(_device)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for up to {num_epochs} epochs...")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        loss = criterion(model(X_train, user_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val, user_val), y_val)

        mlflow.log_metric("train_loss", loss.item(), step=epoch)
        mlflow.log_metric("val_loss", val_loss.item(), step=epoch)

        if val_loss.item() < best_val_loss - min_delta:
            best_val_loss = val_loss.item()
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        # Only print every 5 to the console
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} "
                  f"| Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs (best val loss = {best_val_loss:.5f}).")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    print(f"Training complete. Best model restored from {checkpoint_path}.")
    return model


def find_best_threshold(model, val_tensors):
    """
    Tunes the threshold using the VALIDATION set.
    """
    X_val, _, user_val, true_labels = val_tensors

    model.eval()
    with torch.no_grad():
        nn_preds = model(X_val.to(_device), user_val.to(_device)).cpu().numpy().flatten()

    # Apply ensemble logic (if you have other logic, it goes here)
    ensemble_preds = np.array([ensemble_threat_score(nn, 0) for nn in nn_preds])

    thresholds = np.linspace(EVAL_CFG["start"], EVAL_CFG["end"], EVAL_CFG["steps"])
    f1_scores = []

    for t in thresholds:
        preds_bin = (ensemble_preds >= t).astype(int)
        f1_scores.append(f1_score(true_labels, preds_bin))

    best_idx = int(np.argmax(f1_scores))
    best_t = thresholds[best_idx]

    mlflow.log_param("best_threshold", best_t)
    mlflow.log_metric("val_best_f1", f1_scores[best_idx])

    try:
        out_dir = cfg("preprocessing.output_dir")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, cfg("preprocessing.artifacts.threshold"))

        with open(out_path, "w") as f:
            f.write(f"{best_t:.6f}\n")
    except Exception as e:
        logging.error(f"Failed to write threshold file: {e}")

    return best_t


def evaluate_on_test(model, test_tensors, threshold):
    """
    Final evaluation on the test set.
    """
    X_test, _, user_test, true_labels = test_tensors

    model.eval()
    with torch.no_grad():
        nn_preds = model(X_test.to(_device), user_test.to(_device)).cpu().numpy().flatten()

    ensemble_preds = np.array([ensemble_threat_score(nn, 0) for nn in nn_preds])

    # Apply the threshold found during validation
    final_preds = (ensemble_preds >= threshold).astype(int)

    # Calculate final metrics
    test_f1 = f1_score(true_labels, final_preds)
    test_precision = precision_score(true_labels, final_preds, zero_division=0)
    test_recall = recall_score(true_labels, final_preds, zero_division=0)

    print(f"\nTest evaluation results:")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}\n")

    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)


def main():
    if ENABLE_MLFLOW:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_PATH)
        client = MlflowClient()
        ctx = mlflow.start_run()
    else:
        client = None
        ctx = nullcontext()

    with ctx:
        df = load_training_data(LIMIT)

        (train_tensors, val_tensors, test_tensors, scaler, num_users, user_to_id,
         preprocessor) = preprocess_training_data(df)

        model = train_neural_network(train_tensors, val_tensors, num_users, checkpoint_path=MODEL_PATH)

        # Save artifacts
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(user_to_id, USER_MAP_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)

        best_threshold = find_best_threshold(model, val_tensors)
        print(f"Best threshold (determined on Val): {best_threshold:.3f}")

        evaluate_on_test(model, test_tensors, best_threshold)

        # MLFlow model registration
        if ENABLE_MLFLOW:
            mlflow.log_artifacts(cfg("preprocessing.output_dir", "nn_data"), artifact_path="artifacts")

            # Need this for the signature validation in MLFlow. It just gives MLFlow the shape of our data
            example_input = pd.DataFrame(
                [np.zeros(len(FEATURE_COLUMNS))],
                columns=FEATURE_COLUMNS
            )

            model_info = mlflow.pytorch.log_model(
                pytorch_model=model,
                input_example=example_input
            )

            try:
                mlflow.register_model(model_uri=model_info.model_uri, name=FULL_UC_MODEL_NAME)
            except Exception:
                client.create_registered_model(FULL_UC_MODEL_NAME)
                mlflow.register_model(model_uri=model_info.model_uri, name=FULL_UC_MODEL_NAME)

            versions = client.search_model_versions(f"name='{FULL_UC_MODEL_NAME}'")
            latest_version = max(int(v.version) for v in versions)

            print(f"Model registered: {FULL_UC_MODEL_NAME}, version={latest_version}")


if __name__ == "__main__":
    main()
