import logging
import os
from contextlib import nullcontext

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import joblib

from nn_scripts.feature_preprocessor import FeaturePreprocessor
from helpers.globals import select_device, resolve_path, cfg
from nn_scripts.model import BehavioralEncoder

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

_device = select_device()

FEATURE_COLUMNS = cfg("data.feature_columns")
VAL_SIZE = cfg("data.validation_size")
RANDOM_STATE = cfg("data.random_state")
LIMIT = cfg("data.limit")
MIN_USER_EVENTS = cfg("model.min_user_events")
MIN_DEVICE_EVENTS = cfg("model.min_device_events")

# Paths
MODEL_PATH = resolve_path("NN_MODEL_PATH", os.path.join(cfg("model.output_dir"), cfg("model.checkpoint")))
SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.scaler")))
USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.user_map")))
PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH", os.path.join(cfg("preprocessing.output_dir"),
                                                                      cfg("preprocessing.artifacts.preprocessor")))
CENTROIDS_PATH = resolve_path("NN_CENTROIDS_PATH",
                              os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.centroids")))
DISTANCE_STATS_PATH = resolve_path("NN_DISTANCE_STATS_PATH",
                                   os.path.join(cfg("preprocessing.output_dir"),
                                                cfg("preprocessing.artifacts.distance_stats")))

os.makedirs(cfg("preprocessing.output_dir"), exist_ok=True)
os.makedirs(cfg("model.output_dir"), exist_ok=True)

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


def set_global_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_global_seeds(RANDOM_STATE)


class TripletDataset(Dataset):
    def __init__(self, features: np.ndarray, user_ids: np.ndarray):
        self.features = features.astype(np.float32)
        self.user_ids = user_ids

        # Build per-user index for O(1) positive/negative sampling
        self.user_indices: dict[int, list[int]] = {}
        for idx, uid in enumerate(user_ids):
            self.user_indices.setdefault(uid, []).append(idx)

        self.all_users = list(self.user_indices.keys())

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        anchor = self.features[idx]
        anchor_user = self.user_ids[idx]

        # Positive: same user, different sample
        pos_candidates = self.user_indices[anchor_user]
        pos_idx = idx
        if len(pos_candidates) > 1:
            while pos_idx == idx:
                pos_idx = pos_candidates[np.random.randint(len(pos_candidates))]
        positive = self.features[pos_idx]

        # Negative: different user
        neg_user = anchor_user
        while neg_user == anchor_user:
            neg_user = self.all_users[np.random.randint(len(self.all_users))]
        neg_candidates = self.user_indices[neg_user]
        neg_idx = neg_candidates[np.random.randint(len(neg_candidates))]
        negative = self.features[neg_idx]

        return (
            torch.tensor(anchor, dtype=torch.float32),
            torch.tensor(positive, dtype=torch.float32),
            torch.tensor(negative, dtype=torch.float32),
        )


def load_training_data(limit: int):
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, platform, device_category
        FROM {cfg("data.table")}
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
        df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        mlflow.log_param("source_query", str(query))
        mlflow.log_metric("n_rows", len(df))
    print(f"Loaded {len(df)} rows.")
    return df


def preprocess_training_data(df):
    preprocessor = FeaturePreprocessor()
    preprocessor.fit(df, FEATURE_COLUMNS)
    df = preprocessor.transform_df(df, FEATURE_COLUMNS)

    # Filter to users with enough events
    user_counts = df["username"].value_counts()
    eligible_users = user_counts[user_counts >= MIN_USER_EVENTS].index
    df = df[df["username"].isin(eligible_users)].reset_index(drop=True)
    print(f"After filtering to users with >= {MIN_USER_EVENTS} events: {len(df)} rows, {len(eligible_users)} users.")

    # Encode usernames
    unique_users = df["username"].unique()
    user_to_id = {u: i for i, u in enumerate(unique_users)}
    df["user_id"] = df["username"].map(user_to_id)

    X = df[FEATURE_COLUMNS].astype(np.float32)
    user_ids = df["user_id"].astype(np.int64).values
    device_categories = df["device_category"].values  # may contain None/NaN

    # Normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    return X_scaled, user_ids, device_categories, scaler, user_to_id, preprocessor


def train_encoder(X_scaled, user_ids, checkpoint_path):
    batch_size = int(cfg("training.batch_size", 256))
    triplet_margin = float(cfg("training.triplet_margin", 0.3))
    num_epochs = int(cfg("training.num_epochs"))
    learning_rate = float(cfg("training.learning_rate"))
    patience = int(cfg("training.patience"))
    min_delta = float(cfg("training.min_delta"))

    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("triplet_margin", triplet_margin)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("patience", patience)

    # Split into train/val
    n = len(X_scaled)
    indices = np.random.permutation(n)
    val_count = int(n * VAL_SIZE)
    val_idx, train_idx = indices[:val_count], indices[val_count:]

    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    uid_train, uid_val = user_ids[train_idx], user_ids[val_idx]

    train_dataset = TripletDataset(X_train, uid_train)
    val_dataset = TripletDataset(X_val, uid_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = X_scaled.shape[1]
    model = BehavioralEncoder(input_dim=input_dim).to(_device)

    criterion = torch.nn.TripletMarginLoss(margin=triplet_margin, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for up to {num_epochs} epochs...")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for anchor, positive, negative in train_loader:
            anchor = anchor.to(_device)
            positive = positive.to(_device)
            negative = negative.to(_device)

            a_emb = model(anchor)
            p_emb = model(positive)
            n_emb = model(negative)

            loss = criterion(a_emb, p_emb, n_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validation
        model.eval()
        val_losses = []
        all_embeddings = []
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor = anchor.to(_device)
                positive = positive.to(_device)
                negative = negative.to(_device)

                a_emb = model(anchor)
                p_emb = model(positive)
                n_emb = model(negative)

                val_loss = criterion(a_emb, p_emb, n_emb)
                val_losses.append(val_loss.item())
                all_embeddings.append(a_emb.cpu())

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        # Collapse health: std of embedding norms (should stay near 1.0 for L2-normalized)
        if all_embeddings:
            cat_emb = torch.cat(all_embeddings, dim=0)
            embedding_std = cat_emb.std(dim=0).mean().item()
        else:
            embedding_std = 0.0

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("embedding_std", embedding_std, step=epoch)

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} "
                  f"| Emb Std: {embedding_std:.4f} | Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch + 1} epochs (best val loss = {best_val_loss:.5f}).")
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=_device))
    print(f"Training complete. Best model restored from {checkpoint_path}.")
    return model


def _make_centroid(embs):
    """Compute L2-normalized centroid from an array of embeddings."""
    centroid = embs.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    return centroid


def _make_distance_stats(embs, centroid):
    """Compute cosine-distance statistics for embeddings relative to a centroid."""
    distances = 1.0 - embs @ centroid
    distances = np.clip(distances, 0.0, 2.0)
    return {
        "mean": float(np.mean(distances)),
        "std": float(np.std(distances)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "count": int(len(distances)),
    }, distances


def compute_centroids(model, X_scaled, user_ids, device_categories, user_to_id):
    model.eval()
    id_to_user = {v: k for k, v in user_to_id.items()}

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(_device)

    with torch.no_grad():
        embeddings = model(X_tensor).cpu().numpy()

    centroids = {}
    distance_stats = {}
    all_distances = []

    # --- Level 1: Per-(user, device_category) centroids ---
    device_centroid_count = 0
    for uid in np.unique(user_ids):
        username = id_to_user[uid]
        user_mask = user_ids == uid

        # Find device categories for this user (exclude None/NaN)
        user_device_cats = device_categories[user_mask]
        for cat in np.unique(user_device_cats):
            if cat is None or (isinstance(cat, float) and np.isnan(cat)) or str(cat) == "nan":
                continue
            cat_str = str(cat)
            device_mask = user_mask & np.array([
                str(d) == cat_str if d is not None and not (isinstance(d, float) and np.isnan(d))
                else False
                for d in device_categories
            ])
            device_embs = embeddings[device_mask]
            if len(device_embs) < MIN_DEVICE_EVENTS:
                continue

            centroid = _make_centroid(device_embs)
            key = (username, cat_str)
            centroids[key] = centroid
            stats, dists = _make_distance_stats(device_embs, centroid)
            distance_stats[key] = stats
            device_centroid_count += 1

    # --- Level 2: Per-user centroids (all logins regardless of device) ---
    user_centroid_count = 0
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        user_embs = embeddings[mask]
        username = id_to_user[uid]

        centroid = _make_centroid(user_embs)
        centroids[username] = centroid
        stats, dists = _make_distance_stats(user_embs, centroid)
        distance_stats[username] = stats
        all_distances.extend(dists.tolist())
        user_centroid_count += 1

    # --- Level 3: Population centroid ---
    user_only_centroids = [v for k, v in centroids.items() if isinstance(k, str) and k != "__population__"]
    pop_centroid = _make_centroid(np.stack(user_only_centroids))
    centroids["__population__"] = pop_centroid

    global_threshold = float(np.percentile(all_distances, 99))

    all_dists_arr = np.array(all_distances)
    distance_stats["__population__"] = {
        "mean": float(np.mean(all_dists_arr)),
        "std": float(np.std(all_dists_arr)),
        "p95": float(np.percentile(all_dists_arr, 95)),
        "p99": global_threshold,
        "count": int(len(all_dists_arr)),
    }

    print(f"Computed {device_centroid_count} device-level centroids, {user_centroid_count} user-level centroids.")
    print(f"Global anomaly threshold (p99): {global_threshold:.6f}")
    print(f"Mean distance: {np.mean(all_distances):.6f}, Std: {np.std(all_distances):.6f}")

    mlflow.log_metric("global_anomaly_threshold", global_threshold)
    mlflow.log_metric("num_user_centroids", user_centroid_count)
    mlflow.log_metric("num_device_centroids", device_centroid_count)
    mlflow.log_metric("mean_distance", float(np.mean(all_distances)))

    return centroids, distance_stats, global_threshold


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

        X_scaled, user_ids, device_categories, scaler, user_to_id, preprocessor = preprocess_training_data(df)

        model = train_encoder(X_scaled, user_ids, checkpoint_path=MODEL_PATH)

        # Compute centroids and distance stats
        centroids, distance_stats, global_threshold = compute_centroids(model, X_scaled, user_ids, device_categories, user_to_id)

        # Save artifacts
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(user_to_id, USER_MAP_PATH)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        joblib.dump(centroids, CENTROIDS_PATH)
        joblib.dump(distance_stats, DISTANCE_STATS_PATH)

        # Write threshold file
        try:
            threshold_path = os.path.join(cfg("preprocessing.output_dir"),
                                          cfg("preprocessing.artifacts.threshold"))
            with open(threshold_path, "w") as f:
                f.write(f"{global_threshold:.6f}\n")
        except Exception as e:
            logging.error(f"Failed to write threshold file: {e}")

        print(f"Global anomaly threshold: {global_threshold:.6f}")

        # MLFlow model registration
        if ENABLE_MLFLOW:
            mlflow.log_artifacts(cfg("preprocessing.output_dir", "nn_data"), artifact_path="artifacts")

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
