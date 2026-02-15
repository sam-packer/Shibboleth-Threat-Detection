import os
import logging
import math

import mlflow
import numpy as np
import torch
import joblib
from mlflow import MlflowClient
from dotenv import load_dotenv

from helpers.globals import resolve_path, select_device, cfg
from sklearn.preprocessing import StandardScaler

from nn_scripts.feature_preprocessor import FeaturePreprocessor
from nn_scripts.model import BehavioralEncoder

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

# File based environment
MODEL_PATH = resolve_path("NN_MODEL_PATH",
                          os.path.join(cfg("model.output_dir"), cfg("model.checkpoint"))
                          )

SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(cfg("model.output_dir"),
                                        cfg("preprocessing.artifacts.scaler"))
                           )

USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(cfg("model.output_dir"),
                                          cfg("preprocessing.artifacts.user_map"))
                             )

PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH",
                                 os.path.join(cfg("model.output_dir"),
                                              cfg("preprocessing.artifacts.preprocessor"))
                                 )

CENTROIDS_PATH = resolve_path("NN_CENTROIDS_PATH",
                              os.path.join(cfg("model.output_dir"),
                                           cfg("preprocessing.artifacts.centroids"))
                              )

DISTANCE_STATS_PATH = resolve_path("NN_DISTANCE_STATS_PATH",
                                   os.path.join(cfg("model.output_dir"),
                                                cfg("preprocessing.artifacts.distance_stats"))
                                   )

# MLFlow environment
ENABLE_MLFLOW = cfg("mlflow.enable")
MLFLOW_TRACKING_URI = cfg("mlflow.tracking_uri")
MLFLOW_REGISTRY_URI = cfg("mlflow.registry_uri")
UC_CATALOG = cfg("mlflow.uc_catalog")
UC_SCHEMA = cfg("mlflow.uc_schema")
UC_MODEL_NAME = cfg("mlflow.uc_model_name")
FULL_UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

_device = select_device()
_model = None
_model_version: int | None = None
_scaler: StandardScaler | None = None
_user_to_id = {}
_preprocessor: FeaturePreprocessor | None = None
_centroids: dict | None = None
_distance_stats: dict | None = None


def _load_from_mlflow():
    global _model, _model_version, _scaler, _user_to_id, _preprocessor, _centroids, _distance_stats

    if not ENABLE_MLFLOW:
        return False

    if not MLFLOW_TRACKING_URI or not UC_MODEL_NAME:
        logging.warning("[NN] MLFlow enabled but URIs not set. Skipping MLFlow load.")
        return False

    logging.info(f"[NN] Attempting to load model '{FULL_UC_MODEL_NAME}' from MLFlow...")

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{FULL_UC_MODEL_NAME}'")
        if not versions:
            logging.warning(f"[NN] No versions found for {FULL_UC_MODEL_NAME} in MLFlow.")
            return False

        latest_version_obj = max(versions, key=lambda v: int(v.version))
        run_id = latest_version_obj.run_id
        version_num = latest_version_obj.version

        logging.info(f"[NN] Found Version {version_num} (Run ID: {run_id})")

        local_artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts")

        scaler_p = os.path.join(local_artifacts_dir, "rba_scaler.pkl")
        preproc_p = os.path.join(local_artifacts_dir, "rba_preprocessor.pkl")
        usermap_p = os.path.join(local_artifacts_dir, "rba_user_map.pkl")
        centroids_p = os.path.join(local_artifacts_dir, "rba_centroids.pkl")
        dist_stats_p = os.path.join(local_artifacts_dir, "rba_distance_stats.pkl")

        _scaler = joblib.load(scaler_p)
        _preprocessor = joblib.load(preproc_p)
        _user_to_id = joblib.load(usermap_p)
        _centroids = joblib.load(centroids_p)
        _distance_stats = joblib.load(dist_stats_p)

        model_uri = f"models:/{FULL_UC_MODEL_NAME}/{version_num}"
        loaded_model = mlflow.pytorch.load_model(model_uri, map_location=_device)

        _model = loaded_model.to(_device)
        _model.eval()
        _model_version = int(version_num)

        logging.info(f"[NN] Successfully loaded from MLFlow (Ver: {version_num}). "
                     f"Users: {len(_user_to_id)}, Centroids: {len(_centroids) - 1}")
        return True

    except Exception as e:
        logging.error(f"[NN] MLFlow load failed: {e}. Falling back to local files.")
        return False


def _load_from_local():
    global _model, _scaler, _user_to_id, _preprocessor, _centroids, _distance_stats

    logging.info("[NN] Loading from local file system...")

    try:
        _scaler = joblib.load(SCALER_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _user_to_id = joblib.load(USER_MAP_PATH)
        _centroids = joblib.load(CENTROIDS_PATH)
        _distance_stats = joblib.load(DISTANCE_STATS_PATH)
    except FileNotFoundError as e:
        logging.error(f"[NN] CRITICAL: Missing local artifact: {e}")
        return

    input_dim = len(cfg("data.feature_columns"))
    _model = BehavioralEncoder(input_dim=input_dim)
    try:
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: Model file not found at {MODEL_PATH}")
        _model = None
        return

    _model.to(_device)
    _model.eval()
    logging.info(f"[NN] Local model loaded. Users: {len(_user_to_id)} | Centroids: {len(_centroids) - 1}")


def get_model_version() -> int | None:
    return _model_version


def load_model_and_scaler():
    global _model, _scaler, _preprocessor

    if _model is not None and _scaler is not None and _preprocessor is not None:
        return

    success = _load_from_mlflow()

    if not success:
        _load_from_local()


def _distance_to_score(distance: float, stats: dict) -> float:
    mean_d = stats["mean"]
    p95 = stats["p95"]

    if distance <= mean_d:
        return 0.05

    if distance <= p95:
        # Linear ramp from 0.05 to 0.50
        t = (distance - mean_d) / (p95 - mean_d + 1e-8)
        return 0.05 + t * 0.45

    # Exponential escalation toward 1.0 for distance > p95
    excess = (distance - p95) / (p95 + 1e-8)
    score = 0.50 + 0.50 * (1.0 - math.exp(-3.0 * excess))
    return min(score, 1.0)


def compute_anomaly_score(username: str, device_category: str, features: dict) -> float:
    try:
        load_model_and_scaler()

        if _model is None or _scaler is None or _preprocessor is None or _centroids is None:
            logging.error("[NN] Model/scaler/preprocessor/centroids not loaded. Returning neutral score.")
            return 0.5

        features = _preprocessor.transform_single(features, cfg("data.feature_columns"))

        # Prepare input tensor
        X = np.array([[features[col] for col in cfg("data.feature_columns")]], dtype=np.float32)
        X_scaled = _scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(_device)

        # Forward pass to get embedding
        with torch.no_grad():
            embedding = _model(X_tensor).cpu().numpy().flatten()

        # Three-level centroid lookup:
        # 1. (username, device_category) — most specific
        # 2. username — user-level fallback
        # 3. __population__ — final fallback
        device_key = (username, device_category)
        if device_key in _centroids:
            centroid = _centroids[device_key]
            stats = _distance_stats.get(device_key, _distance_stats["__population__"])
        elif username in _centroids:
            centroid = _centroids[username]
            stats = _distance_stats.get(username, _distance_stats["__population__"])
        else:
            centroid = _centroids["__population__"]
            stats = _distance_stats["__population__"]

        # Cosine distance (for L2-normalized vectors)
        distance = float(1.0 - np.dot(embedding, centroid))
        distance = max(0.0, min(distance, 2.0))

        score = _distance_to_score(distance, stats)
        return max(0.0, min(score, 1.0))

    except Exception as e:
        logging.error(f"[NN] Inference failed: {e}", exc_info=True)
        return 0.5


def test_inference():
    dummy_input = {col: np.random.rand() for col in cfg("data.feature_columns")}
    score = compute_anomaly_score("demo_user", "desktop", dummy_input)
    print(f"NN Score for demo_user: {score:.4f}")


if __name__ == "__main__":
    test_inference()
