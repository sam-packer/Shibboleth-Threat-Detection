import os
import logging

import mlflow
import numpy as np
import torch
import joblib
from mlflow import MlflowClient
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from helpers.globals import CONFIG, resolve_path, FEATURE_COLUMNS, select_device
from sklearn.preprocessing import StandardScaler

from nn_scripts.feature_preprocessor import FeaturePreprocessor
from nn_scripts.model import SimpleRBAModel

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")

# File based environment
MODEL_PATH = resolve_path("NN_MODEL_PATH",
                          os.path.join(CONFIG["model"]["output_dir"], CONFIG["model"]["checkpoint"])
                          )

SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(CONFIG["preprocessing"]["output_dir"],
                                        CONFIG["preprocessing"]["artifacts"]["scaler"])
                           )

USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(CONFIG["preprocessing"]["output_dir"],
                                          CONFIG["preprocessing"]["artifacts"]["user_map"])
                             )

PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH",
                                 os.path.join(CONFIG["preprocessing"]["output_dir"],
                                              CONFIG["preprocessing"]["artifacts"]["preprocessor"])
                                 )
USER_THRESHOLD = CONFIG["model"]["min_user_events"]

# MLFlow environment
MLFLOW_CFG = CONFIG["mlflow"]

ENABLE_MLFLOW = MLFLOW_CFG["enable"]
MLFLOW_TRACKING_URI = MLFLOW_CFG["tracking_uri"]
MLFLOW_REGISTRY_URI = MLFLOW_CFG["registry_uri"]
UC_CATALOG = MLFLOW_CFG["uc_catalog"]
UC_SCHEMA = MLFLOW_CFG["uc_schema"]
UC_MODEL_NAME = MLFLOW_CFG["uc_model_name"]
FULL_UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

_device = select_device()
_model = None
_scaler: StandardScaler | None = None
_user_to_id = {}
_preprocessor: FeaturePreprocessor | None = None


def _load_from_mlflow():
    """
    Attempts to load the Model and auxiliary artifacts from MLFlow Registry.
    Returns True if successful, False otherwise.
    """
    global _model, _scaler, _user_to_id, _num_users, _preprocessor, _embed_dim

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

        # sort by version number to get the latest
        latest_version_obj = max(versions, key=lambda v: int(v.version))
        run_id = latest_version_obj.run_id
        version_num = latest_version_obj.version

        logging.info(f"[NN] Found Version {version_num} (Run ID: {run_id})")

        local_artifacts_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="artifacts")

        scaler_p = os.path.join(local_artifacts_dir, "rba_scaler.pkl")
        preproc_p = os.path.join(local_artifacts_dir, "rba_preprocessor.pkl")
        usermap_p = os.path.join(local_artifacts_dir, "rba_user_map.pkl")

        _scaler = joblib.load(scaler_p)
        _preprocessor = joblib.load(preproc_p)
        _user_to_id = joblib.load(usermap_p)
        _num_users = len(_user_to_id)

        embed_cfg = CONFIG["model"]

        if embed_cfg["embed_dim_scale"] == "log2":
            raw = np.log2(_num_users)
        else:
            raw = embed_cfg["embed_dim_scale"]

        _embed_dim = int(min(max(raw, embed_cfg["min_embed_dim"]), embed_cfg["max_embed_dim"]))

        model_uri = f"models:/{FULL_UC_MODEL_NAME}/{version_num}"
        loaded_model = mlflow.pytorch.load_model(model_uri, map_location=_device)

        _model = loaded_model.to(_device)
        _model.eval()

        logging.info(f"[NN] Successfully loaded from MLFlow (Ver: {version_num}). Users: {_num_users}")
        return True

    except Exception as e:
        logging.error(f"[NN] MLFlow load failed: {e}. Falling back to local files.")
        return False


def _load_from_local():
    """
    Falls back to loading from local file paths defined in .env.
    """
    global _model, _scaler, _user_to_id, _num_users, _preprocessor, _embed_dim

    logging.info("[NN] Loading from local file system...")

    try:
        _scaler = joblib.load(SCALER_PATH)
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        _user_to_id = joblib.load(USER_MAP_PATH)
    except FileNotFoundError as e:
        logging.error(f"[NN] CRITICAL: Missing local artifact: {e}")
        return

    _num_users = len(_user_to_id)
    if _num_users == 0:
        logging.error("[NN] User map is empty.")
        return

    embed_cfg = CONFIG["model"]

    if embed_cfg["embed_dim_scale"] == "log2":
        raw = np.log2(_num_users)
    else:
        raw = embed_cfg["embed_dim_scale"]

    _embed_dim = int(min(max(raw, embed_cfg["min_embed_dim"]), embed_cfg["max_embed_dim"]))

    _model = SimpleRBAModel(input_dim=len(FEATURE_COLUMNS), num_users=_num_users, embed_dim=_embed_dim)
    try:
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: Model file not found at {MODEL_PATH}")
        _model = None
        return

    _model.to(_device)
    _model.eval()
    logging.info(f"[NN] Local model loaded. Users: {_num_users} | Embed dim: {_embed_dim}")


def load_model_and_scaler():
    """
    Main entry point to load resources. Tries MLFlow first, then Local.
    """
    global _model, _scaler, _preprocessor

    if _model is not None and _scaler is not None and _preprocessor is not None:
        return

    success = _load_from_mlflow()

    if not success:
        _load_from_local()


def user_has_sufficient_data(username: str) -> bool:
    """
    Check if a given user has enough login history for a personalized embedding.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"""
                     SELECT COUNT(*) AS n
                     FROM {CONFIG["data"]["table"]}
                     WHERE username = :username
                       AND nn_score >= 0.0
                     """),
                {"username": username}
            ).fetchone()

        count = result.n if result else 0
        return count >= USER_THRESHOLD
    except Exception as e:
        logging.warning(f"[NN] Failed to check user data sufficiency: {e}")
        return False


def compute_nn_score(username: str, features: dict) -> float:
    """
    Compute neural network threat score for a single login event.
    If user does not have sufficient data, defaults to the global model.
    """
    try:
        load_model_and_scaler()

        if _model is None or _scaler is None or _preprocessor is None:
            logging.error("[NN] Model/scaler/preprocessor not loaded. Returning neutral score.")
            return 0.5

        features = _preprocessor.transform_single(features, FEATURE_COLUMNS)

        # Prepare input tensor
        X = np.array([[features[col] for col in FEATURE_COLUMNS]], dtype=np.float32)
        X_scaled = _scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Determine if this user qualifies for an embedding
        if username in _user_to_id and user_has_sufficient_data(username):
            user_id = torch.tensor([_user_to_id[username]], dtype=torch.long)
        else:
            user_id = None  # use global model path

        # Forward pass
        with torch.no_grad():
            y_pred = _model(X_tensor, user_id)
            score = float(y_pred.item())

        # Clamp and smooth the score
        score = max(0.0, min(score, 1.0))
        return score

    except Exception as e:
        logging.error(f"[NN] Inference failed: {e}", exc_info=True)
        return 0.5


def test_inference():
    """
    Local manual test utility.
    """
    dummy_input = {col: np.random.rand() for col in FEATURE_COLUMNS}
    score = compute_nn_score("demo_user", dummy_input)
    print(f"NN Score for demo_user: {score:.4f}")


if __name__ == "__main__":
    test_inference()
