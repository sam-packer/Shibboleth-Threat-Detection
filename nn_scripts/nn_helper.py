import os
import logging
import numpy as np
import torch
import joblib
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from feature_preprocessor import FeaturePreprocessor
from helpers.globals import FEATURE_COLUMNS
from model import SimpleRBAModel
from sklearn.preprocessing import StandardScaler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
MODEL_PATH = os.getenv("NN_MODEL_PATH", "../nn_data/best_rba_model.pt")
SCALER_PATH = os.getenv("NN_SCALER_PATH", "../nn_data/rba_scaler.pkl")
USER_THRESHOLD = int(os.getenv("NN_MIN_LOGINS", 10))
USER_MAP_PATH = os.getenv("NN_USER_MAP_PATH", "../nn_data/rba_user_map.pkl")
PREPROCESSOR_PATH = os.getenv("NN_PREPROCESSOR_PATH", "../nn_data/rba_preprocessor.pkl")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)


def get_best_device():
    """
    Choose the best available PyTorch device (CUDA, MPS, or CPU).
    Priority order:
      1. CUDA (NVIDIA GPUs)
      2. MPS  (Apple Silicon / Metal)
      3. CPU  (fallback)
    """
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logging.info(f"[NN] Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logging.info("[NN] Using Apple MPS backend")
    else:
        dev = torch.device("cpu")
        logging.info("[NN] Using CPU backend")
    return dev


_device = get_best_device()
_model = None
_scaler: StandardScaler | None = None
_user_to_id = {}
_num_users = 0
_embed_dim = 0
_preprocessor: FeaturePreprocessor | None = None


def load_model_and_scaler():
    """
    Load trained neural network, scaler, and user embedding metadata.
    """
    global _model, _scaler, _user_to_id, _num_users, _embed_dim, _preprocessor

    if _model is not None and _scaler is not None and _preprocessor is not None:
        return

    logging.info("[NN] Loading model and scaler...")

    try:
        _scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: Scaler file not found at {SCALER_PATH}")
        return

    try:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: Preprocessor file not found at {PREPROCESSOR_PATH}")
        return

    try:
        _user_to_id = joblib.load(USER_MAP_PATH)
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: User map file not found at {USER_MAP_PATH}")
        _user_to_id = {}

    _num_users = len(_user_to_id)
    if _num_users == 0:
        logging.error("[NN] User map is empty or failed to load. Model cannot be initialized.")
        return

    _embed_dim = int(min(max(np.log2(_num_users), 4), 64))

    _model = SimpleRBAModel(input_dim=len(FEATURE_COLUMNS), num_users=_num_users, embed_dim=_embed_dim)
    try:
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    except FileNotFoundError:
        logging.error(f"[NN] CRITICAL: Model file not found at {MODEL_PATH}")
        _model = None
        return

    _model.eval()
    logging.info(f"[NN] Model ready. Users: {_num_users} | Embed dim: {_embed_dim}")


def user_has_sufficient_data(username: str) -> bool:
    """
    Check if a given user has enough login history for a personalized embedding.
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                     SELECT COUNT(*) AS n
                     FROM rba_login_event
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
