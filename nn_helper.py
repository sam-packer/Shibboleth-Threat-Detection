import os
import logging
import numpy as np
import torch
import joblib
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from model import SimpleRBAModel
from sklearn.preprocessing import StandardScaler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
MODEL_PATH = os.getenv("NN_MODEL_PATH", "best_rba_model.pt")
SCALER_PATH = os.getenv("NN_SCALER_PATH", "rba_scaler.pkl")
USER_THRESHOLD = int(os.getenv("NN_MIN_LOGINS", 10))

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


def load_model_and_scaler():
    """
    Load trained neural network, scaler, and user embedding metadata.
    """
    global _model, _scaler, _user_to_id, _num_users, _embed_dim

    if _model is not None and _scaler is not None:
        return  # already loaded

    logging.info("[NN] Loading model and scaler...")

    _scaler = joblib.load(SCALER_PATH)

    # Fetch user mapping from DB (users seen in training)
    with engine.connect() as conn:
        df_users = pd.read_sql(text("SELECT DISTINCT username FROM rba_login_event WHERE nn_score >= 0.0"), conn)
        _user_to_id = {u: i for i, u in enumerate(df_users["username"].tolist())}
        _num_users = len(_user_to_id)
        _embed_dim = int(min(max(np.log2(max(_num_users, 1)), 4), 64))

    _model = SimpleRBAModel(input_dim=len(FEATURE_COLUMNS), num_users=_num_users, embed_dim=_embed_dim)
    _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
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

        # Ensure all required features are present
        missing = [col for col in FEATURE_COLUMNS if col not in features]
        if missing:
            logging.warning(f"[NN] Missing features: {missing}")
            for col in missing:
                features[col] = 0.0  # fallback safe default

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
        return 0.5  # safe neutral fallback


def test_inference():
    """
    Local manual test utility.
    """
    dummy_input = {col: np.random.rand() for col in FEATURE_COLUMNS}
    score = compute_nn_score("demo_user", dummy_input)
    print(f"NN Score for demo_user: {score:.4f}")


if __name__ == "__main__":
    test_inference()
