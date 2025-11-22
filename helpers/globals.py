import logging
import os

import torch
import yaml


def project_root() -> str:
    """
    Returns the absolute path to the project root directory.
    This file lives in: project/helpers/globals.py
    So project root = dirname(dirname(__file__))
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config() -> dict:
    """
    Load config.yml reliably:
    - If CONFIG_PATH env var is set, use that explicitly.
    - Otherwise load config.yml from the project root.
    """
    # env override
    override = os.getenv("CONFIG_PATH")
    if override and os.path.exists(override):
        with open(override, "r") as f:
            return yaml.safe_load(f)

    # default: project root/config.yml
    default_path = os.path.join(project_root(), "config.yml")

    if not os.path.exists(default_path):
        raise FileNotFoundError(f"config.yml not found at: {default_path}")

    with open(default_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(env_var_name: str, default_relative: str) -> str:
    """
    Resolve a relative or configured path safely.

    Priority:
      1. Environment variable override (absolute or relative).
      2. Resolve relative to project root if directory exists.
      3. Fall back to default_relative unchanged.
    """
    # environment override
    env_val = os.getenv(env_var_name)
    if env_val:
        # normalize user-provided paths (resolve relative env paths too)
        if not os.path.isabs(env_val):
            return os.path.abspath(os.path.join(project_root(), env_val))
        return env_val

    # resolve under project root
    root = project_root()
    resolved = os.path.abspath(os.path.join(root, default_relative))

    # check the directory
    parent_dir = os.path.dirname(resolved)
    if os.path.isdir(parent_dir):
        return resolved

    # fallback to original string if directory doesn't exist
    return default_relative


def select_device() -> torch.device:
    """
    Determine the correct PyTorch device using configuration + auto detection.

    Config behavior:
      - If CONFIG["runtime"]["device"] == "cpu" | "cuda" | "mps":
            Try to use it. If unavailable, fall back to auto.
      - If CONFIG["runtime"]["device"] == "auto":
            Use the best available device.

    Auto selection priority:
      1. CUDA (NVIDIA)
      2. MPS  (Apple Silicon)
      3. CPU  (fallback)
    """
    cfg_device = CONFIG["runtime"]["device"].lower()

    # config override
    if cfg_device != "auto":
        try:
            dev = torch.device(cfg_device)

            # validate availability
            if cfg_device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested, but no CUDA device is available.")

            if cfg_device == "mps" and not (hasattr(torch.backends, "mps")
                                            and torch.backends.mps.is_available()):
                raise RuntimeError("MPS requested, but MPS is not available.")

            logging.info(f"[NN] Using device from config: {cfg_device}")
            return dev

        except Exception as e:
            logging.error(f"[NN] Config-selected device '{cfg_device}' unavailable: {e}")
            logging.info("[NN] Falling back to auto device selection.")

    # auto select
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logging.info(f"[NN] Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
        return dev

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        logging.info("[NN] Auto-selected MPS device (Apple Silicon)")
        return dev

    dev = torch.device("cpu")
    logging.info("[NN] Auto-selected CPU device")
    return dev


# load config globally for application
CONFIG = load_config()

# keep this variable for compatibility
FEATURE_COLUMNS = CONFIG["data"]["feature_columns"]
