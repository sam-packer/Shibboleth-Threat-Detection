import logging
import threading
import time
from mlflow import MlflowClient
import mlflow
from helpers.globals import cfg

MODEL_CACHE = {
    "models": [],
    "last_update": None,
}

MODEL_CACHE_LOCK = threading.Lock()

ENABLE_MLFLOW = cfg("mlflow.enable")
MLFLOW_TRACKING_URI = cfg("mlflow.tracking_uri")
MLFLOW_REGISTRY_URI = cfg("mlflow.registry_uri")

UC_CATALOG = cfg("mlflow.uc_catalog")
UC_SCHEMA = cfg("mlflow.uc_schema")
UC_MODEL_NAME = cfg("mlflow.uc_model_name")

FULL_UC_MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.{UC_MODEL_NAME}"


def fetch_models_from_mlflow():
    try:
        if not ENABLE_MLFLOW:
            raise RuntimeError("MLFlow is not enabled")

        if not MLFLOW_TRACKING_URI or not UC_MODEL_NAME:
            raise RuntimeError("MLFlow configuration missing")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()

        versions = client.search_model_versions(f"name='{FULL_UC_MODEL_NAME}'")
        if not versions:
            raise RuntimeError("No model versions found")

        results = []

        for v in versions:
            version_num = int(v.version)

            run = client.get_run(v.run_id)
            metrics = run.data.metrics or {}

            threshold = metrics.get("global_anomaly_threshold")

            try:
                threshold = float(threshold) if threshold is not None else None
            except:
                threshold = None

            results.append({
                "version": version_num,
                "anomaly_threshold": threshold,
            })

        return sorted(results, key=lambda x: x["version"])

    except Exception as e:
        return e


def refresh_model_cache():
    data = fetch_models_from_mlflow()

    if isinstance(data, Exception):
        logging.error(f"[Models] Refresh failed: {data}")
        return False

    new_cache = {
        "models": data,
        "last_update": time.time(),
    }

    # Atomic swap
    with MODEL_CACHE_LOCK:
        MODEL_CACHE.update(new_cache)

    return True


def _async_refresh_wrapper():
    refresh_model_cache()


def initialize_model_cache():
    success = refresh_model_cache()
    if not success:
        logging.error("[Models] Initial warm-load failed. Using empty cache.")
