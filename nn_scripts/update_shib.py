import argparse
import sys
import mlflow
from mlflow import MlflowClient

from helpers.globals import cfg
from helpers.shib_updater import update_shib_threshold


def main():
    # Ensure MLFlow is required
    enable_mlflow = cfg("mlflow.enable", False)

    if not enable_mlflow:
        print(
            "[Shib Update] Error: Updating the Shibboleth RBA file requires MLFlow to be enabled. We require this to ensure model consistency among Shibboleth and load balanced servers running the neural network.")
        sys.exit(1)

    tracking_uri = cfg("mlflow.tracking_uri")
    registry_uri = cfg("mlflow.registry_uri")
    experiment_path = cfg("mlflow.experiment_path")

    if not tracking_uri or not registry_uri or not experiment_path:
        print("[Shib Update] Error: MLFlow is not fully configured. Cannot proceed.")
        sys.exit(1)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    uc_catalog = cfg("mlflow.uc_catalog")
    uc_schema = cfg("mlflow.uc_schema")
    uc_model_name = cfg("mlflow.uc_model_name")

    full_name = f"{uc_catalog}.{uc_schema}.{uc_model_name}"

    client = MlflowClient()

    # Fetch latest version
    versions = client.search_model_versions(f"name='{full_name}'")

    if not versions:
        print(f"[Shib Update] Error: No registered models found: {full_name}")
        sys.exit(1)

    latest_version = max(int(v.version) for v in versions)
    print(f"[Shib Update] Found latest version = {latest_version}")

    # Load run metadata
    latest_ver_obj = [v for v in versions if int(v.version) == latest_version][0]

    run_id = latest_ver_obj.run_id
    run = client.get_run(run_id)

    if "best_threshold" not in run.data.params:
        print("[Shib Update] Error: best_threshold not logged in MLFlow for this run.")
        sys.exit(1)

    threshold = float(run.data.params["best_threshold"])
    print(f"[Shib Update] Retrieved threshold from MLFlow: {threshold}")

    # Load Shibboleth path from config
    shib_path = cfg("deployment.shibboleth_path")

    if not shib_path:
        print("[Shib Update] Error: deployment.shibboleth_path not configured.")
        sys.exit(1)

    update_shib_threshold(shib_path, threshold)
    print(f"[Shib Update] Successfully updated {shib_path} with threshold={threshold}")


if __name__ == "__main__":
    main()
