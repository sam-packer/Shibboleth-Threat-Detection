import logging

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from helpers.globals import cfg
from nn_scripts.ensembler import ensemble_threat_score
from external_data.geoip_helper import ensure_geoip_up_to_date, enrich_with_geoip
from nn_scripts.nn_helper import compute_nn_score, load_model_and_scaler
from external_data.stopforumspam_helper import ensure_sfs_up_to_date, ip_in_toxic_list
from db.db_helper import db_health_check, record_login_with_scores

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)

PASSTHROUGH_MODE = cfg("api.passthrough_mode", False)


def preflight():
    ensure_geoip_up_to_date()
    ensure_sfs_up_to_date()
    db_health_check()
    load_model_and_scaler()
    return True


preflight()
app = Flask(__name__)


@app.route("/score", methods=["POST"])
def score_endpoint():
    try:
        body = request.get_json(force=True)

        username = body.get("username", "unknown_user")
        client_ip = body.get("ipAddress") or request.remote_addr

        metrics = body.get("metrics", {})
        device_uuid = metrics.get("device_uuid")

        if not metrics or not isinstance(metrics, dict):
            return jsonify({"error": "Invalid JSON"}), 400

        # Enrich with GeoIP
        enriched = enrich_with_geoip(metrics, client_ip)
        enriched["ip_address"] = client_ip
        enriched["username"] = username
        enriched["device_uuid"] = device_uuid

        nn_score = compute_nn_score(username, enriched)
        ip_risk_score = 1.0 if ip_in_toxic_list(client_ip) else 0.0
        impossible_travel = -1

        threat_score = ensemble_threat_score(nn_score, ip_risk_score, PASSTHROUGH_MODE)

        # Insert into DB
        login_id = record_login_with_scores(
            data=enriched,
            username=username,
            ip_address=client_ip,
            device_uuid=device_uuid,
            nn_score=nn_score,
            ip_risk_score=ip_risk_score,
            impossible_travel=impossible_travel,
            final_score=threat_score
        )

        if not login_id:
            return jsonify({"error": "Failed to record login event"}), 500

        return jsonify({"threatScore": threat_score})

    except Exception as e:
        logging.error(f"[API] /score failed: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


def main():
    if preflight():
        host = cfg("api.host")
        port = cfg("api.port")
        debug_mode = False

        logging.info(
            f"Starting Flask on {host}:{port}, debug={debug_mode}, passthrough={PASSTHROUGH_MODE}"
        )

        app.run(host=host, port=port, debug=debug_mode)


if __name__ == "__main__":
    main()
