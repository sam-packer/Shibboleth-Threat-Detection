import os
import logging
import random

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from geoip_helper import ensure_geoip_up_to_date, enrich_with_geoip
from nn_helper import compute_nn_score, load_model_and_scaler
from stopforumspam_helper import ensure_sfs_up_to_date, ip_in_toxic_list
from db_helper import db_health_check, record_login_with_scores

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

PASSTHROUGH_MODE = os.getenv("PASSTHROUGH_MODE", "true").lower() == "true"

app = Flask(__name__)


def preflight():
    ensure_geoip_up_to_date()
    ensure_sfs_up_to_date()
    db_health_check()
    load_model_and_scaler()
    return True


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

        # Compute the neural network score
        if PASSTHROUGH_MODE:
            nn_score = -1
        else:
            nn_score = compute_nn_score(username, enriched)

        # Compute the IP risk score
        ip_risk_score = 1.0 if ip_in_toxic_list(client_ip) else 0.0

        # Escalation only ensemble (never depresses the NN)
        if nn_score < 0:
            threat_score = ip_risk_score  # passthrough mode
        else:
            base = nn_score
            # If IP is toxic, bump risk instead of averaging
            if ip_risk_score >= 1.0:
                # linear bump that caps at 1.0
                threat_score = min(1.0, base + 0.25 * (1.0 - base))
            else:
                threat_score = base

        impossible_travel = -1
        threat_score = max(0.0, min(threat_score, 1.0))

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


if __name__ == "__main__":
    if preflight():
        port = int(os.getenv("PORT", 5001))
        debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        logging.info(f"Starting Flask on 127.0.0.1:{port}, debug={debug_mode}")
        app.run(host="127.0.0.1", port=port, debug=debug_mode)
