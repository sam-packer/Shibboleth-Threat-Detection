import os
import logging
import random

from flask import Flask, request, jsonify
from dotenv import load_dotenv
from geoip2.database import Reader
from geoip_helper import ensure_geoip_up_to_date, get_geoip_paths
from stopforumspam_helper import ensure_sfs_up_to_date
from db_helper import insert_login_event_from_json, db_health_check
from ipaddress import ip_address as ip_parse

# --- Init & config ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

app = Flask(__name__)


def preflight():
    ensure_geoip_up_to_date()
    ensure_sfs_up_to_date()
    db_health_check()
    return True


def enrich_with_geoip(data: dict, client_ip: str) -> dict:
    paths = get_geoip_paths()

    try:
        ip_obj = ip_parse(client_ip)
    except ValueError:
        logging.warning(f"[GeoIP] Invalid IP address: {client_ip}")
        return data

    try:
        with Reader(paths["city"]) as city_reader, Reader(paths["asn"]) as asn_reader:
            city_info = city_reader.city(client_ip)
            asn_info = asn_reader.asn(client_ip)

            data["country"] = city_info.country.iso_code or None
            data["city"] = city_info.city.name or None
            data["asn"] = asn_info.autonomous_system_number or None

    except FileNotFoundError as e:
        logging.error(f"[GeoIP] Missing MaxMind DB file: {e}")
    except Exception as e:
        logging.warning(f"[GeoIP] Lookup failed for {client_ip}: {e}")

    return data


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

        # Placeholder scores
        heuristic_score = 0.9
        nn_score = -1
        ensemble_score = -1

        # Insert into DB
        login_id = insert_login_event_from_json(enriched, username=username, ip_address=client_ip,
                                                device_uuid=device_uuid,
                                                extra_fields={"heuristic_score": heuristic_score, "nn_score": nn_score,
                                                              "ensemble_score": ensemble_score,
                                                              })
        if not login_id:
            return jsonify({"error": "Failed to record login event"}), 500

        # You know, if we get an R2 lower than 0.5, this would be more effective
        threat_score = random.random()

        return jsonify({"threatScore": round(threat_score, 4)})

    except Exception as e:
        logging.error(f"[API] /score failed: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    if preflight():
        port = int(os.getenv("PORT", 5001))
        debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        logging.info(f"Starting Flask on 127.0.0.1:{port}, debug={debug_mode}")
        app.run(host="127.0.0.1", port=port, debug=debug_mode)
