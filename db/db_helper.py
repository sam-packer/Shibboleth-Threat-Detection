import os
import json
import logging
import re
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from typing import Union, Dict, Any, Optional, Tuple
from helpers.globals import resolve_path, project_root

load_dotenv()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment.")

# Standard Postgres connection
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

ALLOWED_JSON_COLUMNS = {
    "metrics_version", "anomaly_score", "human_verified", "focus_changes", "blur_events",
    "click_count", "key_count", "avg_key_delay_ms", "pointer_distance_px",
    "pointer_event_count", "scroll_distance_px", "scroll_event_count",
    "dom_ready_ms", "time_to_first_key_ms", "time_to_first_click_ms",
    "idle_time_total_ms", "total_session_time_ms", "active_time_ms", "input_focus_count", "paste_events",
    "resize_events", "tz_offset_min", "language", "platform",
    "device_memory_gb", "hardware_concurrency", "screen_width_px",
    "screen_height_px", "pixel_ratio", "color_depth", "touch_support",
    "webauthn_supported", "city", "country", "asn",
    "device_category"
}


def get_latest_version_info(seeds_root: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Scans the seeds directory for version folders (v3, v4, etc.),
    orders them numerically, and returns the latest version string and its full path.
    Returns: (version_string, full_path) e.g., ("v4", "/app/seeds/v4")
    """
    if not os.path.exists(seeds_root):
        logging.error(f"[DB] Seeds directory not found at: {seeds_root}")
        return None, None

    versions = []
    # Iterate over items in seeds directory
    for item in os.listdir(seeds_root):
        full_path = os.path.join(seeds_root, item)
        if os.path.isdir(full_path):
            # Match folders like 'v1', 'v2', 'v10' using Regex
            match = re.match(r'^v(\d+)$', item)
            if match:
                version_num = int(match.group(1))
                versions.append((version_num, item, full_path))

    if not versions:
        logging.warning(f"[DB] No versioned directories (e.g. 'v4') found in {seeds_root}")
        return None, None

    # Sort by version number descending (highest first) so v10 comes before v2
    versions.sort(key=lambda x: x[0], reverse=True)

    # Return the string 'vX' and the full path of the highest version
    return versions[0][1], versions[0][2]


def init_db_schema():
    """
    Idempotent database initialization for standard Postgres 17.
    1. Checks if the main table (rba_login_event) exists.
    2. Dynamically finds the latest seed version.
    3. Runs the seed SQL files from that version to create the schema.
    """
    logging.info("[DB] Checking database schema initialization...")

    # Resolve path to the main 'seeds' folder
    seeds_root = resolve_path("seeds", os.path.join(project_root(), "seeds"))

    # Dynamically find the latest version info
    version_str, version_dir = get_latest_version_info(seeds_root)

    if not version_dir:
        logging.error("[DB] Could not determine latest seed version. Skipping initialization.")
        return

    # Construct filenames dynamically based on the version found (e.g. v4_rba_device.sql)
    # Order matters due to Foreign Keys: Device -> Login Event -> Scores
    seed_files = [
        f"{version_str}_rba_device.sql",
        f"{version_str}_rba_login_event.sql",
        f"{version_str}_rba_scores.sql"
    ]

    try:
        with engine.begin() as conn:
            # Check if the schema exists using standard Postgres system catalog functions
            table_check = conn.execute(text("SELECT to_regclass('public.rba_login_event')")).scalar()

            if not table_check:
                logging.info(f"[DB] Schema missing. Initializing tables using version: {version_str}...")
                for file_name in seed_files:
                    file_path = os.path.join(version_dir, file_name)
                    if not os.path.exists(file_path):
                        logging.error(f"[DB] Seed file not found: {file_path}")
                        continue

                    logging.info(f"[DB] Executing seed: {file_name}")
                    with open(file_path, 'r') as f:
                        sql_content = f.read()
                        conn.execute(text(sql_content))
                logging.info("[DB] Schema initialization complete.")
            else:
                logging.info("[DB] Schema already exists. Skipping initialization.")

    except SQLAlchemyError as e:
        logging.error(f"[DB] Critical error during schema initialization: {e}", exc_info=True)
        raise e


def insert_login_event_from_json(
        data: Union[str, Dict[str, Any]],
        username: str,
        ip_address: str = None,
        device_uuid: str = None,
        device_category: str = None,
        extra_fields: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Insert raw login event data and return login_id."""
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            logging.warning(f"[DB] Invalid data type for login event: {type(data)}")
            return None
    except json.JSONDecodeError as e:
        logging.warning(f"[DB] JSON decode error: {e}")
        return None

    base_fields = {
        "username": username,
        "device_uuid": device_uuid,
        "device_category": device_category,
        "ip_address": ip_address,
        "event_timestamp": datetime.now(timezone.utc),
    }

    behavioral_data = {k: v for k, v in data.items() if k in ALLOWED_JSON_COLUMNS}
    score_fields = {k: v for k, v in (extra_fields or {}).items() if k in ALLOWED_JSON_COLUMNS}

    params = {**base_fields, **behavioral_data, **score_fields}
    params = {k: v for k, v in params.items() if v is not None}

    sanitized_params = {}
    for k, v in params.items():
        if hasattr(v, 'item'):
            sanitized_params[k] = v.item()
        else:
            sanitized_params[k] = v

    columns = ", ".join(sanitized_params.keys())
    values = ", ".join([f":{k}" for k in sanitized_params.keys()])

    sql = text(f"INSERT INTO rba_login_event ({columns}) VALUES ({values}) RETURNING login_id")

    try:
        with engine.begin() as conn:
            result = conn.execute(sql, sanitized_params)
            login_id = result.scalar_one()
            return login_id
    except SQLAlchemyError as e:
        logging.error(f"[DB] Error inserting login event: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"[DB] Unexpected error: {e}", exc_info=True)
        return None


def insert_rba_scores(
        login_id: int,
        username: str,
        anomaly_score: float,
        ip_risk_score: float,
        impossible_travel: float,
        final_score: Optional[float] = None,
) -> bool:
    """Insert derived risk scores into rba_scores table."""
    if login_id is None:
        logging.error("[DB] insert_rba_scores called with null login_id")
        return False

    final_score = final_score if final_score is not None else anomaly_score

    sql = text("""
               INSERT INTO rba_scores (login_id, username, anomaly_score, ip_risk_score, impossible_travel, final_score,
                                       created_at)
               VALUES (:login_id, :username, :anomaly_score, :ip_risk_score, :impossible_travel, :final_score, :created_at)
               """)

    params = {
        "login_id": login_id,
        "username": username,
        "anomaly_score": anomaly_score.item() if hasattr(anomaly_score, 'item') else anomaly_score,
        "ip_risk_score": ip_risk_score.item() if hasattr(ip_risk_score, 'item') else ip_risk_score,
        "impossible_travel": impossible_travel.item() if hasattr(impossible_travel, 'item') else impossible_travel,
        "final_score": final_score.item() if hasattr(final_score, 'item') else final_score,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        with engine.begin() as conn:
            conn.execute(sql, params)
        return True
    except SQLAlchemyError as e:
        logging.error(f"[DB] Error inserting into rba_scores: {e}", exc_info=True)
        return False


def record_login_with_scores(
        data: Union[str, Dict[str, Any]],
        username: str,
        ip_address: str,
        device_uuid: str,
        device_category: str,
        anomaly_score: float,
        ip_risk_score: float,
        impossible_travel: float,
        final_score: Optional[float] = None,
) -> Optional[int]:
    """
    Insert both login event and associated risk scores in one go.
    Returns login_id or None on failure.
    """
    login_id = insert_login_event_from_json(
        data,
        username=username,
        ip_address=ip_address,
        device_uuid=device_uuid,
        device_category=device_category,
        extra_fields={"anomaly_score": anomaly_score}
    )
    if not login_id:
        return None

    insert_rba_scores(
        login_id=login_id,
        username=username,
        anomaly_score=anomaly_score,
        ip_risk_score=ip_risk_score,
        impossible_travel=impossible_travel,
        final_score=final_score
    )

    return login_id


def db_health_check() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        logging.error(f"[DB] Health check failed: {e}")
        return False
