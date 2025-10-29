import os
import json
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timezone
from typing import Union, Dict, Any, Optional

load_dotenv()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment.")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

ALLOWED_JSON_COLUMNS = {
    "metrics_version", "heuristic_score", "nn_score", "ensemble_score",
    "impossible_travel", "human_verified", "focus_changes", "blur_events",
    "click_count", "key_count", "avg_key_delay_ms", "pointer_distance_px",
    "pointer_event_count", "scroll_distance_px", "scroll_event_count",
    "dom_ready_ms", "time_to_first_key_ms", "time_to_first_click_ms",
    "idle_time_total_ms", "input_focus_count", "paste_events",
    "resize_events", "tz_offset_min", "language", "platform",
    "device_memory_gb", "hardware_concurrency", "screen_width_px",
    "screen_height_px", "pixel_ratio", "color_depth", "touch_support",
    "webauthn_supported", "city", "country", "asn"
}


def insert_login_event_from_json(
        data: Union[str, Dict[str, Any]],
        username: str,
        ip_address: str = None,
        device_uuid: str = None,
        extra_fields: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    try:
        if isinstance(data, str):
            data = json.loads(data)
        elif not isinstance(data, dict):
            logging.warning(f"[DB] Invalid data type for login event: {type(data)}")
            return None
    except json.JSONDecodeError as e:
        logging.warning(f"[DB] JSON decode error: {e}")
        return None

    # Define static fields provided by the application
    base_fields = {
        "username": username,
        "device_uuid": device_uuid,
        "ip_address": ip_address,
        "event_timestamp": datetime.now(timezone.utc),
    }

    # Only keep whitelisted dynamic columns from client/enrichment
    behavioral_data = {k: v for k, v in data.items() if k in ALLOWED_JSON_COLUMNS}

    # Only keep whitelisted dynamic columns from server-side extras (scores, flags, etc.)
    nn_scores = {k: v for k, v in (extra_fields or {}).items() if k in ALLOWED_JSON_COLUMNS}

    # Merge: static -> client/enrichment -> server extras (extras can override client if needed)
    params = {**base_fields, **behavioral_data, **nn_scores}

    # Strip Nones so DB defaults can apply
    params = {k: v for k, v in params.items() if v is not None}

    # Build the SQL statement
    columns = ", ".join(params.keys())
    values = ", ".join([f":{k}" for k in params.keys()])

    sql = text(f"INSERT INTO rba_login_event ({columns}) VALUES ({values}) RETURNING login_id")

    try:
        with engine.begin() as conn:
            result = conn.execute(sql, params)
            login_id = result.scalar_one()
            return login_id
    except SQLAlchemyError as e:
        logging.error(f"[DB] Error inserting login event: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"[DB] Unexpected error: {e}", exc_info=True)
        return None


def db_health_check() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as e:
        logging.error(f"[DB] Health check failed: {e}")
        return False
