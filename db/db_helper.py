import os
import json
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError, ProgrammingError
from datetime import datetime, timezone
from typing import Union, Dict, Any, Optional

load_dotenv()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment.")

# Ensure future=True for 2.0 style usage
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

ALLOWED_JSON_COLUMNS = {
    "metrics_version", "nn_score", "human_verified", "focus_changes", "blur_events",
    "click_count", "key_count", "avg_key_delay_ms", "pointer_distance_px",
    "pointer_event_count", "scroll_distance_px", "scroll_event_count",
    "dom_ready_ms", "time_to_first_key_ms", "time_to_first_click_ms",
    "idle_time_total_ms", "total_session_time_ms", "active_time_ms", "input_focus_count", "paste_events",
    "resize_events", "tz_offset_min", "language", "platform",
    "device_memory_gb", "hardware_concurrency", "screen_width_px",
    "screen_height_px", "pixel_ratio", "color_depth", "touch_support",
    "webauthn_supported", "city", "country", "asn"
}


def init_db_schema():
    """
    Idempotent database initialization.
    1. Checks if tables exist.
    2. If not, runs seeds from the seeds/ folder.
    3. Enables Citus and distributes tables if not already distributed.
    """
    logging.info("[DB] Checking database schema initialization...")
    
    # Locate seeds folder relative to this file (db/db_helper.py -> ../seeds)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    seeds_dir = os.path.join(base_dir, "seeds")
    
    # Order matters due to Foreign Keys
    seed_files = [
        "v4_rba_device.sql",
        "v4_rba_login_event.sql", 
        "v4_rba_scores.sql"
    ]

    try:
        with engine.begin() as conn:
            # 1. Check if schema exists (using Postgres-specific check)
            # We check for the main table 'rba_login_event'
            table_check = conn.execute(text("SELECT to_regclass('public.rba_login_event')")).scalar()

            if not table_check:
                logging.info("[DB] Schema missing. Running seeds...")
                for file_name in seed_files:
                    file_path = os.path.join(seeds_dir, file_name)
                    if not os.path.exists(file_path):
                        logging.error(f"[DB] Seed file not found: {file_path}")
                        continue
                        
                    logging.info(f"[DB] Executing seed: {file_name}")
                    with open(file_path, 'r') as f:
                        sql_content = f.read()
                        # Split by ; to handle multiple statements if necessary, 
                        # though execute() usually handles scripts if supported by driver.
                        conn.execute(text(sql_content))
                logging.info("[DB] Seeding complete.")
            else:
                logging.info("[DB] Schema already exists. Skipping seeds.")

            # 2. Setup Citus (Sharding)
            try:
                # Enable Extension
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS citus"))
                
                # Check if tables are already distributed to avoid "table is already distributed" errors
                # We check citus_tables system view
                dist_check = conn.execute(text(
                    "SELECT count(*) FROM citus_tables WHERE table_name = 'rba_login_event'"
                )).scalar()

                if dist_check == 0:
                    logging.info("[DB] Distributing tables via Citus...")
                    conn.execute(text("SELECT create_distributed_table('rba_device', 'device_uuid')"))
                    conn.execute(text("SELECT create_distributed_table('rba_login_event', 'username')"))
                    # Note: rba_scores usually joins on login_id/username. 
                    # If needed, distribute it here too based on your specific query patterns.
                    logging.info("[DB] Tables distributed.")
                else:
                    logging.info("[DB] Tables already distributed via Citus.")

            except ProgrammingError as e:
                # This catches cases where Citus might not be installed on the Postgres server
                logging.warning(f"[DB] Citus extension check failed (Database might not support Citus): {e}")
            except Exception as e:
                logging.error(f"[DB] Citus configuration error: {e}")

    except SQLAlchemyError as e:
        logging.error(f"[DB] Critical error during schema initialization: {e}", exc_info=True)
        raise e


def insert_login_event_from_json(
        data: Union[str, Dict[str, Any]],
        username: str,
        ip_address: str = None,
        device_uuid: str = None,
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
        "ip_address": ip_address,
        "event_timestamp": datetime.now(timezone.utc),
    }

    behavioral_data = {k: v for k, v in data.items() if k in ALLOWED_JSON_COLUMNS}
    nn_scores = {k: v for k, v in (extra_fields or {}).items() if k in ALLOWED_JSON_COLUMNS}

    params = {**base_fields, **behavioral_data, **nn_scores}
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
        nn_score: float,
        ip_risk_score: float,
        impossible_travel: float,
        final_score: Optional[float] = None,
) -> bool:
    """Insert derived risk scores into rba_scores table."""
    if login_id is None:
        logging.error("[DB] insert_rba_scores called with null login_id")
        return False

    final_score = final_score if final_score is not None else nn_score

    sql = text("""
               INSERT INTO rba_scores (login_id, username, nn_score, ip_risk_score, impossible_travel, final_score,
                                       created_at)
               VALUES (:login_id, :username, :nn_score, :ip_risk_score, :impossible_travel, :final_score, :created_at)
               """)

    params = {
        "login_id": login_id,
        "username": username,
        "nn_score": nn_score.item() if hasattr(nn_score, 'item') else nn_score,
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
        nn_score: float,
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
        extra_fields={"nn_score": nn_score}
    )
    if not login_id:
        return None

    insert_rba_scores(
        login_id=login_id,
        username=username,
        nn_score=nn_score,
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