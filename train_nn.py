import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load connection info
load_dotenv()
POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

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

def load_training_data(limit: int = 10000):
    """
    Fetch training data from rba_login_event.
    """
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, nn_score
        FROM rba_login_event
        WHERE nn_score >= 0.0
        ORDER BY random()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": limit})
    print(f"Loaded {len(df)} rows.")
    return df

if __name__ == "__main__":
    df = load_training_data()
    print(df.head())
