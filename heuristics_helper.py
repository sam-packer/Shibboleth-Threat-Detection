import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import os

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment.")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

# Feature columns to include in heuristics analysis
FEATURE_COLUMNS = [
    "focus_changes", "blur_events", "click_count", "key_count",
    "avg_key_delay_ms", "pointer_distance_px", "pointer_event_count",
    "scroll_distance_px", "scroll_event_count", "dom_ready_ms",
    "time_to_first_key_ms", "time_to_first_click_ms", "idle_time_total_ms",
    "input_focus_count", "paste_events", "resize_events"
]


def fetch_sample(limit: int = 10000) -> pd.DataFrame:
    """
    Load a random sample of login events from the database.
    """
    try:
        with engine.connect() as conn:
            query = text(f"""
                SELECT {', '.join(FEATURE_COLUMNS)}
                FROM rba_login_event
                WHERE heuristic_score = 0.0
                ORDER BY random()
                LIMIT :limit
            """)
            df = pd.read_sql(query, conn, params={"limit": limit})
            return df.dropna(how="all")
    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] Error fetching data: {e}")
        return pd.DataFrame()


def compute_feature_stats(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for scaling and normalization.
    """
    stats = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            series = df[col].dropna()
            if len(series) > 0:
                stats[col] = {
                    "mean": float(series.mean()),
                    "std": float(series.std(ddof=0)),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "median": float(series.median())
                }
    return stats


def compute_heuristic_score(row, stats: dict) -> float:
    """
    Compute a heuristic risk score from behavioral metrics.
    Higher score → higher likelihood of suspicious activity.
    """

    score = 0.0

    # Example logic — refine later once you have real data distributions.
    try:
        if row.get("key_count", 0) < (stats["key_count"]["median"] * 0.3):
            score += 0.15  # very few keys typed = suspicious

        if row.get("click_count", 0) > (stats["click_count"]["mean"] * 5):
            score += 0.10  # excessive clicks = automation?

        if row.get("idle_time_total_ms", 0) < (stats["idle_time_total_ms"]["mean"] * 0.2):
            score += 0.10  # always active = possibly scripted

        if row.get("pointer_event_count", 0) < (stats["pointer_event_count"]["median"] * 0.1):
            score += 0.20  # little or no pointer movement

        if row.get("avg_key_delay_ms", 0) < 40:
            score += 0.15  # unrealistically fast typing

        # Normalize to 0–1 range
        return max(0.0, min(1.0, score))
    except Exception as e:
        logging.warning(f"[Heuristics] Score computation failed for row: {e}")
        return 0.0


def preview_scores(limit: int = 1000):
    """
    Preview heuristic scores without modifying the DB.
    """
    df = fetch_sample(limit)
    if df.empty:
        print("No data available for preview.")
        return

    stats = compute_feature_stats(df)
    df["heuristic_preview"] = df.apply(lambda row: compute_heuristic_score(row, stats), axis=1)
    print(df[["key_count", "click_count", "idle_time_total_ms", "heuristic_preview"]].head(10))
    print("\nAggregate preview:\n", df["heuristic_preview"].describe())


def retroactively_update_scores(batch_size: int = 5000):
    """
    Retroactively compute and update heuristic_score in the DB.
    Not executed by default.
    """
    df = fetch_sample(batch_size)
    if df.empty:
        print("No data available for update.")
        return

    stats = compute_feature_stats(df)

    try:
        with engine.begin() as conn:
            # Re-select all rows with null or 0 heuristic
            result = conn.execute(text("""
                                       SELECT login_id,
                                              heuristic_score,
                                              key_count,
                                              click_count,
                                              idle_time_total_ms,
                                              pointer_event_count,
                                              avg_key_delay_ms
                                       FROM rba_login_event
                                       WHERE heuristic_score = 0.0
                                          OR heuristic_score IS NULL
                                       """))
            rows = result.mappings().all()

            for row in rows:
                heuristic = compute_heuristic_score(row, stats)
                conn.execute(
                    text("UPDATE rba_login_event SET heuristic_score = :score WHERE login_id = :id"),
                    {"score": heuristic, "id": row["login_id"]}
                )

        print(f"Updated {len(rows)} rows with computed heuristic scores.")
    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] DB update failed: {e}")
