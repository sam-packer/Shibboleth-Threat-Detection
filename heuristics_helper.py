import os
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from scipy.special import expit

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment variables.")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

FEATURE_COLUMNS = [
    "focus_changes", "blur_events", "click_count", "key_count",
    "avg_key_delay_ms", "pointer_distance_px", "pointer_event_count",
    "scroll_distance_px", "scroll_event_count", "dom_ready_ms",
    "time_to_first_key_ms", "time_to_first_click_ms", "idle_time_total_ms",
    "input_focus_count", "paste_events", "resize_events"
]


# --- Data Loading -------------------------------------------------------------
def fetch_baseline_sample(limit: int = 20000) -> pd.DataFrame:
    """
    Fetches a random sample of login events to derive feature distributions.
    We use these distributions to scale and normalize heuristic features.
    """
    query = text(f"""
        SELECT {', '.join(FEATURE_COLUMNS)}
        FROM rba_login_event
        WHERE nn_score = 0.0
        ORDER BY random()
        LIMIT :limit
    """)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})
        df = df.dropna(how="all")
        logging.info(f"Loaded {len(df)} baseline rows for heuristic calibration.")
        return df
    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] DB fetch failed: {e}")
        return pd.DataFrame()


# --- Feature Scaling ----------------------------------------------------------
def compute_feature_statistics(df: pd.DataFrame) -> dict:
    """
    Compute robust statistics for each feature: mean, std, percentiles.
    """
    stats = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].notna().any():
            series = df[col].astype(float)
            stats[col] = {
                "mean": series.mean(),
                "std": series.std(ddof=0),
                "p05": np.percentile(series, 5),
                "p25": np.percentile(series, 25),
                "p50": np.percentile(series, 50),
                "p75": np.percentile(series, 75),
                "p95": np.percentile(series, 95),
                "min": series.min(),
                "max": series.max(),
            }
    return stats


# --- Heuristic Computation ----------------------------------------------------
def compute_dynamic_heuristic(row: dict, stats: dict) -> float:
    """
    Compute a dynamic heuristic score based on normalized z-scores and deviations.
    The idea: users whose behavior is far from the median get higher risk.
    """
    z_scores = []

    for feature, meta in stats.items():
        value = row.get(feature)
        if value is None or np.isnan(value):
            continue

        std = meta["std"] if meta["std"] > 0 else 1.0
        z = abs((value - meta["p50"]) / std)
        z_scores.append(z)

    if not z_scores:
        return 0.0

    # Aggregate deviation across all features
    mean_deviation = np.mean(z_scores)

    # Sigmoid transform to bound between 0–1
    risk = expit(mean_deviation - 1.0)  # shift so typical behavior ≈0.3–0.4
    return float(np.clip(risk, 0.0, 1.0))


# --- Preview Mode -------------------------------------------------------------
def preview_heuristics(sample_limit: int = 2000):
    df = fetch_baseline_sample(sample_limit)
    if df.empty:
        print("No data available.")
        return

    stats = compute_feature_statistics(df)
    df["heuristic_score"] = df.apply(lambda row: compute_dynamic_heuristic(row, stats), axis=1)

    print(df[["key_count", "click_count", "idle_time_total_ms", "heuristic_score"]].head(10))
    print("\nHeuristic summary:")
    print(df["heuristic_score"].describe())


# --- Retroactive Backfill -----------------------------------------------------
def retroactively_backfill_heuristics(batch_size: int = 5000):
    """
    Compute heuristic scores for logins without a neural net score (nn_score=0.0)
    and update them in the DB.
    """
    baseline = fetch_baseline_sample(batch_size * 2)
    if baseline.empty:
        print("No baseline data for heuristics.")
        return

    stats = compute_feature_statistics(baseline)

    try:
        with engine.begin() as conn:
            result = conn.execute(text(f"""
                SELECT login_id, username, {', '.join(FEATURE_COLUMNS)}
                FROM rba_login_event
                WHERE nn_score = 0.0
            """))
            rows = result.mappings().all()
            logging.info(f"Fetched {len(rows)} rows to backfill.")

            updated = 0
            for row in rows:
                score = compute_dynamic_heuristic(row, stats)
                conn.execute(
                    text("""
                        UPDATE rba_login_event
                        SET nn_score = :score
                        WHERE login_id = :id AND username = :username
                    """),
                    {"score": score, "id": row["login_id"], "username": row["username"]}
                )
                updated += 1

            logging.info(f"Updated {updated} rows with data-driven heuristic scores.")

    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] Backfill failed: {e}")


# --- CLI Entrypoint -----------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heuristic backfill for RBA.")
    parser.add_argument("--preview", action="store_true", help="Preview heuristic stats only.")
    parser.add_argument("--batch", type=int, default=5000, help="Batch size for DB update.")
    args = parser.parse_args()

    if args.preview:
        preview_heuristics(sample_limit=args.batch)
    else:
        retroactively_backfill_heuristics(batch_size=args.batch)
