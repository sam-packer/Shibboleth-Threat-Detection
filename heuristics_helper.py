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


def fetch_baseline_sample(limit: int = 20000) -> pd.DataFrame:
    """
    Fetches a random sample of login events to derive feature distributions.
    We use these distributions to scale and normalize heuristic features.
    """
    query = text(f"""
        SELECT {', '.join(FEATURE_COLUMNS)}
        FROM rba_login_event
        WHERE nn_score = -1.0
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


def estimate_user_sample_power(conn, max_margin: float = 0.10) -> pd.DataFrame:
    """
    Estimate how many logins each user has and whether they have enough
    behavioral data for reliable heuristic scoring.
    Uses click_count variance as a proxy for behavioral stability.

    Args:
        conn: active SQLAlchemy connection
        max_margin: allowable relative margin of error for the mean (10% default)
    """
    df = pd.read_sql(text("""
        SELECT username,
               COUNT(*) AS n,
               AVG(click_count) AS mean_clicks,
               STDDEV(click_count) AS std_clicks
        FROM rba_login_event
        WHERE nn_score = -1.0
        GROUP BY username
    """), conn)

    if df.empty:
        logging.warning("[Heuristics] No user data found for sample adequacy check.")
        return df

    # Handle division safely
    df["std_clicks"].fillna(0.0, inplace=True)
    df["mean_clicks"].replace(0.0, np.nan, inplace=True)

    # 95% confidence interval width
    df["margin_of_error"] = 1.96 * (df["std_clicks"] / np.sqrt(df["n"].clip(lower=1)))
    df["rel_margin"] = df["margin_of_error"] / df["mean_clicks"]
    df["sufficient_data"] = df["rel_margin"] < max_margin
    df.loc[df["n"] < 50, "sufficient_data"] = False
    df.loc[df["std_clicks"] == 0, "sufficient_data"] = False

    qualified = df["sufficient_data"].sum()
    total = len(df)
    logging.info(f"[Heuristics] {qualified}/{total} users have stable data "
                 f"(relative margin < {max_margin:.0%}).")
    return df


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


def preview_heuristics(sample_limit: int = 2000):
    """
    Preview heuristic distributions and user sample stability before committing updates.
    """
    df = fetch_baseline_sample(sample_limit)
    if df.empty:
        print("No data available.")
        return

    stats = compute_feature_statistics(df)
    df["heuristic_score"] = df.apply(lambda row: compute_dynamic_heuristic(row, stats), axis=1)

    print(df[["key_count", "click_count", "idle_time_total_ms", "heuristic_score"]].head(10))
    print("\nHeuristic summary:")
    print(df["heuristic_score"].describe())

    try:
        with engine.connect() as conn:
            power_df = estimate_user_sample_power(conn)
            if power_df.empty:
                print("\nNo user sample data found.")
                return

        print("\nUser sample adequacy (first 10):")
        print(
            power_df[["username", "n", "mean_clicks", "rel_margin", "sufficient_data"]]
            .sort_values("n", ascending=False)
            .head(10)
            .to_string(index=False, float_format=lambda x: f"{x:.3f}")
        )

        qualified = power_df["sufficient_data"].sum()
        total = len(power_df)
        print(f"\n{qualified}/{total} users have stable data (relative margin < 10%).")

        print("\nLogin count distribution per user:")
        print(power_df["n"].describe())

    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] Sample adequacy preview failed: {e}")


def retroactively_backfill_heuristics(batch_size: int = 5000):
    """
    Compute heuristic scores for logins without a neural net score (nn_score=-1.0),
    but only for users whose behavioral data is statistically stable.
    """
    baseline = fetch_baseline_sample(batch_size * 2)
    if baseline.empty:
        print("No baseline data for heuristics.")
        return

    stats = compute_feature_statistics(baseline)

    try:
        with engine.connect() as conn:
            power_df = estimate_user_sample_power(conn)
            if power_df.empty:
                logging.warning("[Heuristics] Skipping update, no users qualified.")
                return

            stable_users = set(power_df.query("sufficient_data")["username"])
            logging.info(f"[Heuristics] {len(stable_users)} users qualified for scoring.")

        with engine.begin() as conn:
            result = conn.execute(text(f"""
                SELECT login_id, username, {', '.join(FEATURE_COLUMNS)}
                FROM rba_login_event
                WHERE nn_score = -1.0
            """))
            rows = result.mappings().all()
            logging.info(f"Fetched {len(rows)} candidate rows for backfill.")

            updated = 0
            skipped = 0
            for row in rows:
                if row["username"] not in stable_users:
                    skipped += 1
                    continue

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

            logging.info(f"[Heuristics] Updated {updated} rows. "
                         f"Skipped {skipped} rows due to insufficient data.")

    except SQLAlchemyError as e:
        logging.error(f"[Heuristics] Backfill failed: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Heuristic backfill tool for Risk-Based Authentication (RBA). "
            "It calculates heuristic scores only for logins where nn_score = -1.0 "
            "(typically collected in passthrough mode). Existing scores will not be modified."
        )
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview heuristic statistics and sample scores (no database writes)."
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5000,
        help="Number of rows to sample or update per batch (default: 5000)."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="!!! IRREVERSIBLE, backup recommended !!! Apply heuristic scores to the database."
    )

    args = parser.parse_args()

    # If no args provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.preview:
        preview_heuristics(sample_limit=args.batch)
    elif args.write:
        retroactively_backfill_heuristics(batch_size=args.batch)
    else:
        print("No action specified. Use --preview or --write.\n")
        parser.print_help()
