import os
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from datetime import datetime

from helpers.globals import cfg

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment variables.")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)

# Weight features by their discriminative power for anomaly detection
FEATURE_WEIGHTS = {
    # Highly discriminative - typing patterns are very personal
    "avg_key_delay_ms": 2.0,  # Typing speed/rhythm is unique per person

    # Strong discriminators - interaction patterns
    "active_time_ms": 1.8,  # Bots rush or have unnatural timing
    "total_session_time_ms": 1.7,  # Overall session duration patterns
    "click_count": 1.5,  # Click patterns vary by user
    "key_count": 1.5,  # Different users type different amounts

    # Moderate discriminators - timing signals
    "idle_time_total_ms": 1.4,  # Pauses are behavioral (now at 500ms threshold)
    "time_to_first_key_ms": 1.3,  # Hesitation/reaction time
    "time_to_first_click_ms": 1.3,  # Initial engagement timing
    "pointer_distance_px": 1.2,  # Mouse movement patterns

    # Weaker but useful - interaction details
    "paste_events": 1.2,  # Credential stuffing often uses paste
    "pointer_event_count": 1.0,
    "input_focus_count": 1.0,

    # Context features - less discriminative but still useful
    "focus_changes": 0.8,
    "blur_events": 0.8,
    "scroll_distance_px": 0.7,
    "scroll_event_count": 0.7,
    "resize_events": 0.6,
    # Other features get default weight of 1.0
}

MIN_SAMPLES_FOR_PROFILE = cfg("model.min_user_events")


def build_user_profiles(conn, min_samples: int = MIN_SAMPLES_FOR_PROFILE) -> pd.DataFrame:
    """
    Build behavioral profiles for each user with sufficient login history.

    Returns DataFrame with columns:
    - username
    - n (number of samples)
    - feature_mean (for each feature)
    - feature_std (for each feature)
    - sufficient_data (bool)
    """
    feature_aggs = []
    for feat in cfg("data.feature_columns"):
        feature_aggs.append(f"AVG({feat}) AS {feat}_mean")
        feature_aggs.append(f"STDDEV({feat}) AS {feat}_std")

    query = text(f"""
        SELECT username,
               COUNT(*) AS n,
               {', '.join(feature_aggs)}
        FROM {cfg("data.table")}
        WHERE nn_score = -1.0
        GROUP BY username
        HAVING COUNT(*) >= :min_samples
    """)

    df = pd.read_sql(query, conn, params={"min_samples": min_samples})

    # Mark users with sufficient non-zero variance as having reliable profiles
    df["sufficient_data"] = True
    for feat in cfg("data.feature_columns"):
        std_col = f"{feat}_std"
        if std_col in df.columns:
            # If std is null or 0 for critical features, mark as insufficient
            if feat in ["click_count", "key_count"]:
                df.loc[df[std_col].fillna(0) == 0, "sufficient_data"] = False

    logging.info(f"[Profiles] Built profiles for {len(df)} users with ≥{min_samples} logins.")
    logging.info(f"[Profiles] {df['sufficient_data'].sum()} have sufficient behavioral variance.")

    return df


def get_all_user_stats(conn) -> pd.DataFrame:
    """
    Get login counts for ALL users (including those below threshold).
    """
    query = text(f"""
                 SELECT username,
                        COUNT(*) AS login_count
                 FROM {cfg("data.table")}
                 WHERE nn_score = -1.0
                 GROUP BY username
                 ORDER BY login_count DESC
                 """)

    df = pd.read_sql(query, conn)
    df["has_profile"] = df["login_count"] >= MIN_SAMPLES_FOR_PROFILE
    df["logins_needed"] = np.maximum(0, MIN_SAMPLES_FOR_PROFILE - df["login_count"])
    df["status"] = df.apply(
        lambda row: "USER_PROFILE" if row["has_profile"] else f"POPULATION (needs {int(row['logins_needed'])} more)",
        axis=1
    )

    return df


def compute_population_percentiles(conn, sample_size: int = 20000) -> dict:
    """
    Compute population-level percentiles for fallback scoring (new users).
    """
    query = text(f"""
        SELECT {', '.join(cfg("data.feature_columns"))}
        FROM {cfg("data.table")}
        WHERE nn_score = -1.0
        ORDER BY random()
        LIMIT :limit
    """)

    df = pd.read_sql(query, conn, params={"limit": sample_size})

    percentiles = {}
    for feat in cfg("data.feature_columns"):
        if feat in df.columns:
            series = df[feat].dropna()
            if len(series) > 0:
                percentiles[feat] = {
                    "p25": np.percentile(series, 25),
                    "p50": np.percentile(series, 50),
                    "p75": np.percentile(series, 75),
                    "p90": np.percentile(series, 90),
                    "p95": np.percentile(series, 95)
                }

    logging.info(f"[Population] Computed percentiles from {len(df)} samples.")
    return percentiles


def score_against_user_profile(login_row: dict, profile: dict) -> tuple[float, dict]:
    """
    Score a login based on deviation from the user's personal behavioral profile.

    Returns:
    - risk score [0, 1]
    - dict of per-feature z-scores for transparency
    """
    anomaly_scores = []
    feature_details = {}

    for feat in cfg("data.feature_columns"):
        value = login_row.get(feat)
        mean = profile.get(f"{feat}_mean")
        std = profile.get(f"{feat}_std")

        # Skip if data is missing
        if value is None or mean is None or std is None:
            continue
        if np.isnan(value) or np.isnan(mean) or np.isnan(std):
            continue

        # Skip if no variance (std = 0)
        if std == 0:
            if abs(value - mean) < 0.01:
                z_score = 0.0
            else:
                z_score = 5.0  # Very anomalous
        else:
            z_score = abs(value - mean) / std

        # Apply feature weight
        weight = FEATURE_WEIGHTS.get(feat, 1.0)
        weighted_z = z_score * weight

        anomaly_scores.append(weighted_z)
        feature_details[feat] = {
            "value": value,
            "user_mean": mean,
            "user_std": std,
            "z_score": z_score,
            "weighted_z": weighted_z
        }

    if not anomaly_scores:
        return 0.5, {}

    # Use the 75th percentile of anomaly scores
    anomaly_level = np.percentile(anomaly_scores, 75)

    # Convert to probability
    risk = 1 / (1 + np.exp(-0.8 * (anomaly_level - 2.5)))

    return float(np.clip(risk, 0.0, 1.0)), feature_details


def score_against_population(login_row: dict, percentiles: dict) -> tuple[float, dict]:
    """
    Score a login based on population percentiles (for users without profiles).

    Returns:
    - risk score [0, 1]
    - dict of per-feature deviation details
    """
    outlier_scores = []
    feature_details = {}

    for feat in cfg("data.feature_columns"):
        value = login_row.get(feat)
        if value is None or np.isnan(value):
            continue

        if feat not in percentiles:
            continue

        pct = percentiles[feat]

        # Determine how extreme this value is
        if value < pct["p25"]:
            range_size = pct["p25"] - pct["p50"]
            if range_size > 0:
                deviation = (pct["p25"] - value) / range_size
            else:
                deviation = 1.0
            position = "below_p25"
        elif value > pct["p75"]:
            range_size = pct["p75"] - pct["p50"]
            if range_size > 0:
                deviation = (value - pct["p75"]) / range_size
            else:
                deviation = 1.0
            position = "above_p75"
        else:
            deviation = 0.0
            position = "normal"

        # Flag if beyond 95th percentile
        if value > pct["p95"]:
            deviation = max(deviation, 3.0)
            position = "above_p95"

        weight = FEATURE_WEIGHTS.get(feat, 1.0)
        weighted_dev = deviation * weight
        outlier_scores.append(weighted_dev)

        feature_details[feat] = {
            "value": value,
            "pop_p50": pct["p50"],
            "pop_p75": pct["p75"],
            "pop_p95": pct["p95"],
            "deviation": deviation,
            "weighted_dev": weighted_dev,
            "position": position
        }

    if not outlier_scores:
        return 0.3, {}

    outlier_level = np.percentile(outlier_scores, 75)
    risk = 0.3 + 0.6 / (1 + np.exp(-1.0 * (outlier_level - 1.0)))

    return float(np.clip(risk, 0.0, 1.0)), feature_details


def create_detailed_report(output_file: str = None) -> str | None:
    """
    Generate comprehensive Excel report with multiple sheets for transparency.

    Returns: filename of created report
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"heuristic_report_{timestamp}.xlsx"

    try:
        with engine.connect() as conn:
            logging.info("[Report] Building user profiles...")
            profiles_df = build_user_profiles(conn, min_samples=MIN_SAMPLES_FOR_PROFILE)

            logging.info("[Report] Getting all user statistics...")
            all_users_df = get_all_user_stats(conn)

            logging.info("[Report] Computing population percentiles...")
            pop_pct = compute_population_percentiles(conn, sample_size=20000)

            # Convert profiles to dict for lookup
            profiles_dict = {}
            for _, row in profiles_df.iterrows():
                if row["sufficient_data"]:
                    profiles_dict[row["username"]] = row.to_dict()

            logging.info("[Report] Sampling recent logins for scoring...")
            # Get a reasonable sample of recent logins
            query = text(f"""
                SELECT login_id, username, event_timestamp, {', '.join(cfg("data.feature_columns"))}
                FROM rba_login_event
                WHERE nn_score = -1.0
                ORDER BY login_id DESC
                LIMIT 500
            """)

            sample_df = pd.read_sql(query, conn)

            if sample_df.empty:
                logging.warning("[Report] No login data found.")
                return None

            logging.info(f"[Report] Scoring {len(sample_df)} logins...")
            # Score each login with detailed breakdown
            scored_logins = []
            feature_breakdowns = []

            for _, row in sample_df.iterrows():
                username = row["username"]
                login_id = row["login_id"]

                if username in profiles_dict:
                    score, details = score_against_user_profile(row.to_dict(), profiles_dict[username])
                    method = "USER_PROFILE"
                else:
                    score, details = score_against_population(row.to_dict(), pop_pct)
                    method = "POPULATION"

                # Add to scored logins summary
                scored_logins.append({
                    "login_id": login_id,
                    "username": username,
                    "timestamp": row.get("timestamp"),
                    "risk_score": score,
                    "scoring_method": method,
                    "click_count": row["click_count"],
                    "key_count": row["key_count"],
                    "idle_time_total_ms": row["idle_time_total_ms"],
                    "avg_key_delay_ms": row.get("avg_key_delay_ms")
                })

                # Add feature-level details
                for feat, detail in details.items():
                    breakdown_entry = {
                        "login_id": login_id,
                        "username": username,
                        "scoring_method": method,
                        "feature": feat,
                        "value": detail.get("value"),
                        "feature_weight": FEATURE_WEIGHTS.get(feat, 1.0)
                    }

                    # Add method-specific columns
                    if method == "USER_PROFILE":
                        breakdown_entry.update({
                            "user_mean": detail.get("user_mean"),
                            "user_std": detail.get("user_std"),
                            "z_score": detail.get("z_score"),
                            "weighted_z": detail.get("weighted_z")
                        })
                    else:  # POPULATION
                        breakdown_entry.update({
                            "pop_p50": detail.get("pop_p50"),
                            "pop_p75": detail.get("pop_p75"),
                            "pop_p95": detail.get("pop_p95"),
                            "deviation": detail.get("deviation"),
                            "weighted_dev": detail.get("weighted_dev"),
                            "position": detail.get("position")
                        })

                    feature_breakdowns.append(breakdown_entry)

            scored_df = pd.DataFrame(scored_logins)
            features_df = pd.DataFrame(feature_breakdowns)

            # Create summary statistics
            summary_stats = []

            # Overall statistics
            summary_stats.append({
                "Metric": "Total Logins Analyzed",
                "Value": len(scored_df)
            })
            summary_stats.append({
                "Metric": "Mean Risk Score",
                "Value": f"{scored_df['risk_score'].mean():.3f}"
            })
            summary_stats.append({
                "Metric": "Median Risk Score",
                "Value": f"{scored_df['risk_score'].median():.3f}"
            })
            summary_stats.append({
                "Metric": "High Risk Logins (>0.7)",
                "Value": (scored_df['risk_score'] > 0.7).sum()
            })
            summary_stats.append({
                "Metric": "",
                "Value": ""
            })

            # By method
            for method in ["USER_PROFILE", "POPULATION"]:
                method_df = scored_df[scored_df["scoring_method"] == method]
                if len(method_df) > 0:
                    summary_stats.append({
                        "Metric": f"{method} - Count",
                        "Value": len(method_df)
                    })
                    summary_stats.append({
                        "Metric": f"{method} - Mean Score",
                        "Value": f"{method_df['risk_score'].mean():.3f}"
                    })
                    summary_stats.append({
                        "Metric": f"{method} - Std Dev",
                        "Value": f"{method_df['risk_score'].std():.3f}"
                    })
                    summary_stats.append({
                        "Metric": "",
                        "Value": ""
                    })

            # User coverage
            summary_stats.append({
                "Metric": "Total Users in System",
                "Value": len(all_users_df)
            })
            summary_stats.append({
                "Metric": "Users with Profiles",
                "Value": all_users_df["has_profile"].sum()
            })
            summary_stats.append({
                "Metric": "Users Needing More Data",
                "Value": (~all_users_df["has_profile"]).sum()
            })
            summary_stats.append({
                "Metric": "Min Logins for Profile",
                "Value": MIN_SAMPLES_FOR_PROFILE
            })

            summary_df = pd.DataFrame(summary_stats)

            # Population percentiles sheet
            pop_data = []
            for feat, pcts in pop_pct.items():
                pop_data.append({
                    "Feature": feat,
                    "Weight": FEATURE_WEIGHTS.get(feat, 1.0),
                    "P25": pcts["p25"],
                    "P50_Median": pcts["p50"],
                    "P75": pcts["p75"],
                    "P90": pcts["p90"],
                    "P95": pcts["p95"]
                })
            pop_df = pd.DataFrame(pop_data)

            # Write to Excel
            logging.info(f"[Report] Writing to {output_file}...")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                all_users_df.to_excel(writer, sheet_name="All Users", index=False)
                scored_df.to_excel(writer, sheet_name="Scored Logins", index=False)
                features_df.to_excel(writer, sheet_name="Feature Breakdown", index=False)
                pop_df.to_excel(writer, sheet_name="Population Stats", index=False)

                # If we have user profiles, add them
                if not profiles_df.empty:
                    # Reshape for readability
                    profile_summary = profiles_df[["username", "n", "sufficient_data"]].copy()
                    profile_summary.columns = ["Username", "Login_Count", "Has_Valid_Profile"]
                    profile_summary.to_excel(writer, sheet_name="User Profiles", index=False)

            logging.info(f"[Report] ✓ Report saved to {output_file}")

            # Print console summary
            print("\n" + "=" * 80)
            print(f"HEURISTIC SCORING REPORT GENERATED: {output_file}")
            print("=" * 80)
            print("\n📊 SUMMARY STATISTICS")
            print("-" * 80)
            print(summary_df.to_string(index=False))

            print("\n\n👥 USER COVERAGE")
            print("-" * 80)
            print(all_users_df.to_string(index=False))

            print("\n\n🎯 SCORE DISTRIBUTION BY METHOD")
            print("-" * 80)
            print(scored_df.groupby("scoring_method")["risk_score"].describe())

            print("\n\n🔝 TOP 10 HIGHEST RISK LOGINS")
            print("-" * 80)
            top_risk = scored_df.nlargest(10, "risk_score")[
                ["login_id", "username", "risk_score", "scoring_method", "click_count", "key_count"]
            ]
            print(top_risk.to_string(index=False))

            print("\n\n💡 KEY INSIGHTS")
            print("-" * 80)
            user_profile_count = (scored_df["scoring_method"] == "USER_PROFILE").sum()
            population_count = (scored_df["scoring_method"] == "POPULATION").sum()

            if user_profile_count > 0 and population_count > 0:
                user_mean = scored_df[scored_df["scoring_method"] == "USER_PROFILE"]["risk_score"].mean()
                pop_mean = scored_df[scored_df["scoring_method"] == "POPULATION"]["risk_score"].mean()
                diff = ((pop_mean - user_mean) / user_mean) * 100

                print(f"✓ User-profiled logins score {diff:.1f}% LOWER than population-based scoring")
                print(f"✓ This validates that personalization is critical for accurate risk assessment")
            elif user_profile_count > 0:
                print(f"✓ All {user_profile_count} logins used personalized user profiles for scoring")
                print(f"✓ Mean risk score: {scored_df['risk_score'].mean():.3f} (personalized baseline)")

            users_needing_data = (~all_users_df["has_profile"]).sum()
            if users_needing_data > 0:
                print(f"\n⚠️  {users_needing_data} users still need more logins before personalized scoring")
                print(f"   They need {MIN_SAMPLES_FOR_PROFILE} total logins for reliable profiles")
            else:
                print(f"\n✓ All {len(all_users_df)} users have sufficient data for personalized profiles")
                print(f"   System is fully personalized - no users falling back to population scoring")

            high_risk_count = (scored_df["risk_score"] > 0.7).sum()
            if high_risk_count > 0:
                print(f"\n⚠️  {high_risk_count} high-risk logins detected (score > 0.7)")
                print(f"   Review these in the 'Scored Logins' sheet for potential anomalies")

            print("\n" + "=" * 80)

            return output_file

    except SQLAlchemyError as e:
        logging.error(f"[Report] Failed: {e}")
        return None


def preview_heuristics(sample_limit: int = 2000):
    """
    Generate comprehensive report (replaces simple preview).
    """
    create_detailed_report()


def retroactively_backfill_heuristics(batch_size: int = 5000):
    """
    Compute and store heuristic scores for all logins with nn_score = -1.0
    """
    try:
        with engine.begin() as conn:
            # Build user profiles
            profiles_df = build_user_profiles(conn, min_samples=MIN_SAMPLES_FOR_PROFILE)
            pop_pct = compute_population_percentiles(conn, sample_size=20000)

            # Convert profiles to dict for faster lookup
            profiles_dict = {}
            for _, row in profiles_df.iterrows():
                if row["sufficient_data"]:
                    profiles_dict[row["username"]] = row.to_dict()

            logging.info(f"[Backfill] Loaded {len(profiles_dict)} user profiles.")

            # Process in batches
            base_query = text(f"""
                SELECT login_id, username, {', '.join(cfg("data.feature_columns"))}
                FROM rba_login_event
                WHERE nn_score = -1.0
                ORDER BY login_id
                LIMIT :limit
                OFFSET :offset
            """)

            update_statement = text("""
                                    UPDATE rba_login_event
                                    SET nn_score = :score
                                    WHERE login_id = :id
                                    """)

            total_updated = 0
            current_offset = 0

            while True:
                logging.info(f"Processing batch at offset {current_offset}...")

                result = conn.execute(
                    base_query,
                    {"limit": batch_size, "offset": current_offset}
                )
                rows = result.mappings().all()

                if not rows:
                    logging.info("No more rows to process.")
                    break

                # Score each login
                updates = []
                for row in rows:
                    username = row["username"]

                    if username in profiles_dict:
                        score, _ = score_against_user_profile(row, profiles_dict[username])
                    else:
                        score, _ = score_against_population(row, pop_pct)

                    updates.append({"score": score, "id": row["login_id"]})

                # Batch update
                if updates:
                    conn.execute(update_statement, updates)
                    total_updated += len(updates)
                    logging.info(f"Updated {len(updates)} logins in this batch.")

                current_offset += batch_size

                if len(rows) < batch_size:
                    break

            logging.info(f"[Backfill] Complete. Total updated: {total_updated}")

    except SQLAlchemyError as e:
        logging.error(f"[Backfill] Failed: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="User-centric heuristic scoring for Risk-Based Authentication (RBA)."
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate detailed Excel report with scoring analysis (no database writes)."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output filename for Excel report (default: heuristic_report_TIMESTAMP.xlsx)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=5000,
        help="Number of rows per batch for backfill (default: 5000)."
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply heuristic scores to database (requires backup)."
    )

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.preview:
        create_detailed_report(output_file=args.output)
    elif args.write:
        retroactively_backfill_heuristics(batch_size=args.batch)
    else:
        print("No action specified. Use --preview or --write.\n")
        parser.print_help()
