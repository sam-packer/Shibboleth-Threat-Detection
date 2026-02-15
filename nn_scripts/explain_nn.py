import os

import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from helpers.globals import select_device, resolve_path, cfg
from nn_scripts.model import BehavioralEncoder

load_dotenv()

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
FEATURE_COLUMNS = cfg("data.feature_columns")
LIMIT = cfg("data.limit")

MODEL_PATH = resolve_path("NN_MODEL_PATH", os.path.join(cfg("model.output_dir"), cfg("model.checkpoint")))
SCALER_PATH = resolve_path("NN_SCALER_PATH",
                           os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.scaler")))
USER_MAP_PATH = resolve_path("NN_USER_MAP_PATH",
                             os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.user_map")))
PREPROCESSOR_PATH = resolve_path("NN_PREPROCESSOR_PATH", os.path.join(cfg("preprocessing.output_dir"),
                                                                      cfg("preprocessing.artifacts.preprocessor")))
CENTROIDS_PATH = resolve_path("NN_CENTROIDS_PATH",
                              os.path.join(cfg("preprocessing.output_dir"), cfg("preprocessing.artifacts.centroids")))
DISTANCE_STATS_PATH = resolve_path("NN_DISTANCE_STATS_PATH",
                                   os.path.join(cfg("preprocessing.output_dir"),
                                                cfg("preprocessing.artifacts.distance_stats")))

_device = select_device()


def load_data():
    engine = create_engine(POSTGRES_CONNECTION_STRING)
    query = text(f"""
        SELECT username, {', '.join(FEATURE_COLUMNS)}, platform, device_category
        FROM {cfg("data.table")}
        ORDER BY random()
        LIMIT :limit
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit": LIMIT})
    return df


def main():
    print("Loading neural network artifacts...")
    scaler = joblib.load(SCALER_PATH)
    user_to_id = joblib.load(USER_MAP_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    centroids = joblib.load(CENTROIDS_PATH)
    distance_stats = joblib.load(DISTANCE_STATS_PATH)

    # Load Data
    df = load_data()
    print(f"Loaded {len(df)} rows for explanation.")

    # Preprocess
    df_transformed = preprocessor.transform_df(df.copy(), FEATURE_COLUMNS)
    X_raw = df_transformed[FEATURE_COLUMNS].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # Load Model
    input_dim = len(FEATURE_COLUMNS)
    model = BehavioralEncoder(input_dim=input_dim).to(_device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.eval()
    print("Model loaded successfully.")

    # Compute embeddings
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(_device)
    with torch.no_grad():
        embeddings = model(X_tensor).cpu().numpy()

    # Compute per-sample distances to user centroid
    usernames = df["username"].values
    distances = np.zeros(len(df))
    for i, username in enumerate(usernames):
        if username in centroids:
            centroid = centroids[username]
        else:
            centroid = centroids["__population__"]
        distances[i] = 1.0 - np.dot(embeddings[i], centroid)

    distances = np.clip(distances, 0.0, 2.0)

    # Assign colors: top N users by frequency get unique colors, rest are gray
    user_counts = pd.Series(usernames).value_counts()
    top_users = user_counts.head(10).index.tolist()
    color_map = {}
    cmap = plt.cm.tab10
    for i, u in enumerate(top_users):
        color_map[u] = cmap(i)

    colors = [color_map.get(u, (0.7, 0.7, 0.7, 0.3)) for u in usernames]

    # t-SNE
    print("Computing t-SNE embedding (this may take a moment)...")
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=cfg("data.random_state"))
    tsne_result = tsne.fit_transform(embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # t-SNE Plot — use circles for desktop, triangles for mobile
    device_cats = df["device_category"].values
    desktop_mask = np.array([str(d) != "mobile" for d in device_cats])
    mobile_mask = ~desktop_mask

    if desktop_mask.any():
        ax1.scatter(tsne_result[desktop_mask, 0], tsne_result[desktop_mask, 1],
                    c=[colors[i] for i in range(len(colors)) if desktop_mask[i]],
                    s=8, alpha=0.6, marker="o")
    if mobile_mask.any():
        ax1.scatter(tsne_result[mobile_mask, 0], tsne_result[mobile_mask, 1],
                    c=[colors[i] for i in range(len(colors)) if mobile_mask[i]],
                    s=12, alpha=0.6, marker="^")

    ax1.set_title("t-SNE Embedding Visualization")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    # Add legend for top users + device category markers
    for u in top_users:
        ax1.scatter([], [], c=[color_map[u]], label=u, s=30)
    ax1.scatter([], [], c="gray", marker="o", label="Desktop", s=30)
    ax1.scatter([], [], c="gray", marker="^", label="Mobile", s=30)
    ax1.legend(loc="best", fontsize=7, title="Top Users / Device")

    # Distance Distribution
    ax2.hist(distances, bins=80, color="steelblue", edgecolor="white", alpha=0.8)
    ax2.axvline(x=np.percentile(distances, 95), color="orange", linestyle="--", label="p95")
    ax2.axvline(x=np.percentile(distances, 99), color="red", linestyle="--", label="p99")
    ax2.set_title("Distance-to-Centroid Distribution")
    ax2.set_xlabel("Cosine Distance")
    ax2.set_ylabel("Count")
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
