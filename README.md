# Shibboleth Threat Detection

This is a risk-based authentication system for Shibboleth IdP. It uses triplet metric learning to train an encoder
that maps behavioral login data into an embedding space. Each user's normal logins end up clustering together, so
anomaly detection is just measuring how far a new login is from that user's centroid. The further away it is, the more
suspicious. You don't need any labeled malicious data to train it, which means it works from day one once you've
collected enough logins. The anomaly score gets combined with IP reputation to produce a final threat score.

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
    - [uv setup](#uv-setup)
    - [Environment setup](#environment-setup)
    - [Storage setup](#storage-setup)
    - [Setting up MLFlow](#setting-up-mlflow)
    - [Shibboleth setup](#shibboleth-setup)
- [Dataset](#dataset)
    - [Data source](#data-source)
    - [Features](#features)
    - [Training data recommendations](#training-data-recommendations)
- [Data collection and training](#data-collection-and-training)
    - [Getting to production](#getting-to-production)
    - [Training the model](#training-the-model)
    - [Model architecture](#model-architecture)
    - [Model explainability](#model-explainability)
    - [Deployment modes: MLFlow vs. Local](#deployment-modes-mlflow-vs-local)
    - [Ongoing maintenance](#ongoing-maintenance)
- [Deploying](#deploying)
    - [Bare metal](#bare-metal)
- [Reproducibility](#reproducibility)
- [Considerations](#considerations)
- [License](#license)

## Requirements

- [Shibboleth IdP with the RBA plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin)
- PostgresSQL 17
- Python 3.11+
- `uv`
- Load balancer (recommended)
- Server with CUDA or MPS support (recommended)

## Installation

### uv setup

This project uses [uv](https://docs.astral.sh/uv/) to manage packages. Create a virtual environment and install
dependencies:

```shell
uv venv
uv sync
```

Then activate the virtual environment:

```shell
source .venv/bin/activate     # macOS / Linux
.\.venv\Scripts\activate      # Windows
```

### Environment setup

Copy the `.env.example` file and fill in the values. You'll need a MaxMind License Key. Sign
up [here](https://www.maxmind.com/en/home), go to "Manage License Keys", create a new key, and add it:

```dotenv
MAXMIND_LICENSE_KEY=my_license_key
```

You'll also need a PostgresSQL connection string:

```dotenv
POSTGRES_CONNECTION_STRING=postgresql://your_username_here:your_password_here@127.0.0.1:5432/your_database_here
```

### Storage setup

This project uses Postgres and Citus. You can follow
[this tutorial](https://www.postgresql.org/download/linux/debian/) to set up Postgres on Debian. For the Citus
extension, refer to the [Citus installation guide](https://www.citusdata.com/download/).

The metrics schema is versioned. The current version is v1. When setting up your database, seed it with v1. You can do
this manually (see the `seeds` folder) or with `uv run seed`.

Sharding with Citus is straightforward:

```sql
CREATE EXTENSION IF NOT EXISTS citus;
SELECT create_distributed_table('rba_device', 'device_uuid');
SELECT create_distributed_table('rba_login_event', 'username');
```

You can then add worker nodes and set the shard count depending on how big your environment is.

### Setting up MLFlow

You'll want [MLFlow](https://mlflow.org/) set up for this project. It uses the UC catalog, so you need to create
the schema. In MLFlow, go to New > Query and run:

```sql
CREATE
CATALOG shibboleth_rba;
GRANT ALL PRIVILEGES ON CATALOG shibboleth_rba TO `your-user@domain.com`;

CREATE SCHEMA shibboleth_rba.models;
```

Then create a personal access token (Account Settings > Developer) and add these to your `.env` file. If you're using
Databricks, the host is the same URL you use to access your MLFlow dashboard:

```dotenv
DATABRICKS_HOST=example.cloud.databricks.com
DATABRICKS_TOKEN=<your-personal-access-token>
```

### Shibboleth setup

For the Shibboleth side, see the [RBA plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin).

## Dataset

### Data source

Login data comes from real Shibboleth IdP authentication events. You don't need any labeled malicious data. The system
learns what normal looks like for each user and flags anything that deviates from it.

### Features

25 behavioral metrics including:

- Typing patterns (key_count, avg_key_delay_ms)
- Mouse behavior (pointer_distance_px, pointer_event_count)
- Session timing (total_session_time_ms, idle_time_total_ms, active_time_ms)
- Device fingerprinting (screen dimensions, platform, hardware_concurrency)
- Contextual data (GeoIP, timezone, IP reputation)

### Training data recommendations

- Minimum: 1,000 logins
- Recommended: 5,000+ logins
- Per-user minimum: 10 logins for a personalized centroid (users below this fall back to a population centroid)

## Data collection and training

### Getting to production

1. **Collect data.** Deploy the API and let logins accumulate in the database. Before the model is trained, the API
   returns a neutral score (0.5) for all logins, so nothing gets blocked.
2. **Accumulate logins.** Wait until you have at least ~1,000 logins with a reasonable number of users having 10+
   logins each.
3. **Train.** Run `uv run train`. The encoder learns an embedding space where each user's logins cluster together.
   Centroids and distance thresholds are computed automatically.
4. **Deploy.** Restart the API so it picks up the trained model. Logins will now be scored based on how anomalous they
   are relative to each user's established behavior.

### Training the model

```shell
uv run train
```

This trains the encoder, computes per-user centroids, and determines a global anomaly threshold (p99 of all training
distances). Artifacts saved: encoder, scaler, preprocessor, user map, centroids, distance statistics, and threshold
file.

### Model architecture

The encoder is a feedforward network that projects behavioral features into a 32-dimensional embedding space:

- Input: 25 behavioral features
- Hidden layers: [64, 48] with BatchNorm, ReLU, and Dropout(0.15)
- Output: 32-dim embedding, L2-normalized onto the unit hypersphere
- Loss: TripletMarginLoss (margin=0.3, p=2)
- Optimizer: Adam (lr=0.001) with ReduceLROnPlateau scheduler

There is no user embedding layer. User identity is captured via post-hoc centroids (mean of a user's embeddings),
which means new users work immediately by falling back to a population centroid.

Anomaly scoring works by computing cosine distance from a login's embedding to the user's centroid:
- Distance at or below the user's mean: score ~0.05 (clearly normal)
- Distance between mean and p95: linear ramp from 0.05 to 0.50
- Distance above p95: exponential escalation toward 1.0

### Model explainability

You can generate a t-SNE visualization of the embedding space and a distance distribution histogram:

```shell
uv run explain
```

The t-SNE plot shows user clusters in embedding space (top users colored, others gray). The histogram shows the
distribution of distances to centroids with p95 and p99 lines marked.

### Deployment modes: MLFlow vs. Local

#### MLFlow mode (recommended for production)

In MLFlow mode, MLFlow is the single source of truth for model versions and their thresholds. When you train a model,
the threshold is logged as a metric alongside the model registration. The API keeps an in-memory cache that periodically
queries MLFlow for current model versions and thresholds, and exposes this via the `/models` endpoint.

Shibboleth runs in "dynamic" mode, pulling thresholds directly from the API. When a login is scored, Shibboleth gets
both the risk score and the model version, then queries the API for the threshold for that version. Because both are
resolved at request time and versioned together, there's no race condition between model updates and threshold
propagation, even across load-balanced servers.

Why you'd want this:

- No Shibboleth restarts needed after training
- Safe for multi-node and load-balanced deployments
- Model and threshold always stay in sync
- You can train anywhere and all API instances pick it up

To set it up, enable MLFlow in `config.yml` and configure your Shibboleth `rba-beans.xml` with the API endpoint URL
(not a static threshold value). `uv run shib-update` is not used in this mode.

#### Local mode (development only)

When MLFlow is disabled, the threshold is written to a local file (`nn_data/threshold.txt`) after training. This is
fine for development or simple single-node setups, but it doesn't scale well.

If you want to push the threshold to Shibboleth, run:

```shell
uv run shib-update
```

This reads the local threshold file and updates your Shibboleth XML configuration. You'll need to restart Shibboleth
for it to take effect.

The limitations are what you'd expect: you have to train on the same server running the API, you need manual restarts
after every threshold update, and there's no protection against model/threshold version mismatches across multiple
servers. MLFlow must be disabled in `config.yml`, and `rba-beans.xml` must use a static threshold value. See the RBA
plugin for more instructions.

### Ongoing maintenance

Once the API is running with a trained model, you should set up a scheduler to retrain regularly.
User behavior changes over time, so periodic retraining keeps the centroids fresh. In MLFlow mode, threshold updates
propagate automatically. In Local mode, you need to manually run `uv run shib-update` and restart Shibboleth after
each training run.

## Deploying

You can run the API on the same server as Shibboleth or on separate servers. Once you've done your data collection
and model training, serving the API is straightforward. It runs within Gunicorn (macOS/Linux) or Waitress (Windows),
and the detection is automatic:

```shell
uv run api-prod
```

### Bare metal

Clone the repository, set the environment variables, update `config.yml`, and run the API. This works on any cloud
provider (AWS EC2, GCP Compute Engine, Azure VM, etc.) or on a local server. If you're not using MLFlow, you'll need
to train the model locally before running the API. You don't need to retrain on the server if you're using MLFlow.

Here's a sample systemd unit file. Put it in `/etc/systemd/system/shib-predict.service`:

```ini
[Unit]
Description = Shibboleth IdP RBA Prediction
After = network.target

[Service]
User = [your-user]
WorkingDirectory = /path/to/shib-predict
ExecStart = /path/to/uv run api-prod
Restart = always
RestartSec = 5

Environment = "PYTHONUNBUFFERED=1"

[Install]
WantedBy = multi-user.target
```

## Reproducibility

Everything is configuration-driven. All hyperparameters, model architecture, and training settings live in `config.yml`.
Random seeds are set globally (`RANDOM_STATE=41`) for deterministic splits, and preprocessing pipelines are serialized
alongside models. Dependencies are locked via `uv.lock`.

MLFlow logs every training run with parameters, metrics, and artifacts. You can roll back to any previous model version.
Trained models are stored in Unity Catalog with lineage tracking, and all API instances pull from the same registry, so
load-balanced servers stay in sync without manual file transfers.

The training data is real authentication events and can't be publicly shared for privacy reasons, but the pipeline
itself is fully reproducible for any Shibboleth deployment: deploy the API, let logins accumulate, then run
`uv run train`. The methodology is the same regardless of institution.

## Considerations

You should load balance this application with something like Caddy, then point the Shibboleth plugin at your
load-balanced endpoint. If you're using MLFlow mode, all servers stay in sync automatically.

There is a PostgresSQL requirement. I recommend sharding with Citus, which works well and scales to whatever your
Shibboleth IdP needs.

Each time you start the web application, it checks for updates from MaxMind and StopForumSpam and downloads new files
if available. It also connects to the database and runs a simple query as a preflight check to confirm your environment
is set up correctly.

## License

This software is licensed under the PolyForm Noncommercial License 1.0.0. You may use, copy, and modify this
software for noncommercial purposes only. See [LICENSE.md](LICENSE.md) for the full license text.

Copyright © 2025-2026 Sam Packer. Released under the PolyForm Noncommercial License 1.0.0.

---

![ShibBlue Logo](assets/shibboleth%20blue.png)
