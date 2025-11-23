# Shibboleth Threat Detection

This is a risk-based authentication system designed for use with Shibboleth IdP. Behavioral data is collected and
trained with a neural network. The neural network adapts scoring to each user's behavior and learns "how risky is this
user's login compared to their previous logins". This is an important distinction from a neural network that learns how
risky a user's login is compared to an average login, which is **not** the project's goal. The behavioral data is
ensembled with other heuristics such as how risky an IP is and whether impossible travel occurred.

## Requirements

- [Shibboleth IdP with the RBA plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin)
- PostgresSQL 17
- Python 3.14+
- `uv`
- Load balancer (recommended)
- Server with CUDA or MPS support (recommended)

## Dataset

### Steps to productionalizing

There are multiple steps to productionalizing this. The first is collecting data. You should ensure that you have
passthrough mode turned on. This collects data silently and allows logins through Shibboleth. After you have data (~
1,000 logins minimum), you must calculate the scores for known good logins. We've developed a heuristic helper to do so.
However, it still requires manual intervention and tuning depending on your results.
After calculating known good logins, you must figure out a way to get malicious logins. You have two options: flagging
known malicious logins or using the synthetic generator to create malicious logins. After this, you are ready to train
your neural network! You can turn passthrough mode off after training and ensuring your API is properly setup. Your
model will now start truly scoring logins. At this point, the next step is to set up a scheduler to retrain your neural
network frequently. You should also update the threshold for denying a login. You can do this automatically, and it can
be configured in `config.yml` or using the CLI. The CLI command for automatically training the neural network and
updating the threshold in Shibboleth is as follows:

```shell
uv run train --update-shib /opt/shibboleth-idp/flows/intercept/rba/rba-beans.xml
```

### Columns (features)

- source of data
- feature list / description
- target definition
- example record
- data volume / characteristics

## Storage Setup

This project uses Postgres and Citus. You can follow this
tutorial [here](https://www.postgresql.org/download/linux/debian/) to set up Postgres on Debian. To add the Citus
extension to Postgres, please refer to the Citus installation guide [here](https://www.citusdata.com/download/).

To ensure strong data consistency, the data collected from Shibboleth is versioned. The latest version of the metrics is
v4. When setting up your database, you should seed it with v4. You can seed your database manually or with the
`uv run seed` command. If you choose to do it manually, please refer to the `seeds` folder for the schema.

Sharding the database with Citus is incredibly easy and can be done as follows:

```sql
CREATE EXTENSION IF NOT EXISTS citus;
SELECT create_distributed_table('rba_device', 'device_uuid');
SELECT create_distributed_table('rba_login_event', 'username');
```

You can then add worker nodes and set the shard count depending on how big your environment is.

## Installation

### uv Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage packages. First, create a uv virtual environment and
install the necessary dependencies with the following commands:

```shell
uv venv
uv sync
```

Then, activate the virtual environment using the standard commands:

```shell
source .venv/bin/activate     # macOS / Linux
.\.venv\Scripts\activate      # Windows
```

### Environment setup

Copy the `.env.example` file and fill in the values. This involves putting in a MaxMind License Key in. You can sign
up [here](https://www.maxmind.com/en/home) and once you have an account, go to "Manage License Keys". Then, create a new
key and put it in the environment file as follows:

```dotenv
MAXMIND_LICENSE_KEY=my_license_key
```

You'll also want to add a PostgresSQL connection string to the `.env` file. An example is below:

```dotenv
POSTGRES_CONNECTION_STRING=postgresql://your_username_here:your_password_here@127.0.0.1:5432/your_database_here
```

### Setting up MLFLow

You will want to set up [MLFlow](https://mlflow.org/) for this project. The environment variables to use can mostly be
left the same except for the host and token. This uses the UC catalog, so you'll need to set up the schema. In
MLFlow, go to New > Query and run this:

```sql
CREATE
CATALOG shibboleth_rba;
GRANT ALL PRIVILEGES ON CATALOG shibboleth_rba TO `your-user@domain.com`;

CREATE SCHEMA shibboleth_rba.models;
```

Now, create a personal access token and change the host. You can do this in Account Settings > Developer. The host
depends on your MLFlow setup. Under the assumption you are using Databricks, it is the same as the URL you use to access
your MLFlow dashboard.

## Deploying

There are multiple approaches to deploying. You can run the API on the same server as your Shibboleth server or run it
on separate servers. You will also need to decide if you plan to run this in Docker or bare metal. Under the assumption
you have already done your data collection and model training, serving the API endpoint is quite simple. You will need
to run the API within Gunicorn (macOS / Linux) or Waitress (Windows). This is automatically handled for you and you can
use one simple command to detect your setup and automatically spin up the server:

```shell
uv run api-prod
```

## Bare metal

If you run it on bare metal, the instructions are quite simple. Clone the GitHub repository, and run the API. A sample
systemd script is provided below for your convenience. Depending on your installation and where you choose to clone
the project, you may need to change the paths. Hopefully you follow better practices than me and don't clone your
projects in the root user's home folder. You can put this in `/etc/systemd/system/shib-predict.service`:

```ini
[Unit]
Description = Shibboleth IdP RBA Prediction
After = network.target

[Service]
User = root
WorkingDirectory = /root/shib-predict
ExecStart = /root/.local/bin/uv run api-prod
Restart = always
RestartSec = 5

Environment = "PYTHONUNBUFFERED=1"

[Install]
WantedBy = multi-user.target
```

## Docker

You can run the entire project (Citus, Caddy reverse proxy, and Flask endpoint) using Docker if you desire. The
instructions are as follows:

- coming soon

## Shibboleth Setup

For more details on setting things up on the Shibboleth side, please view
the [RBA plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin).

## Considerations

It is highly recommended to load balance this application with software such as Caddy. Then, configure the Shibboleth
plugin to point to your load balanced endpoint.

There is a PostgresSQL database requirement. It is very easy to shard the database with Citus and what I recommend. This
will perform extremely well. You can scale it as much as you need depending on how active your Shibboleth IdP instance
is. I recommend purging logins after a year (except malicious or human verified ones). Since this isn't time series
data, there isn't a problem with having a cluster of older malicious logins while accumulating newer login data.

Each time you start the web application, it will check for updates from MaxMind and StopForumSpam for new files. If
any are found, they will automatically be downloaded. It will also try and connect to the database and run a simple
query. This is called a "preflight check" and confirms your environment is set up correctly.

---

![ShibBlue Logo](assets/shibboleth%20blue.png)