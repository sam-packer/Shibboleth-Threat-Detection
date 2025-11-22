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

#### Setting up MLFLow

You will want to set up [MLFlow](https://mlflow.org/) for this project. The environment variables to use can mostly be
left the same except for the host, token, and path. This uses the UC catalog, so you'll need to set up the schema. In
MLFlow, go to New > Query and run this:

```sql
CREATE
CATALOG shibboleth_rba;
GRANT ALL PRIVILEGES ON CATALOG shibboleth_rba TO `your-user@domain.com`;

CREATE SCHEMA shibboleth_rba.models;
```

All you'll need to do is create a personal access token and change the host. You should also created a new folder in
your Shared workspace called "Shibboleth RBA".

### Seeding the database

There are different versions of the metrics. Versions 1 and 2 were made prior to data collection. Version 3 is the first
version of data collection. Therefore, the versioning starts at v3. When you are starting a brand-new database, you
should use the latest version.

Look inside the `seeds` folder. The current version is `v4` and you should use that to seed your database. SQL files are
provided, and you can simply execute them to create the tables with the correct schema.

Sharding the database with Citus is incredibly easy and can be done as follows:

```sql
CREATE EXTENSION IF NOT EXISTS citus;
SELECT create_distributed_table('rba_device', 'device_uuid');
SELECT create_distributed_table('rba_login_event', 'username');
```

You can then add worker nodes and set the shard count depending on how big your environment is.

## Bootstrapping data

At first, you should run the project in passthrough mode. This is set in `.env` as `PASSTHROUGH_MODE=true|false`. This
will always assign a score of -1 and allow the login. This will allow you ample time to collect data and tweak the
heuristics script to ensure you have sufficient low risk logins. You'll also want to ensure you mark any known malicious
logins as such by changing the `nn_score` to `1.0` **AND** setting `human_verified` to `true`. You can use the
`create_synthetic_data.py` script to create some synthetic data as well, so the neural network has something to "catch".

## Training the neural network

It is recommended to use a NVIDIA GPU with CUDA or an Apple Silicon machine. Depending on how many rows you have,
training could take a long time. You will want to run the `train_nn.py` file. At the end, you will see a threshold. You
should set that as your IdP threshold in the Shibboleth plugin. That is the most effective threshold for catching
malicious logins. You can automate this entire process by running this command:

```shell
uv run train --update-shib /opt/shibboleth-idp/flows/intercept/rba/rba-beans.xml
```

Then, restart your Shibboleth IdP server and the web application for the changes to apply.

## Running the API

### Development

In development settings, you can run a development server with the following command:

```shell
uv run api
```

### Production

In production settings, you'll want to use Gunicorn on macOS / Linux or Waitress on Windows. Luckily, this entire
process is automated for you. To run the API in production, use the following command:

```shell
uv run api-prod
```

### Daemonizing

A sample `systemd` script is provided for your convenience. Depending on your installation and where you choose to clone
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

### Considerations

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