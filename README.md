# Shibboleth Threat Detection

A Flask webserver that uses a neural network to classify logins based on their risk level. This is the backend that
takes data from Shibboleth IdP and will return the threat score.

You will need the [Shibboleth RBA Plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin) (also developed by me) as
well for this to fully work.

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

Create a `.env` file and put a MaxMind License Key in. You can sign up [here](https://www.maxmind.com/en/home) and once
you have an account, go to "Manage License Keys". Then, create a new key and put it in the environment file as follows:

```dotenv
MAXMIND_LICENSE_KEY=my_license_key
```

You'll also want to add a PostgreSQL connection string to the `.env` file. An example is below:

```dotenv
POSTGRES_CONNECTION_STRING=postgresql://your_username_here:your_password_here@127.0.0.1:5432/your_database_here
```

### Seeding the database

Instructions coming soon...

## Running

You'll want to run the Flask app (the uv command is: `uv run app.py`) and reverse proxy the endpoint with something such
as Caddy. Then, configure the Shibboleth plugin to point to your endpoint. I highly recommend load balancing two
instances of this application.

There is a PostgreSQL database requirement. It is sharded and should perform extremely well. You can scale it as much as
you need and this project uses the Citus extension to do so. I recommend purging logins after a year (except malicious
ones). Since this isn't time series data, there isn't a problem with having a cluster of older malicious logins while
accumulating newer login data.

Each time you start the Flask web server, it will check for updates from MaxMind and StopForumSpam for new files. If
they are found, they will automatically be downloaded. It will also try and connect to the database and run a simple
query. This is called a "preflight check" and confirms your environment is set up correctly.