# Shibboleth Threat Detection

A Flask webserver that uses a neural network to classify logins based on their risk level. This is the backend that
takes data from Shibboleth IdP and will return the score.

You will need the [Shibboleth plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin) as well for this to fully
work.

## Installation

Create a `.env` file and put a MaxMind License Key in. You can sign up [here](https://www.maxmind.com/en/home) and once
you have an account, go to "Manage License Keys". Then, create a new key and put it in the environment file as follows

```dotenv
MAXMIND_LICENSE_KEY=my_license_key
```

More instructions coming soon...

## Running

You'll want to run the Flask app (under `app.py`) and reverse proxy the endpoint with something such as Caddy. Then,
configure the Shibboleth plugin to point to your endpoint. I highly recommend load balancing two instances of this
application. It is stateless and does not require a database to be kept in sync.

It will automatically download the MaxMind GeoIP databases when you run it for the first time. It will also check for
any updates from MaxMind each time you restart the application. If an update was found, it will automatically update the
GeoIP databases for you.