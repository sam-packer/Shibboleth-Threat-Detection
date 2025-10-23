# Shibboleth Threat Detection

A Flask webserver that uses a neural network to classify logins based on their risk level. This is the backend that
takes data from Shibboleth IdP and will return the score.

You will need the [Shibboleth plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin) as well for this to fully
work.

## Installation

You'll need to download the Kaggle dataset from [here](https://www.kaggle.com/datasets/dasgroup/rba-dataset). Then, move
the `archive.zip` folder into the root directory of your Python directory.

Create a `.env` file and put a MaxMind License Key in. You can sign up [here](https://www.maxmind.com/en/home) and once
you have an account, go to "Manage License Keys". Then, create a new key and put it in the environment file as follows

```dotenv
MAXMIND_LICENSE_KEY=my_license_key
```

You can then run the `train.py` file to train the neural network. You'll get `scaler.joblib`, `rba_model.pth`, and
`feature_maps.json`. Keep these files as you will need them when running the Flask app. Note that I recommend at least
32GB of RAM when training the model. 48-64GB of RAM is ideal, but you only need to train the model once. Once it's
trained, the application can run with very little RAM. You *will* be waiting for the archive.zip file to extract and
load into memory, so please be patient. Even if it appears the model training is stuck, it is not.

## Running

You'll want to run the Flask app (under `app.py`) and reverse proxy the endpoint with something such as Caddy. Then,
configure the Shibboleth plugin to point to your endpoint. I highly recommend load balancing two instances of this
application. It is stateless and does not require a database to be kept in sync.

It will automatically download the MaxMind GeoIP databases when you run it for the first time. It will also check for
any updates from MaxMind each time you restart the application. If an update was found, it will automatically update the
GeoIP databases for you.