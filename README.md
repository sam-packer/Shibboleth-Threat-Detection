## Shibboleth Threat Detection

A Flask webserver that uses a neural network to classify logins based on their risk level. This is the backend that takes data from Shibboleth IdP and will return the score.

You will need the [Shibboleth plugin](https://github.com/sam-packer/Shibboleth-RBA-Plugin) as well for this to fully work.

## Installation

You'll need to download the Kaggle dataset from [here](https://www.kaggle.com/datasets/dasgroup/rba-dataset). Then, move the `archive.zip` folder into the root directory of your Python directory.

You can then run the `train.py` file to train the neural network. You'll get `scaler.joblib`, `rba_model.pth`, and `feature_maps.json`. Keep these files as you will need them when running the Flask app.

## Running

You'll want to run the Flask app (under `app.py`) and reverse proxy the endpoint with something such as Caddy. Then, configure the Shibboleth plugin to point to your endpoint.