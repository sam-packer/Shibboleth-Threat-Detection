import torch
import json
from flask import Flask, request, jsonify
from model import RBAModel
import joblib
import geoip2.database
from user_agents import parse
from datetime import datetime
import os
import requests
import tarfile
import io
from dotenv import load_dotenv

# --- 0. LOAD ENVIRONMENT VARIABLES ---
load_dotenv()
print("Loading environment variables...")

app = Flask(__name__)


# --- 1. DOWNLOAD HELPER ---
def download_and_extract_maxmind_db(edition_id, license_key, dest_path):
    """
    Downloads and extracts a MaxMind GeoLite2 database.
    """
    print(f"Attempting to download '{edition_id}' database...")
    download_url = f"https://download.maxmind.com/app/geoip_download?edition_id={edition_id}&license_key={license_key}&suffix=tar.gz"

    try:
        response = requests.get(download_url, timeout=30)

        if response.status_code == 401:
            print(f"  > ERROR: HTTP 401 Unauthorized. Your MAXMIND_LICENSE_KEY is likely invalid or expired.")
            return False

        response.raise_for_status()  # Raise an exception for other bad statuses (4xx, 5xx)

        print(f"  > Download successful. Extracting '{dest_path}'...")

        # Open the .tar.gz file from in-memory content
        with io.BytesIO(response.content) as f:
            with tarfile.open(fileobj=f, mode="r:gz") as tar:
                # Find the .mmdb file in the archive
                db_member = None
                for member in tar.getmembers():
                    if member.name.endswith('.mmdb'):
                        db_member = member
                        break

                if not db_member:
                    print(f"  > ERROR: Could not find a .mmdb file in the downloaded archive for {edition_id}.")
                    return False

                # Extract the .mmdb file to our destination path
                with tar.extractfile(db_member) as source, open(dest_path, "wb") as dest:
                    dest.write(source.read())

        print(f"  > Successfully extracted and saved to '{dest_path}'.")
        return True

    except requests.exceptions.RequestException as e:
        print(f"  > ERROR: Failed to download {edition_id} database: {e}")
        return False
    except tarfile.TarError as e:
        print(f"  > ERROR: Failed to extract {edition_id} database: {e}")
        return False
    except Exception as e:
        print(f"  > ERROR: An unexpected error occurred during download/extraction: {e}")
        return False


# --- 2. LOAD ALL ARTIFACTS ON STARTUP ---
print("Loading RBA model and all required artifacts...")
model = None
scaler = None
feature_maps = None
city_reader = None
asn_reader = None

try:
    # --- A. Check for License Key ---
    MAXMIND_LICENSE_KEY = os.getenv("MAXMIND_LICENSE_KEY")
    if not MAXMIND_LICENSE_KEY or MAXMIND_LICENSE_KEY == "YOUR_KEY_HERE":
        raise ValueError("MAXMIND_LICENSE_KEY not set in .env file. Please get a key from MaxMind.")

    # --- B. Define DB Paths and check/download ---
    CITY_DB_PATH = 'GeoLite2-City.mmdb'
    ASN_DB_PATH = 'GeoLite2-ASN.mmdb'

    if not os.path.exists(CITY_DB_PATH):
        print(f"'{CITY_DB_PATH}' not found.")
        if not download_and_extract_maxmind_db('GeoLite2-City', MAXMIND_LICENSE_KEY, CITY_DB_PATH):
            raise FileNotFoundError(f"Failed to download {CITY_DB_PATH}")

    if not os.path.exists(ASN_DB_PATH):
        print(f"'{ASN_DB_PATH}' not found.")
        if not download_and_extract_maxmind_db('GeoLite2-ASN', MAXMIND_LICENSE_KEY, ASN_DB_PATH):
            raise FileNotFoundError(f"Failed to download {ASN_DB_PATH}")

    # --- C. Check for all required files before proceeding ---
    required_files = ['rba_model.pth', 'feature_maps.json', 'scaler.joblib', CITY_DB_PATH, ASN_DB_PATH]
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required artifact '{f}' is missing.")

    # --- D. Load artifacts into memory ---
    print("Loading feature maps...")
    with open('feature_maps.json', 'r') as f:
        feature_maps = json.load(f)

    print("Loading numerical scaler...")
    scaler = joblib.load('scaler.joblib')

    print("Loading GeoIP databases...")
    city_reader = geoip2.database.Reader(CITY_DB_PATH)
    asn_reader = geoip2.database.Reader(ASN_DB_PATH)

    print("Loading PyTorch model...")
    # Instantiate the model with the correct dimensions from the loaded artifacts
    vocab_sizes = {col.replace('_to_id', ''): len(feature_maps[col]) for col in feature_maps}
    embedding_dims = {
        'username': 50, 'Country': 20, 'City': 30,
        'OS': 20, 'Browser': 20, 'DeviceType': 10, 'ASN': 25
    }
    num_numerical_feats = len(scaler.mean_) if hasattr(scaler, 'mean_') else scaler.n_features_in_
    model = RBAModel(vocab_sizes, embedding_dims, num_numerical_feats=num_numerical_feats)

    # Load the trained weights
    model.load_state_dict(torch.load('rba_model.pth'))
    model.eval()  # Set the model to evaluation mode (very important!)

    print("\nAll artifacts loaded successfully. API is ready.\n")

except (FileNotFoundError, ValueError) as e:
    print("\n--- FATAL ERROR ---")
    print(f"Error during startup: {e}")
    print("Please check your .env file and ensure 'train.py' has been run.")
    print("API will not start correctly.")
    print("-------------------\n")
    model = None  # Prevent the app from running if setup failed


# --- 3. FEATURE ENGINEERING FUNCTION (MUST MIRROR TRAINING SCRIPT) ---
def create_features_from_request(data):
    """
    Takes raw JSON data from Shibboleth and performs the exact same
    feature engineering as the training script.
    """
    # Raw features from IdP
    ip_address = data.get('ipAddress')
    user_agent_string = data.get('userAgent', '')

    # --- IP-based features ---
    country, city, asn_val = 'Unknown', 'Unknown', '0'
    try:
        if city_reader and ip_address:
            city_response = city_reader.city(ip_address)
            country = city_response.country.iso_code or 'Unknown'
            city = city_response.city.name or 'Unknown'

        if asn_reader and ip_address:
            asn_response = asn_reader.asn(ip_address)
            asn_val = str(asn_response.autonomous_system_number or '0')

    except geoip2.errors.AddressNotFoundError:
        # This is expected for private/internal IP addresses
        pass  # Keep defaults

    # --- User Agent based features ---
    user_agent = parse(user_agent_string)
    device_type = 'desktop' if user_agent.is_pc else 'mobile' if user_agent.is_mobile else 'tablet' if user_agent.is_tablet else 'bot' if user_agent.is_bot else 'unknown'
    os_name = user_agent.os.family or 'Unknown'
    browser_name = user_agent.browser.family or 'Unknown'

    # --- Timestamp based features ---
    now = datetime.utcnow()
    login_hour = now.hour
    day_of_week = now.weekday()

    # --- Assemble feature dictionaries ---
    categorical_features = {
        'username': data.get('username'),
        'Country': country,
        'City': city,
        'OS': os_name,
        'Browser': browser_name,
        'DeviceType': device_type,
        'ASN': asn_val
    }

    numerical_features = {
        'LoginHour': login_hour,
        'DayOfWeek': day_of_week
    }

    return categorical_features, numerical_features


# --- 4. FLASK API ENDPOINT ---
@app.route('/score', methods=['POST'])
def get_threat_score():
    if not model or not scaler or not feature_maps:
        return jsonify({'error': 'Model is not loaded due to missing artifacts on startup.'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid request: No JSON payload received.'}), 400

        print(f"\nReceived data for scoring: {json.dumps(data, indent=2)}")

        # 1. Create features from the live request
        cat_features, num_features = create_features_from_request(data)

        # 2. Convert to tensors
        cat_tensors = []
        # The order MUST match the training script's categorical_cols
        for col in ['username', 'Country', 'City', 'OS', 'Browser', 'DeviceType', 'ASN']:
            mapping = feature_maps.get(f'{col}_to_id')
            if mapping is None:
                raise KeyError(f"Feature map missing for column: {col}")
            # If the value is new (e.g., a new city), map it to the 'UNK' token (ID 0)
            cat_tensors.append(mapping.get(cat_features[col], 0))

        X_cat = torch.LongTensor([cat_tensors])

        # Scale numerical features using the loaded scaler
        # The order MUST match the training script's numerical_cols
        num_data_ordered = [num_features[col] for col in ['LoginHour', 'DayOfWeek']]
        num_data_scaled = scaler.transform([num_data_ordered])
        X_num = torch.FloatTensor(num_data_scaled)

        # 3. Get prediction from the model
        with torch.no_grad():
            logits = model(X_cat, X_num)
            probability = torch.sigmoid(logits)
            threat_score = probability.item()

        # 4. Make decision based on the learned threshold
        decision_threshold = 0.7
        decision = 'allow' if threat_score < decision_threshold else 'reject'

        print(f"  > Features extracted: CAT={cat_features}, NUM={num_features}")
        print(
            f"  > Model prediction for '{data.get('username')}': Logit={logits.item():.4f}, Score={threat_score:.4f}, Decision='{decision}'")

        return jsonify({
            'threatScore': threat_score,
            'decision': decision,
            'transactionId': data.get('transactionId')
        })

    except KeyError as e:
        print(f"Error processing scoring request (KeyError): {e}")
        return jsonify({'error': f'Internal server error: Missing expected data. {e}'}), 500
    except Exception as e:
        print(f"Error processing scoring request: {e}")
        return jsonify({'error': f'Internal server error: {e}'}), 500


if __name__ == '__main__':
    # Only run the app if the model loaded successfully
    if model:
        port = int(os.getenv("PORT", 5001))
        debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        print(f"Starting Flask server on host 127.0.0.1, port {port}, debug={debug_mode}")
        app.run(host='127.0.0.1', port=port, debug=debug_mode)
    else:
        print("Flask server not started because model failed to load.")
