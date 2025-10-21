import torch
import json
from flask import Flask, request, jsonify
from model import RBAModel  # Assumes model.py is in the same directory
import joblib
import geoip2.database
from user_agents import parse
from datetime import datetime
import os

# --- 1. LOAD ALL ARTIFACTS ON STARTUP ---
print("Loading RBA model and all required artifacts...")

try:
    # Check for all required files before proceeding
    required_files = ['rba_model.pth', 'feature_maps.json', 'scaler.joblib', 'GeoLite2-City.mmdb']
    for f in required_files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    # Load feature mappings
    with open('feature_maps.json', 'r') as f:
        feature_maps = json.load(f)

    # Load the numerical scaler
    scaler = joblib.load('scaler.joblib')

    # Load GeoIP database
    geoip_reader = geoip2.database.Reader('GeoLite2-City.mmdb')

    # Instantiate the model with the correct dimensions from the loaded artifacts
    vocab_sizes = {col.replace('_to_id', ''): len(feature_maps[col]) for col in feature_maps}
    embedding_dims = {
        'username': 50, 'Country': 20, 'City': 30,
        'OS': 20, 'Browser': 20, 'DeviceType': 10, 'ASN': 25
    }
    # The number of numerical features is determined by the scaler
    model = RBAModel(vocab_sizes, embedding_dims, num_numerical_feats=len(scaler.mean_))

    # Load the trained weights
    model.load_state_dict(torch.load('rba_model.pth'))
    model.eval()  # Set the model to evaluation mode (very important!)
    print("All artifacts loaded successfully. API is ready.")

except FileNotFoundError as e:
    print("\n--- FATAL ERROR ---")
    print(f"Could not find a required artifact: {e.filename}")
    print("Please run 'python3 train.py' first to generate all required files.")
    print("API will not start.")
    print("-------------------\n")
    model = None  # Prevent the app from running

app = Flask(__name__)


# --- 2. FEATURE ENGINEERING FUNCTION (MUST MIRROR TRAINING SCRIPT) ---
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
        if geoip_reader and ip_address:
            city_response = geoip_reader.city(ip_address)
            country = city_response.country.iso_code or 'Unknown'
            city = city_response.city.name or 'Unknown'
            asn_response = geoip_reader.asn(ip_address)
            asn_val = str(asn_response.autonomous_system_number or '0')
    except geoip2.errors.AddressNotFoundError:
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


# --- 3. FLASK API ENDPOINT ---
@app.route('/score', methods=['POST'])
def get_threat_score():
    if not model:
        return jsonify({'error': 'Model is not loaded due to missing artifacts on startup.'}), 500

    try:
        data = request.get_json()
        print(f"\nReceived data for scoring: {json.dumps(data, indent=2)}")

        # 1. Create features from the live request
        cat_features, num_features = create_features_from_request(data)

        # 2. Convert to tensors
        cat_tensors = []
        # The order MUST match the training script's categorical_cols
        for col in ['username', 'Country', 'City', 'OS', 'Browser', 'DeviceType', 'ASN']:
            mapping = feature_maps[f'{col}_to_id']
            # If the value is new (e.g., a new city), map it to the 'UNK' token (ID 0)
            cat_tensors.append(mapping.get(cat_features[col], 0))

        X_cat = torch.LongTensor([cat_tensors])

        # Scale numerical features using the loaded scaler
        # The order MUST match the training script's numerical_cols
        num_data_ordered = [num_features[col] for col in ['LoginHour', 'DayOfWeek']]
        num_data_scaled = scaler.transform([num_data_ordered])
        X_num = torch.FloatTensor(num_data_scaled)

        # 3. Get prediction from the model
        with torch.no_grad():  # Disable gradient calculation for inference
            threat_score = model(X_cat, X_num).item()

        # 4. Make decision based on the learned threshold
        decision_threshold = 0.5
        decision = 'allow' if threat_score < decision_threshold else 'reject'

        print(f"  > Features extracted: CAT={cat_features}, NUM={num_features}")
        print(f"  > Model prediction for '{data.get('username')}': Score={threat_score:.4f}, Decision='{decision}'")

        return jsonify({
            'threatScore': threat_score,
            'decision': decision,
            'transactionId': data.get('transactionId')
        })

    except Exception as e:
        print(f"Error processing scoring request: {e}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Only run the app if the model loaded successfully
    if model:
        app.run(host='127.0.0.1', port=5001, debug=True)

