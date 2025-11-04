import logging
import os
import tarfile
import requests
import shutil
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from ipaddress import ip_address as ip_parse

from geoip2.database import Reader

load_dotenv()

MAXMIND_LICENSE_KEY = os.getenv("MAXMIND_LICENSE_KEY")
GEOIP_DIR = Path(os.getenv("GEOIP_DIR", "../geoip_data"))
CITY_EDITION = "GeoLite2-City"
ASN_EDITION = "GeoLite2-ASN"
MAXMIND_BASE_URL = "https://download.maxmind.com/app/geoip_download"
META_FILE = GEOIP_DIR / "metadata.txt"
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")


def _ensure_geoip_dir():
    GEOIP_DIR.mkdir(parents=True, exist_ok=True)


def _get_metadata() -> dict:
    if not META_FILE.exists():
        return {}
    try:
        meta = {}
        with open(META_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
        return meta
    except Exception:
        return {}


def _save_metadata(meta: dict):
    """Save metadata to file."""
    with open(META_FILE, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")


def _download_and_extract(edition: str):
    """
    Download and extract a MaxMind GeoLite2 database (.tar.gz)
    into GEOIP_DIR as .mmdb.
    """
    if not MAXMIND_LICENSE_KEY:
        raise RuntimeError("MAXMIND_LICENSE_KEY not found in environment.")

    params = {
        "edition_id": edition,
        "license_key": MAXMIND_LICENSE_KEY,
        "suffix": "tar.gz",
    }

    print(f"[GeoIP] Checking for updates: {edition}…")

    # HEAD request for Last-Modified
    head_resp = requests.head(MAXMIND_BASE_URL, params=params, timeout=10)
    if head_resp.status_code != 200:
        print(f"[GeoIP] Warning: unable to check {edition} (HTTP {head_resp.status_code})")
        return

    new_last_modified = head_resp.headers.get("Last-Modified")
    meta = _get_metadata()
    old_last_modified = meta.get(f"{edition}_last_modified")

    target_mmdb = GEOIP_DIR / f"{edition}.mmdb"

    # Skip if no change
    if new_last_modified == old_last_modified and target_mmdb.exists():
        print(f"[GeoIP] {edition} is up to date.")
        return

    # Download new version
    print(f"[GeoIP] Downloading {edition}…")
    resp = requests.get(MAXMIND_BASE_URL, params=params, stream=True, timeout=60)
    if resp.status_code != 200:
        print(f"[GeoIP] Failed to download {edition}: HTTP {resp.status_code}")
        return

    tar_path = GEOIP_DIR / f"{edition}.tar.gz"
    with open(tar_path, "wb") as f:
        shutil.copyfileobj(resp.raw, f)

    # Extract .mmdb file
    print(f"[GeoIP] Extracting {edition}…")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mmdb"):
                extracted_path = GEOIP_DIR / Path(member.name).name
                tar.extract(member, path=GEOIP_DIR)
                extracted_src = GEOIP_DIR / member.name
                shutil.move(extracted_src, target_mmdb)
                break

    # Clean up .tar.gz and extra folders
    for item in GEOIP_DIR.iterdir():
        if item.is_dir() and edition.replace("GeoLite2-", "") in item.name:
            shutil.rmtree(item, ignore_errors=True)
    tar_path.unlink(missing_ok=True)

    # Save metadata
    meta[f"{edition}_last_modified"] = new_last_modified or datetime.utcnow().isoformat()
    meta[f"{edition}_last_updated"] = datetime.utcnow().isoformat()
    _save_metadata(meta)

    print(f"[GeoIP] {edition} updated successfully.")


def ensure_geoip_up_to_date():
    _ensure_geoip_dir()
    _download_and_extract(CITY_EDITION)
    _download_and_extract(ASN_EDITION)
    print("[GeoIP] All databases are up to date.")


def get_geoip_paths() -> dict:
    return {
        "city": GEOIP_DIR / f"{CITY_EDITION}.mmdb",
        "asn": GEOIP_DIR / f"{ASN_EDITION}.mmdb"
    }


def enrich_with_geoip(data: dict, client_ip: str) -> dict:
    paths = get_geoip_paths()

    try:
        ip_obj = ip_parse(client_ip)
    except ValueError:
        logging.warning(f"[GeoIP] Invalid IP address: {client_ip}")
        return data

    try:
        with Reader(paths["city"]) as city_reader, Reader(paths["asn"]) as asn_reader:
            city_info = city_reader.city(client_ip)
            asn_info = asn_reader.asn(client_ip)

            data["country"] = city_info.country.iso_code or None
            data["city"] = city_info.city.name or None
            data["asn"] = asn_info.autonomous_system_number or None

    except FileNotFoundError as e:
        logging.error(f"[GeoIP] Missing MaxMind DB file: {e}")
    except Exception as e:
        logging.warning(f"[GeoIP] Lookup failed for {client_ip}: {e}")

    return data
