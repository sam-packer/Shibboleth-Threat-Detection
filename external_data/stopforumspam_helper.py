import os
import ipaddress
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

# Load env
load_dotenv()

SFS_URL = "https://www.stopforumspam.com/downloads/toxic_ip_cidr.txt"
SFS_DIR = Path(os.getenv("SFS_DIR", "stopforumspam_data"))
SFS_FILE = SFS_DIR / "toxic_ip_cidr.txt"
META_FILE = SFS_DIR / "metadata.txt"

# Default: refresh every 24 hours
REFRESH_INTERVAL_HOURS = int(os.getenv("SFS_REFRESH_INTERVAL_HOURS", "24"))

# Internal cache of parsed networks
_cached_networks = None
_cached_timestamp = None


def _ensure_sfs_dir():
    SFS_DIR.mkdir(parents=True, exist_ok=True)


def _get_metadata() -> dict:
    if not META_FILE.exists():
        return {}
    meta = {}
    try:
        with open(META_FILE, "r") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    meta[k] = v
    except Exception:
        pass
    return meta


def _save_metadata(meta: dict):
    with open(META_FILE, "w") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")


def _needs_refresh() -> bool:
    meta = _get_metadata()
    last_update_str = meta.get("last_updated")
    if not last_update_str:
        return True

    try:
        last_update = datetime.fromisoformat(last_update_str)
        if last_update.tzinfo is None:
            last_update = last_update.replace(tzinfo=timezone.utc)
    except ValueError:
        return True

    now_utc = datetime.now(timezone.utc)
    return now_utc - last_update > timedelta(hours=REFRESH_INTERVAL_HOURS)



def _download_list():
    """Fetch the toxic IP CIDR list from StopForumSpam."""
    print("[StopForumSpam] Downloading toxic IP list…")
    resp = requests.get(SFS_URL, timeout=15)
    if resp.status_code != 200:
        print(f"[StopForumSpam] Failed to fetch list (HTTP {resp.status_code})")
        return False

    SFS_FILE.write_text(resp.text.strip(), encoding="utf-8")

    meta = {
        "last_updated": datetime.utcnow().isoformat(),
        "last_length": str(len(resp.text.splitlines())),
    }
    _save_metadata(meta)
    print("[StopForumSpam] List updated successfully.")
    return True


def _load_networks():
    global _cached_networks, _cached_timestamp

    if not SFS_FILE.exists():
        return []

    with open(SFS_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    networks = []
    for line in lines:
        try:
            net = ipaddress.ip_network(line)
            networks.append(net)
        except ValueError:
            continue

    _cached_networks = networks
    _cached_timestamp = datetime.utcnow()
    print(f"[StopForumSpam] Loaded {len(networks)} CIDR ranges into memory.")
    return networks


def ensure_sfs_up_to_date(force=False):
    _ensure_sfs_dir()

    if force or _needs_refresh() or not SFS_FILE.exists():
        _download_list()
    else:
        print("[StopForumSpam] List is up to date; no download needed.")

    _load_networks()


def ip_in_toxic_list(ip: str) -> bool:
    global _cached_networks

    if _cached_networks is None:
        if not SFS_FILE.exists():
            print("[StopForumSpam] Warning: toxic IP list not found; returning False.")
            return False
        _load_networks()

    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False

    for net in _cached_networks:
        if addr in net:
            return True
    return False
