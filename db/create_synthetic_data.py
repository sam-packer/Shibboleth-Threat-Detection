import os
import random
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import uuid
from typing import Dict, List, Tuple, Optional

# Import the db_helper functions
from db.db_helper import record_login_with_scores, db_health_check

load_dotenv()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
if not POSTGRES_CONNECTION_STRING:
    raise RuntimeError("POSTGRES_CONNECTION_STRING not found in environment variables.")

engine = create_engine(POSTGRES_CONNECTION_STRING, pool_pre_ping=True, future=True)


class MaliciousPatternGenerator:
    """Generate different types of malicious login patterns"""

    # Common bot user agents/platforms
    BOT_PLATFORMS = ["Windows", "Linux", "Mac"]

    # Common attack origin countries (for demonstration - adjust based on your threat model)
    ATTACK_ORIGINS = [
        ("Beijing", "CN", "4134"),
        ("Moscow", "RU", "8359"),
        ("Lagos", "NG", "36873"),
        ("São Paulo", "BR", "18881"),
        ("Mumbai", "IN", "9498"),
        ("Kiev", "UA", "13188"),
        ("Istanbul", "TR", "9121"),
        ("Jakarta", "ID", "17974"),
        ("Ho Chi Minh City", "VN", "45899"),
        ("Cairo", "EG", "36935")
    ]

    # Legitimate user locations (for impossible travel)
    LEGIT_LOCATIONS = [
        ("Durham", "US", "13371"),
        ("New York", "US", "701"),
        ("San Francisco", "US", "7922"),
        ("London", "GB", "2856"),
        ("Paris", "FR", "3215"),
        ("Berlin", "DE", "3209"),
        ("Tokyo", "JP", "2514"),
        ("Sydney", "AU", "7545"),
        ("Toronto", "CA", "577"),
        ("Amsterdam", "NL", "1103")
    ]

    @staticmethod
    def generate_bot_pattern() -> Dict:
        """
        Generate bot/automated attack pattern
        Characteristics:
        - Very fast, consistent typing (if any)
        - No idle time
        - Minimal mouse movement
        - Quick session completion
        """
        return {
            "pattern_type": "bot",
            "focus_changes": 0,
            "blur_events": 0,
            "click_count": random.randint(1, 2),  # Minimal clicks
            "key_count": random.choice([0, random.randint(8, 15)]),  # Either no typing or very fast
            "avg_key_delay_ms": random.randint(10, 50) if random.random() > 0.5 else 0,  # Very fast or instant
            "pointer_distance_px": random.randint(50, 200),  # Minimal movement
            "pointer_event_count": random.randint(2, 5),
            "scroll_distance_px": 0,
            "scroll_event_count": 0,
            "time_to_first_key_ms": random.randint(50, 200),  # Very quick
            "time_to_first_click_ms": random.randint(50, 150),
            "idle_time_total_ms": random.randint(0, 100),  # Almost no idle time
            "input_focus_count": 1,
            "paste_events": random.choice([0, 1, 2]),  # May paste credentials
            "resize_events": 0,
            "total_session_time_ms": random.randint(500, 3000),  # Very quick session
            "active_time_ms": random.randint(500, 2500)
        }

    @staticmethod
    def generate_credential_stuffing_pattern() -> Dict:
        """
        Generate credential stuffing pattern
        Characteristics:
        - Heavy use of paste
        - Very quick interaction
        - No exploration behavior
        - Consistent timing
        """
        return {
            "pattern_type": "credential_stuffing",
            "focus_changes": 0,
            "blur_events": 0,
            "click_count": random.randint(2, 3),  # Login button + maybe password field
            "key_count": random.randint(0, 5),  # Minimal typing (mostly paste)
            "avg_key_delay_ms": 0,  # No natural typing
            "pointer_distance_px": random.randint(100, 400),
            "pointer_event_count": random.randint(3, 8),
            "scroll_distance_px": 0,
            "scroll_event_count": 0,
            "time_to_first_key_ms": random.randint(100, 300),
            "time_to_first_click_ms": random.randint(100, 300),
            "idle_time_total_ms": random.randint(0, 200),
            "input_focus_count": 2,  # Username and password fields
            "paste_events": random.randint(1, 3),  # Always pasting
            "resize_events": 0,
            "total_session_time_ms": random.randint(1000, 4000),
            "active_time_ms": random.randint(800, 3500)
        }

    @staticmethod
    def generate_account_takeover_pattern() -> Dict:
        """
        Generate account takeover pattern
        Characteristics:
        - Unusual behavior for the user
        - May explore the interface
        - Different typing patterns
        - Longer session (looking around)
        """
        return {
            "pattern_type": "account_takeover",
            "focus_changes": random.randint(2, 5),
            "blur_events": random.randint(1, 3),
            "click_count": random.randint(5, 15),  # Exploring interface
            "key_count": random.randint(15, 40),
            "avg_key_delay_ms": random.randint(80, 200),  # Different typing pattern
            "pointer_distance_px": random.randint(1000, 5000),  # Lots of movement
            "pointer_event_count": random.randint(20, 50),
            "scroll_distance_px": random.randint(0, 2000),
            "scroll_event_count": random.randint(0, 10),
            "time_to_first_key_ms": random.randint(500, 2000),
            "time_to_first_click_ms": random.randint(300, 1500),
            "idle_time_total_ms": random.randint(1000, 5000),  # Thinking/exploring
            "input_focus_count": random.randint(2, 5),
            "paste_events": random.randint(0, 1),
            "resize_events": random.randint(0, 1),
            "total_session_time_ms": random.randint(8000, 20000),  # Longer session
            "active_time_ms": random.randint(6000, 15000)
        }

    @staticmethod
    def generate_browser_automation_pattern() -> Dict:
        """
        Generate browser automation pattern (Selenium, Puppeteer, etc.)
        Characteristics:
        - Perfect, consistent timing
        - No natural human variance
        - Sequential, predictable actions
        """
        base_delay = random.randint(100, 200)
        return {
            "pattern_type": "browser_automation",
            "focus_changes": 0,
            "blur_events": 0,
            "click_count": 3,  # Exactly 3 clicks
            "key_count": random.randint(10, 20),
            "avg_key_delay_ms": base_delay,  # Perfectly consistent
            "pointer_distance_px": random.randint(300, 600),
            "pointer_event_count": 6,  # Very predictable
            "scroll_distance_px": 0,
            "scroll_event_count": 0,
            "time_to_first_key_ms": base_delay * 2,
            "time_to_first_click_ms": base_delay,
            "idle_time_total_ms": 0,  # No idle time
            "input_focus_count": 2,
            "paste_events": 0,
            "resize_events": 0,
            "total_session_time_ms": base_delay * random.randint(20, 30),
            "active_time_ms": base_delay * random.randint(18, 28)
        }

    @staticmethod
    def generate_targeted_attack_pattern() -> Dict:
        """
        Generate targeted attack pattern (sophisticated attacker)
        Characteristics:
        - Tries to mimic human behavior but with subtle anomalies
        - May be slower and more careful
        """
        return {
            "pattern_type": "targeted_attack",
            "focus_changes": random.randint(0, 2),
            "blur_events": random.randint(0, 1),
            "click_count": random.randint(3, 6),
            "key_count": random.randint(12, 25),
            "avg_key_delay_ms": random.randint(120, 180),  # Trying to look human
            "pointer_distance_px": random.randint(400, 1200),
            "pointer_event_count": random.randint(10, 25),
            "scroll_distance_px": random.randint(0, 500),
            "scroll_event_count": random.randint(0, 3),
            "time_to_first_key_ms": random.randint(800, 2000),  # Careful
            "time_to_first_click_ms": random.randint(600, 1500),
            "idle_time_total_ms": random.randint(500, 2000),
            "input_focus_count": 2,
            "paste_events": random.choice([0, 0, 0, 1]),  # Occasionally pastes
            "resize_events": 0,
            "total_session_time_ms": random.randint(5000, 12000),
            "active_time_ms": random.randint(4000, 10000)
        }


class SyntheticMaliciousDataGenerator:
    """Main generator for synthetic malicious login data"""

    def __init__(self):
        self.pattern_gen = MaliciousPatternGenerator()
        self.existing_users = self._get_existing_users()
        self.attack_patterns = [
            (self.pattern_gen.generate_bot_pattern, 0.30),  # 30% bots
            (self.pattern_gen.generate_credential_stuffing_pattern, 0.25),  # 25% credential stuffing
            (self.pattern_gen.generate_account_takeover_pattern, 0.20),  # 20% account takeover
            (self.pattern_gen.generate_browser_automation_pattern, 0.15),  # 15% automation
            (self.pattern_gen.generate_targeted_attack_pattern, 0.10)  # 10% targeted
        ]

    def _get_existing_users(self) -> List[str]:
        """Get list of existing usernames from database"""
        try:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT DISTINCT username FROM rba_login_event"))
                users = [row[0] for row in result]
                if users:
                    return users
                else:
                    logging.error("No users found in database!")
                    return []
        except SQLAlchemyError as e:
            logging.error(f"Failed to fetch existing users: {e}")
            return []

    def _generate_ip_address(self, location_info: Optional[Tuple] = None) -> str:
        """Generate a realistic IP address"""
        if location_info and location_info[1] == "US":
            # US IP ranges
            return f"{random.randint(50, 75)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        else:
            # International IP ranges
            first_octet = random.choice([81, 85, 89, 91, 95, 103, 110, 115, 120, 125,
                                         130, 135, 140, 145, 150, 155, 160, 165, 170, 175,
                                         180, 185, 190, 195, 200, 205, 210, 215, 220])
            return f"{first_octet}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"

    def _generate_device_specs(self, is_bot: bool = False) -> Dict:
        """Generate realistic device specifications"""
        if is_bot:
            # Bot-like specs (headless browsers, VMs)
            return {
                "device_memory_gb": random.choice([1, 2, 4]),
                "hardware_concurrency": random.choice([1, 2, 4]),
                "screen_width_px": random.choice([1024, 1280, 1366]),
                "screen_height_px": random.choice([768, 720, 768]),
                "pixel_ratio": 1.0,
                "color_depth": 24,
                "touch_support": False,
                "webauthn_supported": False,
                "language": random.choice(["en-US", "en", "zh-CN", "ru", "pt-BR"]),
                "platform": random.choice(["Linux", "Windows"])
            }
        else:
            # More varied, realistic specs
            return {
                "device_memory_gb": random.choice([2, 4, 8, 16]),
                "hardware_concurrency": random.choice([2, 4, 6, 8, 12]),
                "screen_width_px": random.choice([1280, 1366, 1440, 1920, 2560]),
                "screen_height_px": random.choice([720, 768, 900, 1080, 1440]),
                "pixel_ratio": random.choice([1.0, 1.25, 1.5, 2.0]),
                "color_depth": random.choice([24, 32]),
                "touch_support": random.choices([True, False], weights=[0.3, 0.7])[0],
                "webauthn_supported": random.choices([True, False], weights=[0.6, 0.4])[0],
                "language": random.choice(["en-US", "en-GB", "es", "fr", "de", "zh-CN", "ja", "ko", "ru", "pt-BR"]),
                "platform": random.choices(["Windows", "Mac", "Linux"], weights=[0.7, 0.2, 0.1])[0]
            }

    def generate_malicious_login(self,
                                 username: Optional[str] = None,
                                 impossible_travel: bool = False,
                                 pattern_type: Optional[str] = None) -> Dict:
        """Generate a single malicious login event"""

        # Select username - ONLY use existing users
        if username is None:
            if not self.existing_users:
                raise RuntimeError("No existing users found in database to generate malicious logins for")
            username = random.choice(self.existing_users)

        # Select attack pattern
        if pattern_type:
            pattern_funcs = {
                "bot": self.pattern_gen.generate_bot_pattern,
                "credential_stuffing": self.pattern_gen.generate_credential_stuffing_pattern,
                "account_takeover": self.pattern_gen.generate_account_takeover_pattern,
                "browser_automation": self.pattern_gen.generate_browser_automation_pattern,
                "targeted_attack": self.pattern_gen.generate_targeted_attack_pattern
            }
            pattern_data = pattern_funcs.get(pattern_type, self.pattern_gen.generate_bot_pattern)()
        else:
            # Weighted random selection
            pattern_func = random.choices(
                [f[0] for f in self.attack_patterns],
                weights=[f[1] for f in self.attack_patterns],
                k=1
            )[0]
            pattern_data = pattern_func()

        # Determine if this is a bot-like attack
        is_bot = pattern_data["pattern_type"] in ["bot", "browser_automation", "credential_stuffing"]

        # Generate device UUID (may reuse for bot farms)
        if is_bot and random.random() < 0.3:
            # 30% chance bots reuse device IDs
            device_uuid = random.choice([
                "00000000-0000-0000-0000-000000000000",
                "11111111-1111-1111-1111-111111111111",
                str(uuid.uuid4())
            ])
        else:
            device_uuid = str(uuid.uuid4())

        # Generate location
        if impossible_travel:
            # Pick a far-away location for impossible travel
            location = random.choice(self.pattern_gen.ATTACK_ORIGINS)
        elif is_bot:
            # Bots often come from suspicious locations
            location = random.choice(self.pattern_gen.ATTACK_ORIGINS)
        else:
            # Mix of suspicious and legitimate locations
            if random.random() < 0.7:
                location = random.choice(self.pattern_gen.ATTACK_ORIGINS)
            else:
                location = random.choice(self.pattern_gen.LEGIT_LOCATIONS)

        # Generate device specs
        device_specs = self._generate_device_specs(is_bot)

        # Build the login record
        login_data = {
            "username": username,
            "device_uuid": device_uuid,
            "metrics_version": 4,
            "nn_score": 1.0,  # Malicious score
            "impossible_travel": impossible_travel,
            "human_verified": True,  # Confirmed malicious
            "ip_address": self._generate_ip_address(location),
            "city": location[0],
            "country": location[1],
            "asn": location[2],
            "tz_offset_min": random.choice(
                [-720, -480, -420, -360, -300, -240, -180, -120, -60, 0, 60, 120, 180, 240, 300, 360, 420, 480, 540,
                 600, 660, 720]),
            **pattern_data,  # Include all behavioral metrics
            **device_specs  # Include device specifications
        }

        # Remove pattern_type as it's not in the schema
        if "pattern_type" in login_data:
            del login_data["pattern_type"]

        # Add some variance to make it more realistic
        for key in ["click_count", "key_count", "pointer_distance_px", "total_session_time_ms", "active_time_ms"]:
            if key in login_data and login_data[key] > 0:
                variance = random.uniform(0.8, 1.2)
                login_data[key] = int(login_data[key] * variance)

        return login_data

    def generate_attack_campaign(self,
                                 target_user: str,
                                 num_attempts: int = 10,
                                 time_window_minutes: int = 5) -> List[Dict]:
        """Generate a coordinated attack campaign against a specific user"""
        campaign_data = []
        base_time = datetime.now() - timedelta(minutes=time_window_minutes)

        # Choose a consistent attack pattern for the campaign
        pattern_type = random.choice(["bot", "credential_stuffing", "browser_automation"])

        # Generate device pool (bots often rotate through a few devices)
        device_pool = [str(uuid.uuid4()) for _ in range(min(3, num_attempts // 3 + 1))]

        # Generate IP pool (distributed botnet)
        ip_pool = [self._generate_ip_address() for _ in range(min(5, num_attempts // 2 + 1))]

        for i in range(num_attempts):
            login = self.generate_malicious_login(
                username=target_user,
                pattern_type=pattern_type
            )

            # Use device and IP from pool
            login["device_uuid"] = random.choice(device_pool)
            login["ip_address"] = random.choice(ip_pool)

            # Note: event_timestamp will be handled by db_helper
            # But we can add a delay between attempts if needed

            campaign_data.append(login)

        return campaign_data

    def generate_impossible_travel_scenario(self, username: str) -> List[Dict]:
        """Generate an impossible travel scenario (2 logins from far locations)"""
        scenarios = []

        # First login from legitimate location
        location1 = random.choice(self.pattern_gen.LEGIT_LOCATIONS)
        login1 = self.generate_malicious_login(username=username)
        login1["city"], login1["country"], login1["asn"] = location1
        login1["ip_address"] = self._generate_ip_address(location1)
        login1["impossible_travel"] = False  # First login is not flagged
        scenarios.append(login1)

        # Second login from impossible location (too far in too short time)
        location2 = random.choice(self.pattern_gen.ATTACK_ORIGINS)
        login2 = self.generate_malicious_login(username=username)
        login2["city"], login2["country"], login2["asn"] = location2
        login2["ip_address"] = self._generate_ip_address(location2)
        login2["impossible_travel"] = True  # This one is flagged
        scenarios.append(login2)

        return scenarios

    def insert_to_database(self, login_data: List[Dict]) -> int:
        """Insert generated malicious logins into database using db_helper"""
        if not login_data:
            return 0

        # Check database health first
        if not db_health_check():
            logging.error("[DB] Database health check failed")
            return 0

        inserted_count = 0
        failed_count = 0

        for login in login_data:
            try:
                # Extract fields for the function call
                username = login['username']
                ip_address = str(login['ip_address'])
                device_uuid = str(login['device_uuid'])

                # Calculate other risk scores based on the attack pattern
                if login.get('impossible_travel', False):
                    impossible_travel_score = 1.0
                else:
                    impossible_travel_score = 0.0

                # IP risk score - binary: 1 if IP is in toxic list, 0 otherwise
                # For synthetic malicious data, we'll simulate that ~60% come from toxic IPs
                # This reflects that many attackers use known bad IPs (botnets, proxies, etc.)
                if random.random() < 0.6:
                    ip_risk_score = 1.0  # Simulating a toxic IP
                else:
                    ip_risk_score = 0.0  # Clean IP (sophisticated attacker or compromised residential)

                # Adjust based on country - attacks from suspicious countries more likely to use toxic IPs
                suspicious_countries = ['CN', 'RU', 'NG', 'BR', 'IN', 'UA', 'TR', 'ID', 'VN', 'EG']
                if login.get('country') in suspicious_countries and random.random() < 0.8:
                    ip_risk_score = 1.0  # 80% chance of toxic IP from suspicious countries

                # For impossible travel, almost always from toxic IPs
                if login.get('impossible_travel', False) and random.random() < 0.9:
                    ip_risk_score = 1.0

                # For ground truth malicious logins, nn_score is ALWAYS 1.0
                nn_score = 1.0  # Ground truth malicious

                # Calculate final score using ensemble logic
                if ip_risk_score >= 1.0:
                    # With nn_score=1.0 and toxic IP, ensemble adds 25% of remaining distance to 1.0
                    # Since we're already at 1.0, it stays at 1.0
                    final_score = 1.0
                else:
                    # With nn_score=1.0 and clean IP, score remains 1.0
                    final_score = 1.0

                # Prepare the behavioral data (everything except the special fields)
                behavioral_data = {
                    k: v for k, v in login.items()
                    if k not in ['username', 'ip_address', 'device_uuid', 'nn_score']
                }

                # Set the metrics version
                behavioral_data['metrics_version'] = 4

                behavioral_data['human_verified'] = True  # Ground truth malicious
                behavioral_data['impossible_travel'] = login.get('impossible_travel', False)

                # Use the db_helper function to insert both login event and scores
                login_id = record_login_with_scores(
                    data=behavioral_data,
                    username=username,
                    ip_address=ip_address,
                    device_uuid=device_uuid,
                    nn_score=nn_score,
                    ip_risk_score=ip_risk_score,
                    impossible_travel=impossible_travel_score,
                    final_score=final_score
                )

                if login_id:
                    inserted_count += 1
                    if inserted_count % 10 == 0:
                        logging.info(f"[DB] Inserted {inserted_count} malicious logins...")
                else:
                    failed_count += 1
                    logging.warning(f"[DB] Failed to insert login for user {username}")

            except Exception as e:
                failed_count += 1
                logging.error(f"[DB] Error inserting login: {e}")

        logging.info(f"[DB] Successfully inserted {inserted_count} malicious login records")
        if failed_count > 0:
            logging.warning(f"[DB] Failed to insert {failed_count} records")

        return inserted_count


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic malicious login data for RBA system training"
    )

    parser.add_argument(
        "--count", "-n",
        type=int,
        default=100,
        help="Number of malicious logins to generate (default: 100)"
    )

    parser.add_argument(
        "--campaigns", "-c",
        type=int,
        default=0,
        help="Number of attack campaigns to simulate (default: 0)"
    )

    parser.add_argument(
        "--impossible-travel", "-i",
        type=int,
        default=0,
        help="Number of impossible travel scenarios to generate (default: 0)"
    )

    parser.add_argument(
        "--target-users",
        nargs="+",
        help="Specific usernames to target (optional)"
    )

    parser.add_argument(
        "--pattern",
        choices=["bot", "credential_stuffing", "account_takeover", "browser_automation", "targeted_attack"],
        help="Specific attack pattern to use (optional)"
    )

    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write to database (required for any database operations)"
    )

    parser.add_argument(
        "--export",
        type=str,
        help="Export generated data to CSV file"
    )

    args = parser.parse_args()

    # Check if any generation was requested
    if args.count == 0 and args.campaigns == 0 and args.impossible_travel == 0:
        print("\n❌ No data generation requested. Please specify at least one of:")
        print("  --count N        : Generate N individual malicious logins")
        print("  --campaigns N    : Generate N attack campaigns")
        print("  --impossible-travel N : Generate N impossible travel scenarios")
        print("\nExample usage:")
        print("  python generate_malicious_logins.py --count 100 --write")
        print("  python generate_malicious_logins.py --campaigns 5 --export attacks.csv")
        print("\nUse -h or --help for full options")
        return

    # Warn if no output method specified
    if not args.write and not args.export:
        print("\n⚠️  WARNING: No output specified. Data will be generated but not saved.")
        print("  Add --write to insert to database")
        print("  Add --export filename.csv to save to file")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Initialize generator
    generator = SyntheticMaliciousDataGenerator()

    # Validate we have users to work with
    if not generator.existing_users:
        print("\n❌ ERROR: No existing users found in database!")
        print("  Cannot generate malicious logins without existing users.")
        print("  Please ensure your database has some legitimate user data first.")
        return

    print(f"\n✅ Found {len(generator.existing_users)} existing users in database")
    print(
        f"   Users: {', '.join(generator.existing_users[:10])}{', ...' if len(generator.existing_users) > 10 else ''}")

    all_logins = []

    # Generate individual malicious logins
    if args.count > 0:
        logging.info(f"Generating {args.count} malicious logins...")

        # Validate target users if specified
        valid_targets = None
        if args.target_users:
            valid_targets = [u for u in args.target_users if u in generator.existing_users]
            if not valid_targets:
                print(f"⚠️  Warning: None of the specified target users exist in database")
                print(f"   Will use random existing users instead")
                args.target_users = None  # Clear invalid targets

        for i in range(args.count):
            username = None
            if args.target_users and valid_targets:
                username = random.choice(valid_targets)

            login = generator.generate_malicious_login(
                username=username,
                pattern_type=args.pattern
            )
            all_logins.append(login)

            if (i + 1) % 10 == 0:
                logging.info(f"Generated {i + 1}/{args.count} logins...")

    # Generate attack campaigns
    if args.campaigns > 0:
        logging.info(f"Generating {args.campaigns} attack campaigns...")
        for i in range(args.campaigns):
            if args.target_users:
                # Validate target users exist
                valid_targets = [u for u in args.target_users if u in generator.existing_users]
                if not valid_targets:
                    print(f"⚠️  Warning: None of the specified target users exist in database")
                    print(f"   Using random existing users instead")
                    target_user = random.choice(generator.existing_users)
                else:
                    target_user = random.choice(valid_targets)
            else:
                target_user = random.choice(generator.existing_users)
            campaign_attempts = random.randint(5, 20)
            campaign = generator.generate_attack_campaign(
                target_user=target_user,
                num_attempts=campaign_attempts,
                time_window_minutes=random.randint(2, 10)
            )
            all_logins.extend(campaign)
            logging.info(
                f"Generated campaign {i + 1}/{args.campaigns} targeting {target_user} with {campaign_attempts} attempts")

    # Generate impossible travel scenarios
    if args.impossible_travel > 0:
        logging.info(f"Generating {args.impossible_travel} impossible travel scenarios...")
        for i in range(args.impossible_travel):
            if args.target_users:
                # Validate target users exist
                valid_targets = [u for u in args.target_users if u in generator.existing_users]
                if not valid_targets:
                    print(f"⚠️  Warning: None of the specified target users exist in database")
                    print(f"   Using random existing users instead")
                    username = random.choice(generator.existing_users)
                else:
                    username = random.choice(valid_targets)
            else:
                username = random.choice(generator.existing_users)
            scenarios = generator.generate_impossible_travel_scenario(username)
            all_logins.extend(scenarios)
            logging.info(f"Generated impossible travel scenario {i + 1}/{args.impossible_travel} for {username}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SYNTHETIC MALICIOUS DATA GENERATION SUMMARY")
    print("=" * 80)
    print(f"Total malicious logins generated: {len(all_logins)}")

    if all_logins:
        df = pd.DataFrame(all_logins)

        print("\n📊 ATTACK PATTERNS DISTRIBUTION:")
        print("-" * 80)

        # Categorize by behavioral patterns
        bot_like = df[(df['idle_time_total_ms'] < 200) & (df['avg_key_delay_ms'] < 60)]
        paste_heavy = df[df['paste_events'] > 0]
        long_session = df[df['total_session_time_ms'] > 10000]

        print(f"Bot-like patterns: {len(bot_like)} ({len(bot_like) / len(df) * 100:.1f}%)")
        print(f"Credential stuffing (paste): {len(paste_heavy)} ({len(paste_heavy) / len(df) * 100:.1f}%)")
        print(f"Account takeover (long session): {len(long_session)} ({len(long_session) / len(df) * 100:.1f}%)")
        print(
            f"Impossible travel: {df['impossible_travel'].sum()} ({df['impossible_travel'].sum() / len(df) * 100:.1f}%)")

        print("\n🎯 TARGETED USERS:")
        print("-" * 80)
        user_counts = df['username'].value_counts().head(10)
        for user, count in user_counts.items():
            print(f"  {user}: {count} attempts")

        print("\n🌍 ATTACK ORIGINS:")
        print("-" * 80)
        country_counts = df['country'].value_counts().head(10)
        for country, count in country_counts.items():
            print(f"  {country}: {count} logins")

        print("\n⚠️  THREAT INDICATORS:")
        print("-" * 80)
        print(f"Average clicks per login: {df['click_count'].mean():.2f}")
        print(f"Average keys typed: {df['key_count'].mean():.2f}")
        print(f"Average session time: {df['total_session_time_ms'].mean():.0f}ms")
        print(f"Logins with paste events: {(df['paste_events'] > 0).sum()}")
        print(f"Logins with no idle time: {(df['idle_time_total_ms'] == 0).sum()}")

        # Export if requested
        if args.export:
            df.to_csv(args.export, index=False)
            print(f"\n✅ Data exported to {args.export}")

        # Insert to database if --write flag is set
        if args.write:
            print("\n💾 Inserting to database...")
            inserted = generator.insert_to_database(all_logins)
            if inserted > 0:
                print(f"✅ Successfully inserted {inserted} records to database")
            else:
                print("❌ Failed to insert records to database")
        elif not args.export:
            print("\n⚠️  Data generated but not saved (use --write for database or --export for CSV)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()