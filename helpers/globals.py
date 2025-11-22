import os

FEATURE_COLUMNS = [
    "focus_changes", "blur_events", "click_count", "key_count",
    "avg_key_delay_ms", "pointer_distance_px", "pointer_event_count",
    "scroll_distance_px", "scroll_event_count", "total_session_time_ms",
    "time_to_first_key_ms", "time_to_first_click_ms", "idle_time_total_ms",
    "input_focus_count", "paste_events", "resize_events", "active_time_ms",
    "tz_offset_min", "device_memory_gb", "hardware_concurrency",
    "screen_width_px", "screen_height_px", "pixel_ratio", "color_depth",
    "touch_support", "webauthn_supported"
]

def resolve_path(env_var_name: str, default_relative: str) -> str:
    """
    Resolve paths safely regardless of working directory:
    1. If env var is set, use that.
    2. Otherwise, resolve relative to the project root (file location).
    3. If that doesn't exist, fall back to the literal relative path.
    """
    # environment override
    env_val = os.getenv(env_var_name)
    if env_val:
        return env_val

    # attempt project-root resolution
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_file_dir, ".."))
    project_relative_path = os.path.join(project_root, default_relative.replace("../", ""))

    if os.path.exists(project_relative_path):
        return project_relative_path

    # fallback
    return default_relative
