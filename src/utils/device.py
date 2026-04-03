import re

_MOBILE_UA_PATTERN = re.compile(
    r"Mobile|Android|iPhone|iPad|iPod|webOS|Opera Mini|IEMobile|Windows Phone",
    re.IGNORECASE,
)

# Screen width threshold: devices narrower than this are likely mobile
_MOBILE_MAX_WIDTH = 820


def classify_device(user_agent: str, metrics: dict) -> str:
    """Classify a login as 'desktop' or 'mobile' using UA string and device metrics."""
    ua = user_agent or ""
    touch = metrics.get("touch_support", False)
    screen_w = metrics.get("screen_width_px")
    screen_h = metrics.get("screen_height_px")

    ua_says_mobile = bool(_MOBILE_UA_PATTERN.search(ua))

    # If UA clearly says mobile, trust it
    if ua_says_mobile:
        return "mobile"

    # UA says desktop (or is empty) — cross-reference with hardware signals
    if touch and screen_w is not None and int(screen_w) <= _MOBILE_MAX_WIDTH:
        return "mobile"

    # Fall back to screen heuristic when UA is missing/empty
    if not ua and screen_w is not None:
        if int(screen_w) <= _MOBILE_MAX_WIDTH:
            return "mobile"

    return "desktop"
