import logging

import requests

from config.config import NOMINATIM_REVERSE_URL, NOMINATIM_USER_AGENT


LOGGER = logging.getLogger(__name__)


def reverse_geocode(lat, lon):
    if not NOMINATIM_REVERSE_URL:
        return "Chicago Urban Zone"
    try:
        response = requests.get(
            NOMINATIM_REVERSE_URL,
            params={
                "lat": lat, "lon": lon, "format": "jsonv2",
                "zoom": 18, "addressdetails": 1,
            },
            headers={"User-Agent": NOMINATIM_USER_AGENT},
            timeout=5,
        )
        response.raise_for_status()
        display_name = response.json().get("display_name")
        return display_name.split(",")[0].strip() if display_name else "Chicago Urban Zone"
    except (requests.RequestException, ValueError, TypeError):
        LOGGER.warning("Reverse geocoding unavailable", exc_info=True)
        return "Chicago Urban Zone"
