import time
from datetime import datetime, timedelta

import requests


def iso_literal(value):
    return value.replace(microsecond=0).isoformat()


def parse_timestamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def date_windows(start, end, days=31):
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=days), end)
        yield cursor, window_end
        cursor = window_end


def request_json(url, params, app_token=None, timeout=180, retries=5):
    headers = {"X-App-Token": app_token} if app_token else {}
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    return []
