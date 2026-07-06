import hashlib
import threading
import time


_TTL_SECONDS = 15 * 60
_CACHE = {}
_LOCK = threading.Lock()


def build_context_id(lat, lon):
    normalized = f"{float(lat):.5f}:{float(lon):.5f}"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def get_cached_context(context_id):
    now = time.monotonic()
    with _LOCK:
        entry = _CACHE.get(context_id)
        if not entry:
            return None
        expires_at, value = entry
        if expires_at <= now:
            _CACHE.pop(context_id, None)
            return None
        return value


def set_cached_context(context_id, value):
    with _LOCK:
        _CACHE[context_id] = (time.monotonic() + _TTL_SECONDS, value)

