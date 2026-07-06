import os
from datetime import timedelta


DYNAMIC_LOOKBACK_DAYS = 1095
SUPER_CELL_FACTOR = 3
FORECAST_BLOCK_HOURS = int(os.getenv("FORECAST_BLOCK_HOURS", "3"))
FORECAST_BLOCKS = int(os.getenv("FORECAST_BLOCKS", str(24 // max(FORECAST_BLOCK_HOURS, 1))))
TRAIN_RATIO = 0.70
CALIBRATION_RATIO = 0.15
TEMPORAL_GAP_HOURS = 24


def chronological_boundaries(latest_timestamp):
    """Return the shared leakage-safe boundaries used by every modeling stage."""
    window_start = latest_timestamp - timedelta(days=DYNAMIC_LOOKBACK_DAYS)
    window_duration = latest_timestamp - window_start
    calibration_start = window_start + window_duration * TRAIN_RATIO
    test_start = window_start + window_duration * (TRAIN_RATIO + CALIBRATION_RATIO)
    train_end = calibration_start - timedelta(hours=TEMPORAL_GAP_HOURS)
    calibration_end = test_start - timedelta(hours=TEMPORAL_GAP_HOURS)
    return {
        "window_start": window_start,
        "train_end": train_end,
        "calibration_start": calibration_start,
        "calibration_end": calibration_end,
        "test_start": test_start,
        "window_end": latest_timestamp,
    }
