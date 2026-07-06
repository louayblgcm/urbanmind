"""Runtime forecast service for the deployed next-24-hour crime-pressure model.

This module converts the trained checkpoint into a location-level forecast payload
that the API can expose directly to the frontend. It intentionally keeps the
output shape stable even while the training setup evolves.
"""

import math
import threading
from datetime import timedelta
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import psycopg2
import torch
from psycopg2.extras import RealDictCursor
from psycopg2.extras import Json

from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER
from config.modeling import FORECAST_BLOCK_HOURS, FORECAST_BLOCKS, SUPER_CELL_FACTOR
from models.dynamic_forecaster import (
    SCALAR_FEATURE_NAMES,
    TwoHeadDynamicForecaster,
    parse_embedding_vector,
)


BACKEND_DIR = Path(__file__).resolve().parents[2]
MODEL_PATHS = (
    BACKEND_DIR / "data" / "models" / "hierarchical_forecaster_3h.pt",
    BACKEND_DIR / "data" / "models" / "hierarchical_forecaster_3h.joblib",
)
HISTORY_DAYS = 1095
SEQUENCE_HOURS = 72
MAX_DATA_AGE_HOURS = 48.0
_MODEL_LOCK = threading.Lock()
_MODEL_BUNDLE = None
_MODEL_MTIME = None
SOURCE_WATERMARK_PATH = BACKEND_DIR / "data" / "debug" / "ingestion" / "crime_source_watermark.json"


def _connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def _load_model():
    global _MODEL_BUNDLE, _MODEL_MTIME
    available = [path for path in MODEL_PATHS if path.exists()]
    if not available:
        return None
    model_path = max(available, key=lambda path: path.stat().st_mtime_ns)
    mtime = model_path.stat().st_mtime_ns
    with _MODEL_LOCK:
        if _MODEL_BUNDLE is not None and _MODEL_MTIME == mtime:
            return _MODEL_BUNDLE
        if model_path.suffix == ".joblib":
            checkpoint = joblib.load(model_path)
            _MODEL_BUNDLE = {
                "family": checkpoint.get("model_type", "boosted_hurdle"),
                "path": model_path,
                "checkpoint": checkpoint,
            }
        else:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            config = checkpoint["config"]
            model = TwoHeadDynamicForecaster(
                seq_input_dim=config["seq_input_dim"],
                scalar_input_dim=config["scalar_input_dim"],
                baseline_input_dim=config["baseline_input_dim"],
                output_steps=config.get("output_steps", 24),
                seq_hidden_dim=config["seq_hidden_dim"],
                scalar_hidden_dim=config["scalar_hidden_dim"],
                baseline_hidden_dim=config["baseline_hidden_dim"],
                fusion_hidden_dim=config["fusion_hidden_dim"],
                dropout=config["dropout"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            _MODEL_BUNDLE = {
                "family": "neural_two_head",
                "path": model_path,
                "model": model,
                "checkpoint": checkpoint,
            }
        _MODEL_MTIME = mtime
        return _MODEL_BUNDLE


def _bundle_block_config(model_bundle):
    if model_bundle is None:
        return FORECAST_BLOCK_HOURS, FORECAST_BLOCKS
    checkpoint = model_bundle.get("checkpoint", {})
    config = checkpoint.get("config", {})
    hours = int(config.get("forecast_block_hours", FORECAST_BLOCK_HOURS))
    blocks = int(config.get("output_steps", FORECAST_BLOCKS))
    if hours <= 0 or blocks <= 0 or hours * blocks != 24:
        return FORECAST_BLOCK_HOURS, FORECAST_BLOCKS
    return hours, blocks


def _source_freshness():
    if not SOURCE_WATERMARK_PATH.exists():
        return {}
    try:
        with SOURCE_WATERMARK_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        "crime_source_latest": payload.get("source_latest"),
        "crime_source_age_hours": payload.get("source_age_hours"),
        "crime_database_latest": payload.get("database_latest"),
        "crime_database_age_hours": payload.get("database_age_hours"),
        "crime_database_vs_source_gap_hours": payload.get("database_vs_source_gap_hours"),
    }


def _neighbor_ids(cell_id):
    try:
        x, y = (int(part) for part in str(cell_id).split("_"))
    except (TypeError, ValueError):
        return []
    return [
        f"{x + dx}_{y + dy}"
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        if dx or dy
    ]


def _load_static(cursor, cell_id):
    baseline_columns = ", ".join(
        f"f.baseline_crime_hour_{hour:02d}" for hour in range(24)
    )
    cursor.execute(
        f"""
        SELECT p.cell_id, p.cluster, p.static_crime_score,
               p.static_activity_score, p.embedding_vector, p.urban_profile,
               f.crime_total, f.requests_311_total, f.poi_total,
               f.business_total, f.permits_total, f.traffic_mean,
               {baseline_columns}
        FROM static_gnn_profiles p
        JOIN static_cell_features f ON f.cell_id = p.cell_id
        WHERE p.cell_id = %s
        """,
        (cell_id,),
    )
    return cursor.fetchone()


def _reference_time(cursor):
    cursor.execute("SELECT MAX(timestamp) AS latest FROM crimes")
    crime_latest = cursor.fetchone()["latest"]
    if crime_latest is None:
        raise ValueError("No timestamped crime history is available")
    # Crime is the prediction target, so its watermark defines the forecast
    # origin. A newer 311 watermark must not manufacture trailing crime zeros.
    latest = pd.Timestamp(crime_latest)
    return latest.floor("h") + pd.Timedelta(hours=1)


def _hourly_matrix(cursor, table, timestamp_column, cell_ids, start, end):
    cursor.execute(
        f"""
        SELECT cell_id, date_trunc('hour', {timestamp_column}) AS hour_ts,
               COUNT(*)::DOUBLE PRECISION AS event_count
        FROM {table}
        WHERE cell_id = ANY(%s)
          AND {timestamp_column} >= %s
          AND {timestamp_column} < %s
        GROUP BY cell_id, date_trunc('hour', {timestamp_column})
        """,
        (cell_ids, start.to_pydatetime(), end.to_pydatetime()),
    )
    rows = cursor.fetchall()
    index = pd.date_range(start=start, end=end - pd.Timedelta(hours=1), freq="h")
    matrix = np.zeros((len(cell_ids), len(index)), dtype=np.float32)
    cell_lookup = {value: idx for idx, value in enumerate(cell_ids)}
    time_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(index)}
    for row in rows:
        cell_idx = cell_lookup.get(row["cell_id"])
        time_idx = time_lookup.get(pd.Timestamp(row["hour_ts"]))
        if cell_idx is not None and time_idx is not None:
            matrix[cell_idx, time_idx] = float(row["event_count"])
    return matrix, index


def _seasonal_mean(values, index, attribute, expected):
    groups = np.asarray([getattr(ts, attribute) for ts in index])
    selected = values[groups == expected]
    return float(selected.mean()) if selected.size else 0.0


def _safe_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    return float(numerator / max(denominator, 1e-6))


def _build_inputs(cursor, cell_id, static, reference):
    start = reference - pd.Timedelta(days=HISTORY_DAYS)
    cell_ids = [cell_id] + _neighbor_ids(cell_id)
    crime, index = _hourly_matrix(cursor, "crimes", "timestamp", cell_ids, start, reference)
    requests, _ = _hourly_matrix(
        cursor, "requests_311", "created_date", cell_ids, start, reference
    )
    if crime.shape[1] < SEQUENCE_HOURS:
        raise ValueError("Less than 72 hours of history are available")

    target_crime = crime[0]
    target_requests = requests[0]
    sequence = np.stack(
        [target_crime[-SEQUENCE_HOURS:], target_requests[-SEQUENCE_HOURS:]], axis=1
    )
    ref_hour = reference.hour
    ref_weekday = reference.weekday()
    ref_month = reference.month

    habit_features = [
        _seasonal_mean(target_crime, index, "hour", ref_hour),
        _seasonal_mean(target_crime, index, "weekday", ref_weekday),
        float(target_crime[
            np.asarray([(ts.hour == ref_hour and ts.weekday() == ref_weekday) for ts in index])
        ].mean()) if any(ts.hour == ref_hour and ts.weekday() == ref_weekday for ts in index) else 0.0,
        _seasonal_mean(target_requests, index, "hour", ref_hour),
        _seasonal_mean(target_requests, index, "weekday", ref_weekday),
    ]

    windows = (1, 3, 6, 12, 24, 48, 72)
    crime_recent = [float(target_crime[-window:].sum()) for window in windows]
    request_recent = [float(target_requests[-window:].sum()) for window in windows]
    neighbor_crime = crime[1:]
    neighbor_requests = requests[1:]

    def neighbor_mean(matrix, window):
        return float(matrix[:, -window:].sum(axis=1).mean()) if matrix.shape[0] else 0.0

    neighbor_features = [
        neighbor_mean(neighbor_crime, window) for window in (1, 3, 6, 24)
    ] + [
        neighbor_mean(neighbor_requests, window) for window in (1, 3, 6, 24)
    ]

    static_features = [
        static["cluster"],
        static["static_crime_score"],
        static["static_activity_score"],
        static["crime_total"],
        static["requests_311_total"],
        static["poi_total"],
        static["business_total"],
        static["permits_total"],
        static["traffic_mean"],
    ]
    scalar = np.asarray([
        math.sin(2.0 * math.pi * ref_hour / 24.0),
        math.cos(2.0 * math.pi * ref_hour / 24.0),
        math.sin(2.0 * math.pi * ref_weekday / 7.0),
        math.cos(2.0 * math.pi * ref_weekday / 7.0),
        float(ref_weekday >= 5),
        math.sin(2.0 * math.pi * (ref_month - 1) / 12.0),
        math.cos(2.0 * math.pi * (ref_month - 1) / 12.0),
        *static_features,
        *habit_features,
        *crime_recent,
        *request_recent,
        *neighbor_features,
        *parse_embedding_vector(static["embedding_vector"]),
    ], dtype=np.float32)

    baseline_by_hour = np.asarray([
        static[f"baseline_crime_hour_{hour:02d}"] or 0.0 for hour in range(24)
    ], dtype=np.float32)
    baseline = baseline_by_hour[(ref_hour + np.arange(24)) % 24]
    return reference, sequence, scalar, baseline


def _static_fallback(cell_id, reference, static, reason, data_age_hours):
    baseline_by_hour = np.asarray([
        static[f"baseline_crime_hour_{hour:02d}"] or 0.0 for hour in range(24)
    ], dtype=np.float32)
    baseline = baseline_by_hour[(reference.hour + np.arange(24)) % 24]
    static_average = float(baseline.mean())
    static_total = float(baseline.sum())
    calibration_reference = max(static_average, 1e-8)
    source_freshness = _source_freshness()
    hours, timeline, static_timeline = [], [], []
    for offset in range(24):
        timestamp = reference + pd.Timedelta(hours=offset)
        value = float(np.clip(50 * baseline[offset] / calibration_reference, 0, 100))
        probability = float(1 - np.exp(-baseline[offset]))
        hours.append({
            "forecast_timestamp": timestamp.isoformat(), "hour": int(timestamp.hour),
            "occurrence_probability": round(probability, 6),
            "expected_count": round(float(baseline[offset]), 6),
            "static_baseline_count": round(float(baseline[offset]), 6),
            "relative_to_static_average": round(float(baseline[offset] / calibration_reference), 4),
            "relative_activity_index": round(value, 2),
        })
        timeline.append({"hour": int(timestamp.hour), "value": round(value, 2)})
        static_timeline.append({"hour": int(timestamp.hour), "value": round(value, 2)})
    timeline.sort(key=lambda row: row["hour"])
    static_timeline.sort(key=lambda row: row["hour"])
    return {
        "cell_id": cell_id, "reference_time": reference.isoformat(), "horizon_hours": 24,
        "model": "trained_static_baseline", "forecast_status": "static_fallback",
        "abstained": True, "abstention_reason": reason,
        "target_definition": "reported-crime pressure, not certainty or underlying crime",
        "data_age_hours": round(float(data_age_hours), 2),
        "predicted_crime": round(static_total, 6),
        "expected_crime_count_24h": round(static_total, 6),
        "static_baseline_count_24h": round(static_total, 6),
        "probability_any_crime_24h": round(float(1 - np.exp(-static_total)), 6),
        "prediction_confidence": None,
        "confidence_definition": "Not reported as model certainty",
        "static_average_hourly_count": round(static_average, 6),
        "calibration_source": "trained_cell_static_hourly_average",
        "calibration_definition": "Timeline index 50 equals the trained static hourly average",
        "uncertainty": {"available": False, "reason": reason},
        "source_freshness": source_freshness,
        "hourly_distribution": timeline, "static_hourly_distribution": static_timeline,
        "hourly_forecast": hours,
    }


def _super_cell_id(cell_id):
    x, y = (int(part) for part in str(cell_id).split("_"))
    return f"{x // SUPER_CELL_FACTOR}_{y // SUPER_CELL_FACTOR}"


def _super_cell_members(super_cell_id):
    x, y = (int(part) for part in str(super_cell_id).split("_"))
    return [
        f"{x * SUPER_CELL_FACTOR + dx}_{y * SUPER_CELL_FACTOR + dy}"
        for dx in range(SUPER_CELL_FACTOR)
        for dy in range(SUPER_CELL_FACTOR)
    ]


def _load_super_static(cursor, super_cell_id):
    rows = [
        row for row in (_load_static(cursor, member) for member in _super_cell_members(super_cell_id))
        if row is not None
    ]
    if not rows:
        return None
    result = {}
    for field in ("crime_total", "requests_311_total", "poi_total", "business_total", "permits_total"):
        result[field] = sum(float(row[field] or 0) for row in rows)
    for field in ("cluster", "static_crime_score", "static_activity_score", "traffic_mean"):
        result[field] = float(np.mean([float(row[field] or 0) for row in rows]))
    result["embedding_vector"] = np.mean(
        [parse_embedding_vector(row["embedding_vector"]) for row in rows], axis=0
    )
    for hour in range(24):
        field = f"baseline_crime_hour_{hour:02d}"
        result[field] = sum(float(row[field] or 0) for row in rows)
    return result


def _hourly_super_matrix(cursor, table, timestamp_column, super_cells, start, end):
    member_groups = [_super_cell_members(cell) for cell in super_cells]
    members = [member for group in member_groups for member in group]
    native, index = _hourly_matrix(
        cursor, table, timestamp_column, members, start, end
    )
    lookup = {member: position for position, member in enumerate(members)}
    matrix = np.zeros((len(super_cells), len(index)), dtype=np.float32)
    for row, group in enumerate(member_groups):
        matrix[row] = native[[lookup[member] for member in group]].sum(axis=0)
    return matrix, index


def _build_hierarchical_inputs(cursor, cell_id, target_static, reference, forecast_blocks, forecast_block_hours):
    super_cell = _super_cell_id(cell_id)
    static = _load_super_static(cursor, super_cell)
    if static is None:
        raise LookupError(f"No hierarchical static profile exists for {super_cell}")
    super_cells = [super_cell] + _neighbor_ids(super_cell)
    start = reference - pd.Timedelta(days=HISTORY_DAYS)
    crime, index = _hourly_super_matrix(
        cursor, "crimes", "timestamp", super_cells, start, reference
    )
    requests, _ = _hourly_super_matrix(
        cursor, "requests_311", "created_date", super_cells, start, reference
    )
    target_crime, target_requests = crime[0], requests[0]
    sequence = np.stack([
        target_crime[-SEQUENCE_HOURS:], target_requests[-SEQUENCE_HOURS:]
    ], axis=1)
    ref_hour, ref_weekday, ref_month = reference.hour, reference.weekday(), reference.month
    same_slot = np.asarray([
        ts.hour == ref_hour and ts.weekday() == ref_weekday for ts in index
    ])
    habits = [
        _seasonal_mean(target_crime, index, "hour", ref_hour),
        _seasonal_mean(target_crime, index, "weekday", ref_weekday),
        float(target_crime[same_slot].mean()) if same_slot.any() else 0.0,
        _seasonal_mean(target_requests, index, "hour", ref_hour),
        _seasonal_mean(target_requests, index, "weekday", ref_weekday),
    ]
    windows = (1, 3, 6, 12, 24, 48, 72)
    recent = [float(target_crime[-window:].sum()) for window in windows]
    recent += [float(target_requests[-window:].sum()) for window in windows]
    crime_prev_24 = max(float(target_crime[-48:-24].sum()), 0.0)
    req_prev_24 = max(float(target_requests[-48:-24].sum()), 0.0)
    neighbor_crime_3 = float(crime[1:, -3:].sum(axis=1).mean()) if crime.shape[0] > 1 else 0.0
    neighbor_crime_24 = float(crime[1:, -24:].sum(axis=1).mean()) if crime.shape[0] > 1 else 0.0
    neighbor_crime_prev_24 = (
        float(crime[1:, -48:-24].sum(axis=1).mean())
        if crime.shape[0] > 1 and crime.shape[1] >= 48 else 0.0
    )
    neighbor_max_crime_3 = float(crime[1:, -3:].sum(axis=1).max()) if crime.shape[0] > 1 else 0.0
    neighbor_max_crime_24 = float(crime[1:, -24:].sum(axis=1).max()) if crime.shape[0] > 1 else 0.0
    neighbor_max_crime_prev_24 = (
        float(crime[1:, -48:-24].sum(axis=1).max())
        if crime.shape[0] > 1 and crime.shape[1] >= 48 else 0.0
    )
    neighbor_req_24 = float(requests[1:, -24:].sum(axis=1).mean()) if requests.shape[0] > 1 else 0.0
    neighbor_req_prev_24 = (
        float(requests[1:, -48:-24].sum(axis=1).mean())
        if requests.shape[0] > 1 and requests.shape[1] >= 48 else 0.0
    )
    neighbor_max_req_24 = float(requests[1:, -24:].sum(axis=1).max()) if requests.shape[0] > 1 else 0.0
    neighbor_max_req_prev_24 = (
        float(requests[1:, -48:-24].sum(axis=1).max())
        if requests.shape[0] > 1 and requests.shape[1] >= 48 else 0.0
    )
    neighbors = [
        float(crime[1:, -window:].sum(axis=1).mean()) for window in (1, 3, 6, 24)
    ] + [
        float(requests[1:, -window:].sum(axis=1).mean()) for window in (1, 3, 6, 24)
    ]
    surge_features = [
        crime_prev_24,
        _safe_ratio(recent[1], recent[4]),
        _safe_ratio(recent[4] + 1.0, crime_prev_24 + 1.0),
        req_prev_24,
        _safe_ratio(recent[8], recent[11]),
        _safe_ratio(recent[11] + 1.0, req_prev_24 + 1.0),
        neighbor_crime_prev_24,
        _safe_ratio(neighbor_crime_3, neighbor_crime_24),
        _safe_ratio(neighbor_crime_24 + 1.0, neighbor_crime_prev_24 + 1.0),
        neighbor_req_prev_24,
        _safe_ratio(neighbor_req_24 + 1.0, neighbor_req_prev_24 + 1.0),
        _safe_ratio(recent[4] + 1.0, neighbor_crime_24 + 1.0),
        _safe_ratio(recent[11] + 1.0, neighbor_req_24 + 1.0),
        neighbor_max_crime_3,
        neighbor_max_crime_24,
        max(neighbor_max_crime_prev_24, 0.0),
        neighbor_max_req_24,
        max(neighbor_max_req_prev_24, 0.0),
        _safe_ratio(recent[4] + 1.0, neighbor_max_crime_24 + 1.0),
    ]
    scalar = np.asarray([
        math.sin(2 * math.pi * ref_hour / 24), math.cos(2 * math.pi * ref_hour / 24),
        math.sin(2 * math.pi * ref_weekday / 7), math.cos(2 * math.pi * ref_weekday / 7),
        float(ref_weekday >= 5), math.sin(2 * math.pi * (ref_month - 1) / 12),
        math.cos(2 * math.pi * (ref_month - 1) / 12), static["cluster"],
        static["static_crime_score"], static["static_activity_score"], static["crime_total"],
        static["requests_311_total"], static["poi_total"], static["business_total"],
        static["permits_total"], static["traffic_mean"], *habits, *recent, *neighbors, *surge_features,
        *parse_embedding_vector(static["embedding_vector"]),
    ], dtype=np.float32)
    super_baseline_by_hour = np.asarray([
        static[f"baseline_crime_hour_{hour:02d}"] for hour in range(24)
    ], dtype=np.float32)
    target_baseline_by_hour = np.asarray([
        target_static[f"baseline_crime_hour_{hour:02d}"] or 0 for hour in range(24)
    ], dtype=np.float32)
    order = (ref_hour + np.arange(24)) % 24
    super_hourly = super_baseline_by_hour[order]
    target_hourly = target_baseline_by_hour[order]
    block_baseline = super_hourly.reshape(forecast_blocks, forecast_block_hours).sum(axis=1)
    return sequence, scalar, block_baseline, super_hourly, target_hourly, super_cell


def _record_shadow_forecast(cursor, cell_id, reference, model_version, forecast):
    target_start = pd.Timestamp(reference)
    target_end = target_start + pd.Timedelta(hours=24)
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS dynamic_forecast_shadow (
            cell_id TEXT NOT NULL,
            reference_time TIMESTAMP NOT NULL,
            model_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            forecast_status TEXT NOT NULL,
            abstained BOOLEAN NOT NULL,
            target_start TIMESTAMP NOT NULL,
            target_end TIMESTAMP NOT NULL,
            forecast JSONB NOT NULL,
            realized_total_24h DOUBLE PRECISION,
            realized_any_24h BOOLEAN,
            evaluated_at TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT NOW(),
            PRIMARY KEY (cell_id, reference_time, model_version)
        )
        """
    )
    cursor.execute(
        """
        INSERT INTO dynamic_forecast_shadow (
            cell_id, reference_time, model_version, model_name, forecast_status,
            abstained, target_start, target_end, forecast
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (cell_id, reference_time, model_version)
        DO UPDATE SET
            model_name = EXCLUDED.model_name,
            forecast_status = EXCLUDED.forecast_status,
            abstained = EXCLUDED.abstained,
            target_start = EXCLUDED.target_start,
            target_end = EXCLUDED.target_end,
            forecast = EXCLUDED.forecast,
            created_at = NOW()
        """,
        (
            cell_id,
            target_start.to_pydatetime(),
            model_version,
            str(forecast.get("model", "unknown")),
            str(forecast.get("forecast_status", "unknown")),
            bool(forecast.get("abstained", False)),
            target_start.to_pydatetime(),
            target_end.to_pydatetime(),
            Json(forecast),
        ),
    )


def forecast_next_24_hours(cell_id):
    model_bundle = _load_model()
    forecast_block_hours, forecast_blocks = _bundle_block_config(model_bundle)
    forecast_schema_version = f"hierarchical-500m-{forecast_block_hours}h-v1"
    with _connect() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dynamic_forecast_cache (
                cell_id TEXT NOT NULL,
                reference_time TIMESTAMP NOT NULL,
                model_version TEXT NOT NULL,
                forecast JSONB NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                PRIMARY KEY (cell_id, reference_time, model_version)
            )
            """
        )
        reference = _reference_time(cursor)
        model_version = (
            f"{model_bundle['path'].stat().st_mtime_ns if model_bundle is not None else 'static'}:"
            f"{forecast_schema_version}"
        )
        cursor.execute(
            """
            SELECT forecast FROM dynamic_forecast_cache
            WHERE cell_id = %s AND reference_time = %s AND model_version = %s
            """,
            (cell_id, reference.to_pydatetime(), model_version),
        )
        cached = cursor.fetchone()
        if cached:
            _record_shadow_forecast(cursor, cell_id, reference, model_version, cached["forecast"])
            return cached["forecast"]
        static = _load_static(cursor, cell_id)
        if static is None:
            raise LookupError(f"No trained static profile exists for cell {cell_id}")
        data_age_hours = (pd.Timestamp.now() - reference).total_seconds() / 3600.0
        if model_bundle is None:
            reason = "no_certified_dynamic_model"
            result = _static_fallback(cell_id, reference, static, reason, data_age_hours)
            cursor.execute(
                """
                INSERT INTO dynamic_forecast_cache (
                    cell_id, reference_time, model_version, forecast
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (cell_id, reference_time, model_version)
                DO UPDATE SET forecast = EXCLUDED.forecast, created_at = NOW()
                """,
                (cell_id, reference.to_pydatetime(), model_version, Json(result)),
            )
            _record_shadow_forecast(cursor, cell_id, reference, model_version, result)
            return result
        checkpoint = model_bundle["checkpoint"]
        sequence, scalar, baseline, super_hourly, target_hourly, super_cell = (
            _build_hierarchical_inputs(
                cursor, cell_id, static, reference, forecast_blocks, forecast_block_hours
            )
        )

    if model_bundle["family"] == "neural_two_head":
        checkpoint_names = list(checkpoint.get("scalar_feature_names", []))
        if checkpoint_names and checkpoint_names != list(SCALAR_FEATURE_NAMES):
            raise ValueError("Runtime scalar features do not match the trained checkpoint")
        mean = np.asarray(checkpoint["scalar_mean"], dtype=np.float32).reshape(-1)
        std = np.asarray(checkpoint["scalar_std"], dtype=np.float32).reshape(-1)
        scalar = scalar.reshape(-1)
        if scalar.shape != mean.shape or scalar.shape != std.shape:
            raise ValueError(
                f"Scalar normalization shape mismatch: input={scalar.shape}, "
                f"mean={mean.shape}, std={std.shape}"
            )
        scalar = (scalar - mean) / np.maximum(std, 1e-6)
        with torch.inference_mode():
            _, probability, conditional, expected = model_bundle["model"](
                torch.from_numpy(sequence[None, ...]),
                torch.from_numpy(scalar[None, ...]),
                torch.from_numpy(baseline[None, ...]),
                calibration_scale=float(checkpoint.get("calibration_scale", 1.0)),
                calibration_bias=float(checkpoint.get("calibration_bias", 0.0)),
            )
        block_probability = probability[0].numpy()
        block_conditional = conditional[0].numpy()
        block_expected = expected[0].numpy()
        model_name = f"hierarchical_forecaster_500m_{forecast_block_hours}h_neural"
    else:
        seq_features = np.log1p(np.clip(sequence, 0.0, 20.0)).reshape(1, -1)
        scalar_features = scalar.reshape(1, -1).astype(np.float32)
        baseline_features = np.clip(baseline.reshape(1, -1), 0.0, 20.0).astype(np.float32)
        features = np.concatenate([seq_features, scalar_features, baseline_features], axis=1)
        block_probability = []
        block_conditional = []
        block_expected = []
        for index, classifier in enumerate(checkpoint["classifiers"]):
            raw_probability = float(classifier.predict_proba(features)[:, 1][0])
            raw_probability = float(np.clip(raw_probability, 1e-6, 1.0 - 1e-6))
            raw_logit = math.log(raw_probability / (1.0 - raw_probability))
            calibration = checkpoint["calibration"][index]
            probability = float(
                1.0 / (
                    1.0
                    + math.exp(
                        -(
                            float(calibration.get("scale", 1.0)) * raw_logit
                            + float(calibration.get("bias", 0.0))
                        )
                    )
                )
            )
            regressor = checkpoint["regressors"][index]
            if isinstance(regressor, dict) and regressor.get("kind") == "constant_mean":
                conditional = float(regressor["value"])
            else:
                conditional = float(max(regressor.predict(features)[0], 1.0))
            block_probability.append(probability)
            block_conditional.append(conditional)
            block_expected.append(probability * conditional)
        block_probability = np.asarray(block_probability, dtype=np.float32)
        block_conditional = np.asarray(block_conditional, dtype=np.float32)
        block_expected = np.asarray(block_expected, dtype=np.float32)
        model_name = f"hierarchical_forecaster_500m_{forecast_block_hours}h_boosted"
    expected = np.zeros(24, dtype=np.float32)
    for block in range(forecast_blocks):
        section = slice(block * forecast_block_hours, (block + 1) * forecast_block_hours)
        denominator = float(super_hourly[section].sum())
        if denominator > 1e-8:
            weights = target_hourly[section] / denominator
        else:
            super_total = max(float(super_hourly.sum()), 1e-8)
            spatial_share = float(target_hourly.sum()) / super_total
            weights = np.full(forecast_block_hours, spatial_share / forecast_block_hours)
        expected[section] = block_expected[block] * weights
    probability = 1.0 - np.exp(-expected)
    conditional = np.divide(
        expected, probability, out=np.ones_like(expected), where=probability > 1e-8
    )
    baseline = target_hourly
    probability_any = float(1.0 - np.exp(-expected.sum()))
    static_average = float(target_hourly.mean())
    static_total = float(target_hourly.sum())
    has_static_baseline = static_average > 1e-8
    source_freshness = _source_freshness()
    calibration_reference = (
        static_average
        if has_static_baseline
        else max(float(expected.mean()), 1e-8)
    )

    hours = []
    legacy_timeline = []
    static_timeline = []
    for offset in range(24):
        timestamp = reference + pd.Timedelta(hours=offset)
        relative_index = float(np.clip(
            50.0 * expected[offset] / calibration_reference,
            0,
            100,
        ))
        static_index = float(np.clip(
            50.0 * baseline[offset] / calibration_reference,
            0,
            100,
        ))
        hours.append({
            "forecast_timestamp": timestamp.isoformat(),
            "hour": int(timestamp.hour),
            "occurrence_probability": round(float(probability[offset]), 6),
            "conditional_count": round(float(conditional[offset]), 6),
            "expected_count": round(float(expected[offset]), 6),
            "static_baseline_count": round(float(baseline[offset]), 6),
            "relative_to_static_average": round(
                float(expected[offset] / calibration_reference), 4
            ),
            "relative_activity_index": round(relative_index, 2),
        })
        legacy_timeline.append({"hour": int(timestamp.hour), "value": round(relative_index, 2)})
        static_timeline.append({"hour": int(timestamp.hour), "value": round(static_index, 2)})

    # The UI chart is a clock-hour profile, so midnight must always be column 0
    # even when the rolling forecast starts at another hour.
    legacy_timeline.sort(key=lambda row: row["hour"])
    static_timeline.sort(key=lambda row: row["hour"])

    dynamic_status = (
        "dynamic_uncertified_stale_source"
        if data_age_hours > MAX_DATA_AGE_HOURS
        else "dynamic_certified"
    )
    result = {
        "cell_id": cell_id,
        "reference_time": reference.isoformat(),
        "horizon_hours": 24,
        "model": model_name,
        "forecast_status": dynamic_status,
        "abstained": False,
        "advisory_reason": (
            "official_crime_source_stale"
            if dynamic_status == "dynamic_uncertified_stale_source"
            else None
        ),
        "target_definition": "reported-crime pressure, not certainty or underlying crime",
        "spatial_resolution": "3x3 native cells (~500 m)",
        "temporal_resolution_hours": forecast_block_hours,
        "super_cell_id": super_cell,
        "data_age_hours": round(float(data_age_hours), 2),
        "predicted_crime": round(float(expected.sum()), 6),
        "expected_crime_count_24h": round(float(expected.sum()), 6),
        "static_baseline_count_24h": round(static_total, 6),
        "probability_any_crime_24h": round(probability_any, 6),
        "prediction_confidence": None,
        "confidence_definition": "Not reported as model certainty",
        "static_average_hourly_count": round(static_average, 6),
        "calibration_source": (
            "trained_cell_static_hourly_average"
            if has_static_baseline
            else "forecast_mean_fallback_no_static_crime_history"
        ),
        "calibration_definition": (
            "Timeline index 50 equals the cell's trained static average hourly crime count"
        ),
        "hourly_distribution": legacy_timeline,
        "static_hourly_distribution": static_timeline,
        "hourly_forecast": hours,
        "source_freshness": source_freshness,
        "forecast_blocks": [
            {
                "offset_hours": block * forecast_block_hours,
                "duration_hours": forecast_block_hours,
                "occurrence_probability_super_cell": round(float(block_probability[block]), 6),
                "conditional_count_super_cell": round(float(block_conditional[block]), 6),
                "expected_count_super_cell": round(float(block_expected[block]), 6),
            }
            for block in range(forecast_blocks)
        ],
        "uncertainty": checkpoint.get("uncertainty", {"available": False}),
    }
    with _connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO dynamic_forecast_cache (
                cell_id, reference_time, model_version, forecast
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (cell_id, reference_time, model_version)
            DO UPDATE SET forecast = EXCLUDED.forecast, created_at = NOW()
            """,
            (cell_id, reference.to_pydatetime(), model_version, Json(result)),
        )
        _record_shadow_forecast(cursor, cell_id, reference, model_version, result)
    return result
