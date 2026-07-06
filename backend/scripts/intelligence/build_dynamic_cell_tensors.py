
import sys
import os
import json
import glob
import ast
from datetime import timedelta

import numpy as np
import pandas as pd
import psycopg2
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
from config.modeling import (
    DYNAMIC_LOOKBACK_DAYS, FORECAST_BLOCKS, FORECAST_BLOCK_HOURS,
    SUPER_CELL_FACTOR, chronological_boundaries,
)


# =========================================================
# CONFIG
# =========================================================

LOOKBACK_DAYS = DYNAMIC_LOOKBACK_DAYS
SEQUENCE_HOURS = 72
HORIZON_HOURS = FORECAST_BLOCKS * FORECAST_BLOCK_HOURS

CELL_BATCH_SIZE = 64

# Sampling caps
MAX_POSITIVE_SAMPLES_PER_CELL = 25
MAX_HARD_NEGATIVE_SAMPLES_PER_CELL = 30
MAX_ZERO_SAMPLES_PER_CELL = 4

RANDOM_STATE = 42

DEBUG_MAX_CELLS = (
    int(os.getenv("DYNAMIC_DEBUG_MAX_CELLS"))
    if os.getenv("DYNAMIC_DEBUG_MAX_CELLS") else None
)

OUT_DIR = os.path.join("backend", "data", "processed", "hierarchical_dynamic_chunks")
DEBUG_DIR = os.path.join("backend", "data", "debug", "hierarchical_dynamic_features")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)

STATIC_EMBEDDING_DIM = 64

SCALAR_FEATURE_NAMES = [
    "reference_hour_sin",
    "reference_hour_cos",
    "reference_weekday_sin",
    "reference_weekday_cos",
    "is_weekend",
    "month_sin",
    "month_cos",
    "cluster",
    "static_crime_score",
    "static_activity_score",
    "crime_total_static",
    "requests_311_total_static",
    "poi_total_static",
    "business_total_static",
    "permits_total_static",
    "traffic_mean_static",
    "habit_crime_hour_mean",
    "habit_crime_weekday_mean",
    "habit_crime_hour_weekday_mean",
    "habit_311_hour_mean",
    "habit_311_weekday_mean",
    "crime_last_1h",
    "crime_last_3h",
    "crime_last_6h",
    "crime_last_12h",
    "crime_last_24h",
    "crime_last_48h",
    "crime_last_72h",
    "req311_last_1h",
    "req311_last_3h",
    "req311_last_6h",
    "req311_last_12h",
    "req311_last_24h",
    "req311_last_48h",
    "req311_last_72h",
    "nbr_crime_last_1h",
    "nbr_crime_last_3h",
    "nbr_crime_last_6h",
    "nbr_crime_last_24h",
    "nbr_req311_last_1h",
    "nbr_req311_last_3h",
    "nbr_req311_last_6h",
    "nbr_req311_last_24h",
    "crime_prev_24h",
    "crime_3h_share_of_24h",
    "crime_24h_vs_prev24h_ratio",
    "req311_prev_24h",
    "req311_3h_share_of_24h",
    "req311_24h_vs_prev24h_ratio",
    "nbr_crime_prev_24h",
    "nbr_crime_3h_share_of_24h",
    "nbr_crime_24h_vs_prev24h_ratio",
    "nbr_req311_prev_24h",
    "nbr_req311_24h_vs_prev24h_ratio",
    "crime_24h_vs_neighbor_24h_ratio",
    "req311_24h_vs_neighbor_24h_ratio",
    "nbr_max_crime_last_3h",
    "nbr_max_crime_last_24h",
    "nbr_max_crime_prev_24h",
    "nbr_max_req311_last_24h",
    "nbr_max_req311_prev_24h",
    "crime_24h_vs_neighbor_max_24h_ratio",
] + [f"static_gnn_embedding_{i:02d}" for i in range(STATIC_EMBEDDING_DIM)]


# =========================================================
# CLEANUP
# =========================================================

def clear_old_chunks(out_dir: str) -> None:
    old_files = glob.glob(os.path.join(out_dir, "dynamic_chunk_*.pt"))
    for path in old_files:
        os.remove(path)
    print(f"Deleted {len(old_files)} old chunk files from {out_dir}")


# =========================================================
# DATABASE
# =========================================================

def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )


def get_time_window(cursor):
    cursor.execute("SELECT MAX(timestamp) FROM crimes;")
    end_ts = cursor.fetchone()[0]
    if end_ts is None:
        raise ValueError("No crime data found in crimes table.")
    start_ts = end_ts - timedelta(days=LOOKBACK_DAYS)
    return start_ts, end_ts


def validate_history_coverage(cursor, start_ts):
    cursor.execute("SELECT MIN(timestamp) FROM crimes WHERE timestamp IS NOT NULL")
    crime_start = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(created_date) FROM requests_311 WHERE created_date IS NOT NULL")
    requests_start = cursor.fetchone()[0]
    missing = []
    if crime_start is None or crime_start > start_ts + timedelta(days=1):
        missing.append(f"crime history begins {crime_start}, required {start_ts}")
    if requests_start is None or requests_start > start_ts + timedelta(days=1):
        missing.append(f"311 history begins {requests_start}, required {start_ts}")
    if missing:
        raise ValueError(
            "Refusing to encode unavailable history as zero events: " + "; ".join(missing)
        )
    return crime_start, requests_start


def load_static_final(conn):
    query = """
        SELECT
            p.cell_id,
            p.cluster,
            p.urban_profile,
            p.static_crime_score,
            p.static_activity_score,
            p.embedding_vector,

            f.crime_total,
            f.requests_311_total,
            f.poi_total,
            f.business_total,
            f.permits_total,
            f.traffic_mean,

            f.baseline_crime_hour_00,
            f.baseline_crime_hour_01,
            f.baseline_crime_hour_02,
            f.baseline_crime_hour_03,
            f.baseline_crime_hour_04,
            f.baseline_crime_hour_05,
            f.baseline_crime_hour_06,
            f.baseline_crime_hour_07,
            f.baseline_crime_hour_08,
            f.baseline_crime_hour_09,
            f.baseline_crime_hour_10,
            f.baseline_crime_hour_11,
            f.baseline_crime_hour_12,
            f.baseline_crime_hour_13,
            f.baseline_crime_hour_14,
            f.baseline_crime_hour_15,
            f.baseline_crime_hour_16,
            f.baseline_crime_hour_17,
            f.baseline_crime_hour_18,
            f.baseline_crime_hour_19,
            f.baseline_crime_hour_20,
            f.baseline_crime_hour_21,
            f.baseline_crime_hour_22,
            f.baseline_crime_hour_23
        FROM static_gnn_profiles p
        JOIN static_cell_features f
          ON p.cell_id = f.cell_id
        ORDER BY p.cell_id;
    """
    return pd.read_sql(query, conn)


def load_crime_hourly(conn, start_ts, end_ts):
    query = """
        SELECT
            cell_id,
            date_trunc('hour', timestamp) AS hour_ts,
            COUNT(*)::DOUBLE PRECISION AS crime_count
        FROM crimes
        WHERE timestamp >= %s
          AND timestamp <= %s
          AND cell_id IS NOT NULL
        GROUP BY cell_id, date_trunc('hour', timestamp)
        ORDER BY cell_id, hour_ts;
    """
    return pd.read_sql(query, conn, params=(start_ts, end_ts))


def load_311_hourly(conn, start_ts, end_ts):
    query = """
        SELECT
            cell_id,
            date_trunc('hour', created_date) AS hour_ts,
            COUNT(*)::DOUBLE PRECISION AS req311_count
        FROM requests_311
        WHERE created_date >= %s
          AND created_date <= %s
          AND cell_id IS NOT NULL
        GROUP BY cell_id, date_trunc('hour', created_date)
        ORDER BY cell_id, hour_ts;
    """
    return pd.read_sql(query, conn, params=(start_ts, end_ts))


# =========================================================
# HELPERS
# =========================================================

def sanitize_arr(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, 0.0, None)
    return x


def sanitize_finite_arr(x):
    """Replace invalid values without clipping signed learned embeddings."""
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def safe_ratio(numerator, denominator):
    numerator = np.asarray(numerator, dtype=np.float32)
    denominator = np.asarray(denominator, dtype=np.float32)
    return np.divide(
        numerator,
        np.maximum(denominator, 1e-6),
        out=np.zeros_like(numerator, dtype=np.float32),
        where=np.isfinite(denominator),
    ).astype(np.float32)


def to_super_cell(cell_id):
    """Map the native roughly 150 m grid to a stable 3x3, roughly 500 m grid."""
    try:
        x, y = (int(float(part)) for part in str(cell_id).split("_"))
        return f"{x // SUPER_CELL_FACTOR}_{y // SUPER_CELL_FACTOR}"
    except (TypeError, ValueError):
        return None


def aggregate_hourly_to_super_cells(frame, value_column):
    frame = frame.copy()
    frame["cell_id"] = frame["cell_id"].map(to_super_cell)
    frame = frame.dropna(subset=["cell_id"])
    return frame.groupby(["cell_id", "hour_ts"], as_index=False)[value_column].sum()


def aggregate_static_to_super_cells(frame):
    frame = frame.copy()
    frame["cell_id"] = frame["cell_id"].map(to_super_cell)
    frame = frame.dropna(subset=["cell_id"])
    baseline_columns = [f"baseline_crime_hour_{hour:02d}" for hour in range(24)]
    sum_columns = [
        "crime_total", "requests_311_total", "poi_total", "business_total",
        "permits_total", *baseline_columns,
    ]
    mean_columns = [
        "cluster", "static_crime_score", "static_activity_score", "traffic_mean",
    ]
    records = []
    for super_cell, group in frame.groupby("cell_id", sort=True):
        record = {"cell_id": super_cell, "urban_profile": "hierarchical_500m"}
        for column in sum_columns:
            record[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).sum()
        for column in mean_columns:
            record[column] = pd.to_numeric(group[column], errors="coerce").fillna(0).mean()
        vectors = np.stack([parse_embedding_vector(value) for value in group["embedding_vector"]])
        record["embedding_vector"] = vectors.mean(axis=0).tolist()
        records.append(record)
    return pd.DataFrame.from_records(records)


def parse_embedding_vector(value, expected_dim=STATIC_EMBEDDING_DIM):
    """Convert PostgreSQL/JSON/list embeddings to one fixed-width float vector."""
    if value is None:
        values = []
    elif isinstance(value, str):
        text = value.strip()
        try:
            values = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                values = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                values = [part for part in text.strip("{}").split(",") if part]
    else:
        values = value

    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.size < expected_dim:
        vector = np.pad(vector, (0, expected_dim - vector.size))
    return vector[:expected_dim].astype(np.float32)


def build_hourly_matrix(hourly_df, value_col, cell_ids, full_index):
    if hourly_df.empty:
        return np.zeros((len(cell_ids), len(full_index)), dtype=np.float32)
    pivot = hourly_df.pivot(index="cell_id", columns="hour_ts", values=value_col)
    pivot = pivot.reindex(index=cell_ids, columns=full_index, fill_value=0.0)
    return sanitize_arr(pivot.to_numpy(dtype=np.float32))


def rolling_sum_matrix(arr, window):
    arr = sanitize_arr(arr)
    out = np.zeros_like(arr, dtype=np.float32)
    csum = np.cumsum(arr, axis=1, dtype=np.float32)
    out[:, :window] = csum[:, :window]
    out[:, window:] = csum[:, window:] - csum[:, :-window]
    return sanitize_arr(out)


def future_sum_matrix(arr, horizon):
    """
    For each t, returns sum(arr[:, t:t+horizon]).
    Result valid for columns [0, n-horizon], trailing positions remain 0.
    """
    arr = sanitize_arr(arr)
    n = arr.shape[1]
    out = np.zeros_like(arr, dtype=np.float32)
    rev = arr[:, ::-1]
    rev_roll = rolling_sum_matrix(rev, horizon)
    out[:, :n] = rev_roll[:, ::-1]
    return sanitize_arr(out)


def causal_group_mean_at_positions(values, group_ids, positions):
    """Mean for each row's group using observations strictly before its reference position."""
    values = sanitize_arr(values).reshape(-1)
    group_ids = np.asarray(group_ids)
    positions = np.asarray(positions, dtype=np.int64)
    result = np.zeros(len(positions), dtype=np.float32)

    for group in np.unique(group_ids[positions]):
        output_mask = group_ids[positions] == group
        group_positions = np.flatnonzero(group_ids == group)
        ranks = np.searchsorted(group_positions, positions[output_mask], side="left")
        valid = ranks > 0
        if not valid.any():
            continue
        prefix = np.cumsum(values[group_positions], dtype=np.float64)
        group_result = np.zeros(output_mask.sum(), dtype=np.float32)
        group_result[valid] = (
            prefix[ranks[valid] - 1] / ranks[valid]
        ).astype(np.float32)
        result[output_mask] = group_result
    return sanitize_arr(result)


def get_baseline_matrix(static_df):
    cols = [f"baseline_crime_hour_{h:02d}" for h in range(24)]
    return sanitize_arr(static_df[cols].to_numpy(dtype=np.float32))


def rotate_baseline_matrix_next24(baseline_mat, ref_hours):
    """
    baseline_mat: (n, 24)
    ref_hours: (n,)
    return: (n, 24) where col j is baseline at hour (ref_hour + j) % 24.
    The first baseline value therefore aligns with target_windows[:, ref_hour].
    """
    ref_hours = np.asarray(ref_hours, dtype=np.int64)
    offsets = np.arange(0, 24, dtype=np.int64)[None, :]
    gather_idx = (ref_hours[:, None] + offsets) % 24
    row_idx = np.arange(len(ref_hours))[:, None]
    out = baseline_mat[row_idx, gather_idx]
    out = sanitize_arr(out)
    return out.reshape(len(out), FORECAST_BLOCKS, FORECAST_BLOCK_HOURS).sum(axis=2)


def precompute_neighbor_indices(cell_ids):
    cell_to_idx = {c: i for i, c in enumerate(cell_ids)}
    neighbors = []
    for cell_id in cell_ids:
        try:
            x_str, y_str = str(cell_id).split("_")
            x = int(float(x_str))
            y = int(float(y_str))
        except Exception:
            neighbors.append(np.array([], dtype=np.int64))
            continue

        nbr_idx = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nbr = f"{x + dx}_{y + dy}"
                j = cell_to_idx.get(nbr)
                if j is not None:
                    nbr_idx.append(j)
        neighbors.append(np.array(nbr_idx, dtype=np.int64))
    return neighbors


def direct_neighbor_cell_ids(cell_id, available_cell_ids):
    try:
        x_str, y_str = str(cell_id).split("_")
        x = int(float(x_str))
        y = int(float(y_str))
    except Exception:
        return []

    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = f"{x + dx}_{y + dy}"
            if candidate in available_cell_ids:
                result.append(candidate)
    return result


def build_neighbor_mean_matrices(mats_dict, neighbor_indices):
    out = {name: np.zeros_like(mat, dtype=np.float32) for name, mat in mats_dict.items()}
    for i, nbr_idx in enumerate(neighbor_indices):
        if nbr_idx.size == 0:
            continue
        for name, mat in mats_dict.items():
            out[name][i] = sanitize_arr(mat[nbr_idx].mean(axis=0))
    return out


def build_neighbor_max_matrices(mats_dict, neighbor_indices):
    out = {name: np.zeros_like(mat, dtype=np.float32) for name, mat in mats_dict.items()}
    for i, nbr_idx in enumerate(neighbor_indices):
        if nbr_idx.size == 0:
            continue
        for name, mat in mats_dict.items():
            out[name][i] = sanitize_arr(mat[nbr_idx].max(axis=0))
    return out


def sample_positions(pos_arr, max_count):
    if pos_arr.size <= max_count:
        return pos_arr
    return np.sort(np.random.choice(pos_arr, size=max_count, replace=False))


def save_debug_summary(all_target_totals, all_occ, all_ref_hours, modeling_boundaries):
    if len(all_target_totals) == 0:
        return

    target_total = sanitize_arr(all_target_totals)
    occurrence = sanitize_arr(all_occ)
    ref_hours = np.array(all_ref_hours, dtype=np.int32)

    plt.figure(figsize=(8, 5))
    plt.hist(target_total, bins=60)
    plt.title("Target total next 24h distribution")
    plt.xlabel("Target total 24h")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "target_total_24h_hist.png"), dpi=180)
    plt.close()

    rates = []
    for h in range(24):
        mask = ref_hours == h
        rates.append(float(occurrence[mask].mean()) if mask.any() else 0.0)

    plt.figure(figsize=(10, 5))
    plt.plot(range(24), rates, marker="o")
    plt.title("Future crime occurrence rate by reference hour")
    plt.xlabel("Reference hour")
    plt.ylabel("Share with future crime")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "future_crime_rate_by_hour.png"), dpi=180)
    plt.close()

    diagnostics = {
        "row_count": int(len(target_total)),
        "future_crime_rate": float(occurrence.mean()),
        "avg_target_total_24h": float(target_total.mean()),
        "max_target_total_24h": float(target_total.max()),
        "causal_habit_features": True,
        "sequence_ends_before_reference": True,
        "rolling_features_end_before_reference": True,
        "forecast_target_starts_at_reference": True,
        "debug_max_cells": DEBUG_MAX_CELLS,
        "spatial_resolution": "3x3 native cells (~500 m)",
        "forecast_block_hours": FORECAST_BLOCK_HOURS,
        "modeling_boundaries": {
            key: value.isoformat() for key, value in modeling_boundaries.items()
        },
    }

    with open(os.path.join(DEBUG_DIR, "dynamic_diagnostics.json"), "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2)


# =========================================================
# BATCH BUILDER
# =========================================================

def process_cell_batch(
    batch_idx,
    static_batch,
    crime_hourly,
    req311_hourly,
    full_index,
    available_cell_ids,
    modeling_boundaries,
):
    cell_ids = static_batch["cell_id"].tolist()
    num_cells = len(cell_ids)
    n_hours = len(full_index)

    if num_cells == 0:
        return None

    # Include neighbors outside this storage batch. Previously, those neighbors
    # silently became zeros whenever they fell across a 250-cell batch boundary.
    context_cell_ids = list(cell_ids)
    seen_context = set(context_cell_ids)
    for cell_id in cell_ids:
        for neighbor_id in direct_neighbor_cell_ids(cell_id, available_cell_ids):
            if neighbor_id not in seen_context:
                context_cell_ids.append(neighbor_id)
                seen_context.add(neighbor_id)

    crime_sub = crime_hourly[crime_hourly["cell_id"].isin(context_cell_ids)]
    req_sub = req311_hourly[req311_hourly["cell_id"].isin(context_cell_ids)]

    crime_context = build_hourly_matrix(crime_sub, "crime_count", context_cell_ids, full_index)
    req_context = build_hourly_matrix(req_sub, "req311_count", context_cell_ids, full_index)
    context_lookup = {cell_id: i for i, cell_id in enumerate(context_cell_ids)}
    target_context_idx = np.array([context_lookup[cell_id] for cell_id in cell_ids], dtype=np.int64)
    crime_mat = crime_context[target_context_idx]
    req_mat = req_context[target_context_idx]

    crime_last_1 = rolling_sum_matrix(crime_mat, 1)
    crime_last_3 = rolling_sum_matrix(crime_mat, 3)
    crime_last_6 = rolling_sum_matrix(crime_mat, 6)
    crime_last_12 = rolling_sum_matrix(crime_mat, 12)
    crime_last_24 = rolling_sum_matrix(crime_mat, 24)
    crime_last_48 = rolling_sum_matrix(crime_mat, 48)
    crime_last_72 = rolling_sum_matrix(crime_mat, 72)

    req_last_1 = rolling_sum_matrix(req_mat, 1)
    req_last_3 = rolling_sum_matrix(req_mat, 3)
    req_last_6 = rolling_sum_matrix(req_mat, 6)
    req_last_12 = rolling_sum_matrix(req_mat, 12)
    req_last_24 = rolling_sum_matrix(req_mat, 24)
    req_last_48 = rolling_sum_matrix(req_mat, 48)
    req_last_72 = rolling_sum_matrix(req_mat, 72)

    recent_signal = sanitize_arr(crime_last_24 + req_last_24)
    future_24h_total = future_sum_matrix(crime_mat, HORIZON_HOURS)

    context_neighbor_indices = precompute_neighbor_indices(context_cell_ids)
    neighbor_context_mats = build_neighbor_mean_matrices(
        {
            "nbr_crime_last_1": rolling_sum_matrix(crime_context, 1),
            "nbr_crime_last_3": rolling_sum_matrix(crime_context, 3),
            "nbr_crime_last_6": rolling_sum_matrix(crime_context, 6),
            "nbr_crime_last_24": rolling_sum_matrix(crime_context, 24),
            "nbr_crime_last_48": rolling_sum_matrix(crime_context, 48),
            "nbr_req_last_1": rolling_sum_matrix(req_context, 1),
            "nbr_req_last_3": rolling_sum_matrix(req_context, 3),
            "nbr_req_last_6": rolling_sum_matrix(req_context, 6),
            "nbr_req_last_24": rolling_sum_matrix(req_context, 24),
            "nbr_req_last_48": rolling_sum_matrix(req_context, 48),
        },
        context_neighbor_indices
    )
    neighbor_mats = {
        name: values[target_context_idx]
        for name, values in neighbor_context_mats.items()
    }
    neighbor_max_context_mats = build_neighbor_max_matrices(
        {
            "nbr_max_crime_last_3": rolling_sum_matrix(crime_context, 3),
            "nbr_max_crime_last_24": rolling_sum_matrix(crime_context, 24),
            "nbr_max_crime_last_48": rolling_sum_matrix(crime_context, 48),
            "nbr_max_req_last_24": rolling_sum_matrix(req_context, 24),
            "nbr_max_req_last_48": rolling_sum_matrix(req_context, 48),
        },
        context_neighbor_indices,
    )
    neighbor_max_mats = {
        name: values[target_context_idx]
        for name, values in neighbor_max_context_mats.items()
    }

    baseline_mat = get_baseline_matrix(static_batch)

    hours = np.array([ts.hour for ts in full_index], dtype=np.int32)
    weekdays = np.array([ts.weekday() for ts in full_index], dtype=np.int32)
    months = np.array([ts.month for ts in full_index], dtype=np.int32)
    weekends = (weekdays >= 5).astype(np.int32)
    hour_weekday_groups = weekdays * 24 + hours

    valid_start = SEQUENCE_HOURS
    valid_end = n_hours - HORIZON_HOURS
    if valid_end <= valid_start:
        return None

    valid_slice = slice(valid_start, valid_end)
    candidate_hours = np.arange(valid_start, valid_end, dtype=np.int32)

    future_valid = future_24h_total[:, valid_slice]
    recent_valid = recent_signal[:, valid_slice]
    neighbor_recent_valid = sanitize_arr(
        neighbor_mats["nbr_crime_last_24"][:, valid_slice]
        + neighbor_mats["nbr_req_last_24"][:, valid_slice]
    )

    # Sliding windows for full batch
    seq_src = np.stack([crime_mat, req_mat], axis=2)  # (cells, hours, 2)
    seq_windows = np.lib.stride_tricks.sliding_window_view(
        seq_src, window_shape=SEQUENCE_HOURS, axis=1
    )
    seq_windows = np.transpose(seq_windows, (0, 1, 3, 2)).astype(np.float32)  # (cells, hours-72+1, 72, 2)

    target_windows = np.lib.stride_tricks.sliding_window_view(
        crime_mat, window_shape=HORIZON_HOURS, axis=1
    ).astype(np.float32)  # (cells, hours-24+1, 24)

    # Static arrays
    static_numeric_cols = [
        "cluster",
        "static_crime_score",
        "static_activity_score",
        "crime_total",
        "requests_311_total",
        "poi_total",
        "business_total",
        "permits_total",
        "traffic_mean",
    ]
    static_num = sanitize_arr(static_batch[static_numeric_cols].to_numpy(dtype=np.float32))
    static_embeddings = np.stack(
        [parse_embedding_vector(value) for value in static_batch["embedding_vector"].tolist()],
        axis=0,
    )

    X_seq_blocks = []
    X_scalar_blocks = []
    X_baseline_blocks = []
    Y_blocks = []
    sample_weight_blocks = []
    cell_meta = []
    time_meta = []
    target_totals = []
    occurrence_meta = []
    ref_hours_meta = []
    cluster_meta = []

    batch_pos_count = 0
    batch_hard_neg_count = 0
    batch_zero_count = 0

    for i, cell_id in enumerate(cell_ids):
        positive_all = candidate_hours[future_valid[i] > 0]
        hard_negative_all = candidate_hours[
            (future_valid[i] == 0)
            & ((recent_valid[i] > 0) | (neighbor_recent_valid[i] > 0))
        ]
        pure_zero_all = candidate_hours[
            (future_valid[i] == 0)
            & (recent_valid[i] == 0)
            & (neighbor_recent_valid[i] == 0)
        ]

        positive_pos = sample_positions(positive_all, MAX_POSITIVE_SAMPLES_PER_CELL)
        hard_negative_pos = sample_positions(hard_negative_all, MAX_HARD_NEGATIVE_SAMPLES_PER_CELL)
        pure_zero_pos = sample_positions(pure_zero_all, MAX_ZERO_SAMPLES_PER_CELL)

        batch_pos_count += int(len(positive_pos))
        batch_hard_neg_count += int(len(hard_negative_pos))
        batch_zero_count += int(len(pure_zero_pos))

        selected_unsorted = np.concatenate([positive_pos, hard_negative_pos, pure_zero_pos])
        weight_unsorted = np.concatenate([
            np.full(len(positive_pos), len(positive_all) / max(len(positive_pos), 1), dtype=np.float32),
            np.full(len(hard_negative_pos), len(hard_negative_all) / max(len(hard_negative_pos), 1), dtype=np.float32),
            np.full(len(pure_zero_pos), len(pure_zero_all) / max(len(pure_zero_pos), 1), dtype=np.float32),
        ])
        order = np.argsort(selected_unsorted)
        selected = selected_unsorted[order]
        sample_weight_block = weight_unsorted[order]
        if selected.size == 0:
            continue

        seq_idx = selected - SEQUENCE_HOURS
        target_idx = selected

        seq_block = sanitize_arr(seq_windows[i, seq_idx])
        y_hourly = sanitize_arr(target_windows[i, target_idx])
        y_block = y_hourly.reshape(
            len(selected), FORECAST_BLOCKS, FORECAST_BLOCK_HOURS
        ).sum(axis=2)

        ref_hour_block = hours[selected].astype(np.int32)
        ref_weekday_block = weekdays[selected].astype(np.int32)
        ref_month_block = months[selected].astype(np.int32)
        ref_weekend_block = weekends[selected].astype(np.int32)

        # These seasonal habits are causal: the value at reference time t is
        # computed only from observations before t, never from validation/test future data.
        habit_crime_hour = causal_group_mean_at_positions(crime_mat[i], hours, selected)
        habit_crime_weekday = causal_group_mean_at_positions(crime_mat[i], weekdays, selected)
        habit_crime_hour_weekday = causal_group_mean_at_positions(
            crime_mat[i], hour_weekday_groups, selected
        )
        habit_req_hour = causal_group_mean_at_positions(req_mat[i], hours, selected)
        habit_req_weekday = causal_group_mean_at_positions(req_mat[i], weekdays, selected)

        baseline_block = rotate_baseline_matrix_next24(baseline_mat[i:i+1].repeat(len(selected), axis=0), ref_hour_block)

        scalar_block = np.column_stack([
            np.sin(2.0 * np.pi * ref_hour_block / 24.0),
            np.cos(2.0 * np.pi * ref_hour_block / 24.0),
            np.sin(2.0 * np.pi * ref_weekday_block / 7.0),
            np.cos(2.0 * np.pi * ref_weekday_block / 7.0),
            ref_weekend_block,
            np.sin(2.0 * np.pi * (ref_month_block - 1) / 12.0),
            np.cos(2.0 * np.pi * (ref_month_block - 1) / 12.0),
            np.full(len(selected), static_num[i, 0], dtype=np.float32),
            np.full(len(selected), static_num[i, 1], dtype=np.float32),
            np.full(len(selected), static_num[i, 2], dtype=np.float32),
            np.full(len(selected), static_num[i, 3], dtype=np.float32),
            np.full(len(selected), static_num[i, 4], dtype=np.float32),
            np.full(len(selected), static_num[i, 5], dtype=np.float32),
            np.full(len(selected), static_num[i, 6], dtype=np.float32),
            np.full(len(selected), static_num[i, 7], dtype=np.float32),
            np.full(len(selected), static_num[i, 8], dtype=np.float32),
            habit_crime_hour,
            habit_crime_weekday,
            habit_crime_hour_weekday,
            habit_req_hour,
            habit_req_weekday,
            crime_last_1[i, selected - 1],
            crime_last_3[i, selected - 1],
            crime_last_6[i, selected - 1],
            crime_last_12[i, selected - 1],
            crime_last_24[i, selected - 1],
            crime_last_48[i, selected - 1],
            crime_last_72[i, selected - 1],
            req_last_1[i, selected - 1],
            req_last_3[i, selected - 1],
            req_last_6[i, selected - 1],
            req_last_12[i, selected - 1],
            req_last_24[i, selected - 1],
            req_last_48[i, selected - 1],
            req_last_72[i, selected - 1],
            neighbor_mats["nbr_crime_last_1"][i, selected - 1],
            neighbor_mats["nbr_crime_last_3"][i, selected - 1],
            neighbor_mats["nbr_crime_last_6"][i, selected - 1],
            neighbor_mats["nbr_crime_last_24"][i, selected - 1],
            neighbor_mats["nbr_req_last_1"][i, selected - 1],
            neighbor_mats["nbr_req_last_3"][i, selected - 1],
            neighbor_mats["nbr_req_last_6"][i, selected - 1],
            neighbor_mats["nbr_req_last_24"][i, selected - 1],
            np.maximum(crime_last_48[i, selected - 1] - crime_last_24[i, selected - 1], 0.0),
            safe_ratio(crime_last_3[i, selected - 1], crime_last_24[i, selected - 1]),
            safe_ratio(
                crime_last_24[i, selected - 1] + 1.0,
                np.maximum(crime_last_48[i, selected - 1] - crime_last_24[i, selected - 1], 0.0) + 1.0,
            ),
            np.maximum(req_last_48[i, selected - 1] - req_last_24[i, selected - 1], 0.0),
            safe_ratio(req_last_3[i, selected - 1], req_last_24[i, selected - 1]),
            safe_ratio(
                req_last_24[i, selected - 1] + 1.0,
                np.maximum(req_last_48[i, selected - 1] - req_last_24[i, selected - 1], 0.0) + 1.0,
            ),
            np.maximum(
                neighbor_mats["nbr_crime_last_48"][i, selected - 1]
                - neighbor_mats["nbr_crime_last_24"][i, selected - 1],
                0.0,
            ),
            safe_ratio(
                neighbor_mats["nbr_crime_last_3"][i, selected - 1],
                neighbor_mats["nbr_crime_last_24"][i, selected - 1],
            ),
            safe_ratio(
                neighbor_mats["nbr_crime_last_24"][i, selected - 1] + 1.0,
                np.maximum(
                    neighbor_mats["nbr_crime_last_48"][i, selected - 1]
                    - neighbor_mats["nbr_crime_last_24"][i, selected - 1],
                    0.0,
                ) + 1.0,
            ),
            np.maximum(
                neighbor_mats["nbr_req_last_48"][i, selected - 1]
                - neighbor_mats["nbr_req_last_24"][i, selected - 1],
                0.0,
            ),
            safe_ratio(
                neighbor_mats["nbr_req_last_24"][i, selected - 1] + 1.0,
                np.maximum(
                    neighbor_mats["nbr_req_last_48"][i, selected - 1]
                    - neighbor_mats["nbr_req_last_24"][i, selected - 1],
                    0.0,
                ) + 1.0,
            ),
            safe_ratio(
                crime_last_24[i, selected - 1] + 1.0,
                neighbor_mats["nbr_crime_last_24"][i, selected - 1] + 1.0,
            ),
            safe_ratio(
                req_last_24[i, selected - 1] + 1.0,
                neighbor_mats["nbr_req_last_24"][i, selected - 1] + 1.0,
            ),
            neighbor_max_mats["nbr_max_crime_last_3"][i, selected - 1],
            neighbor_max_mats["nbr_max_crime_last_24"][i, selected - 1],
            np.maximum(
                neighbor_max_mats["nbr_max_crime_last_48"][i, selected - 1]
                - neighbor_max_mats["nbr_max_crime_last_24"][i, selected - 1],
                0.0,
            ),
            neighbor_max_mats["nbr_max_req_last_24"][i, selected - 1],
            np.maximum(
                neighbor_max_mats["nbr_max_req_last_48"][i, selected - 1]
                - neighbor_max_mats["nbr_max_req_last_24"][i, selected - 1],
                0.0,
            ),
            safe_ratio(
                crime_last_24[i, selected - 1] + 1.0,
                neighbor_max_mats["nbr_max_crime_last_24"][i, selected - 1] + 1.0,
            ),
        ]).astype(np.float32)
        scalar_block = np.concatenate(
            [scalar_block, np.repeat(static_embeddings[i:i + 1], len(selected), axis=0)],
            axis=1,
        )
        scalar_block = sanitize_finite_arr(scalar_block)

        row_totals = sanitize_arr(y_block.sum(axis=1))
        occ_block = (row_totals > 0).astype(np.float32)

        X_seq_blocks.append(seq_block)
        X_scalar_blocks.append(scalar_block)
        X_baseline_blocks.append(baseline_block)
        Y_blocks.append(y_block)
        sample_weight_blocks.append(sample_weight_block)

        cell_meta.extend([cell_id] * len(selected))
        time_meta.extend(pd.Index(full_index[selected]).astype(str).tolist())
        target_totals.extend(row_totals.tolist())
        occurrence_meta.extend(occ_block.tolist())
        ref_hours_meta.extend(ref_hour_block.astype(int).tolist())
        cluster_meta.extend([int(round(static_num[i, 0]))] * len(selected))

    if not X_seq_blocks:
        return None

    X_seq_arr = sanitize_arr(np.concatenate(X_seq_blocks, axis=0))
    X_scalar_arr = sanitize_finite_arr(np.concatenate(X_scalar_blocks, axis=0))
    X_baseline_arr = sanitize_arr(np.concatenate(X_baseline_blocks, axis=0))
    Y_arr = sanitize_arr(np.concatenate(Y_blocks, axis=0))
    sample_weight_arr = sanitize_arr(np.concatenate(sample_weight_blocks, axis=0))

    print("Selected positive rows:", batch_pos_count)
    print("Selected hard negatives:", batch_hard_neg_count)
    print("Selected pure zeros:", batch_zero_count)
    print("Chunk raw Y max:", float(np.nanmax(Y_arr)))
    print("Chunk raw Y row-total max:", float(np.nanmax(np.sum(Y_arr, axis=1))))
    print("Chunk positive ratio:", float(np.mean(np.sum(Y_arr, axis=1) > 0)))

    chunk = {
        "X_seq": torch.tensor(X_seq_arr, dtype=torch.float32),
        "X_scalar": torch.tensor(X_scalar_arr, dtype=torch.float32),
        "X_baseline": torch.tensor(X_baseline_arr, dtype=torch.float32),
        "Y": torch.tensor(Y_arr, dtype=torch.float32),
        "sample_weights": torch.tensor(sample_weight_arr, dtype=torch.float32),
        "cell_ids": cell_meta,
        "reference_times": time_meta,
        "cluster_labels": cluster_meta,
        "scalar_feature_names": SCALAR_FEATURE_NAMES,
        "modeling_boundaries": {
            key: value.isoformat() for key, value in modeling_boundaries.items()
        },
        "spatial_resolution": "3x3 native cells (~500 m)",
        "forecast_block_hours": FORECAST_BLOCK_HOURS,
        "forecast_blocks": FORECAST_BLOCKS,
    }

    out_path = os.path.join(OUT_DIR, f"dynamic_chunk_{batch_idx:04d}.pt")
    torch.save(chunk, out_path)

    return {
        "path": out_path,
        "rows": int(X_seq_arr.shape[0]),
        "target_totals": target_totals,
        "occurrences": occurrence_meta,
        "ref_hours": ref_hours_meta
    }


# =========================================================
# MAIN
# =========================================================

def main():
    conn = connect()
    cursor = conn.cursor()

    start_ts, end_ts = get_time_window(cursor)
    crime_history_start, requests_history_start = validate_history_coverage(cursor, start_ts)
    modeling_boundaries = chronological_boundaries(end_ts)
    print(f"Building gated dynamic tensor chunks from {start_ts} to {end_ts}")
    print(f"Verified source coverage: crimes {crime_history_start}; 311 {requests_history_start}")

    static_df = aggregate_static_to_super_cells(load_static_final(conn))
    crime_hourly = aggregate_hourly_to_super_cells(
        load_crime_hourly(conn, start_ts, end_ts), "crime_count"
    )
    req311_hourly = aggregate_hourly_to_super_cells(
        load_311_hourly(conn, start_ts, end_ts), "req311_count"
    )

    print(f"Loaded static rows: {len(static_df)}")
    print(f"Crime hourly rows:  {len(crime_hourly)}")
    print(f"311 hourly rows:    {len(req311_hourly)}")

    signal_cells = sorted(set(crime_hourly["cell_id"].unique()).union(set(req311_hourly["cell_id"].unique())))
    static_df = static_df[static_df["cell_id"].isin(signal_cells)].copy()
    available_cell_ids = set(static_df["cell_id"].tolist())

    if DEBUG_MAX_CELLS is not None:
        static_df = static_df.head(DEBUG_MAX_CELLS).copy()

    print(f"Cells used for dynamic build: {len(static_df)}")

    clear_old_chunks(OUT_DIR)

    full_index = pd.date_range(start=start_ts, end=end_ts, freq="h")

    total_batches = int(np.ceil(len(static_df) / CELL_BATCH_SIZE))
    all_target_totals = []
    all_occ = []
    all_ref_hours = []
    total_rows = 0

    for batch_idx in range(total_batches):
        start_i = batch_idx * CELL_BATCH_SIZE
        end_i = min((batch_idx + 1) * CELL_BATCH_SIZE, len(static_df))
        static_batch = static_df.iloc[start_i:end_i].copy()

        print(f"Processing batch {batch_idx + 1}/{total_batches} | cells {start_i}:{end_i}")

        result = process_cell_batch(
            batch_idx=batch_idx,
            static_batch=static_batch,
            crime_hourly=crime_hourly,
            req311_hourly=req311_hourly,
            full_index=full_index,
            available_cell_ids=available_cell_ids,
            modeling_boundaries=modeling_boundaries,
        )

        if result is None:
            print("  No rows in this batch.")
            continue

        total_rows += result["rows"]
        all_target_totals.extend(result["target_totals"])
        all_occ.extend(result["occurrences"])
        all_ref_hours.extend(result["ref_hours"])

        print(f"  Saved {result['rows']} rows -> {result['path']}")

    save_debug_summary(all_target_totals, all_occ, all_ref_hours, modeling_boundaries)

    cursor.close()
    conn.close()

    print(f"Done. Total rows saved across chunks: {total_rows}")
    print(f"Chunk folder: {OUT_DIR}")
    print(f"Debug folder: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
