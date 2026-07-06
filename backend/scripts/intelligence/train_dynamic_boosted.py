import glob
import json
import math
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


CHUNK_DIR = os.path.join("backend", "data", "processed", "hierarchical_dynamic_chunks")
MODEL_DIR = os.path.join("backend", "data", "models")
DEBUG_DIR = os.path.join("backend", "data", "debug", "hierarchical_dynamic_boosted_trainer")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

RANDOM_STATE = 42
SEQUENCE_CLIP_MAX = 20.0
BASELINE_CLIP_MAX = 20.0
SAMPLE_TRACE_COUNT = 5
PROBABILITY_EPSILON = 1e-6


def load_dataset_from_chunks():
    paths = sorted(glob.glob(os.path.join(CHUNK_DIR, "dynamic_chunk_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No dynamic chunks found in {CHUNK_DIR}")

    seq_parts = []
    scalar_parts = []
    baseline_parts = []
    target_parts = []
    weight_parts = []
    timestamps = []
    cluster_labels = []
    feature_names = None
    modeling_boundaries = None

    for path in paths:
        print(f"Loading {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        rows = data["Y"].shape[0]
        seq_parts.append(data["X_seq"].float())
        scalar_parts.append(data["X_scalar"].float())
        baseline_parts.append(data["X_baseline"].float())
        target_parts.append(data["Y"].float())
        weight_parts.append(data.get("sample_weights", torch.ones(rows)).float())
        timestamps.extend(data["reference_times"])
        cluster_labels.extend(data.get("cluster_labels", [-1] * rows))

        names = list(data.get("scalar_feature_names", []))
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError(f"Scalar feature schema mismatch in {path}; rebuild chunks together")

        chunk_boundaries = data.get("modeling_boundaries")
        if not chunk_boundaries:
            raise ValueError(f"Chunk {path} has no modeling boundaries; rebuild leakage-safe chunks")
        if modeling_boundaries is None:
            modeling_boundaries = dict(chunk_boundaries)
        elif dict(chunk_boundaries) != modeling_boundaries:
            raise ValueError(f"Modeling boundary mismatch in {path}")

    x_seq = torch.nan_to_num(torch.cat(seq_parts), nan=0.0, posinf=0.0, neginf=0.0).numpy()
    x_scalar = torch.nan_to_num(torch.cat(scalar_parts), nan=0.0, posinf=0.0, neginf=0.0).numpy()
    x_baseline = torch.nan_to_num(torch.cat(baseline_parts), nan=0.0, posinf=0.0, neginf=0.0).numpy()
    y = torch.nan_to_num(torch.cat(target_parts), nan=0.0, posinf=0.0, neginf=0.0).numpy()
    sample_weights = torch.nan_to_num(
        torch.cat(weight_parts), nan=1.0, posinf=1.0, neginf=1.0
    ).numpy()

    x_seq = np.log1p(np.clip(x_seq, 0.0, SEQUENCE_CLIP_MAX))
    x_baseline = np.clip(x_baseline, 0.0, BASELINE_CLIP_MAX)
    y = np.clip(y, 0.0, None)
    sample_weights = np.clip(sample_weights, 1e-6, None)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

    if len(timestamps) != y.shape[0]:
        raise ValueError("reference_times and target rows have different lengths")

    seq_flat = x_seq.reshape(x_seq.shape[0], -1)
    features = np.concatenate([seq_flat, x_scalar, x_baseline], axis=1).astype(np.float32)
    feature_labels = (
        [f"seq_crime_tminus_{72 - step:02d}" for step in range(72)]
        + [f"seq_311_tminus_{72 - step:02d}" for step in range(72)]
        + feature_names
        + [f"baseline_block_{idx}" for idx in range(x_baseline.shape[1])]
    )
    return (
        features,
        y.astype(np.float32),
        x_baseline.astype(np.float32),
        sample_weights.astype(np.float64),
        timestamps,
        cluster_labels,
        feature_labels,
        modeling_boundaries,
    )


def chronological_indices(timestamps, modeling_boundaries):
    train_end = np.datetime64(modeling_boundaries["train_end"])
    calibration_start = np.datetime64(modeling_boundaries["calibration_start"])
    calibration_end = np.datetime64(modeling_boundaries["calibration_end"])
    test_start = np.datetime64(modeling_boundaries["test_start"])
    train_idx = np.flatnonzero(timestamps < train_end)
    calibration_idx = np.flatnonzero((timestamps >= calibration_start) & (timestamps < calibration_end))
    test_idx = np.flatnonzero(timestamps >= test_start)
    if train_idx.size == 0 or calibration_idx.size == 0 or test_idx.size == 0:
        raise ValueError("Chronological split produced an empty partition")
    return train_idx, calibration_idx, test_idx


def weighted_mean(values, row_weights):
    weights = np.broadcast_to(row_weights[:, None], values.shape)
    return float(np.sum(values * weights) / np.sum(weights))


def _sigmoid_numpy(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias):
    probability = _sigmoid_numpy(scale * raw_logits + bias)
    weights = row_weights[:, None]
    loss = -(occurrence * np.log(probability + 1e-12) + (1.0 - occurrence) * np.log(1.0 - probability + 1e-12))
    return float(np.sum(loss * weights) / max(np.sum(weights) * occurrence.shape[1], 1.0))


def fit_platt_calibrator(raw_logits, truth, row_weights):
    occurrence = (truth > 0).astype(np.float64)
    scale = 1.0
    bias = 0.0
    before_loss = weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias)

    for _ in range(12):
        probability = _sigmoid_numpy(scale * raw_logits + bias)
        weights = row_weights[:, None]
        residual = weights * (probability - occurrence)
        curvature = weights * probability * (1.0 - probability)
        gradient = np.array([
            np.sum(residual * raw_logits),
            np.sum(residual),
        ], dtype=np.float64)
        hessian = np.array([
            [np.sum(curvature * raw_logits * raw_logits), np.sum(curvature * raw_logits)],
            [np.sum(curvature * raw_logits), np.sum(curvature)],
        ], dtype=np.float64)
        hessian += np.eye(2) * 1e-6
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break

        current_loss = weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias)
        accepted = False
        for fraction in (1.0, 0.5, 0.25, 0.1, 0.05):
            candidate_scale = float(np.clip(scale - fraction * step[0], 0.05, 5.0))
            candidate_bias = float(np.clip(bias - fraction * step[1], -30.0, 30.0))
            candidate_loss = weighted_log_loss(
                raw_logits, occurrence, row_weights, candidate_scale, candidate_bias
            )
            if candidate_loss <= current_loss:
                scale = candidate_scale
                bias = candidate_bias
                accepted = True
                break
        if not accepted or np.max(np.abs(step)) < 1e-5:
            break

    return scale, bias, {
        "scale": scale,
        "bias": bias,
        "weighted_log_loss_before": before_loss,
        "weighted_log_loss_after": weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias),
        "weighted_observed_rate": weighted_mean(occurrence, row_weights),
        "weighted_mean_probability_after": weighted_mean(_sigmoid_numpy(scale * raw_logits + bias), row_weights),
    }


def train_models(features, targets, sample_weights, train_idx, calibration_idx, test_idx):
    x_train = features[train_idx]
    x_cal = features[calibration_idx]
    x_test = features[test_idx]
    y_train = targets[train_idx]
    y_cal = targets[calibration_idx]
    y_test = targets[test_idx]
    w_train = sample_weights[train_idx]
    w_cal = sample_weights[calibration_idx]

    classifiers = []
    regressors = []
    calibration = []
    train_history = []

    raw_logits_cal = np.zeros_like(y_cal, dtype=np.float64)
    raw_logits_test = np.zeros_like(y_test, dtype=np.float64)
    calibrated_probability_test = np.zeros_like(y_test, dtype=np.float64)
    conditional_mean_test = np.ones_like(y_test, dtype=np.float64)
    expected_count_test = np.zeros_like(y_test, dtype=np.float64)

    for block in range(targets.shape[1]):
        print(f"Training boosted hurdle block {block + 1}/{targets.shape[1]}")
        y_occ_train = (y_train[:, block] > 0).astype(np.int32)
        positive_rate = float(np.average(y_occ_train, weights=w_train))
        positive_weight = np.where(y_occ_train > 0, np.clip((1.0 - positive_rate) / max(positive_rate, 1e-6), 1.0, 50.0), 1.0)
        classifier = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_depth=6,
            max_iter=250,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=RANDOM_STATE + block,
            early_stopping=False,
        )
        classifier.fit(x_train, y_occ_train, sample_weight=w_train * positive_weight)
        classifiers.append(classifier)

        prob_cal = classifier.predict_proba(x_cal)[:, 1]
        prob_test = classifier.predict_proba(x_test)[:, 1]
        block_raw_logits_cal = np.log(np.clip(prob_cal, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON) / np.clip(1.0 - prob_cal, PROBABILITY_EPSILON, 1.0))
        block_raw_logits_test = np.log(np.clip(prob_test, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON) / np.clip(1.0 - prob_test, PROBABILITY_EPSILON, 1.0))
        raw_logits_cal[:, block] = block_raw_logits_cal
        raw_logits_test[:, block] = block_raw_logits_test

        scale, bias, diagnostics = fit_platt_calibrator(
            block_raw_logits_cal[:, None],
            y_cal[:, block:block + 1],
            w_cal,
        )
        calibration.append(diagnostics)
        calibrated_probability_test[:, block] = _sigmoid_numpy(scale * block_raw_logits_test + bias)

        positive_mask = y_train[:, block] > 0
        if positive_mask.sum() >= 25:
            regressor = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.05,
                max_depth=6,
                max_iter=250,
                min_samples_leaf=20,
                l2_regularization=0.5,
                random_state=RANDOM_STATE + 100 + block,
                early_stopping=False,
            )
            regressor.fit(
                x_train[positive_mask],
                y_train[positive_mask, block],
                sample_weight=w_train[positive_mask],
            )
            conditional_mean_test[:, block] = np.clip(regressor.predict(x_test), 1.0, None)
            regressors.append(regressor)
            regressor_name = "HistGradientBoostingRegressor"
        else:
            fallback_mean = float(np.average(
                y_train[positive_mask, block] if positive_mask.any() else np.array([1.0]),
                weights=w_train[positive_mask] if positive_mask.any() else np.array([1.0]),
            ))
            conditional_mean_test[:, block] = np.full(y_test.shape[0], max(fallback_mean, 1.0))
            regressors.append({"kind": "constant_mean", "value": max(fallback_mean, 1.0)})
            regressor_name = "constant_mean"

        expected_count_test[:, block] = calibrated_probability_test[:, block] * conditional_mean_test[:, block]
        train_history.append({
            "block": block,
            "train_positive_rate": positive_rate,
            "classifier": "HistGradientBoostingClassifier",
            "regressor": regressor_name,
            "positive_train_rows": int(positive_mask.sum()),
        })

    return {
        "classifiers": classifiers,
        "regressors": regressors,
        "calibration": calibration,
        "train_history": train_history,
        "raw_logits_test": raw_logits_test.astype(np.float32),
        "probabilities_test": calibrated_probability_test.astype(np.float32),
        "conditional_test": conditional_mean_test.astype(np.float32),
        "expected_test": expected_count_test.astype(np.float32),
    }


def save_evaluation(
    probabilities,
    conditional_means,
    expected,
    truth,
    baseline,
    row_weights,
    calibration_diagnostics,
    test_timestamps,
    test_clusters,
):
    occurrence = (truth > 0).astype(np.float32)
    observed_prevalence = weighted_mean(occurrence, row_weights)
    model_mae = weighted_mean(np.abs(expected - truth), row_weights)
    baseline_mae = weighted_mean(np.abs(baseline - truth), row_weights)
    brier = weighted_mean((probabilities - occurrence) ** 2, row_weights)
    constant_brier = weighted_mean((observed_prevalence - occurrence) ** 2, row_weights)

    try:
        from sklearn.metrics import average_precision_score

        flat_weights = np.repeat(row_weights, truth.shape[1])
        average_precision = float(
            average_precision_score(
                occurrence.reshape(-1),
                probabilities.reshape(-1),
                sample_weight=flat_weights,
            )
        )
    except ValueError:
        average_precision = None

    weighted_true_total = float(np.average(truth.sum(axis=1), weights=row_weights))
    weighted_pred_total = float(np.average(expected.sum(axis=1), weights=row_weights))
    weighted_baseline_total = float(np.average(baseline.sum(axis=1), weights=row_weights))

    flat_probability = probabilities.reshape(-1)
    flat_occurrence = occurrence.reshape(-1)
    flat_weights = np.repeat(row_weights, truth.shape[1])
    top_count = max(int(flat_probability.size * 0.01), 1)
    top_idx = np.argpartition(flat_probability, -top_count)[-top_count:]
    weighted_positive = flat_occurrence * flat_weights
    recall_top_1pct = float(weighted_positive[top_idx].sum() / max(weighted_positive.sum(), 1.0))
    precision_top_1pct = float(weighted_positive[top_idx].sum() / max(flat_weights[top_idx].sum(), 1.0))

    quantile_edges = np.unique(np.quantile(flat_probability, np.linspace(0.0, 1.0, 11)))
    calibration_bins = []
    for bin_index in range(max(len(quantile_edges) - 1, 0)):
        lower = quantile_edges[bin_index]
        upper = quantile_edges[bin_index + 1]
        if bin_index == len(quantile_edges) - 2:
            mask = (flat_probability >= lower) & (flat_probability <= upper)
        else:
            mask = (flat_probability >= lower) & (flat_probability < upper)
        if not np.any(mask):
            continue
        calibration_bins.append({
            "mean_probability": float(np.average(flat_probability[mask], weights=flat_weights[mask])),
            "observed_rate": float(np.average(flat_occurrence[mask], weights=flat_weights[mask])),
            "sampled_slots": int(mask.sum()),
        })

    summary = {
        "overall": {
            "model_expected_count_mae": model_mae,
            "seasonal_baseline_mae": baseline_mae,
            "model_better_than_baseline": model_mae < baseline_mae,
            "occurrence_average_precision": average_precision,
            "average_precision_lift_over_prevalence": (
                average_precision / observed_prevalence
                if average_precision is not None and observed_prevalence > 0
                else None
            ),
            "observed_block_prevalence": observed_prevalence,
            "occurrence_brier_score": brier,
            "constant_prevalence_brier_score": constant_brier,
            "recall_in_top_1pct_block_risks": recall_top_1pct,
            "precision_in_top_1pct_block_risks": precision_top_1pct,
            "avg_true_total_24h": weighted_true_total,
            "avg_pred_total_24h": weighted_pred_total,
            "avg_baseline_total_24h": weighted_baseline_total,
        },
        "forecast_blocks": {
            "model_mae": [
                weighted_mean(np.abs(expected[:, h:h + 1] - truth[:, h:h + 1]), row_weights)
                for h in range(truth.shape[1])
            ],
            "mean_occurrence_probability": [
                float(np.average(probabilities[:, h], weights=row_weights))
                for h in range(truth.shape[1])
            ],
            "observed_occurrence_rate": [
                float(np.average(occurrence[:, h], weights=row_weights))
                for h in range(truth.shape[1])
            ],
        },
        "probability_calibration": calibration_diagnostics,
        "probability_calibration_bins": calibration_bins,
    }

    temporal_backtests = []
    ordered_times = np.asarray(test_timestamps)
    for fold, indices in enumerate(np.array_split(np.argsort(ordered_times), 3), start=1):
        if len(indices) == 0:
            continue
        fold_model = weighted_mean(np.abs(expected[indices] - truth[indices]), row_weights[indices])
        fold_baseline = weighted_mean(np.abs(baseline[indices] - truth[indices]), row_weights[indices])
        temporal_backtests.append({
            "fold": fold,
            "start": str(ordered_times[indices].min()),
            "end": str(ordered_times[indices].max()),
            "model_mae": fold_model,
            "seasonal_baseline_mae": fold_baseline,
            "mae_improvement": (fold_baseline - fold_model) / max(fold_baseline, 1e-12),
        })
    summary["rolling_temporal_backtests"] = temporal_backtests

    cluster_calibration = []
    for cluster in np.unique(test_clusters):
        indices = np.flatnonzero(test_clusters == cluster)
        if len(indices) < 50:
            continue
        predicted = weighted_mean(probabilities[indices], row_weights[indices])
        observed = weighted_mean(occurrence[indices], row_weights[indices])
        cluster_calibration.append({
            "cluster": int(cluster),
            "rows": int(len(indices)),
            "mean_probability": predicted,
            "observed_rate": observed,
            "absolute_calibration_error": abs(predicted - observed),
        })
    summary["cluster_calibration"] = cluster_calibration

    total_residual = np.abs(expected.sum(axis=1) - truth.sum(axis=1))
    summary["empirical_uncertainty"] = {
        "absolute_total_error_p80": float(np.quantile(total_residual, 0.80)),
        "absolute_total_error_p95": float(np.quantile(total_residual, 0.95)),
        "coverage_basis": "untouched temporal test set",
    }

    with open(os.path.join(DEBUG_DIR, "metrics_summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    plt.figure(figsize=(10, 5))
    plt.plot(
        np.arange(1, truth.shape[1] + 1),
        summary["forecast_blocks"]["mean_occurrence_probability"],
        marker="o",
        label="Predicted",
    )
    plt.plot(
        np.arange(1, truth.shape[1] + 1),
        summary["forecast_blocks"]["observed_occurrence_rate"],
        marker="o",
        label="Observed",
    )
    plt.xlabel("Forecast 3-hour block")
    plt.ylabel("Crime occurrence probability")
    plt.title("Boosted model probability calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "hourly_probability_calibration.png"), dpi=180)
    plt.close()

    samples = []
    row_risk = expected.sum(axis=1)
    true_total = truth.sum(axis=1)
    selected = []

    def add_sample(reason, index):
        index = int(index)
        if index not in {item[1] for item in selected} and len(selected) < SAMPLE_TRACE_COUNT:
            selected.append((reason, index))

    add_sample("highest_predicted_risk", np.argmax(row_risk))
    positive_rows = np.flatnonzero(true_total > 0)
    if positive_rows.size:
        positive_order = positive_rows[np.argsort(row_risk[positive_rows])]
        add_sample("highest_scoring_positive", positive_order[-1])
        add_sample("typical_positive", positive_order[len(positive_order) // 2])
        add_sample("lowest_scoring_positive", positive_order[0])
    zero_rows = np.flatnonzero(true_total == 0)
    if zero_rows.size:
        add_sample("highest_scoring_false_alarm", zero_rows[np.argmax(row_risk[zero_rows])])

    for reason, index in selected:
        samples.append({
            "test_row_index": index,
            "selection_reason": reason,
            "occurrence_probability_next24": probabilities[index].tolist(),
            "conditional_count_next24": conditional_means[index].tolist(),
            "expected_count_next24": expected[index].tolist(),
            "true_count_next24": truth[index].tolist(),
            "expected_total_24h": float(expected[index].sum()),
            "true_total_24h": float(truth[index].sum()),
            "probability_at_least_one_crime_24h_assuming_conditional_independence":
                float(1.0 - np.prod(1.0 - probabilities[index])),
        })
    with open(os.path.join(DEBUG_DIR, "sample_predictions.json"), "w", encoding="utf-8") as file:
        json.dump(samples, file, indent=2)


def main():
    (
        features,
        targets,
        baseline,
        sample_weights,
        timestamps,
        cluster_labels,
        feature_labels,
        modeling_boundaries,
    ) = load_dataset_from_chunks()
    train_idx, calibration_idx, test_idx = chronological_indices(timestamps, modeling_boundaries)
    trained = train_models(features, targets, sample_weights, train_idx, calibration_idx, test_idx)
    save_evaluation(
        trained["probabilities_test"],
        trained["conditional_test"],
        trained["expected_test"],
        targets[test_idx],
        baseline[test_idx],
        sample_weights[test_idx],
        trained["calibration"],
        timestamps[test_idx],
        cluster_labels[test_idx],
    )
    with open(os.path.join(DEBUG_DIR, "training_history.json"), "w", encoding="utf-8") as file:
        json.dump(trained["train_history"], file, indent=2)

    artifact_path = os.path.join(MODEL_DIR, "hierarchical_forecaster_3h.boosted.candidate.joblib")
    metrics_path = os.path.join(DEBUG_DIR, "metrics_summary.json")
    with open(metrics_path, encoding="utf-8") as file:
        uncertainty = json.load(file)["empirical_uncertainty"]

    joblib.dump({
        "model_type": "boosted_hurdle",
        "classifiers": trained["classifiers"],
        "regressors": trained["regressors"],
        "calibration": trained["calibration"],
        "feature_labels": feature_labels,
        "output_steps": int(targets.shape[1]),
        "forecast_block_hours": 3,
        "super_cell_factor": 3,
        "uncertainty": uncertainty,
    }, artifact_path)
    print(f"Saved boosted 500 m / 3-hour candidate to {artifact_path}")
    print(f"Diagnostics written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
