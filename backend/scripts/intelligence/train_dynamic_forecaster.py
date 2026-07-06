"""Train the dynamic next-24-hour crime-pressure forecaster.

The trainer learns a forecast on top of the static baseline using leakage-safe
historical sequences, neighborhood context, and weighted evaluation logic that
focuses on sparse-but-important positive periods.
"""

import glob
import json
import math
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from config.modeling import FORECAST_BLOCK_HOURS


# =========================================================
# CONFIG
# =========================================================

CHUNK_DIR = os.path.join("backend", "data", "processed", "hierarchical_dynamic_chunks")
MODEL_DIR = os.path.join("backend", "data", "models")
DEBUG_DIR = os.path.join("backend", "data", "debug", "hierarchical_dynamic_trainer")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

RANDOM_STATE = 42
BATCH_SIZE = 256
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 4
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 1e-5

SEQ_HIDDEN_DIM = 64
SCALAR_HIDDEN_DIM = 128
BASELINE_HIDDEN_DIM = 32
FUSION_HIDDEN_DIM = 128
DROPOUT = 0.2
MAX_LOGIT_RESIDUAL = 3.0
MAX_COUNT_LOG_RESIDUAL = 1.15
SURGE_COUNT_THRESHOLD = 1.50
SURGE_ROW_TOTAL_THRESHOLD = 2.50
SURGE_BCE_LOSS_WEIGHT = 0.22
SURGE_SIZE_LOSS_WEIGHT = 0.26

FOCAL_GAMMA = 2.0
MAX_POS_WEIGHT = 30.0
COUNT_LOSS_WEIGHT = 0.38
BURST_COUNT_LOSS_WEIGHT = 0.22
TOTAL_24H_LOSS_WEIGHT = 0.12
HOTSPOT_ROW_LOSS_WEIGHT = 0.28
DEVIATION_SLOT_LOSS_WEIGHT = 0.24
DEVIATION_ROW_LOSS_WEIGHT = 0.16
FALSE_SURGE_SLOT_LOSS_WEIGHT = 0.34
FALSE_SURGE_ROW_LOSS_WEIGHT = 0.22
HOTSPOT_POSITIVE_EMPHASIS = 1.25
HOTSPOT_SEVERITY_EMPHASIS = 0.90
BASELINE_CLIP_MAX = 20.0
SAMPLE_TRACE_COUNT = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# =========================================================
# DATA
# =========================================================

class DynamicCrimeDataset(Dataset):
    def __init__(self, x_seq, x_scalar, x_baseline, y, sample_weights):
        self.x_seq = x_seq
        self.x_scalar = x_scalar
        self.x_baseline = x_baseline
        self.y = y
        self.sample_weights = sample_weights

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (
            self.x_seq[idx],
            self.x_scalar[idx],
            self.x_baseline[idx],
            self.y[idx],
            self.sample_weights[idx],
        )


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
        data = torch.load(path, map_location="cpu")
        rows = data["Y"].shape[0]
        seq_parts.append(data["X_seq"].float())
        scalar_parts.append(data["X_scalar"].float())
        baseline_parts.append(data["X_baseline"].float())
        target_parts.append(data["Y"].float())
        weight_parts.append(data.get("sample_weights", torch.ones(rows)).float())
        timestamps.extend(data["reference_times"])
        cluster_labels.extend(data.get("cluster_labels", [-1] * rows))

        names = data.get("scalar_feature_names", [])
        if feature_names is None:
            feature_names = list(names)
        elif list(names) != feature_names:
            raise ValueError(f"Scalar feature schema mismatch in {path}; rebuild all chunks together")

        chunk_boundaries = data.get("modeling_boundaries")
        if not chunk_boundaries:
            raise ValueError(f"Chunk {path} has no modeling boundaries; rebuild leakage-safe chunks")
        if modeling_boundaries is None:
            modeling_boundaries = dict(chunk_boundaries)
        elif dict(chunk_boundaries) != modeling_boundaries:
            raise ValueError(f"Modeling boundary mismatch in {path}")

    x_seq = torch.nan_to_num(torch.cat(seq_parts), nan=0.0, posinf=0.0, neginf=0.0)
    x_scalar = torch.nan_to_num(torch.cat(scalar_parts), nan=0.0, posinf=0.0, neginf=0.0)
    x_baseline = torch.nan_to_num(torch.cat(baseline_parts), nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(torch.cat(target_parts), nan=0.0, posinf=0.0, neginf=0.0)
    sample_weights = torch.nan_to_num(torch.cat(weight_parts), nan=1.0, posinf=1.0, neginf=1.0)

    # Count inputs benefit from compression; signed GNN embeddings remain untouched.
    x_seq = torch.log1p(torch.clamp(x_seq, min=0.0))
    x_baseline = torch.clamp(x_baseline, min=0.0, max=BASELINE_CLIP_MAX)
    y = torch.clamp(y, min=0.0)
    sample_weights = torch.clamp(sample_weights, min=1e-6)
    timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
    cluster_labels = np.asarray(cluster_labels, dtype=np.int32)

    if len(timestamps) != y.shape[0]:
        raise ValueError("reference_times and target rows have different lengths")

    print(f"Rows: {y.shape[0]:,}")
    print(f"Sequence/scalar/baseline/target: {tuple(x_seq.shape)} / {tuple(x_scalar.shape)} / "
          f"{tuple(x_baseline.shape)} / {tuple(y.shape)}")
    print(
        f"Sampled {FORECAST_BLOCK_HOURS}-hour-block positive rate: "
        f"{(y > 0).float().mean().item():.6f}"
    )
    return (
        x_seq, x_scalar, x_baseline, y, sample_weights,
        timestamps, cluster_labels, feature_names, modeling_boundaries,
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

    print(f"Train:       reference time < {train_end}")
    print(f"Calibration: {calibration_start} <= reference time < {calibration_end}")
    print(f"Test:        reference time >= {test_start}")
    print(
        f"Rows: train {train_idx.size:,}; calibration {calibration_idx.size:,}; "
        f"test {test_idx.size:,}"
    )
    return train_idx, calibration_idx, test_idx


def standardize_from_train(x_scalar, train_idx):
    train_values = x_scalar[torch.as_tensor(train_idx, dtype=torch.long)]
    mean = train_values.mean(dim=0, keepdim=True)
    std = train_values.std(dim=0, keepdim=True)
    std = torch.where(torch.isfinite(std) & (std >= 1e-6), std, torch.ones_like(std))
    mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    scaled = torch.nan_to_num((x_scalar - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return scaled, mean, std


def build_train_sampler(dataset, train_idx):
    idx = torch.as_tensor(train_idx, dtype=torch.long)
    target = dataset.y[idx]
    baseline = dataset.x_baseline[idx]
    row_total = target.sum(dim=1)
    burst_total = torch.clamp(row_total - 1.0, min=0.0, max=10.0)
    slot_burst = torch.clamp(target.max(dim=1).values - 1.0, min=0.0, max=8.0)
    deviation_total = torch.clamp(
        torch.abs((target - baseline).sum(dim=1)), min=0.0, max=8.0
    )
    positive_row = (row_total > 0).float()
    sampler_weights = (
        1.0
        + 1.5 * positive_row
        + 2.5 * burst_total
        + 2.0 * slot_burst
        + 1.8 * deviation_total
    )
    sampler_weights = torch.nan_to_num(sampler_weights, nan=1.0, posinf=1.0, neginf=1.0)
    sampler_weights = sampler_weights.clamp_min(1e-3)
    print(
        "Train sampler mean weight:"
        f" {sampler_weights.mean().item():.3f}"
        f" | positive-row mean: {sampler_weights[positive_row > 0].mean().item() if torch.any(positive_row > 0) else 0.0:.3f}"
        f" | zero-row mean: {sampler_weights[positive_row == 0].mean().item() if torch.any(positive_row == 0) else 0.0:.3f}"
    )
    return WeightedRandomSampler(
        weights=sampler_weights.double(),
        num_samples=len(train_idx),
        replacement=True,
    )


def make_loaders(dataset, train_idx, calibration_idx, test_idx):
    train_sampler = build_train_sampler(dataset, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    calibration_loader = DataLoader(
        Subset(dataset, calibration_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, calibration_loader, test_loader


def effective_positive_weight(y, sample_weights, train_idx):
    idx = torch.as_tensor(train_idx, dtype=torch.long)
    occurrence = (y[idx] > 0).float()
    weights = sample_weights[idx, None]
    positive = (weights * occurrence).sum().item()
    negative = (weights * (1.0 - occurrence)).sum().item()
    prevalence = positive / max(positive + negative, 1.0)
    # Square-root weighting is less extreme than the raw 300:1 imbalance.
    pos_weight = min(math.sqrt(negative / max(positive, 1.0)), MAX_POS_WEIGHT)
    print(
        f"Sampling-corrected {FORECAST_BLOCK_HOURS}-hour-block prevalence: "
        f"{prevalence:.6f}"
    )
    print(f"Occurrence positive weight: {pos_weight:.3f}")
    return float(pos_weight), float(prevalence)


# =========================================================
# MODEL AND LOSS
# =========================================================

class TwoHeadDynamicForecaster(nn.Module):
    """Occurrence + surge-aware conditional forecasting for configurable blocks."""

    def __init__(
        self,
        seq_input_dim,
        scalar_input_dim,
        baseline_input_dim,
        occurrence_bias=0.0,
        output_steps=8,
        seq_hidden_dim=SEQ_HIDDEN_DIM,
        scalar_hidden_dim=SCALAR_HIDDEN_DIM,
        baseline_hidden_dim=BASELINE_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(seq_input_dim, seq_hidden_dim, batch_first=True)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(scalar_hidden_dim, scalar_hidden_dim),
            nn.ReLU(),
        )
        self.baseline_mlp = nn.Sequential(
            nn.Linear(baseline_input_dim, baseline_hidden_dim),
            nn.ReLU(),
            nn.Linear(baseline_hidden_dim, baseline_hidden_dim),
            nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(seq_hidden_dim + scalar_hidden_dim + baseline_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, 64),
            nn.ReLU(),
        )
        self.output_steps = output_steps
        self.occurrence_head = nn.Linear(64, output_steps)
        self.base_count_head = nn.Linear(64, output_steps)
        self.surge_head = nn.Linear(64, output_steps)
        self.surge_size_head = nn.Linear(64, output_steps)
        nn.init.constant_(self.occurrence_head.bias, 0.0)
        nn.init.constant_(self.base_count_head.bias, 0.0)
        nn.init.constant_(self.surge_head.bias, -2.0)
        nn.init.constant_(self.surge_size_head.bias, -1.0)

    def forward(
        self,
        x_seq,
        x_scalar,
        x_baseline,
        calibration_scale=1.0,
        calibration_bias=0.0,
        return_aux=False,
    ):
        _, hidden = self.gru(x_seq)
        sequence_embedding = hidden[-1]
        scalar_embedding = self.scalar_mlp(x_scalar)
        baseline_input = torch.clamp(x_baseline, min=0.0, max=BASELINE_CLIP_MAX)
        baseline_embedding = self.baseline_mlp(torch.log1p(baseline_input))
        shared = self.shared(torch.cat([sequence_embedding, scalar_embedding, baseline_embedding], dim=1))
        baseline_probability = 1.0 - torch.exp(-baseline_input.clamp_min(1e-6))
        baseline_probability = baseline_probability.clamp(1e-4, 1.0 - 1e-4)
        baseline_logit = torch.logit(baseline_probability)
        occurrence_delta = MAX_LOGIT_RESIDUAL * torch.tanh(self.occurrence_head(shared))
        occurrence_logits_raw = baseline_logit + occurrence_delta
        occurrence_logits = calibration_scale * occurrence_logits_raw + calibration_bias
        occurrence_probability = torch.sigmoid(occurrence_logits)
        base_conditional = torch.clamp(
            1.0 + F.softplus(self.base_count_head(shared)),
            min=1.0,
            max=3.0,
        )
        surge_probability = torch.sigmoid(self.surge_head(shared))
        surge_size = torch.clamp(
            F.softplus(self.surge_size_head(shared)),
            min=0.0,
            max=8.0,
        )
        conditional_mean = torch.clamp(
            base_conditional + surge_probability * surge_size,
            min=1.0,
            max=10.0,
        )
        surge_expected = occurrence_probability * surge_probability * surge_size
        expected_count = occurrence_probability * base_conditional + surge_expected
        if return_aux:
            return (
                occurrence_logits_raw,
                occurrence_probability,
                conditional_mean,
                expected_count,
                {
                    "base_conditional": base_conditional,
                    "surge_probability": surge_probability,
                    "surge_size": surge_size,
                    "surge_expected": surge_expected,
                },
            )
        return occurrence_logits_raw, occurrence_probability, conditional_mean, expected_count


def compute_loss(outputs, target, baseline, sample_weights, pos_weight):
    logits_raw, calibrated_probability, conditional_mean, expected_count, aux = outputs
    base_conditional = aux["base_conditional"]
    surge_probability = aux["surge_probability"]
    surge_size = aux["surge_size"]
    surge_expected = aux["surge_expected"]
    occurrence = (target > 0).float()
    row_weights = sample_weights[:, None]
    normalized_weights = row_weights / row_weights.mean().clamp_min(1e-6)
    row_total = target.sum(dim=1, keepdim=True)
    severity = torch.clamp(torch.log1p(row_total), min=0.0, max=2.5)
    target_deviation = target - baseline
    expected_deviation = expected_count - baseline
    positive_residual = torch.clamp(target_deviation, min=0.0)
    row_positive_residual = positive_residual.sum(dim=1, keepdim=True)
    surge_target = (
        (positive_residual >= SURGE_COUNT_THRESHOLD).float()
        * (row_positive_residual >= SURGE_ROW_TOTAL_THRESHOLD).float()
    )
    deviation_magnitude = torch.abs(target_deviation)
    burst_strength = torch.clamp(target - 1.0, min=0.0, max=8.0)
    false_surge_amount = torch.clamp(surge_expected - positive_residual, min=0.0)
    quiet_slot = (target <= 0).float()
    quiet_row = (row_total[:, 0] <= 0).float()
    hotspot_slot_weight = 1.0 + occurrence * (
        HOTSPOT_POSITIVE_EMPHASIS + HOTSPOT_SEVERITY_EMPHASIS * severity
    )
    deviation_slot_weight = (
        1.0
        + 1.35 * torch.clamp(deviation_magnitude, min=0.0, max=3.0)
        + 0.60 * torch.clamp(burst_strength, min=0.0, max=4.0)
    )
    deviation_row_signal = target_deviation.sum(dim=1, keepdim=True).abs()
    deviation_row_weight = normalized_weights[:, 0] * (
        1.0
        + 1.10 * torch.clamp(deviation_row_signal[:, 0], min=0.0, max=4.0)
        + 0.55 * severity[:, 0]
    )
    false_surge_slot_weight = normalized_weights * (
        1.0
        + 2.2 * quiet_slot
        + 1.2 * torch.clamp(expected_deviation, min=0.0, max=6.0)
    )
    false_surge_row_weight = normalized_weights[:, 0] * (
        1.0
        + 2.8 * quiet_row
        + 1.5 * torch.clamp(expected_deviation.sum(dim=1), min=0.0, max=8.0)
    )

    bce = F.binary_cross_entropy_with_logits(
        logits_raw,
        occurrence,
        pos_weight=torch.tensor(pos_weight, device=target.device),
        reduction="none",
    )
    focal_factor = torch.where(
        occurrence > 0,
        (1.0 - calibrated_probability).pow(FOCAL_GAMMA),
        calibrated_probability.pow(FOCAL_GAMMA),
    )
    occurrence_weights = normalized_weights * hotspot_slot_weight
    occurrence_denominator = occurrence_weights.expand_as(bce).sum().clamp_min(1.0)
    occurrence_loss = (bce * focal_factor * occurrence_weights).sum() / occurrence_denominator

    positive_mask = occurrence * normalized_weights * (
        1.0 + 0.6 * severity + 0.75 * torch.clamp(burst_strength, min=0.0, max=4.0)
    )
    base_target = torch.clamp(target, min=1.0, max=2.5)
    poisson = (
        base_conditional
        - base_target * torch.log(base_conditional.clamp_min(1e-6))
        + torch.lgamma(base_target + 1.0)
    )
    count_loss = (poisson * positive_mask).sum() / positive_mask.sum().clamp_min(1.0)
    burst_loss = (
        F.smooth_l1_loss(
            torch.log1p(base_conditional),
            torch.log1p(torch.clamp(target, min=1.0)),
            reduction="none",
        )
        * positive_mask
        * (1.0 + torch.clamp(burst_strength, min=0.0, max=5.0))
    ).sum() / positive_mask.sum().clamp_min(1.0)
    surge_pos_weight = min(pos_weight * 2.2, MAX_POS_WEIGHT)
    surge_logits = torch.logit(surge_probability.clamp(1e-6, 1.0 - 1e-6))
    surge_bce = F.binary_cross_entropy_with_logits(
        surge_logits,
        surge_target,
        pos_weight=torch.tensor(surge_pos_weight, device=target.device),
        reduction="none",
    )
    surge_weights = normalized_weights * (
        1.0
        + 2.0 * surge_target
        + 1.1 * torch.clamp(positive_residual, min=0.0, max=6.0)
    )
    surge_loss = (surge_bce * surge_weights).sum() / surge_weights.sum().clamp_min(1.0)
    surge_size_loss = (
        F.smooth_l1_loss(
            torch.log1p(surge_size),
            torch.log1p(positive_residual),
            reduction="none",
        )
        * surge_target
        * surge_weights
    ).sum() / (surge_target * surge_weights).sum().clamp_min(1.0)

    total_error = F.smooth_l1_loss(expected_count.sum(dim=1), target.sum(dim=1), reduction="none")
    total_loss_24h = (total_error * normalized_weights[:, 0]).mean()
    hotspot_row_weight = normalized_weights[:, 0] * (
        1.0 + 0.75 * (row_total[:, 0] > 0).float() + 0.50 * severity[:, 0]
    )
    hotspot_total_loss = (total_error * hotspot_row_weight).sum() / hotspot_row_weight.sum().clamp_min(1.0)
    deviation_slot_loss = (
        F.smooth_l1_loss(expected_deviation, target_deviation, reduction="none")
        * normalized_weights
        * deviation_slot_weight
    ).sum() / (normalized_weights * deviation_slot_weight).sum().clamp_min(1.0)
    deviation_total_loss = (
        F.smooth_l1_loss(
            expected_deviation.sum(dim=1),
            target_deviation.sum(dim=1),
            reduction="none",
        )
        * deviation_row_weight
    ).sum() / deviation_row_weight.sum().clamp_min(1.0)
    false_surge_slot_loss = (
        false_surge_amount.pow(2) * false_surge_slot_weight
    ).sum() / false_surge_slot_weight.sum().clamp_min(1.0)
    false_surge_row_loss = (
        torch.clamp(
            expected_deviation.sum(dim=1) - target_deviation.sum(dim=1),
            min=0.0,
        ).pow(2)
        * false_surge_row_weight
    ).sum() / false_surge_row_weight.sum().clamp_min(1.0)
    total = (
        occurrence_loss
        + COUNT_LOSS_WEIGHT * count_loss
        + BURST_COUNT_LOSS_WEIGHT * burst_loss
        + SURGE_BCE_LOSS_WEIGHT * surge_loss
        + SURGE_SIZE_LOSS_WEIGHT * surge_size_loss
        + TOTAL_24H_LOSS_WEIGHT * total_loss_24h
        + HOTSPOT_ROW_LOSS_WEIGHT * hotspot_total_loss
        + DEVIATION_SLOT_LOSS_WEIGHT * deviation_slot_loss
        + DEVIATION_ROW_LOSS_WEIGHT * deviation_total_loss
        + FALSE_SURGE_SLOT_LOSS_WEIGHT * false_surge_slot_loss
        + FALSE_SURGE_ROW_LOSS_WEIGHT * false_surge_row_loss
    )
    parts = {
        "occurrence": occurrence_loss.detach().item(),
        "count": count_loss.detach().item(),
        "burst_count": burst_loss.detach().item(),
        "surge_bce": surge_loss.detach().item(),
        "surge_size": surge_size_loss.detach().item(),
        "total_24h": total_loss_24h.detach().item(),
        "hotspot_24h": hotspot_total_loss.detach().item(),
        "deviation_slot": deviation_slot_loss.detach().item(),
        "deviation_24h": deviation_total_loss.detach().item(),
        "false_surge_slot": false_surge_slot_loss.detach().item(),
        "false_surge_24h": false_surge_row_loss.detach().item(),
    }
    return total, parts


def run_epoch(model, loader, pos_weight, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    training_calibration_bias = -math.log(pos_weight)
    totals = {
        "loss": 0.0,
        "occurrence": 0.0,
        "count": 0.0,
        "burst_count": 0.0,
        "surge_bce": 0.0,
        "surge_size": 0.0,
        "total_24h": 0.0,
        "hotspot_24h": 0.0,
        "deviation_slot": 0.0,
        "deviation_24h": 0.0,
        "false_surge_slot": 0.0,
        "false_surge_24h": 0.0,
    }
    batches = 0

    for x_seq, x_scalar, x_baseline, target, sample_weights in loader:
        x_seq = x_seq.to(DEVICE, non_blocking=True)
        x_scalar = x_scalar.to(DEVICE, non_blocking=True)
        x_baseline = x_baseline.to(DEVICE, non_blocking=True)
        target = target.to(DEVICE, non_blocking=True)
        sample_weights = sample_weights.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(
                x_seq,
                x_scalar,
                x_baseline,
                calibration_scale=1.0,
                calibration_bias=training_calibration_bias,
                return_aux=True,
            )
            loss, parts = compute_loss(outputs, target, x_baseline, sample_weights, pos_weight)
            if not torch.isfinite(loss):
                continue
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        totals["loss"] += loss.item()
        for name, value in parts.items():
            totals[name] += value
        batches += 1

    return {name: value / max(batches, 1) for name, value in totals.items()}


# =========================================================
# EVALUATION
# =========================================================

def collect_outputs(model, loader, calibration_scale=1.0, calibration_bias=0.0):
    model.eval()
    probabilities = []
    conditional_means = []
    expected_counts = []
    targets = []
    baselines = []
    weights = []
    raw_logits = []

    with torch.no_grad():
        for x_seq, x_scalar, x_baseline, target, sample_weights in loader:
            outputs = model(
                x_seq.to(DEVICE),
                x_scalar.to(DEVICE),
                x_baseline.to(DEVICE),
                calibration_scale=calibration_scale,
                calibration_bias=calibration_bias,
            )
            logits_raw, probability, conditional_mean, expected_count = outputs
            raw_logits.append(logits_raw.cpu())
            probabilities.append(probability.cpu())
            conditional_means.append(conditional_mean.cpu())
            expected_counts.append(expected_count.cpu())
            targets.append(target)
            baselines.append(x_baseline)
            weights.append(sample_weights)

    return tuple(torch.cat(parts).numpy() for parts in (
        raw_logits, probabilities, conditional_means, expected_counts, targets, baselines, weights
    ))


def _sigmoid_numpy(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias, chunk_rows=100_000):
    total = 0.0
    denominator = 0.0
    for start in range(0, raw_logits.shape[0], chunk_rows):
        stop = min(start + chunk_rows, raw_logits.shape[0])
        probability = _sigmoid_numpy(scale * raw_logits[start:stop] + bias)
        target = occurrence[start:stop]
        weights = row_weights[start:stop, None]
        loss = -(target * np.log(probability + 1e-12) + (1.0 - target) * np.log(1.0 - probability + 1e-12))
        total += float(np.sum(loss * weights))
        denominator += float(np.sum(weights) * target.shape[1])
    return total / max(denominator, 1.0)


def fit_platt_calibrator(raw_logits, truth, row_weights, initial_bias):
    """Fit two scalar Platt parameters on the calibration period only."""
    occurrence = (truth > 0).astype(np.float64)
    raw_logits = raw_logits.astype(np.float64, copy=False)
    row_weights = row_weights.astype(np.float64, copy=False)
    scale = 1.0
    bias = float(initial_bias)
    before_loss = weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias)

    for _ in range(12):
        gradient = np.zeros(2, dtype=np.float64)
        hessian = np.zeros((2, 2), dtype=np.float64)
        for start in range(0, raw_logits.shape[0], 100_000):
            stop = min(start + 100_000, raw_logits.shape[0])
            x = raw_logits[start:stop]
            target = occurrence[start:stop]
            weights = row_weights[start:stop, None]
            probability = _sigmoid_numpy(scale * x + bias)
            residual = weights * (probability - target)
            curvature = weights * probability * (1.0 - probability)
            gradient[0] += np.sum(residual * x)
            gradient[1] += np.sum(residual)
            hessian[0, 0] += np.sum(curvature * x * x)
            hessian[0, 1] += np.sum(curvature * x)
            hessian[1, 1] += np.sum(curvature)

        hessian[1, 0] = hessian[0, 1]
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
                scale, bias = candidate_scale, candidate_bias
                accepted = True
                break
        if not accepted or np.max(np.abs(step)) < 1e-5:
            break

    after_loss = weighted_log_loss(raw_logits, occurrence, row_weights, scale, bias)
    calibrated = _sigmoid_numpy(scale * raw_logits + bias)
    diagnostics = {
        "scale": scale,
        "bias": bias,
        "weighted_log_loss_before": before_loss,
        "weighted_log_loss_after": after_loss,
        "weighted_observed_rate": weighted_mean(occurrence, row_weights),
        "weighted_mean_probability_after": weighted_mean(calibrated, row_weights),
    }
    print(
        f"Probability calibration: scale={scale:.5f}, bias={bias:.5f}, "
        f"log-loss {before_loss:.6f} -> {after_loss:.6f}"
    )
    return scale, bias, diagnostics


def weighted_mean(values, row_weights):
    weights = np.broadcast_to(row_weights[:, None], values.shape)
    return float(np.sum(values * weights) / np.sum(weights))


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
        average_precision = float(average_precision_score(
            occurrence.reshape(-1), probabilities.reshape(-1), sample_weight=flat_weights
        ))
    except (ImportError, ValueError):
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
    precision_top_1pct = float(
        weighted_positive[top_idx].sum() / max(flat_weights[top_idx].sum(), 1.0)
    )

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
                if average_precision is not None and observed_prevalence > 0 else None
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
            "model_mae": [weighted_mean(np.abs(expected[:, h:h + 1] - truth[:, h:h + 1]), row_weights)
                          for h in range(truth.shape[1])],
            "mean_occurrence_probability": [float(np.average(probabilities[:, h], weights=row_weights))
                                            for h in range(truth.shape[1])],
            "observed_occurrence_rate": [float(np.average(occurrence[:, h], weights=row_weights))
                                         for h in range(truth.shape[1])],
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
        cluster_probability = probabilities[indices]
        cluster_occurrence = occurrence[indices]
        predicted = weighted_mean(cluster_probability, row_weights[indices])
        observed = weighted_mean(cluster_occurrence, row_weights[indices])
        cluster_calibration.append({
            "cluster": int(cluster), "rows": int(len(indices)),
            "mean_probability": predicted, "observed_rate": observed,
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

    x = np.arange(1, truth.shape[1] + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, summary["forecast_blocks"]["mean_occurrence_probability"], marker="o", label="Predicted")
    plt.plot(x, summary["forecast_blocks"]["observed_occurrence_rate"], marker="o", label="Observed")
    plt.xlabel(f"Forecast {FORECAST_BLOCK_HOURS}-hour block")
    plt.ylabel("Crime occurrence probability")
    plt.title("Hourly probability calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "hourly_probability_calibration.png"), dpi=180)
    plt.close()

    if calibration_bins:
        predicted_bin = [item["mean_probability"] for item in calibration_bins]
        observed_bin = [item["observed_rate"] for item in calibration_bins]
        upper = max(max(predicted_bin), max(observed_bin), 1e-6)
        plt.figure(figsize=(7, 6))
        plt.plot([0.0, upper], [0.0, upper], linestyle="--", color="gray", label="Perfect")
        plt.plot(predicted_bin, observed_bin, marker="o", label="Calibrated model")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed occurrence rate")
        plt.title("Probability reliability on untouched test period")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(DEBUG_DIR, "probability_reliability.png"), dpi=180)
        plt.close()

    samples = []
    row_risk = expected.sum(axis=1)
    true_total = truth.sum(axis=1)
    positive_rows = np.flatnonzero(true_total > 0)
    zero_rows = np.flatnonzero(true_total == 0)
    selected = []

    def add_sample(reason, index):
        index = int(index)
        if index not in {item[1] for item in selected} and len(selected) < SAMPLE_TRACE_COUNT:
            selected.append((reason, index))

    add_sample("highest_predicted_risk", np.argmax(row_risk))
    if positive_rows.size:
        positive_order = positive_rows[np.argsort(row_risk[positive_rows])]
        add_sample("highest_scoring_positive", positive_order[-1])
        add_sample("typical_positive", positive_order[len(positive_order) // 2])
        add_sample("lowest_scoring_positive", positive_order[0])
    if zero_rows.size:
        add_sample("highest_scoring_false_alarm", zero_rows[np.argmax(row_risk[zero_rows])])
    for index in range(truth.shape[0]):
        add_sample("fallback_test_row", index)
        if len(selected) >= SAMPLE_TRACE_COUNT:
            break

    for reason, i in selected:
        samples.append({
            "test_row_index": i,
            "selection_reason": reason,
            "occurrence_probability_next24": probabilities[i].tolist(),
            "conditional_count_next24": conditional_means[i].tolist(),
            "expected_count_next24": expected[i].tolist(),
            "true_count_next24": truth[i].tolist(),
            "expected_total_24h": float(expected[i].sum()),
            "true_total_24h": float(truth[i].sum()),
            "probability_at_least_one_crime_24h_assuming_conditional_independence":
                float(1.0 - np.prod(1.0 - probabilities[i])),
        })
    with open(os.path.join(DEBUG_DIR, "sample_predictions.json"), "w", encoding="utf-8") as file:
        json.dump(samples, file, indent=2)


def save_training_history(history):
    with open(os.path.join(DEBUG_DIR, "training_history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["calibration_loss"], label="Calibration period")
    plt.xlabel("Epoch")
    plt.ylabel("Two-head loss")
    plt.title("Dynamic forecaster training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "training_curves.png"), dpi=180)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Using device: {DEVICE}")
    (
        x_seq, x_scalar, x_baseline, y, sample_weights,
        timestamps, cluster_labels, feature_names, modeling_boundaries,
    ) = load_dataset_from_chunks()
    train_idx, calibration_idx, test_idx = chronological_indices(timestamps, modeling_boundaries)
    x_scalar, scalar_mean, scalar_std = standardize_from_train(x_scalar, train_idx)
    pos_weight, train_prevalence = effective_positive_weight(y, sample_weights, train_idx)

    dataset = DynamicCrimeDataset(x_seq, x_scalar, x_baseline, y, sample_weights)
    train_loader, calibration_loader, test_loader = make_loaders(
        dataset, train_idx, calibration_idx, test_idx
    )
    initial_bias = 0.0
    output_steps = int(y.shape[1])
    model = TwoHeadDynamicForecaster(
        seq_input_dim=x_seq.shape[2],
        scalar_input_dim=x_scalar.shape[1],
        baseline_input_dim=x_baseline.shape[1],
        occurrence_bias=initial_bias,
        output_steps=output_steps,
        seq_hidden_dim=SEQ_HIDDEN_DIM,
        scalar_hidden_dim=SCALAR_HIDDEN_DIM,
        baseline_hidden_dim=BASELINE_HIDDEN_DIM,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {
        "epoch": [],
        "train_loss": [],
        "calibration_loss": [],
        "train_parts": [],
        "calibration_parts": [],
    }
    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(EPOCHS):
        train_metrics = run_epoch(model, train_loader, pos_weight, optimizer)
        calibration_metrics = run_epoch(model, calibration_loader, pos_weight)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["calibration_loss"].append(calibration_metrics["loss"])
        history["train_parts"].append(train_metrics)
        history["calibration_parts"].append(calibration_metrics)
        print(
            f"Epoch {epoch:02d} | train {train_metrics['loss']:.5f} | "
            f"calibration {calibration_metrics['loss']:.5f}"
        )

        if calibration_metrics["loss"] < best_loss:
            best_loss = calibration_metrics["loss"]
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print("Early stopping")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a finite model")
    model.load_state_dict(best_state)

    calibration_outputs = collect_outputs(
        model,
        calibration_loader,
        calibration_scale=1.0,
        calibration_bias=-math.log(pos_weight),
    )
    calibration_raw_logits = calibration_outputs[0]
    calibration_truth = calibration_outputs[4]
    calibration_row_weights = calibration_outputs[6]
    calibration_scale, calibration_bias, calibration_diagnostics = fit_platt_calibrator(
        calibration_raw_logits,
        calibration_truth,
        calibration_row_weights,
        initial_bias=-math.log(pos_weight),
    )

    save_training_history(history)
    test_outputs = collect_outputs(
        model,
        test_loader,
        calibration_scale=calibration_scale,
        calibration_bias=calibration_bias,
    )
    _, probabilities, conditional_means, expected, truth, baseline, row_weights = test_outputs
    save_evaluation(
        probabilities, conditional_means, expected, truth, baseline, row_weights,
        calibration_diagnostics, timestamps[test_idx], cluster_labels[test_idx],
    )
    with open(os.path.join(DEBUG_DIR, "metrics_summary.json"), encoding="utf-8") as file:
        evaluation_summary = json.load(file)

    model_path = os.path.join(MODEL_DIR, "hierarchical_forecaster_3h.candidate.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
        "scalar_feature_names": feature_names,
        "positive_weight": pos_weight,
        "calibration_scale": calibration_scale,
        "calibration_bias": calibration_bias,
        "calibration_diagnostics": calibration_diagnostics,
        "uncertainty": evaluation_summary["empirical_uncertainty"],
        "output_definition": {
            "occurrence_probability": (
                f"P(reported crime count > 0) for each next "
                f"{FORECAST_BLOCK_HOURS}-hour block"
            ),
            "conditional_mean": (
                f"E[reported crime count | count > 0] per "
                f"{FORECAST_BLOCK_HOURS}-hour block"
            ),
            "expected_count": "occurrence_probability * conditional_mean",
        },
        "feature_timing": {
            "sequence_end": "reference_time - 1 hour",
            "rolling_feature_end": "reference_time - 1 hour",
            "target_start": "reference_time",
        },
        "config": {
            "seq_input_dim": x_seq.shape[2],
            "scalar_input_dim": x_scalar.shape[1],
            "baseline_input_dim": x_baseline.shape[1],
            "output_steps": output_steps,
            "forecast_block_hours": FORECAST_BLOCK_HOURS,
            "super_cell_factor": 3,
            "seq_hidden_dim": SEQ_HIDDEN_DIM,
            "scalar_hidden_dim": SCALAR_HIDDEN_DIM,
            "baseline_hidden_dim": BASELINE_HIDDEN_DIM,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "dropout": DROPOUT,
        },
    }, model_path)

    print(
        f"Saved hierarchical 500 m / {FORECAST_BLOCK_HOURS}-hour candidate "
        f"to {model_path}"
    )
    print(f"Diagnostics written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
