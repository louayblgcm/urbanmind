"""Inference-safe architecture shared by the deployed dynamic checkpoint."""

import ast
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


STATIC_EMBEDDING_DIM = 64
MAX_LOGIT_RESIDUAL = 3.0
MAX_COUNT_LOG_RESIDUAL = 1.15
SCALAR_FEATURE_NAMES = [
    "reference_hour_sin", "reference_hour_cos", "reference_weekday_sin",
    "reference_weekday_cos", "is_weekend", "month_sin", "month_cos",
    "cluster", "static_crime_score", "static_activity_score",
    "crime_total_static", "requests_311_total_static", "poi_total_static",
    "business_total_static", "permits_total_static", "traffic_mean_static",
    "habit_crime_hour_mean", "habit_crime_weekday_mean",
    "habit_crime_hour_weekday_mean", "habit_311_hour_mean",
    "habit_311_weekday_mean", "crime_last_1h", "crime_last_3h",
    "crime_last_6h", "crime_last_12h", "crime_last_24h", "crime_last_48h",
    "crime_last_72h", "req311_last_1h", "req311_last_3h",
    "req311_last_6h", "req311_last_12h", "req311_last_24h",
    "req311_last_48h", "req311_last_72h", "nbr_crime_last_1h",
    "nbr_crime_last_3h", "nbr_crime_last_6h", "nbr_crime_last_24h",
    "nbr_req311_last_1h", "nbr_req311_last_3h", "nbr_req311_last_6h",
    "nbr_req311_last_24h", "crime_prev_24h", "crime_3h_share_of_24h",
    "crime_24h_vs_prev24h_ratio", "req311_prev_24h",
    "req311_3h_share_of_24h", "req311_24h_vs_prev24h_ratio",
    "nbr_crime_prev_24h", "nbr_crime_3h_share_of_24h",
    "nbr_crime_24h_vs_prev24h_ratio", "nbr_req311_prev_24h",
    "nbr_req311_24h_vs_prev24h_ratio", "crime_24h_vs_neighbor_24h_ratio",
    "req311_24h_vs_neighbor_24h_ratio", "nbr_max_crime_last_3h",
    "nbr_max_crime_last_24h", "nbr_max_crime_prev_24h",
    "nbr_max_req311_last_24h", "nbr_max_req311_prev_24h",
    "crime_24h_vs_neighbor_max_24h_ratio",
] + [f"static_gnn_embedding_{index:02d}" for index in range(STATIC_EMBEDDING_DIM)]


def parse_embedding_vector(value, expected_dim=STATIC_EMBEDDING_DIM):
    if value is None:
        values = []
    elif isinstance(value, str):
        try:
            values = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                values = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                values = []
    else:
        values = value
    vector = np.asarray(values, dtype=np.float32).reshape(-1)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    if vector.size < expected_dim:
        vector = np.pad(vector, (0, expected_dim - vector.size))
    return vector[:expected_dim]


class TwoHeadDynamicForecaster(nn.Module):
    def __init__(
        self, seq_input_dim, scalar_input_dim, baseline_input_dim,
        occurrence_bias=0.0, output_steps=24, seq_hidden_dim=64, scalar_hidden_dim=128,
        baseline_hidden_dim=32, fusion_hidden_dim=128, dropout=0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(seq_input_dim, seq_hidden_dim, batch_first=True)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_input_dim, scalar_hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(scalar_hidden_dim, scalar_hidden_dim), nn.ReLU(),
        )
        self.baseline_mlp = nn.Sequential(
            nn.Linear(baseline_input_dim, baseline_hidden_dim), nn.ReLU(),
            nn.Linear(baseline_hidden_dim, baseline_hidden_dim), nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(seq_hidden_dim + scalar_hidden_dim + baseline_hidden_dim, fusion_hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_hidden_dim, 64), nn.ReLU(),
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
        baseline_input = torch.clamp(x_baseline, min=0.0, max=20.0)
        baseline_embedding = self.baseline_mlp(torch.log1p(baseline_input))
        shared = self.shared(torch.cat(
            [sequence_embedding, scalar_embedding, baseline_embedding], dim=1
        ))
        baseline_probability = 1.0 - torch.exp(-baseline_input.clamp_min(1e-6))
        baseline_probability = baseline_probability.clamp(1e-4, 1.0 - 1e-4)
        baseline_logit = torch.logit(baseline_probability)
        raw_logits = baseline_logit + MAX_LOGIT_RESIDUAL * torch.tanh(self.occurrence_head(shared))
        probability = torch.sigmoid(calibration_scale * raw_logits + calibration_bias)
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
        surge_expected = probability * surge_probability * surge_size
        expected_count = probability * base_conditional + surge_expected
        if return_aux:
            return raw_logits, probability, conditional_mean, expected_count, {
                "base_conditional": base_conditional,
                "surge_probability": surge_probability,
                "surge_size": surge_size,
                "surge_expected": surge_expected,
            }
        return raw_logits, probability, conditional_mean, expected_count
