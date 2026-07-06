import json
import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import torch
import torch.nn as nn
import torch.nn.functional as F
from psycopg2.extras import Json, execute_values
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GCNConv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from config.config import DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


# =========================================================
# CONFIG
# =========================================================

MODEL_DIR = os.path.join("backend", "data", "models")
DEBUG_DIR = os.path.join("backend", "data", "debug", "static_gnn")
STATIC_FEATURE_METADATA_PATH = os.path.join(
    "backend", "data", "debug", "static_features", "build_metadata.json"
)
MODEL_PATH = os.path.join(MODEL_DIR, "static_gnn.pt")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

RANDOM_STATE = 42
TRAIN_RATIO = 0.8
HIDDEN_DIM = 128
EMBEDDING_DIM = 64
EPOCHS = 250
EARLY_STOPPING_PATIENCE = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SMOOTHNESS_WEIGHT = 0.005
CLUSTER_CANDIDATES = range(4, 9)
SILHOUETTE_SAMPLE_SIZE = 5000
PROFILE_SAMPLE_ROWS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


NON_LOG_FEATURES = {
    "arrest_ratio",
    "domestic_ratio",
    "grid_x",
    "grid_y",
}

CRIME_SCORE_COLUMNS = [
    "crime_total",
    "crime_per_day",
    "violent_crime_count",
    "property_crime_count",
    "vehicle_crime_count",
    "narcotics_crime_count",
    "disorder_crime_count",
    "crime_diversity",
]

ACTIVITY_SCORE_COLUMNS = [
    "requests_311_total",
    "requests_311_per_day",
    "requests_311_type_diversity",
    "poi_total",
    "business_total",
    "permits_total",
    "traffic_mean",
]


# =========================================================
# DATABASE AND FEATURES
# =========================================================

def connect():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


def parse_grid_coordinates(cell_id):
    try:
        x_text, y_text = str(cell_id).split("_")
        return float(x_text), float(y_text)
    except (TypeError, ValueError):
        return 0.0, 0.0


def load_static_features(conn):
    frame = pd.read_sql("SELECT * FROM static_cell_features ORDER BY cell_id", conn)
    if frame.empty:
        raise ValueError("static_cell_features is empty; run build_static_cell_features.py first")

    coordinates = np.asarray([parse_grid_coordinates(value) for value in frame["cell_id"]], dtype=np.float32)
    frame["grid_x"] = coordinates[:, 0]
    frame["grid_y"] = coordinates[:, 1]
    return frame


def prepare_feature_matrix(frame):
    excluded = {"cell_id", "created_at"}
    candidate_columns = [column for column in frame.columns if column not in excluded]
    numeric = frame[candidate_columns].apply(pd.to_numeric, errors="coerce")
    nan_counts = numeric.isna().sum().sort_values(ascending=False).head(20).to_dict()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    zero_variance = [column for column in candidate_columns if numeric[column].nunique(dropna=False) <= 1]
    feature_columns = [column for column in candidate_columns if column not in zero_variance]
    transformed = numeric[feature_columns].copy()

    log1p_columns = []
    for column in feature_columns:
        if column in NON_LOG_FEATURES:
            continue
        values = transformed[column].to_numpy(dtype=np.float64)
        if np.nanmin(values) >= 0.0:
            transformed[column] = np.log1p(values)
            log1p_columns.append(column)

    scaler = StandardScaler()
    matrix = scaler.fit_transform(transformed.to_numpy(dtype=np.float64)).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    metadata = {
        "feature_columns": feature_columns,
        "zero_variance_columns_removed": zero_variance,
        "log1p_columns": log1p_columns,
        "top_nan_columns_before_fill": {key: int(value) for key, value in nan_counts.items()},
    }
    return matrix, scaler, metadata


# =========================================================
# GRAPH
# =========================================================

def build_grid_edge_index(cell_ids):
    lookup = {str(cell_id): index for index, cell_id in enumerate(cell_ids)}
    sources = []
    targets = []

    for index, cell_id in enumerate(cell_ids):
        x, y = parse_grid_coordinates(cell_id)
        x = int(x)
        y = int(y)
        sources.append(index)
        targets.append(index)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor_index = lookup.get(f"{x + dx}_{y + dy}")
                if neighbor_index is not None:
                    sources.append(index)
                    targets.append(neighbor_index)

    return torch.tensor([sources, targets], dtype=torch.long)


def make_node_masks(node_count):
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    order = torch.randperm(node_count, generator=generator)
    train_count = max(int(node_count * TRAIN_RATIO), 1)
    train_mask = torch.zeros(node_count, dtype=torch.bool)
    validation_mask = torch.zeros(node_count, dtype=torch.bool)
    train_mask[order[:train_count]] = True
    validation_mask[order[train_count:]] = True
    if not validation_mask.any():
        validation_mask[order[-1]] = True
        train_mask[order[-1]] = False
    return train_mask, validation_mask


# =========================================================
# MODEL
# =========================================================

class StaticGraphAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_1 = GCNConv(input_dim, HIDDEN_DIM, add_self_loops=False)
        self.encoder_2 = GCNConv(HIDDEN_DIM, EMBEDDING_DIM, add_self_loops=False)
        self.decoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, input_dim),
        )

    def encode(self, x, edge_index):
        hidden = F.relu(self.encoder_1(x, edge_index))
        return self.encoder_2(hidden, edge_index)

    def forward(self, x, edge_index):
        embedding = self.encode(x, edge_index)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


def embedding_smoothness(embedding, edge_index):
    source, target = edge_index
    non_self = source != target
    if not non_self.any():
        return embedding.new_tensor(0.0)
    differences = embedding[source[non_self]] - embedding[target[non_self]]
    return differences.pow(2).mean()


def train_model(features, edge_index):
    x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
    edges = edge_index.to(DEVICE)
    train_mask, validation_mask = make_node_masks(features.shape[0])
    train_mask = train_mask.to(DEVICE)
    validation_mask = validation_mask.to(DEVICE)

    model = StaticGraphAutoencoder(features.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history = {"epoch": [], "train_loss": [], "validation_loss": []}
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        reconstruction, embedding = model(x, edges)
        reconstruction_loss = F.mse_loss(reconstruction[train_mask], x[train_mask])
        smoothness_loss = embedding_smoothness(embedding, edges)
        loss = reconstruction_loss + SMOOTHNESS_WEIGHT * smoothness_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_reconstruction, _ = model(x, edges)
            validation_loss = F.mse_loss(
                validation_reconstruction[validation_mask], x[validation_mask]
            ).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(float(loss.item()))
        history["validation_loss"].append(float(validation_loss))
        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:03d} | train {loss.item():.5f} | "
                f"validation {validation_loss:.5f}"
            )

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("Static GNN did not produce a finite model")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(x, edges).cpu().numpy().astype(np.float32)
    return model, embeddings, history, best_validation


# =========================================================
# SCORES, CLUSTERING, AND LABELS
# =========================================================

def percentile_composite(frame, columns):
    available = [column for column in columns if column in frame.columns]
    if not available:
        return np.zeros(len(frame), dtype=np.float32)
    ranks = []
    for column in available:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).clip(lower=0.0)
        # Empty cells should receive zero rather than the midpoint rank of a large tie.
        positive_ranks = values.where(values > 0.0).rank(pct=True).fillna(0.0)
        ranks.append(positive_ranks.to_numpy(dtype=np.float32))
    return np.mean(np.stack(ranks, axis=1), axis=1).astype(np.float32) * 100.0


def graph_smooth_scores(values, edge_index, own_weight=0.65):
    values = np.asarray(values, dtype=np.float32)
    source = edge_index[0].numpy()
    target = edge_index[1].numpy()
    non_self = source != target
    neighbor_sum = np.zeros_like(values)
    neighbor_count = np.zeros_like(values)
    np.add.at(neighbor_sum, source[non_self], values[target[non_self]])
    np.add.at(neighbor_count, source[non_self], 1.0)
    neighbor_mean = np.divide(
        neighbor_sum,
        neighbor_count,
        out=values.copy(),
        where=neighbor_count > 0,
    )
    return (own_weight * values + (1.0 - own_weight) * neighbor_mean).astype(np.float32)


def choose_clusters(embeddings):
    results = []
    fitted = {}
    sample_size = min(SILHOUETTE_SAMPLE_SIZE, embeddings.shape[0])
    for cluster_count in CLUSTER_CANDIDATES:
        model = KMeans(n_clusters=cluster_count, random_state=RANDOM_STATE, n_init=20)
        labels = model.fit_predict(embeddings)
        score = float(
            silhouette_score(
                embeddings,
                labels,
                sample_size=sample_size if sample_size < embeddings.shape[0] else None,
                random_state=RANDOM_STATE,
            )
        )
        results.append({"k": int(cluster_count), "score": score})
        fitted[cluster_count] = (model, labels)
        print(f"Silhouette k={cluster_count}: {score:.5f}")

    best_k = max(results, key=lambda item: item["score"])["k"]
    _, labels = fitted[best_k]
    return labels.astype(np.int32), results, int(best_k)


def stabilize_cluster_ids(labels, crime_scores, activity_scores):
    unique = np.unique(labels)
    ordering = sorted(
        unique.tolist(),
        key=lambda cluster: (
            float(np.mean(crime_scores[labels == cluster]) + np.mean(activity_scores[labels == cluster])),
            float(np.mean(crime_scores[labels == cluster])),
        ),
    )
    mapping = {old: new for new, old in enumerate(ordering)}
    return np.asarray([mapping[int(value)] for value in labels], dtype=np.int32)


def rank_cluster_series(summary, column):
    return summary[column].rank(method="dense", pct=True)


def assign_urban_profiles(summary):
    summary = summary.copy()
    summary["crime_rank"] = rank_cluster_series(summary, "static_crime_score")
    summary["activity_rank"] = rank_cluster_series(summary, "static_activity_score")
    summary["transport_rank"] = rank_cluster_series(summary, "traffic_mean")
    summary["nightlife_rank"] = rank_cluster_series(summary, "poi_food_drink")
    summary["retail_rank"] = rank_cluster_series(summary, "poi_retail")
    summary["growth_rank"] = rank_cluster_series(summary, "permits_total")

    quiet_cluster = int((summary["crime_rank"] + summary["activity_rank"]).idxmin())
    transport_candidates = [index for index in summary.index if index != quiet_cluster]
    transport_cluster = int(summary.loc[transport_candidates, "transport_rank"].idxmax())
    remaining = [index for index in summary.index if index not in {quiet_cluster, transport_cluster}]
    commercial_cluster = None
    if remaining:
        commercial_cluster = int(
            summary.loc[remaining, ["activity_rank", "nightlife_rank", "retail_rank"]]
            .mean(axis=1)
            .idxmax()
        )

    names = {}
    for cluster in summary.index:
        if cluster == quiet_cluster:
            names[int(cluster)] = "Quiet Residential Zone"
        elif cluster == transport_cluster:
            names[int(cluster)] = "Transit Corridor"
        elif commercial_cluster is not None and cluster == commercial_cluster:
            names[int(cluster)] = "Commercial Mixed Zone"
        else:
            names[int(cluster)] = "Balanced Urban Zone"
    summary["urban_profile"] = [names[int(cluster)] for cluster in summary.index]
    return summary, names


def build_profiles(frame, embeddings, labels, edge_index):
    crime_scores = graph_smooth_scores(percentile_composite(frame, CRIME_SCORE_COLUMNS), edge_index)
    activity_scores = graph_smooth_scores(percentile_composite(frame, ACTIVITY_SCORE_COLUMNS), edge_index)
    labels = stabilize_cluster_ids(labels, crime_scores, activity_scores)

    profiles = frame.copy()
    profiles["static_crime_score"] = crime_scores
    profiles["static_activity_score"] = activity_scores
    profiles["cluster"] = labels
    profiles["embedding_vector"] = [vector.astype(float).tolist() for vector in embeddings]

    summary_columns = [
        "static_crime_score",
        "static_activity_score",
        "crime_total",
        "requests_311_total",
        "poi_total",
        "poi_food_drink",
        "poi_retail",
        "poi_transport",
        "poi_leisure_culture",
        "business_total",
        "permits_total",
        "traffic_mean",
    ]
    available_summary = [column for column in summary_columns if column in profiles.columns]
    summary = profiles.groupby("cluster")[available_summary].mean()
    summary["cell_count"] = profiles.groupby("cluster").size()
    summary, names = assign_urban_profiles(summary)
    profiles["urban_profile"] = profiles["cluster"].map(names)
    return profiles, summary.reset_index()


# =========================================================
# STORAGE AND DIAGNOSTICS
# =========================================================

def save_profiles_to_database(conn, profiles):
    with conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS static_gnn_profiles;")
        cursor.execute("""
            CREATE TABLE static_gnn_profiles (
                cell_id TEXT PRIMARY KEY,
                static_crime_score DOUBLE PRECISION NOT NULL,
                static_activity_score DOUBLE PRECISION NOT NULL,
                cluster INTEGER NOT NULL,
                embedding_vector JSONB NOT NULL,
                urban_profile TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """)
        cursor.execute("TRUNCATE TABLE static_gnn_profiles;")
        rows = [
            (
                str(row.cell_id),
                float(row.static_crime_score),
                float(row.static_activity_score),
                int(row.cluster),
                Json(row.embedding_vector),
                str(row.urban_profile),
            )
            for row in profiles.itertuples(index=False)
        ]
        execute_values(
            cursor,
            """
                INSERT INTO static_gnn_profiles (
                    cell_id, static_crime_score, static_activity_score,
                    cluster, embedding_vector, urban_profile
                ) VALUES %s
            """,
            rows,
            page_size=1000,
        )
    conn.commit()


def save_diagnostics(
    profiles,
    cluster_summary,
    embeddings,
    history,
    silhouette_scores,
    feature_metadata,
    node_count,
    edge_count,
    best_validation,
    best_k,
):
    with open(os.path.join(DEBUG_DIR, "training_history.json"), "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    diagnostics = {
        "timestamp": datetime.now().isoformat(),
        "device": str(DEVICE),
        "num_nodes": int(node_count),
        "num_edges": int(edge_count),
        "num_features": int(len(feature_metadata["feature_columns"])),
        **feature_metadata,
        "best_validation_loss": float(best_validation),
        "selected_cluster_count": int(best_k),
        "silhouette_scores": silhouette_scores,
    }
    if os.path.exists(STATIC_FEATURE_METADATA_PATH):
        with open(STATIC_FEATURE_METADATA_PATH, "r", encoding="utf-8") as file:
            diagnostics["static_feature_snapshot"] = json.load(file)
    with open(os.path.join(DEBUG_DIR, "diagnostics.json"), "w", encoding="utf-8") as file:
        json.dump(diagnostics, file, indent=2)

    profiles.head(PROFILE_SAMPLE_ROWS).to_csv(
        os.path.join(DEBUG_DIR, "profile_sample.csv"), index=False
    )
    cluster_summary.to_csv(os.path.join(DEBUG_DIR, "cluster_summary.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train")
    plt.plot(history["epoch"], history["validation_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction loss")
    plt.title("Static graph autoencoder training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "training_curves.png"), dpi=180)
    plt.close()

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    embedding_2d = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=profiles["cluster"], s=4, cmap="tab10")
    plt.xlabel("Embedding PC1")
    plt.ylabel("Embedding PC2")
    plt.title("Static GNN embedding clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "embedding_pca.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(
        [item["k"] for item in silhouette_scores],
        [item["score"] for item in silhouette_scores],
        marker="o",
    )
    plt.xlabel("Cluster count")
    plt.ylabel("Silhouette score")
    plt.title("Static embedding cluster selection")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "silhouette_scores.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        profiles["static_crime_score"],
        profiles["static_activity_score"],
        c=profiles["cluster"],
        s=5,
        cmap="tab10",
    )
    plt.xlabel("Static crime score")
    plt.ylabel("Static activity score")
    plt.title("Static profile scores")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "score_scatter.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.bar(cluster_summary["cluster"].astype(str), cluster_summary["cell_count"])
    plt.xlabel("Cluster")
    plt.ylabel("Cells")
    plt.title("Static cluster sizes")
    plt.tight_layout()
    plt.savefig(os.path.join(DEBUG_DIR, "cluster_sizes.png"), dpi=180)
    plt.close()


def save_model(model, scaler, feature_metadata, best_k):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": feature_metadata["feature_columns"],
            "zero_variance_columns_removed": feature_metadata["zero_variance_columns_removed"],
            "log1p_columns": feature_metadata["log1p_columns"],
            "scaler_mean": scaler.mean_.astype(np.float32),
            "scaler_scale": scaler.scale_.astype(np.float32),
            "selected_cluster_count": best_k,
            "config": {
                "input_dim": len(feature_metadata["feature_columns"]),
                "hidden_dim": HIDDEN_DIM,
                "embedding_dim": EMBEDDING_DIM,
            },
        },
        MODEL_PATH,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print(f"Using device: {DEVICE}")
    conn = connect()
    try:
        frame = load_static_features(conn)
        features, scaler, feature_metadata = prepare_feature_matrix(frame)
        edge_index = build_grid_edge_index(frame["cell_id"].tolist())
        print(
            f"Static graph: {len(frame):,} cells, {edge_index.shape[1]:,} directed edges, "
            f"{features.shape[1]} features"
        )

        model, embeddings, history, best_validation = train_model(features, edge_index)
        raw_labels, silhouette_scores, best_k = choose_clusters(embeddings)
        profiles, cluster_summary = build_profiles(frame, embeddings, raw_labels, edge_index)
        save_profiles_to_database(conn, profiles)
        save_model(model, scaler, feature_metadata, best_k)
        save_diagnostics(
            profiles=profiles,
            cluster_summary=cluster_summary,
            embeddings=embeddings,
            history=history,
            silhouette_scores=silhouette_scores,
            feature_metadata=feature_metadata,
            node_count=len(frame),
            edge_count=edge_index.shape[1],
            best_validation=best_validation,
            best_k=best_k,
        )
    finally:
        conn.close()

    print(f"Saved static GNN model to {MODEL_PATH}")
    print("Rebuilt static_gnn_profiles for the dynamic forecaster")
    print(f"Diagnostics written to {DEBUG_DIR}")


if __name__ == "__main__":
    main()
