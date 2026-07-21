"""Embedding parsing, dimensionality reduction, elbow, and clustering helpers."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def parse_embedding(value: object) -> np.ndarray | None:
    """Parse a serialized embedding value safely."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if not isinstance(value, str) or not value.strip():
        return None
    parsed = ast.literal_eval(value)
    arr = np.asarray(parsed, dtype=float)
    return arr if arr.size else None


def load_embedding_matrix(csv_path: Path, embedding_column: str = "Embedding") -> tuple[pd.DataFrame, np.ndarray]:
    """Load records and embedding matrix from CSV."""

    df = pd.read_csv(csv_path)
    embeddings = df[embedding_column].map(parse_embedding)
    valid = embeddings.map(lambda item: item is not None)
    df = df[valid].copy()
    matrix = np.vstack(embeddings[valid].to_list())
    return df, matrix


def reduce_embeddings(matrix: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to two dimensions, using t-SNE when sample size permits."""

    if len(matrix) < 3:
        padded = np.zeros((len(matrix), 2))
        padded[:, : min(matrix.shape[1], 2)] = matrix[:, : min(matrix.shape[1], 2)]
        return padded
    scaled = StandardScaler().fit_transform(matrix)
    perplexity = max(1, min(30, len(matrix) - 1))
    try:
        return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state, learning_rate="auto").fit_transform(scaled)
    except Exception:
        return PCA(n_components=2, random_state=random_state).fit_transform(scaled)


def elbow_scores(points: np.ndarray, max_k: int = 10, random_state: int = 42) -> list[dict[str, float]]:
    """Calculate WCSS values for an elbow plot."""

    max_k = max(1, min(max_k, len(points)))
    scores = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(points)
        scores.append({"k": k, "wcss": float(model.inertia_)})
    return scores


def cluster_points(points: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Assign cluster labels to two-dimensional points."""

    n_clusters = max(1, min(int(n_clusters), len(points)))
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(points)


def cluster_csv(input_csv: Path, output_csv: Path, n_clusters: int = 3) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Load embeddings, reduce, cluster, and save a labeled CSV."""

    df, matrix = load_embedding_matrix(input_csv)
    points = reduce_embeddings(matrix)
    scores = elbow_scores(points)
    labels = cluster_points(points, n_clusters)
    df["TSNE_1"] = points[:, 0]
    df["TSNE_2"] = points[:, 1]
    df["Cluster"] = labels
    df.to_csv(output_csv, index=False)
    return df, scores
