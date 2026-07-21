"""Reproducible t-SNE clustering with persistent, navigable run history."""

from __future__ import annotations

import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

STATE_FILENAME = "clustering_state.json"
RUN_DIRECTORY = Path("visualizations") / "clustering_runs"


def parse_embedding(value: object) -> np.ndarray | None:
    """Parse a serialized embedding value safely."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=float)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
        arr = np.asarray(parsed, dtype=float)
    except (SyntaxError, ValueError, TypeError):
        return None
    return arr if arr.size else None


def _point_id(row: pd.Series, index: int) -> str:
    if pd.notna(row.get("PointID")) and str(row.get("PointID")).strip():
        return str(row["PointID"])
    identity = "\n".join(
        [str(row.get("RecordID", index)), str(row.get("Title", "")), str(row.get("Abstract", "")), str(index)]
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def load_embedding_matrix(csv_path: Path, embedding_column: str = "Embedding") -> tuple[pd.DataFrame, np.ndarray]:
    """Load records and a consistently shaped embedding matrix from CSV."""

    df = pd.read_csv(csv_path).reset_index(drop=True)
    if embedding_column not in df.columns:
        raise ValueError(f"Embedding column {embedding_column!r} not found")
    embeddings = df[embedding_column].map(parse_embedding)
    valid = embeddings.map(lambda item: item is not None)
    df = df[valid].copy().reset_index(drop=True)
    embeddings = embeddings[valid].reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid embeddings were found")
    dimensions = embeddings.map(len)
    if dimensions.nunique() != 1:
        raise ValueError("Embedding vectors must all have the same dimensions")
    df["PointID"] = [_point_id(row, index) for index, row in df.iterrows()]
    if df["PointID"].duplicated().any():
        raise ValueError("Each record must have a unique point identifier")
    return df, np.vstack(embeddings.to_list())


def reduce_embeddings(
    matrix: np.ndarray,
    random_state: int = 42,
    perplexity: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run t-SNE reproducibly, or preserve coordinates for fewer than three points."""

    if len(matrix) < 3:
        points = np.zeros((len(matrix), 2))
        points[:, : min(matrix.shape[1], 2)] = matrix[:, : min(matrix.shape[1], 2)]
        return points, {"method": "direct coordinates (too few records for t-SNE)", "perplexity": None}
    effective_perplexity = float(perplexity if perplexity is not None else min(30, max(2, (len(matrix) - 1) // 3)))
    if not 0 < effective_perplexity < len(matrix):
        raise ValueError(f"t-SNE perplexity must be greater than 0 and less than the {len(matrix)} selected records")
    scaled = StandardScaler().fit_transform(matrix)
    points = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=random_state,
        learning_rate="auto",
        metric="cosine",
    ).fit_transform(scaled)
    return points, {"method": "t-SNE", "perplexity": effective_perplexity}


def elbow_scores(points: np.ndarray, max_k: int = 10, random_state: int = 42) -> list[dict[str, float]]:
    """Calculate reproducible WCSS values for an elbow plot."""

    max_k = max(1, min(max_k, len(points)))
    scores = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(points)
        scores.append({"k": k, "wcss": float(model.inertia_)})
    return scores


def cluster_points(points: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    """Assign reproducible K-Means labels to two-dimensional points."""

    if not 1 <= int(n_clusters) <= len(points):
        raise ValueError(f"Cluster count must be between 1 and the {len(points)} selected records")
    return KMeans(n_clusters=int(n_clusters), random_state=random_state, n_init=10).fit_predict(points)


def load_clustering_state(project_path: Path) -> dict[str, Any]:
    state_path = project_path / STATE_FILENAME
    if not state_path.exists():
        return {"active_run_id": None, "runs": []}
    state = json.loads(state_path.read_text())
    if not isinstance(state.get("runs"), list):
        raise ValueError("Clustering history is invalid")
    return state


def save_clustering_state(project_path: Path, state: dict[str, Any]) -> None:
    (project_path / STATE_FILENAME).write_text(json.dumps(state, indent=2, sort_keys=True))


def get_run(state: dict[str, Any], run_id: str | None = None) -> dict[str, Any] | None:
    target = run_id or state.get("active_run_id")
    return next((run for run in state["runs"] if run["run_id"] == target), None)


def _next_run_id(state: dict[str, Any]) -> str:
    return f"run_{len(state['runs']) + 1:03d}"


def create_clustering_run(
    project_path: Path,
    source_csv: Path,
    *,
    n_clusters: int,
    random_state: int = 42,
    perplexity: float | None = None,
    parent_run_id: str | None = None,
    parent_clusters: list[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Create and persist one immutable clustering run and its exact parameters."""

    df, matrix = load_embedding_matrix(source_csv)
    points, reduction = reduce_embeddings(matrix, random_state=random_state, perplexity=perplexity)
    labels = cluster_points(points, n_clusters, random_state=random_state)
    scores = elbow_scores(points, random_state=random_state)
    df["TSNE_1"] = points[:, 0]
    df["TSNE_2"] = points[:, 1]
    df["Cluster"] = labels.astype(int)

    state = load_clustering_state(project_path)
    parent = get_run(state, parent_run_id) if parent_run_id else None
    if parent_run_id and parent is None:
        raise ValueError("The parent clustering run no longer exists")
    run_id = _next_run_id(state)
    relative_output = RUN_DIRECTORY / f"{run_id}.csv"
    output_path = project_path / relative_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    run = {
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "depth": int(parent["depth"]) + 1 if parent else 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(source_csv.relative_to(project_path)),
        "output_file": str(relative_output),
        "row_count": len(df),
        "n_clusters": int(n_clusters),
        "random_state": int(random_state),
        "perplexity": reduction["perplexity"],
        "method": reduction["method"],
        "metric": "cosine" if reduction["method"] == "t-SNE" else None,
        "selected_parent_clusters": sorted(parent_clusters or []),
        "elbow_scores": scores,
        "sklearn_version": sklearn.__version__,
    }
    state["runs"].append(run)
    state["active_run_id"] = run_id
    save_clustering_state(project_path, state)
    return df, run, state


def create_subclustering_run(
    project_path: Path,
    *,
    parent_run_id: str,
    selected_clusters: list[int],
    n_clusters: int,
    random_state: int = 42,
    perplexity: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Create a child run from selected clusters of an existing run."""

    state = load_clustering_state(project_path)
    parent = get_run(state, parent_run_id)
    if parent is None:
        raise ValueError("The selected parent clustering run no longer exists")
    parent_df = pd.read_csv(project_path / parent["output_file"])
    available = {int(value) for value in parent_df["Cluster"].unique()}
    selected = {int(value) for value in selected_clusters}
    if not selected or not selected <= available:
        raise ValueError("Select one or more clusters from the active run")
    subset = parent_df[parent_df["Cluster"].isin(selected)].copy()
    if n_clusters > len(subset):
        raise ValueError(f"Subcluster count cannot exceed the {len(subset)} selected records")
    input_path = project_path / RUN_DIRECTORY / f"{_next_run_id(state)}_input.csv"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(input_path, index=False)
    return create_clustering_run(
        project_path,
        input_path,
        n_clusters=n_clusters,
        random_state=random_state,
        perplexity=perplexity,
        parent_run_id=parent_run_id,
        parent_clusters=sorted(selected),
    )


def activate_run(project_path: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make an existing run active without deleting later branches."""

    state = load_clustering_state(project_path)
    run = get_run(state, run_id)
    if run is None:
        raise ValueError("The requested clustering run does not exist")
    state["active_run_id"] = run_id
    save_clustering_state(project_path, state)
    return run, state


def cluster_csv(
    input_csv: Path,
    output_csv: Path,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    """Compatibility helper for a standalone reproducible clustering export."""

    df, matrix = load_embedding_matrix(input_csv)
    points, _metadata = reduce_embeddings(matrix, random_state=random_state)
    scores = elbow_scores(points, random_state=random_state)
    df["TSNE_1"] = points[:, 0]
    df["TSNE_2"] = points[:, 1]
    df["Cluster"] = cluster_points(points, n_clusters, random_state=random_state)
    df.to_csv(output_csv, index=False)
    return df, scores
