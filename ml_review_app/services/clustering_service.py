"""Reproducible t-SNE clustering with persistent, navigable run history."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from rapidfuzz import fuzz
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

STATE_FILENAME = "clustering_state.json"
RUN_DIRECTORY = Path("visualizations") / "clustering_runs"
DRAFT_DIRECTORY = Path("visualizations") / "clustering_draft"
DRAFT_INPUT_FILE = DRAFT_DIRECTORY / "input.csv"
DRAFT_OUTPUT_FILE = DRAFT_DIRECTORY / "projection.csv"
MAX_CLUSTER_SEARCH_KEYWORDS = 20
MAX_CLUSTER_KEYWORD_LENGTH = 100
MAX_STUDY_SEARCH_LENGTH = 200
STUDY_SEARCH_LIMIT = 25
STUDY_FUZZY_THRESHOLD = 60


def normalize_search_text(value: object) -> str:
    """Normalize Unicode, whitespace, and case for deterministic local search."""

    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def parse_cluster_keywords(raw_keywords: str | None) -> list[str]:
    """Parse, validate, normalize, and deduplicate comma/newline-separated terms."""

    keywords: list[str] = []
    seen: set[str] = set()
    for raw_keyword in re.split(r"[,\r\n]+", raw_keywords or ""):
        keyword = normalize_search_text(raw_keyword)
        if not keyword or keyword in seen:
            continue
        if len(keyword) > MAX_CLUSTER_KEYWORD_LENGTH:
            raise ValueError(f"Each keyword must be {MAX_CLUSTER_KEYWORD_LENGTH} characters or fewer")
        seen.add(keyword)
        keywords.append(keyword)
    if len(keywords) > MAX_CLUSTER_SEARCH_KEYWORDS:
        raise ValueError(f"Enter no more than {MAX_CLUSTER_SEARCH_KEYWORDS} keywords")
    return keywords


def _keyword_matches(keyword: str, text: str) -> bool:
    escaped = re.escape(keyword)
    pattern = rf"(?<!\w){escaped}(?!\w)" if " " not in keyword else escaped
    return re.search(pattern, text) is not None


def _row_text(row: pd.Series, column: str) -> str:
    return normalize_search_text(row.get(column, ""))


def _study_match(row: pd.Series, query: str, fuzzy_threshold: int) -> dict[str, Any] | None:
    title = _row_text(row, "Title")
    identifiers = [
        (label, _row_text(row, label), str(row.get(label, "")))
        for label in ("PMID", "RecordID", "DOI")
        if _row_text(row, label)
    ]
    exact_identifier = next(((label, display) for label, value, display in identifiers if value == query), None)
    prefix_identifier = next(((label, display) for label, value, display in identifiers if value.startswith(query)), None)
    direct_title = bool(title and query in title)
    fuzzy_score = int(round(fuzz.WRatio(query, title))) if title else 0

    if exact_identifier:
        priority, match_type, score, identifier = 0, "identifier_exact", 100, exact_identifier[1]
    elif direct_title:
        priority, match_type, score = 1, "title_substring", 100
        identifier = next((display for _label, _value, display in identifiers), "")
    elif prefix_identifier:
        priority, match_type, score, identifier = 2, "identifier_prefix", 99, prefix_identifier[1]
    elif fuzzy_score >= fuzzy_threshold:
        priority, match_type, score = 3, "title_fuzzy", fuzzy_score
        identifier = next((display for _label, _value, display in identifiers), "")
    else:
        return None

    return {
        "point_id": str(row["PointID"]),
        "cluster": int(row["Cluster"]),
        "title": "" if not title else str(row.get("Title", "")),
        "identifier": identifier,
        "match_type": match_type,
        "match_score": score,
        "_priority": priority,
    }


def build_cluster_search(
    df: pd.DataFrame,
    *,
    run_id: str,
    raw_keywords: str | None = None,
    match_mode: str = "any",
    study_query: str | None = None,
    study_limit: int = STUDY_SEARCH_LIMIT,
    fuzzy_threshold: int = STUDY_FUZZY_THRESHOLD,
) -> dict[str, Any]:
    """Build read-only keyword prevalence and ranked study-search results."""

    required = {"PointID", "Cluster"}
    if not required <= set(df.columns):
        raise ValueError("The active clustering run is missing point or cluster identifiers")
    mode = normalize_search_text(match_mode) or "any"
    if mode not in {"any", "all"}:
        raise ValueError("Keyword matching must use either any or all")
    keywords = parse_cluster_keywords(raw_keywords)
    query = normalize_search_text(study_query)
    if len(query) > MAX_STUDY_SEARCH_LENGTH:
        raise ValueError(f"Study search must be {MAX_STUDY_SEARCH_LENGTH} characters or fewer")

    working = df.copy()
    title_text = working.get("Title", pd.Series("", index=working.index)).map(normalize_search_text)
    abstract_text = working.get("Abstract", pd.Series("", index=working.index)).map(normalize_search_text)
    searchable_text = (title_text + " " + abstract_text).str.strip()
    keyword_masks = {
        keyword: searchable_text.map(lambda text, term=keyword: _keyword_matches(term, text))
        for keyword in keywords
    }
    if keyword_masks:
        keyword_frame = pd.DataFrame(keyword_masks, index=working.index)
        combined_mask = keyword_frame.any(axis=1) if mode == "any" else keyword_frame.all(axis=1)
    else:
        combined_mask = pd.Series(False, index=working.index, dtype=bool)

    clusters = []
    for cluster_value, cluster_rows in working.groupby("Cluster", sort=True):
        indices = cluster_rows.index
        size = len(cluster_rows)
        keyword_stats = []
        for keyword in keywords:
            count = int(keyword_masks[keyword].loc[indices].sum())
            keyword_stats.append(
                {
                    "keyword": keyword,
                    "count": count,
                    "prevalence": count / size if size else 0.0,
                    "percentage": round(count / size * 100, 1) if size else 0.0,
                }
            )
        combined_count = int(combined_mask.loc[indices].sum())
        clusters.append(
            {
                "cluster": int(cluster_value),
                "size": size,
                "keywords": keyword_stats,
                "combined_count": combined_count,
                "combined_prevalence": combined_count / size if size else 0.0,
                "combined_percentage": round(combined_count / size * 100, 1) if size else 0.0,
            }
        )

    study_matches = []
    if query:
        for _, row in working.iterrows():
            match = _study_match(row, query, fuzzy_threshold)
            if match:
                study_matches.append(match)
        study_matches.sort(
            key=lambda item: (
                item["_priority"],
                -item["match_score"],
                normalize_search_text(item["title"]),
                item["point_id"],
            )
        )
    study_point_ids = [item["point_id"] for item in study_matches]
    ranked_matches = []
    for item in study_matches[:study_limit]:
        public_item = {key: value for key, value in item.items() if key != "_priority"}
        ranked_matches.append(public_item)

    return {
        "run_id": run_id,
        "keywords": keywords,
        "match_mode": mode,
        "study_query": query,
        "clusters": clusters,
        "keyword_matched_point_ids": working.loc[combined_mask, "PointID"].astype(str).tolist(),
        "study_matched_point_ids": study_point_ids,
        "study_matches": ranked_matches,
    }


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


def elbow_scores(points: np.ndarray, max_k: int = 20, random_state: int = 42) -> list[dict[str, float]]:
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
        return {"active_run_id": None, "runs": [], "draft": None}
    state = json.loads(state_path.read_text())
    if not isinstance(state.get("runs"), list):
        raise ValueError("Clustering history is invalid")
    state.setdefault("draft", None)
    return state


def save_clustering_state(project_path: Path, state: dict[str, Any]) -> None:
    (project_path / STATE_FILENAME).write_text(json.dumps(state, indent=2, sort_keys=True))


def get_run(state: dict[str, Any], run_id: str | None = None) -> dict[str, Any] | None:
    target = run_id or state.get("active_run_id")
    return next((run for run in state["runs"] if run["run_id"] == target), None)


def _next_run_id(state: dict[str, Any]) -> str:
    return f"run_{len(state['runs']) + 1:03d}"


def analyze_clustering_source(
    project_path: Path,
    source_csv: Path,
    *,
    random_state: int = 42,
    perplexity: float | None = None,
    parent_run_id: str | None = None,
    parent_clusters: list[int] | None = None,
    source_label: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Save an exact projection and WCSS curve before cluster count is chosen."""

    state = load_clustering_state(project_path)
    parent = get_run(state, parent_run_id) if parent_run_id else None
    if parent_run_id and parent is None:
        raise ValueError("The parent clustering run no longer exists")
    df, matrix = load_embedding_matrix(source_csv)
    points, reduction = reduce_embeddings(matrix, random_state=random_state, perplexity=perplexity)
    scores = elbow_scores(points, random_state=random_state)
    df["TSNE_1"] = points[:, 0]
    df["TSNE_2"] = points[:, 1]
    draft_path = project_path / DRAFT_OUTPUT_FILE
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(draft_path, index=False)
    draft = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": source_label or str(source_csv.relative_to(project_path)),
        "projection_file": str(DRAFT_OUTPUT_FILE),
        "row_count": len(df),
        "random_state": int(random_state),
        "perplexity": reduction["perplexity"],
        "method": reduction["method"],
        "metric": "cosine" if reduction["method"] == "t-SNE" else None,
        "parent_run_id": parent_run_id,
        "depth": int(parent["depth"]) + 1 if parent else 0,
        "selected_parent_clusters": sorted(parent_clusters or []),
        "elbow_scores": scores,
        "sklearn_version": sklearn.__version__,
    }
    state["draft"] = draft
    save_clustering_state(project_path, state)
    return df, draft, state


def analyze_subclustering(
    project_path: Path,
    *,
    parent_run_id: str,
    selected_clusters: list[int],
    random_state: int = 42,
    perplexity: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Analyze selected parent clusters and expose WCSS before creating a child run."""

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
    input_path = project_path / DRAFT_INPUT_FILE
    input_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(input_path, index=False)
    return analyze_clustering_source(
        project_path,
        input_path,
        random_state=random_state,
        perplexity=perplexity,
        parent_run_id=parent_run_id,
        parent_clusters=sorted(selected),
        source_label=parent["output_file"],
    )


def finalize_clustering_draft(
    project_path: Path,
    *,
    n_clusters: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Create an immutable run from the displayed projection and WCSS analysis."""

    state = load_clustering_state(project_path)
    draft = state.get("draft")
    if not draft:
        raise ValueError("Analyze t-SNE and WCSS before choosing a cluster count")
    projection_path = project_path / draft["projection_file"]
    if not projection_path.exists():
        raise ValueError("The analyzed clustering projection is missing; run the WCSS analysis again")
    df = pd.read_csv(projection_path)
    points = df[["TSNE_1", "TSNE_2"]].to_numpy(dtype=float)
    df["Cluster"] = cluster_points(points, n_clusters, random_state=int(draft["random_state"])).astype(int)
    parent = get_run(state, draft.get("parent_run_id")) if draft.get("parent_run_id") else None
    if draft.get("parent_run_id") and parent is None:
        raise ValueError("The analyzed parent run no longer exists")
    run_id = _next_run_id(state)
    relative_output = RUN_DIRECTORY / f"{run_id}.csv"
    output_path = project_path / relative_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    run = {
        "run_id": run_id,
        "parent_run_id": draft.get("parent_run_id"),
        "depth": draft["depth"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_file": draft["source_file"],
        "output_file": str(relative_output),
        "row_count": len(df),
        "n_clusters": int(n_clusters),
        "random_state": int(draft["random_state"]),
        "perplexity": draft["perplexity"],
        "method": draft["method"],
        "metric": draft["metric"],
        "selected_parent_clusters": draft["selected_parent_clusters"],
        "elbow_scores": draft["elbow_scores"],
        "sklearn_version": draft["sklearn_version"],
    }
    state["runs"].append(run)
    state["active_run_id"] = run_id
    state["draft"] = None
    save_clustering_state(project_path, state)
    return df, run, state


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

    analyze_clustering_source(
        project_path,
        source_csv,
        random_state=random_state,
        perplexity=perplexity,
        parent_run_id=parent_run_id,
        parent_clusters=parent_clusters,
    )
    return finalize_clustering_draft(project_path, n_clusters=n_clusters)


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

    analyze_subclustering(
        project_path,
        parent_run_id=parent_run_id,
        selected_clusters=selected_clusters,
        random_state=random_state,
        perplexity=perplexity,
    )
    return finalize_clustering_draft(project_path, n_clusters=n_clusters)


def activate_run(project_path: Path, run_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Make an existing run active without deleting later branches."""

    state = load_clustering_state(project_path)
    run = get_run(state, run_id)
    if run is None:
        raise ValueError("The requested clustering run does not exist")
    state["active_run_id"] = run_id
    state["draft"] = None
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
