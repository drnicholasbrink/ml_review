"""Build and inspect deterministic Evidence Atlas artifacts."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import umap
from sklearn.neighbors import NearestNeighbors


ATLAS_VERSION = "0.22.0"
ATLAS_SCHEMA_VERSION = 2
ATLAS_DIRECTORY = "evidence_atlas"
ATLAS_MANIFEST = "manifest.json"
EMBEDDINGS_KEY = "embeddings"
NEIGHBOR_COUNT = 15
UMAP_PARAMETERS = {
    "n_components": 2,
    "metric": "cosine",
    "n_neighbors": 15,
    "min_dist": 0.1,
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
}

TEXT_COLUMNS = (
    "Title",
    "Abstract",
    "Authors",
    "Journal",
    "Date",
    "DOI",
    "Source",
    "EmbeddingModel",
    "ai_decision",
    "ai_confidence",
    "ai_exclusion_reason",
)
NUMERIC_COLUMNS = ("Year", "Cluster")
ARTIFACT_SCHEMA = (
    ("atlas_id", "string"),
    ("atlas_x", "float64"),
    ("atlas_y", "float64"),
    ("neighbors", "struct<ids:list<int64>,distances:list<float64>>"),
    ("search_text", "string"),
    *((name, "string") for name in TEXT_COLUMNS),
    *((name, "int64") for name in NUMERIC_COLUMNS),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_project_file(project_path: Path, filename: str | None) -> Path | None:
    if not filename:
        return None
    project_root = project_path.resolve()
    candidate = (project_path / filename).resolve()
    if candidate != project_root and project_root not in candidate.parents:
        return None
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(project_path: Path, project_manifest: dict[str, Any]) -> list[dict[str, str]]:
    files = project_manifest.get("files", {})
    sources: list[dict[str, str]] = []
    for role, key in (
        ("embeddings", EMBEDDINGS_KEY),
        ("clusters", "labeled_clusters"),
        ("screening", "ai_screening_full_results"),
    ):
        path = _safe_project_file(project_path, files.get(key))
        if path and path.is_file():
            sources.append(
                {
                    "role": role,
                    "filename": str(path.relative_to(project_path.resolve())),
                    "sha256": _file_sha256(path),
                }
            )
    return sources


def atlas_fingerprint(project_path: Path, project_manifest: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    """Return the fingerprint for the current embedding and optional metadata inputs."""
    sources = _source_files(project_path, project_manifest)
    if not sources or sources[0]["role"] != "embeddings":
        raise ValueError("Generate embeddings before building the Evidence Atlas.")
    payload = {
        "atlas_version": ATLAS_VERSION,
        "schema_version": ATLAS_SCHEMA_VERSION,
        "schema": ARTIFACT_SCHEMA,
        "parameters": {"neighbors": NEIGHBOR_COUNT, "umap": UMAP_PARAMETERS},
        "sources": sources,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), sources


def _read_atlas_manifest(project_path: Path) -> dict[str, Any] | None:
    path = project_path / ATLAS_DIRECTORY / ATLAS_MANIFEST
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def atlas_status(
    project_path: Path,
    project_manifest: dict[str, Any],
    *,
    assets_ready: bool = True,
) -> dict[str, Any]:
    """Describe whether Atlas can be opened and whether its artifact is current."""
    try:
        fingerprint, sources = atlas_fingerprint(project_path, project_manifest)
    except ValueError as exc:
        return {"state": "unavailable", "message": str(exc), "assets_ready": assets_ready}

    embeddings_path = _safe_project_file(project_path, sources[0]["filename"])
    try:
        if embeddings_path is not None and pd.read_csv(embeddings_path, usecols=["Embedding"], nrows=1).empty:
            return {
                "state": "empty",
                "message": "The embeddings CSV does not contain any records.",
                "fingerprint": fingerprint,
                "sources": sources,
                "assets_ready": assets_ready,
            }
    except (ValueError, pd.errors.EmptyDataError):
        pass

    record = _read_atlas_manifest(project_path)
    if not record:
        state = "not_built"
    else:
        artifact = project_path / ATLAS_DIRECTORY / f"{record.get('fingerprint', '')}.parquet"
        state = "ready" if record.get("fingerprint") == fingerprint and artifact.is_file() else "stale"
    return {
        "state": state,
        "fingerprint": fingerprint,
        "sources": sources,
        "record": record,
        "assets_ready": assets_ready,
    }


def _parse_embedding(value: Any, row_number: int) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:
            raise ValueError(f"Embedding row {row_number} is not a valid vector.") from exc
    if not isinstance(value, (list, tuple, np.ndarray)) or not len(value):
        raise ValueError(f"Embedding row {row_number} is empty or invalid.")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Embedding row {row_number} contains a non-numeric value.") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"Embedding row {row_number} contains a non-finite value.")
    return vector


def _validated_vectors(frame: pd.DataFrame) -> np.ndarray:
    if "Embedding" not in frame.columns:
        raise ValueError("The embeddings CSV does not contain an Embedding column.")
    if frame.empty:
        raise ValueError("The embeddings CSV does not contain any records.")
    vectors = [_parse_embedding(value, index + 2) for index, value in enumerate(frame["Embedding"])]
    dimensions = {len(vector) for vector in vectors}
    if len(dimensions) != 1:
        raise ValueError("Every embedding vector must have the same dimension.")
    return np.asarray(vectors, dtype=np.float64)


def _clean_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def _first_value(row: pd.Series, names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]) and str(row[name]).strip():
            return row[name]
    return ""


def _stable_ids(frame: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for index, row in frame.iterrows():
        supplied = _clean_text(row.get("PointID", "")).strip()
        if supplied:
            identifier = supplied
        else:
            seed = "\n".join(
                (
                    str(row.get("RecordID", index)),
                    str(row.get("Title", "")),
                    str(row.get("Abstract", "")),
                    str(index),
                )
            )
            identifier = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:20]
        if identifier in seen:
            raise ValueError(f"Atlas record identifiers are not unique: {identifier}")
        seen.add(identifier)
        ids.append(identifier)
    return ids


def _projection(vectors: np.ndarray) -> tuple[np.ndarray, int]:
    count = len(vectors)
    if count == 1:
        return np.asarray([[0.0, 0.0]], dtype=np.float64), 0
    if count == 2:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float64), 1
    effective_neighbors = min(UMAP_PARAMETERS["n_neighbors"], count - 1)
    reducer = umap.UMAP(
        n_components=UMAP_PARAMETERS["n_components"],
        metric=UMAP_PARAMETERS["metric"],
        n_neighbors=effective_neighbors,
        min_dist=UMAP_PARAMETERS["min_dist"],
        random_state=UMAP_PARAMETERS["random_state"],
        transform_seed=UMAP_PARAMETERS["transform_seed"],
        n_jobs=UMAP_PARAMETERS["n_jobs"],
    )
    coordinates = np.asarray(reducer.fit_transform(vectors), dtype=np.float64)
    if coordinates.shape != (count, 2) or not np.isfinite(coordinates).all():
        raise ValueError("UMAP did not produce a valid two-dimensional projection.")
    return coordinates, effective_neighbors


def _neighbors(vectors: np.ndarray) -> list[dict[str, list[Any]]]:
    count = len(vectors)
    if count == 1:
        return [{"ids": [], "distances": []}]
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=1)
    model.fit(vectors)
    distances, indices = model.kneighbors(vectors, n_neighbors=min(count, NEIGHBOR_COUNT + 1))
    result: list[dict[str, list[Any]]] = []
    for row_index, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        pairs = [
            (float(distance), int(neighbor_index))
            for distance, neighbor_index in zip(row_distances, row_indices, strict=True)
            if int(neighbor_index) != row_index and math.isfinite(float(distance))
        ]
        pairs.sort(key=lambda pair: (pair[0], pair[1]))
        pairs = pairs[:NEIGHBOR_COUNT]
        result.append(
            {
                "ids": [identifier for _, identifier in pairs],
                "distances": [distance for distance, _ in pairs],
            }
        )
    return result


def _metadata_index(frame: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    point_map: dict[str, pd.Series] = {}
    record_map: dict[str, pd.Series] = {}
    if "PointID" in frame.columns:
        point_map = {
            _clean_text(row["PointID"]): row
            for _, row in frame.iterrows()
            if _clean_text(row["PointID"]).strip()
        }
    if "RecordID" in frame.columns:
        counts = frame["RecordID"].astype(str).value_counts()
        record_map = {
            _clean_text(row["RecordID"]): row
            for _, row in frame.iterrows()
            if _clean_text(row["RecordID"]).strip() and counts.get(str(row["RecordID"]), 0) == 1
        }
    return point_map, record_map


def _read_optional_metadata(project_path: Path, filename: str | None) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    path = _safe_project_file(project_path, filename)
    if not path or not path.is_file():
        return {}, {}
    try:
        return _metadata_index(pd.read_csv(path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read Atlas metadata from {path.name}.") from exc


def _year_value(row: pd.Series) -> int | None:
    candidate = _first_value(row, ("Year", "PublicationYear", "publication_year"))
    if not _clean_text(candidate).strip():
        candidate = _first_value(row, ("Date", "PublicationDate", "publication_date"))
    text = _clean_text(candidate).strip()
    for token in text.replace("/", "-").split("-"):
        if len(token) == 4 and token.isdigit():
            return int(token)
    try:
        return int(float(text)) if text else None
    except ValueError:
        return None


def _artifact_rows(
    frame: pd.DataFrame,
    identifiers: list[str],
    coordinates: np.ndarray,
    neighbors: list[dict[str, list[Any]]],
    cluster_maps: tuple[dict[str, pd.Series], dict[str, pd.Series]],
    screening_maps: tuple[dict[str, pd.Series], dict[str, pd.Series]],
) -> dict[str, list[Any]]:
    columns: dict[str, list[Any]] = {name: [] for name, _ in ARTIFACT_SCHEMA}
    cluster_point, cluster_record = cluster_maps
    screening_point, screening_record = screening_maps
    aliases = {
        "Authors": ("Authors", "Author", "authors"),
        "Journal": ("Journal", "journal"),
        "Date": ("Date", "PublicationDate", "publication_date"),
        "DOI": ("DOI", "doi"),
        "Source": ("Source", "source", "source_database", "Database"),
        "EmbeddingModel": ("EmbeddingModel", "embedding_model", "Model"),
    }
    for position, (_, row) in enumerate(frame.iterrows()):
        identifier = identifiers[position]
        record_id = _clean_text(row.get("RecordID", ""))
        cluster_row = cluster_point.get(identifier)
        if cluster_row is None:
            cluster_row = cluster_record.get(record_id)
        screening_row = screening_point.get(identifier)
        if screening_row is None:
            screening_row = screening_record.get(record_id)

        title = _clean_text(_first_value(row, ("Title", "title")))
        abstract = _clean_text(_first_value(row, ("Abstract", "abstract")))
        columns["atlas_id"].append(identifier)
        columns["atlas_x"].append(float(coordinates[position, 0]))
        columns["atlas_y"].append(float(coordinates[position, 1]))
        columns["neighbors"].append(neighbors[position])
        columns["search_text"].append(f"{title}\n\n{abstract}".strip())
        columns["Title"].append(title)
        columns["Abstract"].append(abstract)
        for name, names in aliases.items():
            columns[name].append(_clean_text(_first_value(row, names)))
        columns["Year"].append(_year_value(row))
        cluster_value = _first_value(cluster_row, ("Cluster", "cluster")) if cluster_row is not None else ""
        try:
            columns["Cluster"].append(int(float(cluster_value)) if _clean_text(cluster_value).strip() else None)
        except ValueError:
            columns["Cluster"].append(None)
        screening_aliases = {
            "ai_decision": ("ai_decision", "decision", "Decision"),
            "ai_confidence": ("ai_confidence", "confidence", "Confidence"),
            "ai_exclusion_reason": ("ai_exclusion_reason", "exclusion_reason", "ExclusionReason"),
        }
        for name, names in screening_aliases.items():
            value = _first_value(screening_row, names) if screening_row is not None else ""
            columns[name].append(_clean_text(value))
    return columns


def _arrow_table(columns: dict[str, list[Any]]) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    neighbor_type = pa.struct(
        [
            pa.field("ids", pa.list_(pa.int64())),
            pa.field("distances", pa.list_(pa.float64())),
        ]
    )
    for name, kind in ARTIFACT_SCHEMA:
        if kind == "string":
            arrays[name] = pa.array(columns[name], type=pa.string())
        elif kind == "int64":
            arrays[name] = pa.array(columns[name], type=pa.int64())
        elif name == "neighbors":
            arrays[name] = pa.array(columns[name], type=neighbor_type)
        else:
            arrays[name] = pa.array(columns[name], type=pa.float64())
    return pa.table(arrays)


def _write_parquet_atomic(table: pa.Table, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".parquet.tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(value: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent, suffix=".json.tmp")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def build_atlas(project_path: Path, project_manifest: dict[str, Any]) -> dict[str, Any]:
    """Build a current Atlas artifact, or reuse one with an identical fingerprint."""
    fingerprint, sources = atlas_fingerprint(project_path, project_manifest)
    atlas_directory = project_path / ATLAS_DIRECTORY
    destination = atlas_directory / f"{fingerprint}.parquet"
    existing = _read_atlas_manifest(project_path)
    if existing and existing.get("fingerprint") == fingerprint and destination.is_file():
        return {**existing, "reused": True}

    embeddings_path = _safe_project_file(project_path, project_manifest.get("files", {}).get(EMBEDDINGS_KEY))
    if not embeddings_path or not embeddings_path.is_file():
        raise ValueError("Generate embeddings before building the Evidence Atlas.")
    try:
        frame = pd.read_csv(embeddings_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("The embeddings CSV could not be read.") from exc

    vectors = _validated_vectors(frame)
    identifiers = _stable_ids(frame)
    coordinates, effective_neighbors = _projection(vectors)
    neighbor_values = _neighbors(vectors)
    files = project_manifest.get("files", {})
    cluster_maps = _read_optional_metadata(project_path, files.get("labeled_clusters"))
    screening_maps = _read_optional_metadata(project_path, files.get("ai_screening_full_results"))
    table = _arrow_table(
        _artifact_rows(
            frame,
            identifiers,
            coordinates,
            neighbor_values,
            cluster_maps,
            screening_maps,
        )
    )

    record = {
        "atlas_version": ATLAS_VERSION,
        "schema_version": ATLAS_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "artifact": destination.name,
        "built_at": _utc_now(),
        "rows": len(frame),
        "embedding_dimension": int(vectors.shape[1]),
        "sources": sources,
        "parameters": {
            "neighbors": NEIGHBOR_COUNT,
            "umap": {**UMAP_PARAMETERS, "effective_n_neighbors": effective_neighbors},
        },
    }
    _write_parquet_atomic(table, destination)
    try:
        _write_json_atomic(record, atlas_directory / ATLAS_MANIFEST)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return {**record, "reused": False}
