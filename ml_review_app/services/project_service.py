"""Project/session storage helpers."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


INVALIDATION_STAGES: dict[str, tuple[set[str], set[str]]] = {
    "upload": (
        {"column_mapping", "normalized_records", "deduplicated_records", "duplicate_report", "embeddings", "labeled_clusters", "selected_records", "ai_screening_full_results"},
        {"column_mapping", "normalized_rows", "deduplication", "embedding_rows", "elbow_scores", "cluster_rows", "cluster_count", "selected_clusters", "selected_rows", "screening_rows", "screening_decision_counts"},
    ),
    "mapping": (
        {"deduplicated_records", "duplicate_report", "embeddings", "labeled_clusters", "selected_records", "ai_screening_full_results"},
        {"deduplication", "embedding_rows", "elbow_scores", "cluster_rows", "cluster_count", "selected_clusters", "selected_rows", "screening_rows", "screening_decision_counts"},
    ),
    "records": (
        {"embeddings", "labeled_clusters", "selected_records", "ai_screening_full_results"},
        {"embedding_rows", "elbow_scores", "cluster_rows", "cluster_count", "selected_clusters", "selected_rows", "screening_rows", "screening_decision_counts"},
    ),
    "embeddings": (
        {"labeled_clusters", "selected_records", "ai_screening_full_results"},
        {"elbow_scores", "cluster_rows", "cluster_count", "selected_clusters", "selected_rows", "screening_rows", "screening_decision_counts"},
    ),
    "clustering": (
        {"selected_records", "ai_screening_full_results"},
        {"selected_clusters", "selected_rows", "screening_rows", "screening_decision_counts"},
    ),
    "selection": (
        {"ai_screening_full_results"},
        {"screening_rows", "screening_decision_counts"},
    ),
}


def ensure_runtime_dir(runtime_dir: Path) -> None:
    """Create the runtime project directory if it does not exist."""

    (runtime_dir / "projects").mkdir(parents=True, exist_ok=True)


def create_project(runtime_dir: Path, name: str) -> dict[str, Any]:
    """Create a new review project and return its manifest."""

    ensure_runtime_dir(runtime_dir)
    project_id = uuid.uuid4().hex[:12]
    project_dir = runtime_dir / "projects" / project_id
    (project_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    manifest = {
        "project_id": project_id,
        "name": name.strip() or "Untitled review",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": {},
        "column_mapping": {},
    }
    save_manifest(project_dir, manifest)
    return manifest


def project_dir(runtime_dir: Path, project_id: str) -> Path:
    """Return the directory for a project ID."""

    if not re.fullmatch(r"[a-f0-9]{12}", project_id):
        raise FileNotFoundError("Invalid project ID")
    path = runtime_dir / "projects" / project_id
    if not path.is_dir():
        raise FileNotFoundError("Project not found")
    return path


def manifest_path(project_path: Path) -> Path:
    """Return the manifest path for a project directory."""

    return project_path / "manifest.json"


def load_manifest(project_path: Path) -> dict[str, Any]:
    """Load a project manifest."""

    path = manifest_path(project_path)
    if not path.is_file():
        raise FileNotFoundError("Project manifest not found")
    return json.loads(path.read_text())


def save_manifest(project_path: Path, manifest: dict[str, Any]) -> None:
    """Persist a project manifest."""

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path(project_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def invalidate_outputs(manifest: dict[str, Any], stage: str) -> None:
    """Remove stale downstream references from a manifest without deleting files."""

    file_keys, metadata_keys = INVALIDATION_STAGES[stage]
    files = manifest.setdefault("files", {})
    for key in file_keys:
        files.pop(key, None)
    for key in metadata_keys:
        manifest.pop(key, None)


def list_projects(runtime_dir: Path) -> list[dict[str, Any]]:
    """List existing projects sorted newest first."""

    ensure_runtime_dir(runtime_dir)
    projects = []
    for path in (runtime_dir / "projects").iterdir():
        if path.is_dir() and manifest_path(path).exists():
            try:
                projects.append(load_manifest(path))
            except (json.JSONDecodeError, OSError):
                continue
    return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)


def update_project_file(project_path: Path, key: str, value: str) -> dict[str, Any]:
    """Update a file entry in the manifest."""

    manifest = load_manifest(project_path)
    manifest.setdefault("files", {})[key] = value
    save_manifest(project_path, manifest)
    return manifest
