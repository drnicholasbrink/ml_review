"""Project/session storage helpers."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    return runtime_dir / "projects" / project_id


def manifest_path(project_path: Path) -> Path:
    """Return the manifest path for a project directory."""

    return project_path / "manifest.json"


def load_manifest(project_path: Path) -> dict[str, Any]:
    """Load a project manifest."""

    return json.loads(manifest_path(project_path).read_text())


def save_manifest(project_path: Path, manifest: dict[str, Any]) -> None:
    """Persist a project manifest."""

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path(project_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def list_projects(runtime_dir: Path) -> list[dict[str, Any]]:
    """List existing projects sorted newest first."""

    ensure_runtime_dir(runtime_dir)
    projects = []
    for path in (runtime_dir / "projects").iterdir():
        if path.is_dir() and manifest_path(path).exists():
            projects.append(load_manifest(path))
    return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)


def update_project_file(project_path: Path, key: str, value: str) -> dict[str, Any]:
    """Update a file entry in the manifest."""

    manifest = load_manifest(project_path)
    manifest.setdefault("files", {})[key] = value
    save_manifest(project_path, manifest)
    return manifest
