"""Evidence Atlas page, build, and artifact routes."""

from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
from flask import Blueprint, Response, current_app, jsonify, redirect, render_template, request, send_file, url_for

from ..services.atlas_service import ATLAS_DIRECTORY, atlas_status, build_atlas
from ..services.project_service import load_manifest, project_dir

bp = Blueprint("atlas", __name__, url_prefix="/projects/<project_id>/atlas")
FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _assets_ready() -> bool:
    static_folder = Path(current_app.static_folder or "")
    return (static_folder / "js" / "atlas_explorer.js").is_file()


def _render(project_id: str, *, error: str | None = None, status_code: int = 200):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    status = atlas_status(path, manifest, assets_ready=_assets_ready())
    return render_template("atlas.html", manifest=manifest, atlas_status=status, error=error), status_code


@bp.get("")
def atlas(project_id: str):
    return _render(project_id)


@bp.post("/build")
def build(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        build_atlas(path, manifest)
    except ValueError as exc:
        return _render(project_id, error=str(exc), status_code=400)
    return redirect(url_for("atlas.atlas", project_id=project_id))


@bp.get("/data/<fingerprint>.parquet")
def data(project_id: str, fingerprint: str):
    if not FINGERPRINT_PATTERN.fullmatch(fingerprint):
        return Response(status=404)
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    status = atlas_status(path, manifest, assets_ready=_assets_ready())
    record = status.get("record") or {}
    if status.get("state") != "ready" or record.get("fingerprint") != fingerprint:
        return Response(status=404)
    artifact = path / ATLAS_DIRECTORY / f"{fingerprint}.parquet"
    if not artifact.is_file():
        return Response(status=404)
    response = send_file(artifact, mimetype="application/vnd.apache.parquet", conditional=True)
    response.headers["Cache-Control"] = "private, max-age=31536000, immutable"
    return response


def _current_artifact(project_id: str, fingerprint: str) -> Path | None:
    if not FINGERPRINT_PATTERN.fullmatch(fingerprint):
        return None
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    status = atlas_status(path, manifest, assets_ready=_assets_ready())
    record = status.get("record") or {}
    artifact = path / ATLAS_DIRECTORY / f"{fingerprint}.parquet"
    if status.get("state") != "ready" or record.get("fingerprint") != fingerprint or not artifact.is_file():
        return None
    return artifact


@bp.get("/preview/<fingerprint>.json")
def preview(project_id: str, fingerprint: str):
    """Serve the compact, project-scoped data used by the browser-native Atlas."""

    artifact = _current_artifact(project_id, fingerprint)
    if artifact is None:
        return Response(status=404)
    columns = [
        "atlas_id", "atlas_x", "atlas_y", "neighbors", "Title", "Authors", "Journal", "Date",
        "DOI", "Source", "EmbeddingModel", "Year", "Cluster", "ai_decision", "ai_confidence",
        "ai_exclusion_reason",
    ]
    rows = pq.read_table(artifact, columns=columns).to_pylist()
    return jsonify({"fingerprint": fingerprint, "rows": rows})


@bp.get("/preview/<fingerprint>/records/<path:atlas_id>.json")
def preview_record(project_id: str, fingerprint: str, atlas_id: str):
    artifact = _current_artifact(project_id, fingerprint)
    if artifact is None:
        return Response(status=404)
    rows = pq.read_table(artifact, columns=["atlas_id", "Abstract"]).to_pylist()
    match = next((row for row in rows if row["atlas_id"] == atlas_id), None)
    return jsonify(match) if match is not None else (jsonify({"error": "Atlas record not found"}), 404)


@bp.after_request
def atlas_headers(response: Response) -> Response:
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data: blob:; "
        "font-src 'self' data:; "
        "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
    )
    return response
