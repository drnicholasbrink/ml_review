"""Download routes."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, send_from_directory

from ..services.project_service import load_manifest, project_dir

bp = Blueprint("exports", __name__, url_prefix="/projects/<project_id>/exports")


@bp.get("/<path:filename>")
def download(project_id: str, filename: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    if filename not in set(manifest.get("files", {}).values()):
        abort(404)
    return send_from_directory(path, filename, as_attachment=True)
