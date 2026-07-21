"""Download routes."""

from __future__ import annotations

from flask import Blueprint, current_app, send_from_directory

from ..services.project_service import project_dir

bp = Blueprint("exports", __name__, url_prefix="/projects/<project_id>/exports")


@bp.get("/<path:filename>")
def download(project_id: str, filename: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    return send_from_directory(path, filename, as_attachment=True)
