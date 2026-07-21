"""Background task status and history routes."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from ..services.project_service import load_manifest, project_dir
from ..services.task_service import list_tasks, load_task

bp = Blueprint("tasks", __name__, url_prefix="/projects/<project_id>/tasks")


@bp.get("")
def tasks(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    return render_template("tasks.html", manifest=load_manifest(path), tasks=list_tasks(path, limit=100))


@bp.get("/<task_id>.json")
def task_status(project_id: str, task_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    task = load_task(path, task_id)
    response = jsonify(task)
    response.headers["Cache-Control"] = "no-store"
    return response
