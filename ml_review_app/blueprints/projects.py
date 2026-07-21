"""Project creation and setup routes."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.project_service import create_project, list_projects, project_dir, load_manifest, save_manifest

bp = Blueprint("projects", __name__, url_prefix="/projects")


@bp.get("")
def projects_index():
    projects = list_projects(current_app.config["RUNTIME_DIR"])
    return render_template("projects.html", projects=projects)


@bp.post("")
def projects_create():
    manifest = create_project(current_app.config["RUNTIME_DIR"], request.form.get("name", ""))
    return redirect(url_for("projects.setup", project_id=manifest["project_id"]))


@bp.route("/<project_id>/setup", methods=["GET", "POST"])
def setup(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    if request.method == "POST":
        search_strategy = request.form.get("search_strategy", "")
        criteria = request.form.get("inclusion_criteria", "")
        (path / "search_strategy.txt").write_text(search_strategy)
        (path / "inclusion_criteria.txt").write_text(criteria)
        manifest.setdefault("files", {})["search_strategy"] = "search_strategy.txt"
        manifest.setdefault("files", {})["inclusion_criteria"] = "inclusion_criteria.txt"
        save_manifest(path, manifest)
        return redirect(url_for("imports.import_data", project_id=project_id))
    search_strategy = (path / "search_strategy.txt").read_text() if (path / "search_strategy.txt").exists() else ""
    criteria = (path / "inclusion_criteria.txt").read_text() if (path / "inclusion_criteria.txt").exists() else ""
    return render_template("setup.html", manifest=manifest, search_strategy=search_strategy, criteria=criteria)
