"""Project creation and setup routes."""

from __future__ import annotations

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..services.project_service import (
    create_project,
    delete_project,
    invalidate_outputs,
    list_projects,
    load_manifest,
    project_dir,
    save_manifest,
)
from ..services.task_service import TaskConflictError
from ..services.validation_service import validate_text
from ..services.workflow_service import build_workflow_steps

bp = Blueprint("projects", __name__, url_prefix="/projects")


@bp.get("")
def projects_index():
    projects = list_projects(current_app.config["RUNTIME_DIR"])
    return render_template("projects.html", projects=projects, error=None)


@bp.post("")
def projects_create():
    try:
        name = validate_text(
            request.form.get("name"),
            "Project name",
            current_app.config["MAX_PROJECT_NAME_LENGTH"],
        )
    except ValueError as exc:
        projects = list_projects(current_app.config["RUNTIME_DIR"])
        return render_template("projects.html", projects=projects, error=str(exc)), 400
    manifest = create_project(current_app.config["RUNTIME_DIR"], name)
    return redirect(url_for("projects.setup", project_id=manifest["project_id"]))


@bp.get("/<project_id>")
def dashboard(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    steps = build_workflow_steps(path, manifest)
    return render_template("project_dashboard.html", manifest=manifest, workflow_steps=steps)


@bp.get("/<project_id>/delete")
def project_delete_confirmation(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    return render_template("project_delete.html", manifest=manifest, error=None)


@bp.post("/<project_id>/delete")
def project_delete(project_id: str):
    runtime_dir = current_app.config["RUNTIME_DIR"]
    path = project_dir(runtime_dir, project_id)
    manifest = load_manifest(path)
    confirmation = request.form.get("confirmation", "")
    if confirmation != manifest["name"]:
        return render_template(
            "project_delete.html",
            manifest=manifest,
            error="Enter the project name exactly as shown to confirm permanent deletion.",
        ), 400
    try:
        current_app.extensions["task_manager"].run_project_exclusive(
            path,
            lambda: delete_project(runtime_dir, project_id),
        )
    except TaskConflictError as exc:
        return render_template(
            "error.html",
            title="Background task running",
            message=f"{exc}. Wait for it to finish before deleting this project.",
        ), 409
    except (OSError, ValueError):
        current_app.logger.exception("Could not delete project %s", project_id)
        return render_template(
            "project_delete.html",
            manifest=manifest,
            error="Deletion could not be completed. Some project files may remain; check the application logs before retrying.",
        ), 500
    flash(f'Deleted "{manifest["name"]}" and all associated project data.')
    return redirect(url_for("projects.projects_index"))


@bp.route("/<project_id>/setup", methods=["GET", "POST"])
def setup(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    error = None
    if request.method == "POST":
        previous_search_strategy = (path / "search_strategy.txt").read_text() if (path / "search_strategy.txt").exists() else ""
        previous_criteria = (path / "inclusion_criteria.txt").read_text() if (path / "inclusion_criteria.txt").exists() else ""
        search_strategy = request.form.get("search_strategy", "")
        criteria = request.form.get("inclusion_criteria", "")
        try:
            search_strategy = validate_text(
                search_strategy,
                "PubMed search strategy",
                current_app.config["MAX_SEARCH_STRATEGY_LENGTH"],
                required=True,
            )
            criteria = validate_text(
                criteria,
                "Inclusion/exclusion criteria",
                current_app.config["MAX_INCLUSION_CRITERIA_LENGTH"],
                required=True,
            )
        except ValueError as exc:
            error = str(exc)
            return render_template(
                "setup.html",
                manifest=manifest,
                search_strategy=search_strategy,
                criteria=criteria,
                error=error,
            ), 400
        (path / "search_strategy.txt").write_text(search_strategy)
        (path / "inclusion_criteria.txt").write_text(criteria)
        if search_strategy != previous_search_strategy:
            manifest.setdefault("files", {}).pop("pubmed_results_complete", None)
            manifest.pop("pubmed_rows", None)
            manifest.pop("last_pubmed_count", None)
            manifest.pop("last_pubmed_parameters", None)
            if manifest.get("record_source") == "pubmed":
                invalidate_outputs(manifest, "records")
        if criteria != previous_criteria:
            invalidate_outputs(manifest, "selection")
        manifest.setdefault("files", {})["search_strategy"] = "search_strategy.txt"
        manifest.setdefault("files", {})["inclusion_criteria"] = "inclusion_criteria.txt"
        save_manifest(path, manifest)
        return redirect(url_for("pubmed.pubmed", project_id=project_id))
    search_strategy = (path / "search_strategy.txt").read_text() if (path / "search_strategy.txt").exists() else ""
    criteria = (path / "inclusion_criteria.txt").read_text() if (path / "inclusion_criteria.txt").exists() else ""
    return render_template("setup.html", manifest=manifest, search_strategy=search_strategy, criteria=criteria, error=error)
