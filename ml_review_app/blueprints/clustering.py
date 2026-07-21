"""Reproducible clustering exploration routes."""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, current_app, flash, jsonify, redirect, render_template, request, url_for

from ..services.clustering_service import (
    activate_run,
    analyze_clustering_source,
    analyze_subclustering,
    build_cluster_search,
    finalize_clustering_draft,
    get_run,
    load_clustering_state,
)
from ..services.project_service import invalidate_outputs, load_manifest, project_dir, save_manifest
from ..services.validation_service import parse_bounded_int

bp = Blueprint("clustering", __name__, url_prefix="/projects/<project_id>/clustering")


def _parse_perplexity(raw_value: str | None) -> float | None:
    if raw_value is None or not raw_value.strip():
        return None
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("Perplexity must be a number") from exc
    if not 0 < value <= 1000:
        raise ValueError("Perplexity must be greater than 0 and no more than 1,000")
    return value


def _activate_in_manifest(path, manifest, run, *, invalidate: bool) -> None:
    if invalidate:
        invalidate_outputs(manifest, "clustering")
    manifest.setdefault("files", {})["labeled_clusters"] = run["output_file"]
    manifest["files"]["clustering_state"] = "clustering_state.json"
    manifest["active_clustering_run"] = run["run_id"]
    manifest["elbow_scores"] = run["elbow_scores"]
    manifest["cluster_rows"] = run["row_count"]
    manifest["cluster_count"] = run["n_clusters"]
    save_manifest(path, manifest)


def _view_data(path, manifest):
    state = load_clustering_state(path)
    active_run = get_run(state)
    rows = None
    cluster_summaries = []
    if active_run and (path / active_run["output_file"]).exists():
        df = pd.read_csv(path / active_run["output_file"])
        rows = df[["PointID", "TSNE_1", "TSNE_2", "Cluster", "Title", "RecordID"]].fillna("").to_dict(orient="records")
        cluster_summaries = [
            {"cluster": int(cluster), "size": int(size)}
            for cluster, size in df.groupby("Cluster", sort=True).size().items()
        ]
    return state, active_run, state.get("draft"), rows, cluster_summaries


@bp.route("", methods=["GET", "POST"])
def clustering(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    error = None
    if request.method == "POST":
        try:
            random_state = parse_bounded_int(
                request.form.get("random_state"),
                "Random seed",
                minimum=0,
                maximum=2_147_483_647,
                default=current_app.config["DEFAULT_TSNE_RANDOM_STATE"],
            )
            perplexity = _parse_perplexity(request.form.get("perplexity"))
            embeddings_name = manifest.get("files", {}).get("embeddings")
            embeddings_path = path / embeddings_name if embeddings_name else None
            if not embeddings_path or not embeddings_path.exists():
                raise ValueError("Generate embeddings before running clustering")
            _df, _draft, _state = analyze_clustering_source(
                path,
                embeddings_path,
                random_state=random_state,
                perplexity=perplexity,
            )
        except ValueError as exc:
            error = str(exc)
    state, active_run, draft, rows, cluster_summaries = _view_data(path, manifest)
    status_code = 400 if error else 200
    return render_template(
        "clustering.html",
        manifest=manifest,
        state=state,
        active_run=active_run,
        draft=draft,
        scores=active_run["elbow_scores"] if active_run else None,
        rows=rows,
        cluster_summaries=cluster_summaries,
        error=error,
    ), status_code


@bp.get("/search")
def search(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    active = get_run(load_clustering_state(path))
    if active is None:
        return jsonify({"error": "Run clustering before searching the cluster explorer"}), 404
    output_path = path / active["output_file"]
    if not output_path.exists():
        return jsonify({"error": "The active clustering run could not be found"}), 404
    try:
        result = build_cluster_search(
            pd.read_csv(output_path),
            run_id=active["run_id"],
            raw_keywords=request.args.get("keywords"),
            match_mode=request.args.get("mode", "any"),
            study_query=request.args.get("study"),
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@bp.post("/finalize")
def finalize(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        n_clusters = parse_bounded_int(
            request.form.get("n_clusters"),
            "Cluster count",
            minimum=1,
            maximum=current_app.config["MAX_CLUSTER_COUNT"],
            default=3,
        )
        df, run, _state = finalize_clustering_draft(path, n_clusters=n_clusters)
        _activate_in_manifest(path, manifest, run, invalidate=True)
        manifest["cluster_rows"] = len(df)
        save_manifest(path, manifest)
        branch_type = "root run" if run["parent_run_id"] is None else f"branch from {run['parent_run_id']}"
        flash(f"Created {run['run_id']} as a {branch_type} with {run['n_clusters']} clusters.")
    except ValueError as exc:
        flash(str(exc))
    return redirect(url_for("clustering.clustering", project_id=project_id))


@bp.post("/subcluster")
def subcluster(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    try:
        state = load_clustering_state(path)
        active = get_run(state)
        if active is None:
            raise ValueError("Run clustering before creating subclusters")
        selected = [int(value) for value in request.form.getlist("clusters")]
        random_state = parse_bounded_int(
            request.form.get("random_state"),
            "Random seed",
            minimum=0,
            maximum=2_147_483_647,
            default=current_app.config["DEFAULT_TSNE_RANDOM_STATE"],
        )
        df, draft, _state = analyze_subclustering(
            path,
            parent_run_id=active["run_id"],
            selected_clusters=selected,
            random_state=random_state,
            perplexity=_parse_perplexity(request.form.get("perplexity")),
        )
        flash(f"WCSS analysis is ready for a new branch from {draft['parent_run_id']} across {len(df)} records. Choose the cluster count next.")
    except ValueError as exc:
        flash(str(exc))
    return redirect(url_for("clustering.clustering", project_id=project_id))


@bp.post("/back")
def step_back(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    state = load_clustering_state(path)
    active = get_run(state)
    if not active or not active.get("parent_run_id"):
        flash("The active run has no parent to step back to.")
        return redirect(url_for("clustering.clustering", project_id=project_id))
    run, _state = activate_run(path, active["parent_run_id"])
    _activate_in_manifest(path, manifest, run, invalidate=True)
    return redirect(url_for("clustering.clustering", project_id=project_id))


@bp.post("/activate/<run_id>")
def activate(project_id: str, run_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        run, _state = activate_run(path, run_id)
        _activate_in_manifest(path, manifest, run, invalidate=True)
    except ValueError as exc:
        flash(str(exc))
    return redirect(url_for("clustering.clustering", project_id=project_id))


@bp.get("/points/<point_id>")
def point_details(project_id: str, point_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    active = get_run(load_clustering_state(path))
    if active is None:
        return jsonify({"error": "No active clustering run"}), 404
    df = pd.read_csv(path / active["output_file"])
    matches = df[df["PointID"].astype(str) == point_id]
    if matches.empty:
        return jsonify({"error": "Point not found in the active run"}), 404
    row = matches.iloc[0]
    fields = ["PointID", "RecordID", "PMID", "Title", "Abstract", "Authors", "Date", "Journal", "DOI", "Cluster"]
    details = {}
    for key in fields:
        if key not in df.columns:
            continue
        value = row.get(key)
        if pd.isna(value):
            details[key] = None
        else:
            details[key] = value.item() if hasattr(value, "item") else value
    return jsonify(details)


@bp.post("/select")
def select_clusters(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    active = get_run(load_clustering_state(path))
    if active is None or not (path / active["output_file"]).exists():
        flash("Run clustering before selecting clusters.")
        return redirect(url_for("clustering.clustering", project_id=project_id))
    raw_selected = request.form.getlist("clusters")
    if not raw_selected:
        flash("Select at least one cluster to continue to screening.")
        return redirect(url_for("clustering.clustering", project_id=project_id))
    df = pd.read_csv(path / active["output_file"])
    try:
        selected = {int(value) for value in raw_selected}
    except ValueError:
        flash("One or more selected clusters were invalid.")
        return redirect(url_for("clustering.clustering", project_id=project_id))
    available = {int(value) for value in df["Cluster"].unique()}
    if not selected <= available:
        flash("One or more selected clusters no longer exist.")
        return redirect(url_for("clustering.clustering", project_id=project_id))
    selected_df = df[df["Cluster"].isin(selected)].copy()
    selected_df.to_csv(path / "selected_records.csv", index=False)
    invalidate_outputs(manifest, "selection")
    manifest.setdefault("files", {})["selected_records"] = "selected_records.csv"
    manifest["selected_clusters"] = sorted(selected)
    manifest["selected_rows"] = len(selected_df)
    manifest["selection_clustering_run"] = active["run_id"]
    save_manifest(path, manifest)
    return redirect(url_for("screening.screening", project_id=project_id))
