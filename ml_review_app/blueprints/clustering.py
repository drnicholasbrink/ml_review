"""Clustering routes."""

from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.clustering_service import cluster_csv
from ..services.project_service import load_manifest, project_dir, save_manifest

bp = Blueprint("clustering", __name__, url_prefix="/projects/<project_id>/clustering")


@bp.route("", methods=["GET", "POST"])
def clustering(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    scores = manifest.get("elbow_scores")
    rows = None
    if request.method == "POST":
        n_clusters = int(request.form.get("n_clusters", "3"))
        df, scores = cluster_csv(path / "pubmed_results_with_embeddings.csv", path / "labeled_clusters.csv", n_clusters=n_clusters)
        manifest.setdefault("files", {})["labeled_clusters"] = "labeled_clusters.csv"
        manifest["elbow_scores"] = scores
        manifest["cluster_rows"] = len(df)
        manifest["cluster_count"] = n_clusters
        save_manifest(path, manifest)
        rows = df.head(100).to_dict(orient="records")
    elif (path / "labeled_clusters.csv").exists():
        import pandas as pd
        rows = pd.read_csv(path / "labeled_clusters.csv").head(100).to_dict(orient="records")
    return render_template("clustering.html", manifest=manifest, scores=scores, rows=rows)


@bp.post("/select")
def select_clusters(project_id: str):
    import pandas as pd
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    selected = {int(value) for value in request.form.getlist("clusters")}
    df = pd.read_csv(path / "labeled_clusters.csv")
    selected_df = df[df["Cluster"].isin(selected)].copy()
    selected_df.to_csv(path / "selected_records.csv", index=False)
    manifest.setdefault("files", {})["selected_records"] = "selected_records.csv"
    manifest["selected_clusters"] = sorted(selected)
    manifest["selected_rows"] = len(selected_df)
    save_manifest(path, manifest)
    return redirect(url_for("screening.screening", project_id=project_id))
