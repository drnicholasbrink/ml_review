"""Embedding routes."""

from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, url_for

from ..services.embedding_service import add_embeddings_offline
from ..services.project_service import load_manifest, project_dir, save_manifest

bp = Blueprint("embeddings", __name__, url_prefix="/projects/<project_id>/embeddings")


@bp.route("", methods=["GET", "POST"])
def embeddings(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    source_name = manifest.get("files", {}).get("deduplicated_records") or manifest.get("files", {}).get("normalized_records") or manifest.get("files", {}).get("pubmed_results_complete")
    if request.method == "POST" and source_name:
        df = add_embeddings_offline(path / source_name, path / "pubmed_results_with_embeddings.csv")
        manifest.setdefault("files", {})["embeddings"] = "pubmed_results_with_embeddings.csv"
        manifest["embedding_rows"] = len(df)
        save_manifest(path, manifest)
        return redirect(url_for("clustering.clustering", project_id=project_id))
    return render_template("embeddings.html", manifest=manifest, source_name=source_name)
