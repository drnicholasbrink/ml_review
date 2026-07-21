"""PubMed search/fetch routes."""

from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.project_service import load_manifest, project_dir, save_manifest
from ..services.pubmed_service import count_pubmed, fetch_pubmed_records

bp = Blueprint("pubmed", __name__, url_prefix="/projects/<project_id>/pubmed")


@bp.route("", methods=["GET", "POST"])
def pubmed(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    query_path = path / "search_strategy.txt"
    query = query_path.read_text() if query_path.exists() else ""
    error = None
    count = None
    if request.method == "POST":
        action = request.form.get("action")
        api_key = request.form.get("pubmed_api_key", "")
        mindate = request.form.get("mindate") or None
        maxdate = request.form.get("maxdate") or None
        try:
            if action == "count":
                count = count_pubmed(query, api_key=api_key, mindate=mindate, maxdate=maxdate)
                manifest["last_pubmed_count"] = count
                save_manifest(path, manifest)
            elif action == "fetch":
                retmax = int(request.form.get("retmax", "500"))
                df = fetch_pubmed_records(query, path / "pubmed_results_complete.csv", api_key=api_key, retmax=retmax, mindate=mindate, maxdate=maxdate)
                manifest.setdefault("files", {})["pubmed_results_complete"] = "pubmed_results_complete.csv"
                manifest["pubmed_rows"] = len(df)
                save_manifest(path, manifest)
                return redirect(url_for("embeddings.embeddings", project_id=project_id))
        except Exception as exc:
            error = str(exc)
    return render_template("pubmed.html", manifest=manifest, query=query, count=count, error=error)
