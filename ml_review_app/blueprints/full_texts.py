"""Safe per-record full-text PDF upload, access, and removal routes."""

from __future__ import annotations

import pandas as pd
from flask import Blueprint, abort, current_app, flash, redirect, request, send_file, url_for

from ..services.full_text_service import apply_full_text_reviews, full_text_path, remove_full_text_pdf, save_full_text_pdf
from ..services.project_service import invalidate_extraction, load_manifest, project_dir, save_manifest
from ..services.screening_service import apply_human_reviews

bp = Blueprint("full_texts", __name__, url_prefix="/projects/<project_id>/full-text")


def _screening_records(path, manifest: dict) -> pd.DataFrame:
    files = manifest.get("files", {})
    source_name = files.get("human_screening_reviewed_results") or files.get("ai_screening_full_results")
    if not source_name or not (path / source_name).is_file():
        raise ValueError("Run screening before adding full text")
    frame = pd.read_csv(path / source_name)
    return frame if "record_key" in frame else apply_human_reviews(frame)


def _return_url(project_id: str):
    if request.form.get("return_to") == "extraction":
        return url_for("extraction.extraction", project_id=project_id)
    parameters = {
        "stage": "full_text",
        "view": request.form.get("view", "focus"),
        "record": request.form.get("record", ""),
        "review_filter": request.form.get("review_filter", "all"),
        "decision_filter": request.form.get("decision_filter", "any"),
        "ai_filter": request.form.get("ai_filter", "any"),
        "q": request.form.get("q", ""),
        "page": request.form.get("page", "1"),
    }
    return url_for("screening.screening", project_id=project_id, **parameters)


def _invalidate_full_text_ai(path, manifest: dict) -> None:
    """Hide stale PDF-backed AI decisions while preserving human full-text overrides."""

    files = manifest.setdefault("files", {})
    files.pop("ai_full_text_screening_results", None)
    for key in ("full_text_ai_screening_rows", "full_text_ai_candidate_rows", "full_text_ai_model"):
        manifest.pop(key, None)
    reviews_name = files.get("human_full_text_decisions")
    if reviews_name and (path / reviews_name).is_file():
        abstract = _screening_records(path, manifest)
        reviews = pd.read_csv(path / reviews_name, dtype=str)
        final = apply_full_text_reviews(abstract, reviews)
        final.to_csv(path / "full_text_screening_results.csv", index=False)
        files["full_text_screening_results"] = "full_text_screening_results.csv"
    else:
        files.pop("full_text_screening_results", None)


@bp.post("/<record_key>/upload")
def upload(project_id: str, record_key: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        records = _screening_records(path, manifest)
        metadata = save_full_text_pdf(
            path,
            manifest,
            record_key,
            request.files.get("full_text_pdf"),
            valid_record_keys=set(records["record_key"].astype(str)),
        )
        manifest.setdefault("full_text_documents", {})[record_key] = metadata
        manifest["full_text_document_count"] = len(manifest["full_text_documents"])
        _invalidate_full_text_ai(path, manifest)
        invalidate_extraction(manifest)
        save_manifest(path, manifest)
        flash("Full-text PDF saved. Existing extraction outputs were marked stale.")
    except ValueError as exc:
        flash(str(exc))
    return redirect(_return_url(project_id))


@bp.post("/<record_key>/remove")
def remove(project_id: str, record_key: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        remove_full_text_pdf(path, manifest, record_key)
        manifest.setdefault("full_text_documents", {}).pop(record_key, None)
        manifest["full_text_document_count"] = len(manifest["full_text_documents"])
        _invalidate_full_text_ai(path, manifest)
        invalidate_extraction(manifest)
        save_manifest(path, manifest)
        flash("Full-text PDF removed. Existing extraction outputs were marked stale.")
    except ValueError as exc:
        flash(str(exc))
    return redirect(_return_url(project_id))


@bp.get("/<record_key>.pdf")
def view(project_id: str, record_key: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    try:
        source = full_text_path(path, manifest, record_key)
    except ValueError:
        source = None
    if source is None:
        abort(404)
    metadata = manifest.get("full_text_documents", {}).get(record_key, {})
    download_name = metadata.get("original_filename") or f"{record_key}.pdf"
    return send_file(
        source,
        mimetype="application/pdf",
        as_attachment=request.args.get("download") == "1",
        download_name=download_name,
        conditional=True,
    )
