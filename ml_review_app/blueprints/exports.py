"""Download routes."""

from __future__ import annotations

import io
import json
import re
import zipfile

from flask import Blueprint, abort, current_app, send_file, send_from_directory

from ..services.project_service import load_manifest, project_dir

bp = Blueprint("exports", __name__, url_prefix="/projects/<project_id>/exports")

PUBLICATION_FILE_KEYS = (
    "search_strategy", "inclusion_criteria", "pubmed_results_complete", "normalized_records",
    "deduplicated_records", "duplicate_report", "labeled_clusters", "selected_records",
    "ai_screening_full_results", "human_screening_decisions", "human_screening_reviewed_results",
    "human_full_text_decisions", "full_text_screening_results",
    "human_evaluation_metrics", "human_evaluation_comparison", "ai_extraction_full_results",
    "ai_extraction_full_results_json", "study_characteristics", "effect_estimates", "extraction_summary",
)


@bp.get("/publication-bundle.zip")
def publication_bundle(project_id: str):
    """Download a focused audit and publication handoff without embeddings or secrets."""

    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    buffer = io.BytesIO()
    included: list[str] = []
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key in PUBLICATION_FILE_KEYS:
            filename = manifest.get("files", {}).get(key)
            source = path / filename if filename else None
            try:
                relative_source = source.resolve().relative_to(path.resolve()) if source else None
            except ValueError:
                relative_source = None
            if source and relative_source and source.is_file():
                archive.write(source, f"artifacts/{relative_source}")
                included.append(key)
        for filename in ("clustering_state.json",):
            source = path / filename
            if source.is_file():
                archive.write(source, f"artifacts/{filename}")
        audit_manifest = {
            "project_id": manifest.get("project_id"),
            "name": manifest.get("name"),
            "created_at": manifest.get("created_at"),
            "updated_at": manifest.get("updated_at"),
            "screening_model": manifest.get("screening_model"),
            "screening_decision_counts": manifest.get("screening_decision_counts"),
            "human_review_rows": manifest.get("human_review_rows", 0),
            "human_review_pending_rows": manifest.get("human_review_pending_rows", 0),
            "final_screening_decision_counts": manifest.get("final_screening_decision_counts"),
            "full_text_document_count": manifest.get("full_text_document_count", 0),
            "full_text_review_rows": manifest.get("full_text_review_rows", 0),
            "full_text_review_pending_rows": manifest.get("full_text_review_pending_rows", 0),
            "full_text_screening_decision_counts": manifest.get("full_text_screening_decision_counts"),
            "extraction_model": manifest.get("extraction_model"),
            "extraction_rows": manifest.get("extraction_rows"),
            "included_file_keys": included,
        }
        archive.writestr("audit_manifest.json", json.dumps(audit_manifest, indent=2, sort_keys=True) + "\n")
        archive.writestr(
            "README.txt",
            "ML Review publication handoff\n\n"
            "This bundle preserves search criteria, selected records, AI outputs, human adjudications, "
            "evaluation, and structured extraction artifacts that were available at export time. "
            "Embeddings, credentials, and uploaded full-text PDFs are deliberately excluded. Full-text "
            "review decisions remain in the included CSV audit artifacts without redistributing copyrighted "
            "source documents. AI decisions and extractions are not final scientific judgments; verify them "
            "against the protocol and source full text.\n",
        )
    buffer.seek(0)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", manifest.get("name", "ml-review")).strip("-") or "ml-review"
    return send_file(buffer, mimetype="application/zip", as_attachment=True, download_name=f"{safe_name}-publication-bundle.zip")


@bp.get("/<path:filename>")
def download(project_id: str, filename: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    if filename not in set(manifest.get("files", {}).values()):
        abort(404)
    return send_from_directory(path, filename, as_attachment=True)
