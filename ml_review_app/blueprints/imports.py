"""CSV import and deduplication routes."""

from __future__ import annotations

import json

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.deduplication_service import deduplicate_records
from ..services.import_service import CANONICAL_COLUMNS, build_column_mapping, normalize_records, profile_csv, read_csv_preview, save_column_mapping, save_uploaded_csv
from ..services.project_service import load_manifest, project_dir, save_manifest

bp = Blueprint("imports", __name__, url_prefix="/projects/<project_id>/import")


@bp.route("", methods=["GET", "POST"])
def import_data(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    preview = None
    columns = []
    profile = None
    error = None
    upload_path = path / "uploaded_source.csv"
    if request.method == "POST" and "csv_file" in request.files:
        upload = request.files["csv_file"]
        if upload and upload.filename:
            upload_path = save_uploaded_csv(upload, path)
            manifest.setdefault("files", {})["uploaded_source"] = upload_path.name
            save_manifest(path, manifest)
    if upload_path.exists():
        preview_df, columns = read_csv_preview(upload_path)
        preview = preview_df.to_dict(orient="records")
        profile = profile_csv(upload_path, request.form.get("RecordID") or None)
    return render_template("import.html", manifest=manifest, columns=columns, canonical_columns=CANONICAL_COLUMNS, preview=preview, profile=profile, error=error)


@bp.post("/map")
def map_columns(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    upload_path = path / "uploaded_source.csv"
    _, columns = read_csv_preview(upload_path)
    manifest = load_manifest(path)
    try:
        mapping = build_column_mapping(request.form, columns)
        save_column_mapping(path, mapping)
        normalized = normalize_records(upload_path, mapping, path / "normalized_records.csv")
        manifest["column_mapping"] = mapping
        manifest.setdefault("files", {})["column_mapping"] = "column_mapping.json"
        manifest.setdefault("files", {})["normalized_records"] = "normalized_records.csv"
        manifest["normalized_rows"] = len(normalized)
        save_manifest(path, manifest)
        return redirect(url_for("imports.deduplicate", project_id=project_id))
    except ValueError as exc:
        preview_df, columns = read_csv_preview(upload_path)
        return render_template("import.html", manifest=manifest, columns=columns, canonical_columns=CANONICAL_COLUMNS, preview=preview_df.to_dict(orient="records"), profile=profile_csv(upload_path), error=str(exc))


@bp.route("/deduplicate", methods=["GET", "POST"])
def deduplicate(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    input_path = path / "normalized_records.csv"
    df_preview = read_csv_preview(input_path)[0] if input_path.exists() else None
    columns = list(df_preview.columns) if df_preview is not None else []
    error = None
    report_rows = None
    if request.method == "POST":
        match_columns = request.form.getlist("match_columns") or ["Title"]
        try:
            kept, report = deduplicate_records(input_path, path / "deduplicated_records.csv", path / "duplicate_report.csv", "RecordID", match_columns)
            manifest.setdefault("files", {})["deduplicated_records"] = "deduplicated_records.csv"
            manifest.setdefault("files", {})["duplicate_report"] = "duplicate_report.csv"
            manifest["deduplication"] = {"match_columns": match_columns, "kept_rows": len(kept), "duplicate_report_rows": len(report)}
            save_manifest(path, manifest)
            report_rows = report.head(50).to_dict(orient="records")
        except ValueError as exc:
            error = str(exc)
    return render_template("deduplicate.html", manifest=manifest, columns=columns, error=error, report_rows=report_rows)
