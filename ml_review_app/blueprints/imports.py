"""CSV import and deduplication routes."""

from __future__ import annotations

import pandas as pd

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for

from ..services.deduplication_service import deduplicate_records
from ..services.import_service import CANONICAL_COLUMNS, build_column_mapping, normalize_records, profile_csv, read_csv_preview, save_column_mapping, save_uploaded_csv
from ..services.project_service import invalidate_outputs, load_manifest, project_dir, save_manifest

bp = Blueprint("imports", __name__, url_prefix="/projects/<project_id>/import")


@bp.route("", methods=["GET", "POST"])
def import_data(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    preview = None
    columns = []
    profile = None
    error = None
    status_code = 200
    upload_path = path / "uploaded_source.csv"
    if request.method == "POST":
        upload = request.files.get("csv_file")
        if not upload or not upload.filename:
            error = "Choose a CSV file to upload"
            status_code = 400
        else:
            try:
                upload_path = save_uploaded_csv(upload, path)
                invalidate_outputs(manifest, "upload")
                manifest.setdefault("files", {})["uploaded_source"] = upload_path.name
                save_manifest(path, manifest)
            except ValueError as exc:
                error = str(exc)
                status_code = 400
    if upload_path.exists():
        try:
            preview_df, columns = read_csv_preview(upload_path)
            preview = preview_df.to_dict(orient="records")
            profile = profile_csv(upload_path, request.form.get("RecordID") or None)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            error = "The saved upload could not be read as CSV. Upload a replacement file."
            status_code = 400
    mapping = {
        canonical: source
        for canonical, source in manifest.get("column_mapping", {}).items()
        if canonical in CANONICAL_COLUMNS and source in columns
    }
    return render_template("import.html", manifest=manifest, columns=columns, canonical_columns=CANONICAL_COLUMNS, mapping=mapping, preview=preview, profile=profile, error=error), status_code


@bp.post("/map")
def map_columns(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    upload_path = path / "uploaded_source.csv"
    if not upload_path.exists():
        flash("Upload a CSV before mapping columns.")
        return redirect(url_for("imports.import_data", project_id=project_id))
    _, columns = read_csv_preview(upload_path)
    manifest = load_manifest(path)
    try:
        mapping = build_column_mapping(request.form, columns)
        save_column_mapping(path, mapping)
        normalized = normalize_records(upload_path, mapping, path / "normalized_records.csv")
        invalidate_outputs(manifest, "mapping")
        manifest["record_source"] = "csv"
        manifest["column_mapping"] = mapping
        manifest.setdefault("files", {})["column_mapping"] = "column_mapping.json"
        manifest.setdefault("files", {})["normalized_records"] = "normalized_records.csv"
        manifest["normalized_rows"] = len(normalized)
        save_manifest(path, manifest)
        return redirect(url_for("imports.deduplicate", project_id=project_id))
    except ValueError as exc:
        preview_df, columns = read_csv_preview(upload_path)
        submitted_mapping = {
            canonical: request.form.get(canonical, "").strip()
            for canonical in CANONICAL_COLUMNS
            if request.form.get(canonical, "").strip() in columns
        }
        return render_template("import.html", manifest=manifest, columns=columns, canonical_columns=CANONICAL_COLUMNS, mapping=submitted_mapping, preview=preview_df.to_dict(orient="records"), profile=profile_csv(upload_path), error=str(exc)), 400


@bp.route("/deduplicate", methods=["GET", "POST"])
def deduplicate(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    input_path = path / "normalized_records.csv"
    df_preview = read_csv_preview(input_path)[0] if input_path.exists() else None
    columns = list(df_preview.columns) if df_preview is not None else []
    error = None
    report_rows = None
    selected_match_columns = manifest.get("deduplication", {}).get("match_columns", ["Title"])
    if request.method == "POST":
        match_columns = request.form.getlist("match_columns") or ["Title"]
        selected_match_columns = match_columns
        try:
            if not input_path.exists():
                raise ValueError("Upload and map a CSV before deduplicating records")
            kept, report = deduplicate_records(input_path, path / "deduplicated_records.csv", path / "duplicate_report.csv", "RecordID", match_columns)
            invalidate_outputs(manifest, "records")
            manifest["record_source"] = "csv"
            manifest.setdefault("files", {})["deduplicated_records"] = "deduplicated_records.csv"
            manifest.setdefault("files", {})["duplicate_report"] = "duplicate_report.csv"
            manifest["deduplication"] = {"match_columns": match_columns, "kept_rows": len(kept), "duplicate_report_rows": len(report)}
            save_manifest(path, manifest)
            report_rows = report.head(50).to_dict(orient="records")
        except ValueError as exc:
            error = str(exc)
    if report_rows is None:
        report_name = manifest.get("files", {}).get("duplicate_report")
        report_path = path / report_name if report_name else None
        if report_path and report_path.is_file():
            report_rows = pd.read_csv(report_path).head(50).fillna("").to_dict(orient="records")
    status_code = 400 if error else 200
    return render_template("deduplicate.html", manifest=manifest, columns=columns, selected_match_columns=selected_match_columns, error=error, report_rows=report_rows), status_code
