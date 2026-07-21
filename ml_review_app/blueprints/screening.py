"""OpenAI screening routes."""

from __future__ import annotations

from flask import Blueprint, current_app, render_template, request
from openai import OpenAIError

from ..services.credential_service import credential_available, resolve_api_key
from ..services.project_service import load_manifest, project_dir, save_manifest
from ..services.screening_service import screen_csv
from ..services.validation_service import validate_text

bp = Blueprint("screening", __name__, url_prefix="/projects/<project_id>/screening")


@bp.route("", methods=["GET", "POST"])
def screening(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    criteria_path = path / "inclusion_criteria.txt"
    criteria = criteria_path.read_text() if criteria_path.exists() else ""
    source = manifest.get("files", {}).get("selected_records") or manifest.get("files", {}).get("labeled_clusters") or manifest.get("files", {}).get("deduplicated_records")
    source_ready = bool(source and (path / source).exists())
    rows = None
    error = None
    fallback_available = credential_available("OPENAI_API_KEY")
    if request.method == "POST":
        if not source_ready:
            error = "Select or deduplicate records before running screening"
        else:
            try:
                criteria = validate_text(
                    criteria,
                    "Inclusion/exclusion criteria",
                    current_app.config["MAX_INCLUSION_CRITERIA_LENGTH"],
                    required=True,
                )
                submitted_key = validate_text(
                    request.form.get("openai_api_key"),
                    "OpenAI API key",
                    current_app.config["MAX_API_KEY_LENGTH"],
                )
                api_key, key_source = resolve_api_key(submitted_key, "OPENAI_API_KEY", required=True)
                model = validate_text(request.form.get("model"), "Screening model", 100, required=True)
                allowed_models = {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"}
                if model not in allowed_models:
                    raise ValueError("Choose a supported OpenAI screening model")
                df = screen_csv(
                    path / source,
                    criteria,
                    path / "ai_screening_full_results.csv",
                    api_key=api_key,
                    model=model,
                    resume=request.form.get("resume") == "on",
                )
                manifest.setdefault("files", {})["ai_screening_full_results"] = "ai_screening_full_results.csv"
                manifest["screening_rows"] = len(df)
                manifest["screening_decision_counts"] = df["ai_decision"].value_counts().to_dict()
                manifest["screening_truncated_rows"] = int(df["ai_input_truncated"].fillna(False).sum())
                manifest["screening_model"] = model
                manifest["openai_key_source"] = key_source
                save_manifest(path, manifest)
                rows = df.head(100).to_dict(orient="records")
            except ValueError as exc:
                error = str(exc)
            except OpenAIError:
                current_app.logger.exception("OpenAI screening request failed")
                error = "OpenAI screening failed. Check the key, account access, and record content, then try again."
        if error:
            return render_template("screening.html", manifest=manifest, criteria=criteria, rows=rows, error=error, source_ready=source_ready, fallback_available=fallback_available), 400
    return render_template("screening.html", manifest=manifest, criteria=criteria, rows=rows, error=error, source_ready=source_ready, fallback_available=fallback_available)
