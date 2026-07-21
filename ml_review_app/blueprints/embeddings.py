"""Embedding routes."""

from __future__ import annotations

from flask import Blueprint, current_app, redirect, render_template, request, url_for
from openai import OpenAIError

from ..services.credential_service import credential_available, resolve_api_key
from ..services.embedding_service import add_embeddings
from ..services.project_service import invalidate_outputs, load_manifest, project_dir, save_manifest
from ..services.validation_service import parse_bounded_int, validate_text

bp = Blueprint("embeddings", __name__, url_prefix="/projects/<project_id>/embeddings")


@bp.route("", methods=["GET", "POST"])
def embeddings(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    files = manifest.get("files", {})
    if manifest.get("record_source") == "pubmed":
        candidates = [files.get("pubmed_results_complete")]
    elif manifest.get("record_source") == "csv":
        candidates = [files.get("deduplicated_records"), files.get("normalized_records")]
    else:
        candidates = [files.get("deduplicated_records"), files.get("normalized_records"), files.get("pubmed_results_complete")]
    source_name = next((name for name in candidates if name and (path / name).exists()), None)
    error = None
    fallback_available = credential_available("OPENAI_API_KEY")
    if request.method == "POST":
        if not source_name or not (path / source_name).exists():
            error = "Fetch PubMed records or prepare a CSV before generating embeddings"
        else:
            try:
                submitted_key = validate_text(
                    request.form.get("openai_api_key"),
                    "OpenAI API key",
                    current_app.config["MAX_API_KEY_LENGTH"],
                )
                api_key, key_source = resolve_api_key(submitted_key, "OPENAI_API_KEY", required=True)
                model = validate_text(request.form.get("model"), "Embedding model", 100, required=True)
                allowed_models = {"text-embedding-3-small", "text-embedding-3-large"}
                if model not in allowed_models:
                    raise ValueError("Choose a supported OpenAI embedding model")
                batch_size = parse_bounded_int(
                    request.form.get("batch_size"),
                    "Batch size",
                    minimum=1,
                    maximum=2048,
                    default=current_app.config["DEFAULT_EMBEDDING_BATCH_SIZE"],
                )
                df = add_embeddings(
                    path / source_name,
                    path / "pubmed_results_with_embeddings.csv",
                    api_key=api_key,
                    model=model,
                    batch_size=batch_size,
                )
                invalidate_outputs(manifest, "embeddings")
                manifest["embedding_source"] = source_name
                manifest.setdefault("files", {})["embeddings"] = "pubmed_results_with_embeddings.csv"
                manifest["embedding_rows"] = len(df)
                manifest["embedding_model"] = model
                manifest["embedding_truncated_rows"] = int(df["EmbeddingInputTruncated"].sum())
                manifest["openai_key_source"] = key_source
                save_manifest(path, manifest)
                return redirect(url_for("clustering.clustering", project_id=project_id))
            except ValueError as exc:
                error = str(exc)
            except OpenAIError:
                current_app.logger.exception("OpenAI embedding request failed")
                error = "OpenAI could not generate embeddings. Check the key, account access, and input, then try again."
        return render_template(
            "embeddings.html",
            manifest=manifest,
            source_name=source_name,
            error=error,
            fallback_available=fallback_available,
        ), 400
    return render_template(
        "embeddings.html",
        manifest=manifest,
        source_name=source_name,
        error=error,
        fallback_available=fallback_available,
    )
