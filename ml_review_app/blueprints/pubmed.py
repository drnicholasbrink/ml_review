"""PubMed search/fetch routes."""

from __future__ import annotations

import requests

from flask import Blueprint, current_app, redirect, render_template, request, url_for

from ..services.credential_service import credential_available, resolve_api_key
from ..services.project_service import invalidate_outputs, load_manifest, project_dir, save_manifest
from ..services.pubmed_service import PubMedResponseError, count_pubmed, fetch_pubmed_records
from ..services.validation_service import parse_bounded_int, validate_date_range, validate_text

bp = Blueprint("pubmed", __name__, url_prefix="/projects/<project_id>/pubmed")


@bp.route("", methods=["GET", "POST"])
def pubmed(project_id: str):
    path = project_dir(current_app.config["RUNTIME_DIR"], project_id)
    manifest = load_manifest(path)
    query_path = path / "search_strategy.txt"
    query = query_path.read_text() if query_path.exists() else ""
    error = None
    count = manifest.get("last_pubmed_count")
    status_code = 200
    fallback_available = credential_available("PUBMED_API_KEY")
    if request.method == "POST":
        action = request.form.get("action")
        try:
            query = validate_text(
                query,
                "PubMed search strategy",
                current_app.config["MAX_SEARCH_STRATEGY_LENGTH"],
                required=True,
            )
            submitted_key = validate_text(
                request.form.get("pubmed_api_key"),
                "PubMed API key",
                current_app.config["MAX_API_KEY_LENGTH"],
            )
            api_key, key_source = resolve_api_key(submitted_key, "PUBMED_API_KEY", required=False)
            mindate, maxdate = validate_date_range(request.form.get("mindate"), request.form.get("maxdate"))
            if action == "count":
                count = count_pubmed(query, api_key=api_key, mindate=mindate, maxdate=maxdate)
                manifest["last_pubmed_count"] = count
                manifest["pubmed_key_source"] = key_source
                save_manifest(path, manifest)
            elif action == "fetch":
                retmax = parse_bounded_int(
                    request.form.get("retmax"),
                    "Maximum records",
                    minimum=1,
                    maximum=current_app.config["MAX_PUBMED_RECORDS"],
                    default=500,
                )
                df = fetch_pubmed_records(query, path / "pubmed_results_complete.csv", api_key=api_key, retmax=retmax, mindate=mindate, maxdate=maxdate)
                invalidate_outputs(manifest, "records")
                manifest["record_source"] = "pubmed"
                manifest.setdefault("files", {})["pubmed_results_complete"] = "pubmed_results_complete.csv"
                manifest["pubmed_rows"] = len(df)
                manifest["pubmed_key_source"] = key_source
                save_manifest(path, manifest)
                return redirect(url_for("embeddings.embeddings", project_id=project_id))
            else:
                raise ValueError("Choose Count records or Fetch records")
        except ValueError as exc:
            error = str(exc)
            status_code = 400
        except requests.RequestException:
            error = "PubMed could not be reached. Check the connection and try again."
            status_code = 502
        except (PubMedResponseError, KeyError, TypeError):
            current_app.logger.exception("Unexpected PubMed response")
            error = "PubMed returned an unexpected response. Try again later."
            status_code = 502
    return render_template("pubmed.html", manifest=manifest, query=query, count=count, error=error, fallback_available=fallback_available), status_code
