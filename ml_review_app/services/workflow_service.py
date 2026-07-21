"""Project workflow state used by the dashboard and navigation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def build_workflow_steps(project_path: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe every workflow section and its current dependency state."""

    files = manifest.get("files", {})
    def file_ready(key: str) -> bool:
        filename = files.get(key)
        return bool(filename and (project_path / filename).is_file())

    search_ready = bool(
        files.get("search_strategy")
        and (project_path / files["search_strategy"]).exists()
        and (project_path / files["search_strategy"]).read_text().strip()
    )
    criteria_ready = bool(
        files.get("inclusion_criteria")
        and (project_path / files["inclusion_criteria"]).exists()
        and (project_path / files["inclusion_criteria"]).read_text().strip()
    )
    normalized_ready = file_ready("normalized_records")
    record_source_ready = any(
        files.get(key) and (project_path / files[key]).exists()
        for key in ("deduplicated_records", "normalized_records", "pubmed_results_complete")
    )
    embeddings_ready = file_ready("embeddings")
    clusters_ready = file_ready("labeled_clusters")
    screening_source_ready = any(
        files.get(key) and (project_path / files[key]).exists()
        for key in ("selected_records", "labeled_clusters", "deduplicated_records")
    )
    screening_ready = file_ready("ai_screening_full_results")

    return [
        {
            "id": "setup",
            "number": "1",
            "title": "Review setup",
            "description": "Define the search strategy and review criteria.",
            "endpoint": "projects.setup",
            "enabled": True,
            "complete": search_ready and criteria_ready,
            "blocked_reason": "",
        },
        {
            "id": "pubmed",
            "number": "2A",
            "title": "PubMed search",
            "description": "Count and fetch records from the saved query.",
            "endpoint": "pubmed.pubmed",
            "enabled": search_ready,
            "complete": file_ready("pubmed_results_complete"),
            "blocked_reason": "Add a search strategy in Review setup.",
        },
        {
            "id": "import",
            "number": "2B",
            "title": "CSV import",
            "description": "Upload and map records from another source.",
            "endpoint": "imports.import_data",
            "enabled": True,
            "complete": normalized_ready,
            "blocked_reason": "",
        },
        {
            "id": "deduplicate",
            "number": "3",
            "title": "Deduplicate",
            "description": "Review and collapse duplicate imported records.",
            "endpoint": "imports.deduplicate",
            "enabled": normalized_ready,
            "complete": file_ready("deduplicated_records"),
            "blocked_reason": "Upload and map a CSV first.",
        },
        {
            "id": "embeddings",
            "number": "4",
            "title": "Embeddings",
            "description": "Generate vectors from the selected record source.",
            "endpoint": "embeddings.embeddings",
            "enabled": record_source_ready,
            "complete": embeddings_ready,
            "blocked_reason": "Fetch PubMed records or prepare a CSV first.",
        },
        {
            "id": "clustering",
            "number": "5",
            "title": "Clustering",
            "description": "Explore the records and select relevant clusters.",
            "endpoint": "clustering.clustering",
            "enabled": embeddings_ready,
            "complete": clusters_ready,
            "blocked_reason": "Generate embeddings first.",
        },
        {
            "id": "screening",
            "number": "6",
            "title": "Screening",
            "description": "Run and review structured screening decisions.",
            "endpoint": "screening.screening",
            "enabled": screening_source_ready and criteria_ready,
            "complete": screening_ready,
            "blocked_reason": "Select records and save inclusion criteria first.",
        },
        {
            "id": "evaluation",
            "number": "7",
            "title": "Evaluation",
            "description": "Explore screening instruments and compare with human decisions.",
            "endpoint": "evaluation.evaluation",
            "enabled": screening_ready,
            "complete": screening_ready,
            "blocked_reason": "Run screening first.",
        },
    ]
