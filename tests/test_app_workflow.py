from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd

from ml_review_app import create_app
from ml_review_app.config import TestConfig
from ml_review_app.services.project_service import load_manifest


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "review_records.csv"


def make_app(tmp_path: Path, **overrides):
    settings = {"RUNTIME_DIR": tmp_path / "runtime", **overrides}
    config = type("WorkflowTestConfig", (TestConfig,), settings)
    return create_app(config)


def create_project(client) -> tuple[str, str]:
    response = client.post("/projects", data={"name": "Workflow review"})
    assert response.status_code == 302
    setup_url = response.headers["Location"]
    project_id = setup_url.split("/")[2]
    return project_id, setup_url


def test_pages_and_prerequisite_messages(tmp_path: Path):
    app = make_app(tmp_path)
    client = app.test_client()

    assert client.get("/").status_code == 200
    assert client.get("/health").json == {"status": "ok"}
    assert client.get("/projects").status_code == 200
    too_long_name = client.post("/projects", data={"name": "x" * 201})
    assert too_long_name.status_code == 400
    assert b"Project name must be 200 characters or fewer" in too_long_name.data

    project_id, setup_url = create_project(client)
    dashboard = client.get(f"/projects/{project_id}")
    assert dashboard.status_code == 200
    assert b"Project overview" in dashboard.data
    assert b"Add a search strategy in Review setup" in dashboard.data
    assert b"CSV import" in dashboard.data

    setup_page = client.get(setup_url)
    assert b"Save and continue to PubMed" in setup_page.data
    assert b"Use PubMed search" not in setup_page.data
    assert b"Upload CSV</a>" not in setup_page.data

    oversized_setup = client.post(
        setup_url,
        data={"search_strategy": "x" * 10_001, "inclusion_criteria": "Include studies."},
    )
    assert oversized_setup.status_code == 400
    assert b"must be 10,000 characters or fewer" in oversized_setup.data

    assert client.get(f"/projects/{project_id}/embeddings").status_code == 200
    embeddings = client.post(f"/projects/{project_id}/embeddings")
    assert embeddings.status_code == 400
    assert b"prepare a CSV before generating embeddings" in embeddings.data

    clustering = client.post(f"/projects/{project_id}/clustering", data={"n_clusters": "2"})
    assert clustering.status_code == 400
    assert b"Generate embeddings before running clustering" in clustering.data

    invalid_cluster_count = client.post(f"/projects/{project_id}/clustering", data={"n_clusters": "21"})
    assert invalid_cluster_count.status_code == 400
    assert b"Cluster count must be between 1 and 20" in invalid_cluster_count.data

    screening = client.post(f"/projects/{project_id}/screening")
    assert screening.status_code == 400
    assert b"Select or deduplicate records before running screening" in screening.data

    selection = client.post(f"/projects/{project_id}/clustering/select", follow_redirects=True)
    assert selection.status_code == 200
    assert b"Run clustering before selecting clusters" in selection.data

    missing_mapping = client.post(f"/projects/{project_id}/import/map", follow_redirects=True)
    assert missing_mapping.status_code == 200
    assert b"Upload a CSV before mapping columns" in missing_mapping.data

    assert client.get("/projects/not-a-project").status_code == 404


def test_form_validation_errors_are_actionable(tmp_path: Path):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)

    missing_criteria = client.post(
        setup_url,
        data={"search_strategy": "pregnancy AND heat", "inclusion_criteria": ""},
    )
    assert missing_criteria.status_code == 400
    assert b"Inclusion/exclusion criteria is required" in missing_criteria.data

    valid_setup = client.post(
        setup_url,
        data={"search_strategy": "pregnancy AND heat", "inclusion_criteria": "Include maternal outcomes."},
    )
    assert valid_setup.status_code == 302
    assert valid_setup.headers["Location"].endswith(f"/projects/{project_id}/pubmed")

    invalid_dates = client.post(
        f"/projects/{project_id}/pubmed",
        data={"action": "count", "mindate": "2025/12/31", "maxdate": "2025/01/01"},
    )
    assert invalid_dates.status_code == 400
    assert b"Start date must not be after end date" in invalid_dates.data

    missing_upload = client.post(
        f"/projects/{project_id}/import",
        data={},
        content_type="multipart/form-data",
    )
    assert missing_upload.status_code == 400
    assert b"Choose a CSV file to upload" in missing_upload.data

    wrong_extension = client.post(
        f"/projects/{project_id}/import",
        data={"csv_file": (BytesIO(b"id,title\n1,Example\n"), "records.txt")},
        content_type="multipart/form-data",
    )
    assert wrong_extension.status_code == 400
    assert b"Upload must be a .csv file" in wrong_extension.data

    empty_upload = client.post(
        f"/projects/{project_id}/import",
        data={"csv_file": (BytesIO(b""), "records.csv")},
        content_type="multipart/form-data",
    )
    assert empty_upload.status_code == 400
    assert b"empty or is not a readable CSV" in empty_upload.data

    small_upload_app = make_app(tmp_path / "small-upload", MAX_CONTENT_LENGTH=64)
    small_upload_client = small_upload_app.test_client()
    small_project_id, _ = create_project(small_upload_client)
    too_large_upload = small_upload_client.post(
        f"/projects/{small_project_id}/import",
        data={"csv_file": (BytesIO(b"id,title\n1," + b"x" * 200 + b"\n"), "records.csv")},
        content_type="multipart/form-data",
    )
    assert too_large_upload.status_code == 413
    assert b"exceeds the 64 byte limit" in too_large_upload.data


def test_csv_workflow_end_to_end(tmp_path: Path, monkeypatch):
    def fake_add_embeddings(input_csv, output_csv, **_kwargs):
        df = pd.read_csv(input_csv)
        df["Embedding"] = [f"[{index}.0, 1.0, 2.0]" for index in range(len(df))]
        df["EmbeddingModel"] = "text-embedding-3-small"
        df["EmbeddingInputTruncated"] = False
        df.to_csv(output_csv, index=False)
        return df

    def fake_screen_csv(input_csv, _criteria, output_csv, **_kwargs):
        df = pd.read_csv(input_csv)
        df["ai_decision"] = "uncertain"
        df["ai_confidence"] = "low"
        df["ai_exclusion_reason"] = None
        df["ai_reasoning"] = "Requires human review."
        df["ai_population_match"] = False
        df["ai_exposure_match"] = False
        df["ai_outcome_match"] = False
        df["ai_study_design_appropriate"] = False
        df["ai_input_truncated"] = False
        df.to_csv(output_csv, index=False)
        return df

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ml_review_app.blueprints.embeddings.add_embeddings", fake_add_embeddings)
    monkeypatch.setattr("ml_review_app.blueprints.screening.screen_csv", fake_screen_csv)
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)

    response = client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND heat",
            "inclusion_criteria": "Include maternal pregnancy heat exposure outcomes.",
        },
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/projects/{project_id}/pubmed")
    assert client.get(f"/projects/{project_id}/pubmed").status_code == 200

    dashboard = client.get(f"/projects/{project_id}")
    assert dashboard.status_code == 200
    assert b"Review setup" in dashboard.data
    assert b"Complete" in dashboard.data
    assert b"Generate embeddings first" in dashboard.data

    response = client.post(
        f"/projects/{project_id}/import",
        data={"csv_file": (BytesIO(FIXTURE_PATH.read_bytes()), "review_records.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"6 rows" in response.data
    assert b"source_id" in response.data

    response = client.post(
        f"/projects/{project_id}/import/map",
        data={
            "RecordID": "source_id",
            "Title": "article_title",
            "Abstract": "summary",
            "Authors": "authors",
            "Date": "published",
            "Journal": "journal",
            "DOI": "doi",
        },
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/import/deduplicate")

    response = client.post(
        f"/projects/{project_id}/import/deduplicate",
        data={"match_columns": "Title"},
    )
    assert response.status_code == 200
    assert b"Download deduplicated CSV" in response.data

    project_path = tmp_path / "runtime" / "projects" / project_id
    deduplicated = pd.read_csv(project_path / "deduplicated_records.csv")
    assert len(deduplicated) == 5

    response = client.get(f"/projects/{project_id}/exports/deduplicated_records.csv")
    assert response.status_code == 200
    assert response.headers["Content-Disposition"].startswith("attachment;")

    response = client.post(
        f"/projects/{project_id}/embeddings",
        data={"model": "text-embedding-3-small", "batch_size": "100"},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/clustering")

    response = client.post(
        f"/projects/{project_id}/clustering",
        data={"n_clusters": "2"},
    )
    assert response.status_code == 200
    assert b"Elbow scores" in response.data
    assert b"Refine selected clusters" in response.data

    manifest = load_manifest(project_path)
    clustered = pd.read_csv(project_path / manifest["files"]["labeled_clusters"])
    selected_clusters = sorted(clustered["Cluster"].unique())
    point = client.get(f"/projects/{project_id}/clustering/points/{clustered.loc[0, 'PointID']}")
    assert point.status_code == 200
    assert point.json["Abstract"]

    subcluster = client.post(
        f"/projects/{project_id}/clustering/subcluster",
        data={
            "clusters": [str(value) for value in selected_clusters],
            "n_clusters": "2",
            "random_state": "42",
        },
    )
    assert subcluster.status_code == 302
    manifest = load_manifest(project_path)
    assert manifest["active_clustering_run"] == "run_002"
    assert (project_path / "visualizations" / "clustering_runs" / "run_001.csv").exists()
    assert (project_path / "visualizations" / "clustering_runs" / "run_002.csv").exists()

    back = client.post(f"/projects/{project_id}/clustering/back")
    assert back.status_code == 302
    manifest = load_manifest(project_path)
    assert manifest["active_clustering_run"] == "run_001"

    response = client.post(
        f"/projects/{project_id}/clustering/select",
        data={"clusters": [str(value) for value in selected_clusters]},
    )
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/screening")

    response = client.post(
        f"/projects/{project_id}/screening",
        data={"model": "gpt-5.6-luna", "resume": "on"},
    )
    assert response.status_code == 200
    assert b"Decision counts" in response.data
    assert b"human reviewer" in response.data

    screened = pd.read_csv(project_path / "ai_screening_full_results.csv")
    assert len(screened) == 5
    assert set(screened["ai_decision"]) <= {"include", "exclude", "uncertain"}
    assert screened["ai_confidence"].notna().all()

    download = client.get(f"/projects/{project_id}/exports/ai_screening_full_results.csv")
    assert download.status_code == 200
    assert download.headers["Content-Disposition"].startswith("attachment;")

    changed_criteria = client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND heat",
            "inclusion_criteria": "Updated criteria that require human revalidation.",
        },
    )
    assert changed_criteria.status_code == 302
    manifest = load_manifest(project_path)
    assert "ai_screening_full_results" not in manifest["files"]
    assert "labeled_clusters" in manifest["files"]
    assert client.get(f"/projects/{project_id}/exports/ai_screening_full_results.csv").status_code == 404

    replacement_upload = client.post(
        f"/projects/{project_id}/import",
        data={"csv_file": (BytesIO(FIXTURE_PATH.read_bytes()), "replacement.csv")},
        content_type="multipart/form-data",
    )
    assert replacement_upload.status_code == 200
    manifest = load_manifest(project_path)
    assert "normalized_records" not in manifest["files"]
    assert "deduplicated_records" not in manifest["files"]
    assert "embeddings" not in manifest["files"]
    stale_download = client.get(f"/projects/{project_id}/exports/deduplicated_records.csv")
    assert stale_download.status_code == 404
