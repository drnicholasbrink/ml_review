from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from ml_review_app import create_app
from ml_review_app.config import TestConfig
from ml_review_app.services.clustering_service import load_clustering_state
from ml_review_app.services.project_service import load_manifest, save_manifest


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
    assert client.get("/ready").json == {"status": "ready", "runtime_writable": True}
    plotly_bundle = client.get("/vendor/plotly.min.js")
    assert plotly_bundle.status_code == 200
    assert plotly_bundle.mimetype == "text/javascript"
    home = client.get("/")
    assert b"cdn.plot.ly" not in home.data
    assert home.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in home.headers["Content-Security-Policy"]
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
    bundle = client.get(f"/projects/{project_id}/exports/publication-bundle.zip")
    assert bundle.status_code == 200
    with zipfile.ZipFile(BytesIO(bundle.data)) as archive:
        assert {"audit_manifest.json", "README.txt"} <= set(archive.namelist())

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

    invalid_cluster_count = client.post(f"/projects/{project_id}/clustering/finalize", data={"n_clusters": "21"}, follow_redirects=True)
    assert invalid_cluster_count.status_code == 200
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


def test_pubmed_count_preserves_the_approved_fetch_scope(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)
    client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND heat",
            "inclusion_criteria": "Include maternal outcomes.",
        },
    )
    monkeypatch.setattr("ml_review_app.blueprints.pubmed.count_pubmed", lambda *args, **kwargs: 58)

    counted = client.post(
        f"/projects/{project_id}/pubmed",
        data={
            "action": "count",
            "mindate": "2024/01/01",
            "maxdate": "2024/03/31",
            "retmax": "25",
        },
    )
    assert counted.status_code == 200
    assert b'value="2024/01/01"' in counted.data
    assert b'value="2024/03/31"' in counted.data
    assert b'value="25"' in counted.data
    assert b"Scope: 2024/01/01 through 2024/03/31" in counted.data
    assert b"Fetch remains capped at 25 records" in counted.data

    persisted = client.get(f"/projects/{project_id}/pubmed")
    assert b'value="2024/01/01"' in persisted.data
    manifest = load_manifest(tmp_path / "runtime" / "projects" / project_id)
    assert manifest["last_pubmed_parameters"] == {
        "mindate": "2024/01/01",
        "maxdate": "2024/03/31",
        "retmax": 25,
    }

    client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND humidity",
            "inclusion_criteria": "Include maternal outcomes.",
        },
    )
    manifest = load_manifest(tmp_path / "runtime" / "projects" / project_id)
    assert "last_pubmed_parameters" not in manifest


def test_pubmed_fetch_runs_as_a_tracked_background_task(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)
    client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND heat",
            "inclusion_criteria": "Include maternal outcomes.",
        },
    )

    def fake_fetch(_query, output, **kwargs):
        frame = pd.DataFrame(
            [{"RecordID": "1", "PMID": "1", "Title": "Fixture", "Abstract": "Abstract"}]
        )
        kwargs["progress_callback"](0, 1, "Found one record")
        frame.to_csv(output, index=False)
        kwargs["progress_callback"](1, 1, "Saved")
        return frame

    monkeypatch.setattr("ml_review_app.blueprints.pubmed.fetch_pubmed_records", fake_fetch)
    response = client.post(
        f"/projects/{project_id}/pubmed",
        data={"action": "fetch", "retmax": "1"},
    )
    assert response.status_code == 302
    assert "/pubmed?task=" in response.headers["Location"]
    completed = client.get(response.headers["Location"])
    assert b"Completed successfully" in completed.data
    assert b"Task history" in completed.data
    manifest = load_manifest(tmp_path / "runtime" / "projects" / project_id)
    assert manifest["record_source"] == "pubmed"
    assert manifest["pubmed_rows"] == 1


def test_extraction_route_runs_a_resumable_test_and_exposes_exports(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)
    client.post(
        setup_url,
        data={
            "search_strategy": "pregnancy AND heat",
            "inclusion_criteria": "Include primary pregnancy heat studies.",
        },
    )
    project_path = tmp_path / "runtime" / "projects" / project_id
    screened = pd.DataFrame(
        {
            "RecordID": ["one", "two"],
            "PMID": ["1", "2"],
            "Title": ["Included study", "Excluded study"],
            "Abstract": ["Included abstract", "Excluded abstract"],
            "ai_decision": ["include", "exclude"],
            "ai_confidence": ["high", "high"],
        }
    )
    screened.to_csv(project_path / "ai_screening_full_results.csv", index=False)
    manifest = load_manifest(project_path)
    manifest["files"]["ai_screening_full_results"] = "ai_screening_full_results.csv"
    save_manifest(project_path, manifest)

    def fake_extract(_source, _criteria, output, **_kwargs):
        result = screened.iloc[[0]].copy()
        result["extraction_record_key"] = "recordid:one"
        result["country"] = "South Africa"
        result["study_design"] = "cohort"
        result["sample_size"] = "1,000"
        result["extraction_confidence"] = "medium"
        result["data_completeness"] = "partial"
        result["key_finding_summary"] = "Heat exposure was associated with miscarriage."
        result["effect_estimates"] = '[{"estimate_type":"odds ratio","value":"1.20"}]'
        result["notes"] = "Validate against full text."
        result["extraction_json"] = '{"population_description":"Pregnant participants","location":{"country":"South Africa"}}'
        result["extraction_model"] = "gpt-5.6-luna"
        result["extraction_timestamp"] = "2026-07-21T00:00:00+00:00"
        result["text_source"] = "abstract"
        result.to_csv(output, index=False)
        return result, 1

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("ml_review_app.blueprints.extraction.extract_csv", fake_extract)
    page = client.get(f"/projects/{project_id}/extraction")
    assert page.status_code == 200
    assert b"1</strong><span>Eligible records" in page.data
    response = client.post(
        f"/projects/{project_id}/extraction",
        data={"model": "gpt-5.6-luna", "scope": "test", "test_limit": "1", "resume": "on"},
    )
    assert response.status_code == 302
    assert "/extraction?task=" in response.headers["Location"]
    completed = client.get(response.headers["Location"])
    assert b"Extraction exports" in completed.data
    assert b"Validate against full text" in completed.data
    assert b"Completed successfully" in completed.data
    manifest = load_manifest(project_path)
    assert manifest["extraction_rows"] == 1
    assert manifest["files"]["effect_estimates"] == "effect_estimates.csv"
    dashboard = client.get(f"/projects/{project_id}")
    assert b"Extraction" in dashboard.data
    assert b"Complete" in dashboard.data
    bundle = client.get(f"/projects/{project_id}/exports/publication-bundle.zip")
    with zipfile.ZipFile(BytesIO(bundle.data)) as archive:
        names = set(archive.namelist())
        assert "artifacts/ai_extraction_full_results.csv" in names
        assert "artifacts/effect_estimates.csv" in names
        audit = json.loads(archive.read("audit_manifest.json"))
        assert audit["extraction_model"] == "gpt-5.6-luna"


def test_cluster_search_route_is_read_only_validated_and_scoped_to_active_run(tmp_path: Path):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, _setup_url = create_project(client)
    project_path = tmp_path / "runtime" / "projects" / project_id
    search_url = f"/projects/{project_id}/clustering/search"

    inactive = client.get(search_url)
    assert inactive.status_code == 404
    assert inactive.json == {"error": "Run clustering before searching the cluster explorer"}

    run_directory = project_path / "visualizations" / "clustering_runs"
    run_directory.mkdir(parents=True, exist_ok=True)
    run_one = pd.DataFrame(
        [
            {
                "PointID": "point-heat",
                "RecordID": "R-HEAT",
                "PMID": "10001",
                "DOI": "10.1000/heat",
                "Title": "Heat exposure in pregnancy",
                "Abstract": "PRIVATE ABSTRACT maternal outcomes",
                "TSNE_1": 0.0,
                "TSNE_2": 1.0,
                "Cluster": 0,
            },
            {
                "PointID": "point-air",
                "RecordID": "R-AIR",
                "PMID": "10002",
                "DOI": "10.1000/air",
                "Title": "Air pollution and asthma",
                "Abstract": "PRIVATE ABSTRACT respiratory outcomes",
                "TSNE_1": 1.0,
                "TSNE_2": 0.0,
                "Cluster": 1,
            },
        ]
    )
    run_two = run_one.iloc[[1]].copy()
    run_one.to_csv(run_directory / "run_001.csv", index=False)
    run_two.to_csv(run_directory / "run_002.csv", index=False)

    def run_metadata(run_id: str, output_file: str, parent_run_id: str | None, row_count: int, depth: int):
        return {
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "depth": depth,
            "row_count": row_count,
            "n_clusters": row_count,
            "output_file": output_file,
            "selected_parent_clusters": [] if parent_run_id is None else [1],
            "method": "test projection",
            "perplexity": None,
            "random_state": 42,
            "sklearn_version": "test",
            "elbow_scores": [{"k": 1, "wcss": 0.0}],
        }

    state = {
        "active_run_id": "run_001",
        "draft": None,
        "runs": [
            run_metadata("run_001", "visualizations/clustering_runs/run_001.csv", None, 2, 0),
            run_metadata("run_002", "visualizations/clustering_runs/run_002.csv", "run_001", 1, 1),
        ],
    }
    state_path = project_path / "clustering_state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    manifest = load_manifest(project_path)
    manifest["files"].update(
        {
            "embeddings": "visualizations/clustering_runs/run_001.csv",
            "labeled_clusters": "visualizations/clustering_runs/run_001.csv",
            "clustering_state": "clustering_state.json",
        }
    )
    manifest["active_clustering_run"] = "run_001"
    save_manifest(project_path, manifest)

    state_before_search = state_path.read_text()
    response = client.get(
        search_url,
        query_string={"keywords": " HEAT, maternal outcomes, heat ", "mode": "all", "study": "10001"},
    )
    assert response.status_code == 200
    assert set(response.json) == {
        "run_id",
        "keywords",
        "match_mode",
        "study_query",
        "clusters",
        "keyword_matched_point_ids",
        "study_matched_point_ids",
        "study_matches",
    }
    assert response.json["run_id"] == "run_001"
    assert response.json["keywords"] == ["heat", "maternal outcomes"]
    assert response.json["keyword_matched_point_ids"] == ["point-heat"]
    assert response.json["study_matched_point_ids"] == ["point-heat"]
    assert response.json["study_matches"][0]["match_type"] == "identifier_exact"
    assert "PRIVATE ABSTRACT" not in response.get_data(as_text=True)
    assert "Abstract" not in response.get_data(as_text=True)
    assert state_path.read_text() == state_before_search

    no_matches = client.get(search_url, query_string={"keywords": "xylophone-quasar", "study": "zzzzzzzzzzzz"})
    assert no_matches.status_code == 200
    assert no_matches.json["keyword_matched_point_ids"] == []
    assert no_matches.json["study_matches"] == []

    too_many = client.get(search_url, query_string={"keywords": ",".join(f"term-{index}" for index in range(21))})
    assert too_many.status_code == 400
    assert too_many.json["error"] == "Enter no more than 20 keywords"
    long_keyword = client.get(search_url, query_string={"keywords": "k" * 101})
    assert long_keyword.status_code == 400
    assert "100 characters or fewer" in long_keyword.json["error"]
    long_study = client.get(search_url, query_string={"study": "s" * 201})
    assert long_study.status_code == 400
    assert "200 characters or fewer" in long_study.json["error"]
    invalid_mode = client.get(search_url, query_string={"mode": "sometimes"})
    assert invalid_mode.status_code == 400
    assert "either any or all" in invalid_mode.json["error"]

    clustering_page = client.get(f"/projects/{project_id}/clustering")
    assert clustering_page.status_code == 200
    assert b"Search and compare clusters" in clustering_page.data
    assert b'id="cluster-search-form"' in clustering_page.data
    assert b'class="cluster-table"' in clustering_page.data
    assert clustering_page.data.count(b'name="clusters"') == 2
    assert b"Plotly.react" in clustering_page.data
    assert b"Keyword match" in clustering_page.data
    assert b"Study match" in clustering_page.data

    state["active_run_id"] = "run_002"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True))
    branch_search = client.get(search_url, query_string={"keywords": "heat", "study": "10001"})
    assert branch_search.status_code == 200
    assert branch_search.json["run_id"] == "run_002"
    assert branch_search.json["keyword_matched_point_ids"] == []
    assert branch_search.json["study_matches"] == []
    branch_page = client.get(f"/projects/{project_id}/clustering")
    assert branch_page.data.count(b'name="clusters"') == 1
    assert b'id="cluster-keywords" name="keywords"' in branch_page.data


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

    reopened_import = client.get(f"/projects/{project_id}/import")
    assert b'<option value="source_id" selected>source_id</option>' in reopened_import.data
    assert b'<option value="article_title" selected>article_title</option>' in reopened_import.data

    response = client.post(
        f"/projects/{project_id}/import/deduplicate",
        data={"match_columns": "Title"},
    )
    assert response.status_code == 200
    assert b"Download deduplicated CSV" in response.data
    assert b"Records retained" in response.data
    assert b"Duplicate report preview" in response.data

    reopened_deduplication = client.get(f"/projects/{project_id}/import/deduplicate")
    assert b'name="match_columns" value="Title" checked' in reopened_deduplication.data
    assert b"Duplicate report preview" in reopened_deduplication.data

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
    assert "/embeddings?task=" in response.headers["Location"]
    completed_embeddings = client.get(response.headers["Location"])
    assert b"Completed successfully" in completed_embeddings.data

    initial_clustering = client.get(f"/projects/{project_id}/clustering")
    assert b"Analyze new root WCSS" in initial_clustering.data
    assert b"Cluster count after reviewing WCSS" not in initial_clustering.data

    response = client.post(
        f"/projects/{project_id}/clustering",
        data={"random_state": "42"},
    )
    assert response.status_code == 200
    assert b"WCSS before choosing K" in response.data
    assert b"Cluster count after reviewing WCSS" in response.data
    manifest = load_manifest(project_path)
    assert "labeled_clusters" not in manifest["files"]

    finalize_root = client.post(
        f"/projects/{project_id}/clustering/finalize",
        data={"n_clusters": "2"},
    )
    assert finalize_root.status_code == 302
    clustered_page = client.get(f"/projects/{project_id}/clustering")
    assert b"Create a child branch from run_001" in clustered_page.data

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
            "random_state": "42",
        },
    )
    assert subcluster.status_code == 302
    state = load_clustering_state(project_path)
    assert state["active_run_id"] == "run_001"
    assert state["draft"]["parent_run_id"] == "run_001"

    finalize_subcluster = client.post(
        f"/projects/{project_id}/clustering/finalize",
        data={"n_clusters": "2"},
    )
    assert finalize_subcluster.status_code == 302
    manifest = load_manifest(project_path)
    assert manifest["active_clustering_run"] == "run_002"
    assert (project_path / "visualizations" / "clustering_runs" / "run_001.csv").exists()
    assert (project_path / "visualizations" / "clustering_runs" / "run_002.csv").exists()

    back = client.post(f"/projects/{project_id}/clustering/back")
    assert back.status_code == 302
    manifest = load_manifest(project_path)
    assert manifest["active_clustering_run"] == "run_001"

    sibling_analysis = client.post(
        f"/projects/{project_id}/clustering/subcluster",
        data={
            "clusters": [str(value) for value in selected_clusters],
            "random_state": "7",
        },
    )
    assert sibling_analysis.status_code == 302
    sibling_finalize = client.post(
        f"/projects/{project_id}/clustering/finalize",
        data={"n_clusters": "2"},
    )
    assert sibling_finalize.status_code == 302
    state = load_clustering_state(project_path)
    assert state["active_run_id"] == "run_003"
    assert [run["parent_run_id"] for run in state["runs"]] == [None, "run_001", "run_001"]
    assert state["runs"][1]["random_state"] == 42
    assert state["runs"][2]["random_state"] == 7

    back_to_seed = client.post(f"/projects/{project_id}/clustering/back")
    assert back_to_seed.status_code == 302
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
    assert response.status_code == 302
    assert "/screening?task=" in response.headers["Location"]
    completed_screening = client.get(response.headers["Location"])
    assert b"Screening status" in completed_screening.data
    assert b"Human adjudication queue" in completed_screening.data
    assert b'id="screening-table"' in completed_screening.data
    assert b"<pre>[{" not in completed_screening.data

    screened = pd.read_csv(project_path / "ai_screening_full_results.csv")
    assert len(screened) == 5
    assert set(screened["ai_decision"]) <= {"include", "exclude", "uncertain"}
    assert screened["ai_confidence"].notna().all()

    screening_page = client.get(f"/projects/{project_id}/screening?review_filter=needs_review")
    assert screening_page.status_code == 200
    assert b"Required review pending" in screening_page.data
    record_key = screening_page.data.split(b'name="record_key" value="', 1)[1].split(b'"', 1)[0].decode()
    adjudicated = client.post(
        f"/projects/{project_id}/screening",
        data={
            "action": "review", "record_key": record_key, "human_decision": "include",
            "human_note": "Validated against the source text.", "review_filter": "all", "page": "1",
        },
        follow_redirects=True,
    )
    assert adjudicated.status_code == 200
    assert b"Human screening decision saved" in adjudicated.data
    reviewed = pd.read_csv(project_path / "human_screening_reviewed_results.csv")
    assert reviewed["ai_decision"].notna().all()
    assert reviewed["final_decision_source"].eq("human").sum() == 1
    assert client.get(f"/projects/{project_id}/exports/human_screening_reviewed_results.csv").status_code == 200

    evaluation = client.get(f"/projects/{project_id}/evaluation")
    assert evaluation.status_code == 200
    assert b"Screening funnel" in evaluation.data
    assert b'id="criteria-chart"' in evaluation.data
    human_reference = pd.DataFrame(
        {"Title": screened["Title"], "status": ["include"] + ["exclude"] * (len(screened) - 1)}
    ).to_csv(index=False).encode()
    human_evaluation = client.post(
        f"/projects/{project_id}/evaluation",
        data={
            "human_screening_csv": (BytesIO(human_reference), "human-screening.csv"),
            "threshold": "85",
            "uncertain_is_positive": "on",
        },
        content_type="multipart/form-data",
    )
    assert human_evaluation.status_code == 302
    evaluation_page = client.get(f"/projects/{project_id}/evaluation")
    assert b"Human-reference metrics" in evaluation_page.data
    assert b"Download comparison CSV" in evaluation_page.data

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


def test_long_content_renders_in_responsive_containers_without_truncation(tmp_path: Path):
    app = make_app(tmp_path)
    client = app.test_client()
    project_id, setup_url = create_project(client)
    project_path = tmp_path / "runtime" / "projects" / project_id
    long_token = "unbroken" + "x" * 500
    long_url = f"https://example.test/articles/{long_token}?source=systematic-review"
    long_doi = f"10.1234/{long_token}"
    long_title = f"A deliberately long title for responsive review tables {long_token} {long_url}"
    long_abstract = f"First abstract paragraph with {long_doi}.\n\nSecond paragraph with {long_url}."
    long_rationale = f"First rationale paragraph.\n\nSecond rationale paragraph contains {long_token}."
    long_reason = f"outside_scope_{long_token}"

    setup = client.post(
        setup_url,
        data={
            "search_strategy": f'("pregnancy"[Title]) AND {long_token}',
            "inclusion_criteria": f"Include relevant studies.\n\nIdentifier: {long_doi}",
        },
    )
    assert setup.status_code == 302

    long_column = f"source_{long_token}"
    csv_bytes = pd.DataFrame([{long_column: long_url, "article_title": long_title, "doi": long_doi}]).to_csv(index=False).encode()
    imported = client.post(
        f"/projects/{project_id}/import",
        data={"csv_file": (BytesIO(csv_bytes), f"review-export-{'f' * 120}.csv")},
        content_type="multipart/form-data",
    )
    assert imported.status_code == 200
    assert b'class="table-scroll"' in imported.data
    assert b'class="preview-table"' in imported.data
    assert long_column.encode() in imported.data
    assert long_url.encode() in imported.data

    screened = pd.DataFrame(
        [
            {
                "RecordID": long_token,
                "PMID": "12345678",
                "Title": long_title,
                "Abstract": long_abstract,
                "DOI": long_doi,
                "ai_decision": "exclude",
                "ai_confidence": "low",
                "ai_exclusion_category": "population",
                "ai_exclusion_reason": long_reason,
                "ai_population_match": False,
                "ai_exposure_match": True,
                "ai_outcome_match": False,
                "ai_study_design_appropriate": True,
                "ai_reasoning": long_rationale,
                "ai_input_truncated": False,
                "TSNE_1": 0.5,
                "TSNE_2": -0.5,
            }
        ]
    )
    screened.to_csv(project_path / "ai_screening_full_results.csv", index=False)
    screened.to_csv(project_path / "deduplicated_records.csv", index=False)
    comparison = pd.DataFrame(
        [
            {
                "ai_title": long_title,
                "ai_decision": "exclude",
                "human_title": long_title,
                "match_score": 100,
                "match_status": f"matched_{long_token}",
            }
        ]
    )
    comparison.to_csv(project_path / "human_evaluation_comparison.csv", index=False)
    manifest = load_manifest(project_path)
    manifest["files"].update(
        {
            "deduplicated_records": "deduplicated_records.csv",
            "ai_screening_full_results": "ai_screening_full_results.csv",
            "human_evaluation_comparison": "human_evaluation_comparison.csv",
        }
    )
    manifest["screening_decision_counts"] = {"exclude": 1}
    manifest["human_evaluation"] = {
        "human_file_type": "full decisions",
        "matched_records": 1,
        "match_coverage": 1.0,
        "sensitivity": None,
        "precision": None,
        "f1_score": None,
        "specificity": 1.0,
        "match_threshold": 85,
        "uncertain_is_positive": True,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
        "true_negatives": 1,
    }
    save_manifest(project_path, manifest)

    screening_page = client.get(f"/projects/{project_id}/screening")
    assert screening_page.status_code == 200
    assert b'class="data-table screening-table review-table"' in screening_page.data
    assert b'class="col-instrument" span="4"' in screening_page.data
    assert long_title.encode() in screening_page.data
    assert long_rationale.encode() in screening_page.data
    assert long_doi.encode() in screening_page.data
    assert b"Population" in screening_page.data

    evaluation_page = client.get(f"/projects/{project_id}/evaluation")
    assert evaluation_page.status_code == 200
    assert b'class="data-table comparison-table"' in evaluation_page.data
    assert long_title.encode() in evaluation_page.data
    assert f"matched_{long_token}".encode() in evaluation_page.data
    assert b"automargin:true" in evaluation_page.data
    assert b"Broad exclusion categories" in evaluation_page.data

    run_id = f"run_{long_token}"
    run_output = Path("visualizations/clustering_runs/long-content.csv")
    (project_path / run_output).parent.mkdir(parents=True, exist_ok=True)
    clustered = screened.assign(PointID=long_token, Cluster=0)
    clustered.to_csv(project_path / run_output, index=False)
    state = {
        "active_run_id": run_id,
        "draft": None,
        "runs": [
            {
                "run_id": run_id,
                "parent_run_id": None,
                "depth": 0,
                "row_count": 1,
                "n_clusters": 1,
                "output_file": str(run_output),
                "selected_parent_clusters": [],
                "method": f"projection_{long_token}",
                "perplexity": None,
                "random_state": 42,
                "sklearn_version": "test",
                "elbow_scores": [{"k": 1, "wcss": 0.0}],
            }
        ],
    }
    (project_path / "clustering_state.json").write_text(json.dumps(state))
    manifest = load_manifest(project_path)
    manifest["files"]["embeddings"] = "deduplicated_records.csv"
    manifest["files"]["labeled_clusters"] = str(run_output)
    manifest["active_clustering_run"] = run_id
    save_manifest(project_path, manifest)

    clustering_page = client.get(f"/projects/{project_id}/clustering")
    assert clustering_page.status_code == 200
    assert run_id.encode() in clustering_page.data
    assert b"abstract.className = 'abstract-text'" in clustering_page.data
    assert b"automargin:true" in clustering_page.data
    point = client.get(f"/projects/{project_id}/clustering/points/{long_token}")
    assert point.status_code == 200
    assert point.json["Abstract"] == long_abstract
    assert point.json["DOI"] == long_doi

    css = client.get("/static/css/app.css")
    assert css.status_code == 200
    assert b"overflow-wrap: anywhere" in css.data
    assert b"white-space: pre-wrap" in css.data
    assert b"position: sticky" in css.data
    assert b"@media (max-width: 520px)" in css.data
