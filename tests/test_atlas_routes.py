from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml_review_app import create_app
from ml_review_app.config import TestConfig
from ml_review_app.services.project_service import create_project, load_manifest, save_manifest


def make_app(tmp_path: Path, **overrides):
    settings = {"RUNTIME_DIR": tmp_path / "runtime", **overrides}
    config = type("AtlasTestConfig", (TestConfig,), settings)
    return create_app(config)


def project_with_embeddings(app, title: str = "Atlas study"):
    manifest = create_project(app.config["RUNTIME_DIR"], "Atlas review")
    path = app.config["RUNTIME_DIR"] / "projects" / manifest["project_id"]
    pd.DataFrame(
        [
            {
                "RecordID": "one",
                "Title": title,
                "Abstract": "An abstract",
                "Embedding": json.dumps([1.0, 0.0]),
                "EmbeddingModel": "fixture",
            },
            {
                "RecordID": "two",
                "Title": "Second study",
                "Abstract": "Another abstract",
                "Embedding": json.dumps([0.0, 1.0]),
                "EmbeddingModel": "fixture",
            },
        ]
    ).to_csv(path / "embeddings.csv", index=False)
    manifest["files"]["embeddings"] = "embeddings.csv"
    save_manifest(path, manifest)
    return manifest, path


def test_atlas_routes_build_serve_cache_and_stale_without_mutating_project_manifest(tmp_path: Path, monkeypatch):
    app = make_app(tmp_path)
    monkeypatch.setattr("ml_review_app.blueprints.atlas._assets_ready", lambda: True)
    manifest, path = project_with_embeddings(app)
    client = app.test_client()
    page_url = f"/projects/{manifest['project_id']}/atlas"

    initial = client.get(page_url)
    assert initial.status_code == 200
    assert b"Build evidence atlas" in initial.data
    assert b"cdn.plot.ly" not in initial.data
    assert initial.headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    project_manifest_before = load_manifest(path)

    built = client.post(f"{page_url}/build")
    assert built.status_code == 302
    assert load_manifest(path) == project_manifest_before
    ready = client.get(page_url)
    assert ready.status_code == 200
    assert b'id="atlas-root"' in ready.data
    assert b"Atlas selections stay in this browser" in ready.data

    atlas_manifest = json.loads((path / "evidence_atlas" / "manifest.json").read_text())
    fingerprint = atlas_manifest["fingerprint"]
    artifact = client.get(f"{page_url}/data/{fingerprint}.parquet")
    assert artifact.status_code == 200
    assert artifact.headers["Cache-Control"] == "private, max-age=31536000, immutable"
    assert artifact.headers["Cross-Origin-Resource-Policy"] == "same-origin"
    assert client.get(f"{page_url}/data/{'0' * 64}.parquet").status_code == 404
    assert client.get(f"{page_url}/data/not-a-hash.parquet").status_code == 404

    source = pd.read_csv(path / "embeddings.csv")
    source.loc[0, "Title"] = "Changed title"
    source.to_csv(path / "embeddings.csv", index=False)
    stale = client.get(page_url)
    assert b"The Atlas is out of date" in stale.data
    assert b"Rebuild evidence atlas" in stale.data
    assert client.get(f"{page_url}/data/{fingerprint}.parquet").status_code == 404


def test_atlas_data_is_project_scoped(tmp_path: Path):
    app = make_app(tmp_path)
    first, first_path = project_with_embeddings(app, "First project")
    second, _second_path = project_with_embeddings(app, "Second project")
    client = app.test_client()
    client.post(f"/projects/{first['project_id']}/atlas/build")
    fingerprint = json.loads((first_path / "evidence_atlas" / "manifest.json").read_text())["fingerprint"]

    response = client.get(f"/projects/{second['project_id']}/atlas/data/{fingerprint}.parquet")
    assert response.status_code == 404


def test_atlas_build_requires_csrf_when_protection_is_enabled(tmp_path: Path):
    app = make_app(tmp_path, WTF_CSRF_ENABLED=True)
    manifest, _path = project_with_embeddings(app)
    response = app.test_client().post(f"/projects/{manifest['project_id']}/atlas/build")
    assert response.status_code == 400
    assert b"missing or has an expired security token" in response.data


def test_atlas_is_optional_and_blocked_until_embeddings_exist(tmp_path: Path):
    app = make_app(tmp_path)
    manifest = create_project(app.config["RUNTIME_DIR"], "Empty review")
    client = app.test_client()
    dashboard = client.get(f"/projects/{manifest['project_id']}")
    assert b"Evidence Atlas" in dashboard.data
    assert b"Generate embeddings first" in dashboard.data
    page = client.get(f"/projects/{manifest['project_id']}/atlas")
    assert page.status_code == 200
    assert b"Embeddings required" in page.data


def test_atlas_has_an_empty_evidence_state(tmp_path: Path):
    app = make_app(tmp_path)
    manifest = create_project(app.config["RUNTIME_DIR"], "Empty embedded review")
    path = app.config["RUNTIME_DIR"] / "projects" / manifest["project_id"]
    pd.DataFrame(columns=["RecordID", "Title", "Abstract", "Embedding"]).to_csv(
        path / "embeddings.csv", index=False
    )
    manifest["files"]["embeddings"] = "embeddings.csv"
    save_manifest(path, manifest)
    page = app.test_client().get(f"/projects/{manifest['project_id']}/atlas")
    assert page.status_code == 200
    assert b"No embedded records" in page.data
