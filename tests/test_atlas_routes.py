from __future__ import annotations

import base64
import json
import zlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def test_atlas_routes_build_launch_download_and_stale_without_mutating_project_manifest(tmp_path: Path):
    app = make_app(tmp_path, PUBLIC_BASE_URL="https://reviews.example")
    manifest, path = project_with_embeddings(app)
    client = app.test_client()
    page_url = f"/projects/{manifest['project_id']}/atlas"

    initial = client.get(page_url)
    assert initial.status_code == 200
    assert b"Build evidence atlas" in initial.data
    assert b"cdn.plot.ly" not in initial.data
    assert "Cross-Origin-Embedder-Policy" not in initial.headers
    assert "wasm-unsafe-eval" not in initial.headers["Content-Security-Policy"]
    project_manifest_before = load_manifest(path)

    built = client.post(f"{page_url}/build")
    assert built.status_code == 302
    assert load_manifest(path) == project_manifest_before
    ready = client.get(page_url)
    assert ready.status_code == 200
    assert b"Open preloaded Atlas" in ready.data
    assert b"Precomputed UMAP" in ready.data
    assert b"atlas.js" not in ready.data

    atlas_manifest = json.loads((path / "evidence_atlas" / "manifest.json").read_text())
    fingerprint = atlas_manifest["fingerprint"]
    artifact = client.get(f"{page_url}/data/{fingerprint}.parquet")
    assert artifact.status_code == 200
    assert artifact.headers["Cache-Control"] == "private, max-age=31536000, immutable"
    assert artifact.headers["Access-Control-Allow-Origin"] == "https://apple.github.io"
    assert artifact.headers["Cross-Origin-Resource-Policy"] == "cross-origin"
    assert client.get(f"{page_url}/data/{'0' * 64}.parquet").status_code == 404
    assert client.get(f"{page_url}/data/not-a-hash.parquet").status_code == 404

    launch = client.get(f"{page_url}/open/{fingerprint}")
    assert launch.status_code == 302
    location = urlparse(launch.headers["Location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == "https://apple.github.io/embedding-atlas/app/"
    params = parse_qs(location.fragment.removeprefix("?"))
    assert params["data"] == [f"https://reviews.example{page_url}/data/{fingerprint}.parquet"]
    encoded_settings = params["settings"][0]
    compressed = base64.urlsafe_b64decode(encoded_settings + "=" * (-len(encoded_settings) % 4))
    settings = json.loads(zlib.decompress(compressed, wbits=-zlib.MAX_WBITS))
    assert settings == {
        "version": "0.22.0",
        "text": "search_text",
        "embedding": {
            "precomputed": {"x": "atlas_x", "y": "atlas_y", "neighbors": "neighbors"}
        },
    }
    assert client.get(f"{page_url}/open/{'0' * 64}").status_code == 404

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
    assert client.get(f"/projects/{second['project_id']}/atlas/open/{fingerprint}").status_code == 404


def test_local_http_atlas_uses_download_and_drop_handoff(tmp_path: Path):
    app = make_app(tmp_path)
    manifest, path = project_with_embeddings(app)
    client = app.test_client()
    page_url = f"/projects/{manifest['project_id']}/atlas"
    client.post(f"{page_url}/build")

    ready = client.get(page_url)
    assert ready.status_code == 200
    assert b"Download Atlas Parquet" in ready.data
    assert b"Open official file viewer" in ready.data
    assert b"Open preloaded Atlas" not in ready.data
    assert b"secure web page cannot fetch data from this local HTTP server" in ready.data

    local_tls = client.get(page_url, base_url="https://localhost")
    assert b"Open official file viewer" in local_tls.data
    assert b"Open preloaded Atlas" not in local_tls.data

    fingerprint = json.loads((path / "evidence_atlas" / "manifest.json").read_text())["fingerprint"]
    launch = client.get(f"{page_url}/open/{fingerprint}")
    assert launch.status_code == 409
    assert b"One-click preload requires a publicly reachable HTTPS address" in launch.data
    assert client.get(f"{page_url}/open/{fingerprint}", base_url="https://localhost").status_code == 409


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
