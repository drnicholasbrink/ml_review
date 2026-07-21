from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from ml_review_app.services import atlas_service
from ml_review_app.services.atlas_service import atlas_status, build_atlas


def write_embeddings(path: Path, vectors: list[list[float]]) -> dict:
    frame = pd.DataFrame(
        [
            {
                "RecordID": f"record-{index}",
                "Title": f"Study {index}",
                "Abstract": f"Abstract text {index}",
                "Date": f"202{index}-01-01",
                "Source": "fixture",
                "EmbeddingModel": "test-model",
                "Embedding": json.dumps(vector),
            }
            for index, vector in enumerate(vectors)
        ]
    )
    frame.to_csv(path / "embeddings.csv", index=False)
    return {"files": {"embeddings": "embeddings.csv"}}


@pytest.mark.parametrize(
    "values, message",
    [
        (["not-a-vector"], "not a valid vector"),
        (["[]"], "empty or invalid"),
        (["[1, 1e309]"], "non-finite"),
        (["[1, 2]", "[1, 2, 3]"], "same dimension"),
    ],
)
def test_embedding_validation_rejects_the_whole_source(tmp_path: Path, values: list[str], message: str):
    pd.DataFrame({"Title": ["Study"] * len(values), "Embedding": values}).to_csv(
        tmp_path / "embeddings.csv", index=False
    )
    with pytest.raises(ValueError, match=message):
        build_atlas(tmp_path, {"files": {"embeddings": "embeddings.csv"}})
    assert not (tmp_path / "evidence_atlas" / "manifest.json").exists()


@pytest.mark.parametrize(
    "vectors, expected",
    [
        ([[1.0, 0.0]], [[0.0, 0.0]]),
        ([[1.0, 0.0], [0.0, 1.0]], [[-0.5, 0.0], [0.5, 0.0]]),
    ],
)
def test_tiny_datasets_use_deterministic_projection_fallbacks(tmp_path: Path, vectors, expected):
    manifest = write_embeddings(tmp_path, vectors)
    record = build_atlas(tmp_path, manifest)
    table = pq.read_table(tmp_path / "evidence_atlas" / record["artifact"]).to_pydict()
    assert np.asarray([table["atlas_x"], table["atlas_y"]]).T.tolist() == expected
    assert len(table["neighbors"]) == len(vectors)
    assert all(len(value["ids"]) == max(0, len(vectors) - 1) for value in table["neighbors"])
    assert "embedding" not in table


def test_umap_and_neighbor_artifact_is_deterministic_and_reused(tmp_path: Path):
    vectors = [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0.9, 0.1], [0, 0, 1]]
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first_manifest = write_embeddings(first_path, vectors)
    second_manifest = write_embeddings(second_path, vectors)

    first = build_atlas(first_path, first_manifest)
    reused = build_atlas(first_path, first_manifest)
    second = build_atlas(second_path, second_manifest)
    first_table = pq.read_table(first_path / "evidence_atlas" / first["artifact"]).to_pydict()
    second_table = pq.read_table(second_path / "evidence_atlas" / second["artifact"]).to_pydict()

    assert first["fingerprint"] == second["fingerprint"]
    assert reused["reused"] is True
    assert np.allclose(first_table["atlas_x"], second_table["atlas_x"])
    assert np.allclose(first_table["atlas_y"], second_table["atlas_y"])
    for row_index, neighbors in enumerate(first_table["neighbors"]):
        assert row_index not in neighbors["ids"]
        assert all(isinstance(identifier, int) for identifier in neighbors["ids"])
        assert neighbors["distances"] == sorted(neighbors["distances"])
        assert len(neighbors["ids"]) == 4


def test_changed_source_is_stale_and_failed_rebuild_preserves_current_artifact(tmp_path: Path, monkeypatch):
    manifest = write_embeddings(tmp_path, [[1, 0], [0, 1]])
    current = build_atlas(tmp_path, manifest)
    old_manifest = (tmp_path / "evidence_atlas" / "manifest.json").read_text()
    old_artifact = tmp_path / "evidence_atlas" / current["artifact"]
    frame = pd.read_csv(tmp_path / "embeddings.csv")
    frame.loc[0, "Embedding"] = "[0.8, 0.2]"
    frame.to_csv(tmp_path / "embeddings.csv", index=False)

    assert atlas_status(tmp_path, manifest)["state"] == "stale"

    def fail_manifest(*_args, **_kwargs):
        raise OSError("simulated manifest failure")

    monkeypatch.setattr(atlas_service, "_write_json_atomic", fail_manifest)
    with pytest.raises(OSError, match="simulated"):
        build_atlas(tmp_path, manifest)
    assert (tmp_path / "evidence_atlas" / "manifest.json").read_text() == old_manifest
    assert old_artifact.is_file()
    assert list((tmp_path / "evidence_atlas").glob("*.tmp")) == []


def test_cluster_and_screening_metadata_are_joined_without_changing_source(tmp_path: Path):
    manifest = write_embeddings(tmp_path, [[1, 0], [0, 1]])
    source = pd.read_csv(tmp_path / "embeddings.csv")
    source["PointID"] = ["point-a", "point-b"]
    source.to_csv(tmp_path / "embeddings.csv", index=False)
    pd.DataFrame({"PointID": ["point-a", "point-b"], "Cluster": [2, 3]}).to_csv(
        tmp_path / "clusters.csv", index=False
    )
    pd.DataFrame(
        {
            "RecordID": ["record-0"],
            "ai_decision": ["include"],
            "ai_confidence": ["high"],
            "ai_exclusion_category": [""],
            "ai_exclusion_reason": [""],
        }
    ).to_csv(tmp_path / "screening.csv", index=False)
    manifest["files"].update(
        {"labeled_clusters": "clusters.csv", "ai_screening_full_results": "screening.csv"}
    )

    record = build_atlas(tmp_path, manifest)
    table = pq.read_table(tmp_path / "evidence_atlas" / record["artifact"]).to_pydict()
    assert table["Cluster"] == [2, 3]
    assert table["ai_decision"] == ["include", ""]
    assert table["ai_exclusion_category"] == ["", ""]
