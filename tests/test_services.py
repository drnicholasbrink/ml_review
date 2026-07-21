from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ml_review_app.services.clustering_service import cluster_csv
from ml_review_app.services.deduplication_service import deduplicate_records, normalize_text
from ml_review_app.services.embedding_service import add_embeddings
from ml_review_app.services.import_service import build_column_mapping, normalize_records, profile_csv
from ml_review_app.services.screening_service import ScreeningDecision, screen_csv


def test_import_mapping_normalize_and_profile(tmp_path: Path):
    source = tmp_path / "source.csv"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "title": ["Heat exposure", "Cold exposure"],
            "abstract": ["Maternal heat outcome", "Other topic"],
            "year": [2020, 2021],
        }
    ).to_csv(source, index=False)

    mapping = build_column_mapping({"RecordID": "id", "Title": "title", "Abstract": "abstract", "Date": "year"}, ["id", "title", "abstract", "year"])
    assert mapping["RecordID"] == "id"

    output = tmp_path / "normalized.csv"
    normalized = normalize_records(source, mapping, output)
    assert output.exists()
    assert list(normalized["RecordID"]) == ["a", "b"]
    assert "Journal" in normalized.columns

    profiled = profile_csv(source, "id")
    assert profiled["rows"] == 2
    assert profiled["duplicate_unique_ids"] == 0


def test_deduplicate_records_by_normalized_title(tmp_path: Path):
    source = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "RecordID": ["1", "2", "3"],
            "Title": ["Heat and Health!", "heat and health", "Different title"],
            "Abstract": ["a", "b", "c"],
        }
    ).to_csv(source, index=False)

    kept, report = deduplicate_records(source, tmp_path / "dedup.csv", tmp_path / "report.csv", match_columns=["Title"])
    assert len(kept) == 2
    assert len(report) == 2
    assert normalize_text("Heat and Health!") == "heat and health"


def test_deduplicate_does_not_collapse_missing_match_values(tmp_path: Path):
    source = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "RecordID": ["1", "2", "3"],
            "Title": [None, "", "A real title"],
            "Abstract": ["first", "second", "third"],
        }
    ).to_csv(source, index=False)

    kept, report = deduplicate_records(
        source,
        tmp_path / "dedup.csv",
        tmp_path / "report.csv",
        match_columns=["Title"],
    )
    assert len(kept) == 3
    assert report.empty


def test_openai_embedding_orchestration_and_clustering(tmp_path: Path):
    source = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "RecordID": ["1", "2", "3", "4"],
            "Title": ["heat pregnancy", "heat birth", "air quality", "temperature health"],
            "Abstract": ["maternal outcome", "pregnancy outcome", "pollution", "heat morbidity"],
        }
    ).to_csv(source, index=False)

    class FakeEmbeddings:
        def create(self, *, input, model, encoding_format):
            data = [SimpleNamespace(index=index, embedding=[float(index), 1.0, 2.0]) for index, _text in enumerate(input)]
            return SimpleNamespace(data=data)

    client = SimpleNamespace(embeddings=FakeEmbeddings())
    embedded = add_embeddings(source, tmp_path / "embedded.csv", api_key="test-key", client=client)
    assert "Embedding" in embedded.columns
    assert embedded["EmbeddingModel"].eq("text-embedding-3-small").all()

    clustered, scores = cluster_csv(tmp_path / "embedded.csv", tmp_path / "clustered.csv", n_clusters=2)
    repeated, _repeated_scores = cluster_csv(tmp_path / "embedded.csv", tmp_path / "clustered-again.csv", n_clusters=2)
    assert {"TSNE_1", "TSNE_2", "Cluster"}.issubset(clustered.columns)
    assert len(scores) >= 1
    assert clustered[["TSNE_1", "TSNE_2", "Cluster"]].equals(repeated[["TSNE_1", "TSNE_2", "Cluster"]])


def test_structured_screening_outputs_schema_and_csv(tmp_path: Path):
    class FakeResponses:
        def parse(self, **_kwargs):
            decision = ScreeningDecision(
                decision="include",
                confidence="high",
                exclusion_reason=None,
                reasoning="The title and abstract match the criteria.",
                population_match=True,
                exposure_match=True,
                outcome_match=True,
                study_design_appropriate=True,
            )
            return SimpleNamespace(output_parsed=decision)

    source = tmp_path / "selected.csv"
    pd.DataFrame({"RecordID": ["1"], "Title": ["Heat pregnancy"], "Abstract": ["Maternal heat outcome"]}).to_csv(source, index=False)
    client = SimpleNamespace(responses=FakeResponses())
    screened = screen_csv(
        source,
        "heat maternal pregnancy",
        tmp_path / "screened.csv",
        api_key="test-key",
        model="gpt-5.6-luna",
        client=client,
    )
    assert screened.loc[0, "ai_decision"] == "include"


def test_parse_pubmed_xml_fixture():
    from ml_review_app.services.pubmed_service import parse_pubmed_xml

    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Heat study</ArticleTitle><Abstract><AbstractText>Maternal outcome.</AbstractText></Abstract><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal><AuthorList><Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName></Author></AuthorList></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    df = parse_pubmed_xml(xml)
    assert df.loc[0, "RecordID"] == "123"
    assert df.loc[0, "Title"] == "Heat study"
    assert df.loc[0, "DOI"] == "10.1/example"


def test_parse_pubmed_xml_rejects_malformed_response():
    import pytest

    from ml_review_app.services.pubmed_service import PubMedResponseError, parse_pubmed_xml

    with pytest.raises(PubMedResponseError, match="malformed"):
        parse_pubmed_xml("<PubmedArticleSet>")
