from pathlib import Path

import pandas as pd

from ml_review_app.services.clustering_service import cluster_csv
from ml_review_app.services.deduplication_service import deduplicate_records, normalize_text
from ml_review_app.services.embedding_service import add_embeddings_offline, deterministic_embedding
from ml_review_app.services.import_service import build_column_mapping, normalize_records, profile_csv
from ml_review_app.services.screening_service import local_screen, screen_csv_offline


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


def test_offline_embeddings_and_clustering(tmp_path: Path):
    source = tmp_path / "records.csv"
    pd.DataFrame(
        {
            "RecordID": ["1", "2", "3", "4"],
            "Title": ["heat pregnancy", "heat birth", "air quality", "temperature health"],
            "Abstract": ["maternal outcome", "pregnancy outcome", "pollution", "heat morbidity"],
        }
    ).to_csv(source, index=False)

    embedded = add_embeddings_offline(source, tmp_path / "embedded.csv")
    assert "Embedding" in embedded.columns
    assert deterministic_embedding("same") == deterministic_embedding("same")

    clustered, scores = cluster_csv(tmp_path / "embedded.csv", tmp_path / "clustered.csv", n_clusters=2)
    assert {"TSNE_1", "TSNE_2", "Cluster"}.issubset(clustered.columns)
    assert len(scores) >= 1


def test_local_screening_outputs_schema_and_csv(tmp_path: Path):
    decision = local_screen("Heat pregnancy", "Maternal heat outcome", "heat maternal pregnancy")
    assert decision.decision == "include"

    source = tmp_path / "selected.csv"
    pd.DataFrame({"RecordID": ["1"], "Title": ["Heat pregnancy"], "Abstract": ["Maternal heat outcome"]}).to_csv(source, index=False)
    screened = screen_csv_offline(source, "heat maternal pregnancy", tmp_path / "screened.csv")
    assert screened.loc[0, "ai_decision"] == "include"


def test_parse_pubmed_xml_fixture():
    from ml_review_app.services.pubmed_service import parse_pubmed_xml

    xml = """<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>123</PMID><Article><ArticleTitle>Heat study</ArticleTitle><Abstract><AbstractText>Maternal outcome.</AbstractText></Abstract><Journal><Title>Journal</Title><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal><AuthorList><Author><ForeName>Ada</ForeName><LastName>Lovelace</LastName></Author></AuthorList></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1/example</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
    df = parse_pubmed_xml(xml)
    assert df.loc[0, "RecordID"] == "123"
    assert df.loc[0, "Title"] == "Heat study"
    assert df.loc[0, "DOI"] == "10.1/example"
