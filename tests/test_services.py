from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from ml_review_app.services.clustering_service import build_cluster_search, cluster_csv, parse_cluster_keywords
from ml_review_app.services.deduplication_service import deduplicate_records, normalize_text
from ml_review_app.services.embedding_service import MAX_EMBEDDING_TEXT_LENGTH, add_embeddings
from ml_review_app.services.evaluation_service import build_screening_evaluation, compare_with_human
from ml_review_app.services.extraction_service import (
    DataExtraction,
    EffectEstimate,
    ExposureMetric,
    GeographicLocation,
    OutcomeMeasure,
    extract_csv,
    write_extraction_exports,
)
from ml_review_app.services.import_service import build_column_mapping, normalize_records, profile_csv
from ml_review_app.services.screening_service import (
    MAX_SCREENING_RECORD_LENGTH,
    ScreeningDecision,
    apply_human_reviews,
    save_human_review,
    screen_csv,
)


def test_human_screening_review_preserves_ai_audit_and_sets_final_decision(tmp_path: Path):
    screening_path = tmp_path / "screening.csv"
    reviews_path = tmp_path / "reviews.csv"
    reviewed_path = tmp_path / "reviewed.csv"
    screening = pd.DataFrame([
        {
            "RecordID": "one", "PMID": "123", "Title": "Heat and pregnancy", "Abstract": "Study abstract.",
            "ai_decision": "uncertain", "ai_confidence": "low", "ai_reasoning": "Insufficient detail.",
        }
    ])
    screening.to_csv(screening_path, index=False)
    keyed = apply_human_reviews(screening)
    assert bool(keyed.loc[0, "requires_human_review"])
    assert keyed.loc[0, "final_decision"] == "uncertain"

    reviewed = save_human_review(
        screening_path, reviews_path, reviewed_path,
        record_key=keyed.loc[0, "record_key"], decision="include", note="Confirmed against full text.",
    )
    assert reviewed.loc[0, "ai_decision"] == "uncertain"
    assert reviewed.loc[0, "final_decision"] == "include"
    assert reviewed.loc[0, "final_decision_source"] == "human"
    assert not bool(reviewed.loc[0, "requires_human_review"])
    assert reviewed.loc[0, "human_note"] == "Confirmed against full text."

    cleared = save_human_review(
        screening_path, reviews_path, reviewed_path,
        record_key=keyed.loc[0, "record_key"], decision="", note="",
    )
    assert cleared.loc[0, "final_decision"] == "uncertain"
    assert cleared.loc[0, "final_decision_source"] == "ai"


def test_structured_extraction_resumes_and_writes_publication_exports(tmp_path: Path):
    source = tmp_path / "screening.csv"
    pd.DataFrame(
        {
            "RecordID": ["one", "two", "three"],
            "PMID": ["1", "2", "3"],
            "Title": ["Included one", "Uncertain two", "Included three"],
            "Abstract": ["Abstract one", "Abstract two", "Abstract three"],
            "ai_decision": ["include", "uncertain", "include"],
            "ai_confidence": ["high", "low", "medium"],
        }
    ).to_csv(source, index=False)
    parsed = DataExtraction(
        population_description="Pregnant participants",
        location=GeographicLocation(country="South Africa"),
        exposures=[ExposureMetric(metric_type="temperature", unit="°C")],
        outcomes=[OutcomeMeasure(outcome_type="pregnancy outcome", specific_outcome="miscarriage")],
        sample_size="1,000 pregnancies",
        study_design="cohort",
        statistical_methods=["logistic regression"],
        effect_estimates=[EffectEstimate(estimate_type="odds ratio", value="1.20", confidence_interval="95% CI 1.05-1.37")],
        key_finding_summary="Higher heat exposure was associated with miscarriage.",
        data_completeness="partial",
        extraction_confidence="medium",
    )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        def parse(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(output_parsed=parsed)

    responses = FakeResponses()
    fake_client = SimpleNamespace(responses=responses)
    output = tmp_path / "ai_extraction_full_results.csv"
    test_result, candidate_count = extract_csv(
        source,
        "Include eligible pregnancy studies.",
        output,
        api_key="test-key",
        model="gpt-5.6-luna",
        include_uncertain=True,
        limit=1,
        client=fake_client,
    )
    assert candidate_count == 3
    assert len(test_result) == 1

    full_result, candidate_count = extract_csv(
        source,
        "Include eligible pregnancy studies.",
        output,
        api_key="test-key",
        model="gpt-5.6-luna",
        include_uncertain=True,
        client=fake_client,
    )
    assert candidate_count == 3
    assert len(full_result) == 3
    assert len(responses.calls) == 3
    assert responses.calls[0]["store"] is False
    assert full_result["country"].tolist() == ["South Africa"] * 3
    assert len(full_result.loc[0, "effect_estimates"]) > 10

    reviewed_source = pd.read_csv(source)
    reviewed_source["final_decision"] = ["exclude", "uncertain", "include"]
    reviewed_source.to_csv(source, index=False)
    filtered_result, candidate_count = extract_csv(
        source,
        "Include eligible pregnancy studies.",
        output,
        api_key="test-key",
        model="gpt-5.6-luna",
        include_uncertain=False,
        client=fake_client,
    )
    assert candidate_count == 1
    assert filtered_result["RecordID"].tolist() == ["three"]
    assert len(responses.calls) == 3

    exports = write_extraction_exports(full_result, tmp_path)
    assert set(exports) == {
        "ai_extraction_full_results_json",
        "study_characteristics",
        "effect_estimates",
        "extraction_summary",
    }
    assert len(pd.read_csv(tmp_path / "effect_estimates.csv")) == 3


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
    embedding_progress = []
    embedded = add_embeddings(
        source,
        tmp_path / "embedded.csv",
        api_key="test-key",
        client=client,
        progress_callback=lambda completed, total, message: embedding_progress.append(
            (completed, total, message)
        ),
    )
    assert "Embedding" in embedded.columns
    assert embedded["EmbeddingModel"].eq("text-embedding-3-small").all()
    assert embedding_progress[-1] == (4, 4, "Embedding artifact saved")

    clustered, scores = cluster_csv(tmp_path / "embedded.csv", tmp_path / "clustered.csv", n_clusters=2)
    repeated, _repeated_scores = cluster_csv(tmp_path / "embedded.csv", tmp_path / "clustered-again.csv", n_clusters=2)
    assert {"TSNE_1", "TSNE_2", "Cluster"}.issubset(clustered.columns)
    assert len(scores) >= 1
    assert clustered[["TSNE_1", "TSNE_2", "Cluster"]].equals(repeated[["TSNE_1", "TSNE_2", "Cluster"]])


def test_cluster_search_uses_document_prevalence_and_normalized_matching():
    clustered = pd.DataFrame(
        [
            {"PointID": "p1", "Cluster": 0, "Title": "Café heat exposure", "Abstract": "Maternal   outcomes after exposure"},
            {"PointID": "p2", "Cluster": 0, "Title": "Heating systems", "Abstract": None},
            {"PointID": "p3", "Cluster": 0, "Title": None, "Abstract": "HEAT exposure without maternal outcomes"},
            {"PointID": "p4", "Cluster": 1, "Title": "Maternal heat exposure in pregnancy", "Abstract": "Birth outcomes"},
            {"PointID": "p5", "Cluster": 1, "Title": "Air pollution", "Abstract": "Respiratory outcomes"},
        ]
    )

    any_result = build_cluster_search(
        clustered,
        run_id="run_001",
        raw_keywords=" HEAT, heat\n maternal   outcomes ",
        match_mode="any",
    )
    assert any_result["keywords"] == ["heat", "maternal outcomes"]
    assert any_result["keyword_matched_point_ids"] == ["p1", "p3", "p4"]
    cluster_zero = any_result["clusters"][0]
    assert cluster_zero["size"] == 3
    assert cluster_zero["keywords"] == [
        {"keyword": "heat", "count": 2, "prevalence": 2 / 3, "percentage": 66.7},
        {"keyword": "maternal outcomes", "count": 2, "prevalence": 2 / 3, "percentage": 66.7},
    ]
    assert cluster_zero["combined_count"] == 2
    assert cluster_zero["combined_percentage"] == 66.7

    all_result = build_cluster_search(
        clustered,
        run_id="run_001",
        raw_keywords="heat, maternal outcomes",
        match_mode="all",
    )
    assert all_result["keyword_matched_point_ids"] == ["p1", "p3"]
    assert all_result["clusters"][0]["combined_count"] == 2
    assert all_result["clusters"][1]["combined_count"] == 0

    unicode_result = build_cluster_search(clustered, run_id="run_001", raw_keywords="ＣＡＦÉ")
    assert unicode_result["keywords"] == ["café"]
    assert unicode_result["keyword_matched_point_ids"] == ["p1"]
    assert "p2" not in any_result["keyword_matched_point_ids"]


def test_cluster_search_ranks_direct_titles_and_identifier_matches_before_fuzzy_titles():
    clustered = pd.DataFrame(
        [
            {"PointID": "direct", "Cluster": 0, "RecordID": "R-100", "PMID": "12345", "DOI": "10.1/direct", "Title": "Maternal heat exposure review", "Abstract": ""},
            {"PointID": "fuzzy", "Cluster": 1, "RecordID": "R-200", "PMID": "67890", "DOI": "10.1/fuzzy", "Title": "Review of maternal exposure to heat", "Abstract": ""},
            {"PointID": "other", "Cluster": 1, "RecordID": "R-201", "PMID": "", "DOI": "", "Title": "Air pollution", "Abstract": ""},
        ]
    )

    title_result = build_cluster_search(
        clustered,
        run_id="run_title",
        study_query="maternal heat exposure",
    )
    assert [match["point_id"] for match in title_result["study_matches"]][:2] == ["direct", "fuzzy"]
    assert title_result["study_matches"][0]["match_type"] == "title_substring"
    assert title_result["study_matches"][1]["match_type"] == "title_fuzzy"

    exact_identifier = build_cluster_search(clustered, run_id="run_id", study_query="10.1/direct")
    assert exact_identifier["study_matches"][0]["point_id"] == "direct"
    assert exact_identifier["study_matches"][0]["match_type"] == "identifier_exact"
    prefix_identifier = build_cluster_search(clustered, run_id="run_id", study_query="R-20")
    assert {match["point_id"] for match in prefix_identifier["study_matches"]} == {"fuzzy", "other"}
    assert all(match["match_type"] == "identifier_prefix" for match in prefix_identifier["study_matches"])


def test_cluster_search_validation_and_result_cap():
    import pytest

    clustered = pd.DataFrame(
        [
            {"PointID": f"p{index}", "Cluster": index % 2, "Title": f"Shared study title {index}", "Abstract": ""}
            for index in range(30)
        ]
    )
    result = build_cluster_search(clustered, run_id="run_001", study_query="shared study title")
    assert len(result["study_matches"]) == 25
    assert len(result["study_matched_point_ids"]) == 30
    assert result["keyword_matched_point_ids"] == []

    assert parse_cluster_keywords("Heat, HEAT, heat") == ["heat"]
    with pytest.raises(ValueError, match="no more than 20"):
        parse_cluster_keywords(",".join(f"keyword-{index}" for index in range(21)))
    with pytest.raises(ValueError, match="100 characters"):
        parse_cluster_keywords("x" * 101)
    with pytest.raises(ValueError, match="either any or all"):
        build_cluster_search(clustered, run_id="run_001", match_mode="some")
    with pytest.raises(ValueError, match="200 characters"):
        build_cluster_search(clustered, run_id="run_001", study_query="x" * 201)


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
    screening_progress = []
    screened = screen_csv(
        source,
        "heat maternal pregnancy",
        tmp_path / "screened.csv",
        api_key="test-key",
        model="gpt-5.6-luna",
        client=client,
        progress_callback=lambda completed, total, message: screening_progress.append(
            (completed, total, message)
        ),
    )
    assert screened.loc[0, "ai_decision"] == "include"
    assert pd.isna(screened.loc[0, "ai_exclusion_category"])
    assert screening_progress[-1] == (1, 1, "Screening CSV saved")


def test_excluded_screening_decisions_require_a_broad_category():
    with pytest.raises(ValueError, match="require an exclusion category"):
        ScreeningDecision(
            decision="exclude",
            confidence="high",
            exclusion_reason="Animal model rather than the target population.",
            reasoning="The study uses animals.",
            population_match=False,
            exposure_match=True,
            outcome_match=True,
            study_design_appropriate=True,
        )

    with pytest.raises(ValueError, match="require a concise exclusion reason"):
        ScreeningDecision(
            decision="exclude",
            confidence="high",
            exclusion_category="population",
            reasoning="The study uses animals.",
            population_match=False,
            exposure_match=True,
            outcome_match=True,
            study_design_appropriate=True,
        )

    decision = ScreeningDecision(
        decision="exclude",
        confidence="high",
        exclusion_category="population",
        exclusion_reason="Animal model rather than the target population.",
        reasoning="The study uses animals.",
        population_match=False,
        exposure_match=True,
        outcome_match=True,
        study_design_appropriate=True,
    )
    assert decision.exclusion_category == "population"


def test_overlength_api_inputs_are_truncated_without_altering_stored_abstracts(tmp_path: Path):
    embedding_inputs = []

    class FakeEmbeddings:
        def create(self, *, input, model, encoding_format):
            embedding_inputs.extend(input)
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.0, 2.0])])

    long_embedding_abstract = "e" * (MAX_EMBEDDING_TEXT_LENGTH + 500)
    embedding_source = tmp_path / "long-embedding.csv"
    pd.DataFrame([{"RecordID": "long-1", "Title": "Title", "Abstract": long_embedding_abstract}]).to_csv(embedding_source, index=False)
    embedded = add_embeddings(
        embedding_source,
        tmp_path / "long-embedded.csv",
        api_key="test-key",
        client=SimpleNamespace(embeddings=FakeEmbeddings()),
    )
    assert len(embedding_inputs[0]) == MAX_EMBEDDING_TEXT_LENGTH
    assert embedded.loc[0, "Abstract"] == long_embedding_abstract
    assert bool(embedded.loc[0, "EmbeddingInputTruncated"])

    screening_inputs = []

    class FakeResponses:
        def parse(self, **kwargs):
            screening_inputs.append(kwargs["input"])
            return SimpleNamespace(
                output_parsed=ScreeningDecision(
                    decision="uncertain",
                    confidence="low",
                    exclusion_reason=None,
                    reasoning="The supplied excerpt requires human review.",
                    population_match=False,
                    exposure_match=False,
                    outcome_match=False,
                    study_design_appropriate=False,
                )
            )

    long_screening_abstract = "s" * (MAX_SCREENING_RECORD_LENGTH + 500)
    screening_source = tmp_path / "long-screening.csv"
    pd.DataFrame([{"RecordID": "long-2", "Title": "Title", "Abstract": long_screening_abstract}]).to_csv(screening_source, index=False)
    screened = screen_csv(
        screening_source,
        "Include relevant studies.",
        tmp_path / "long-screened.csv",
        api_key="test-key",
        model="gpt-5.6-luna",
        client=SimpleNamespace(responses=FakeResponses()),
    )
    assert "LEADING EXCERPT; INPUT WAS TRUNCATED" in screening_inputs[0]
    assert screened.loc[0, "Abstract"] == long_screening_abstract
    assert bool(screened.loc[0, "ai_input_truncated"])
    assert screened.loc[0, "ai_input_characters"] == MAX_SCREENING_RECORD_LENGTH
    assert screened.loc[0, "ai_input_original_characters"] > MAX_SCREENING_RECORD_LENGTH


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


def test_screening_evaluation_and_human_comparison():
    screened = pd.DataFrame(
        {
            "Title": ["Heat and pregnancy", "Air pollution study", "Animal experiment"],
            "ai_decision": ["include", "uncertain", "exclude"],
            "ai_confidence": ["high", "low", "high"],
            "ai_exclusion_reason": [None, None, "animal_study"],
            "ai_population_match": [True, True, False],
            "ai_exposure_match": [True, True, True],
            "ai_outcome_match": [True, False, True],
            "ai_study_design_appropriate": [True, True, False],
            "TSNE_1": [1.0, 2.0, 3.0],
            "TSNE_2": [3.0, 2.0, 1.0],
        }
    )
    evaluation = build_screening_evaluation(screened)
    assert evaluation["funnel"]["values"] == [3, 2, 1]
    assert evaluation["manual_review_count"] == 1
    assert evaluation["exclusion_categories"] == [
        {"category": "population", "label": "Population", "count": 1}
    ]
    assert len(evaluation["tsne_points"]) == 3

    human = pd.DataFrame(
        {
            "Title": ["Heat & Pregnancy", "Air pollution study", "Animal experiment"],
            "status": ["include", "exclude", "exclude"],
        }
    )
    metrics, comparison = compare_with_human(screened, human, uncertain_is_positive=True)
    assert metrics["matched_records"] == 3
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["true_negatives"] == 1
    assert len(comparison) == 3


def test_screening_evaluation_aggregates_specific_and_legacy_exclusion_reasons():
    screened = pd.DataFrame(
        {
            "ai_decision": ["exclude", "exclude", "exclude", "exclude"],
            "ai_exclusion_category": ["population", "population", None, None],
            "ai_exclusion_reason": [
                "Wrong age group",
                "Animal study",
                "Non-pregnant population",
                "Narrative review rather than original research",
            ],
        }
    )

    evaluation = build_screening_evaluation(screened)

    assert evaluation["exclusion_categories"] == [
        {"category": "population", "label": "Population", "count": 3},
        {"category": "publication_type", "label": "Publication type", "count": 1},
    ]
