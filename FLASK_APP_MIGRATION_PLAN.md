# Flask Application Migration Plan

## Objective

Create a new development branch for transforming the current notebook-first systematic-review workflow into a standalone Flask web application, optionally packaged in Docker, while preserving the underlying data-processing logic as closely as possible.

The target application should guide a non-technical reviewer through the same core workflow currently represented by the notebooks:

1. define a PubMed search strategy;
2. define inclusion/exclusion criteria;
3. run PubMed search/fetch or upload an existing CSV;
4. map the unique ID, title/abstract/text, and deduplication-match columns;
5. download fetched, uploaded, normalized, or deduplicated metadata;
6. generate embeddings with a user-provided OpenAI API key;
7. visualize embeddings with t-SNE and Plotly;
8. display elbow-method plots so the reviewer can choose cluster counts;
9. select included clusters and iteratively re-run/subset t-SNE as in the current exploratory notebook;
10. export selected records to AI screening;
11. run AI screening with structured outputs;
12. download screening results and reports; and
13. optionally continue to structured AI data extraction.

## Branching plan

Create implementation work on a dedicated development branch:

```bash
git checkout -b deve/flask-app
```

Use small, reviewable commits by milestone rather than one large conversion commit. Keep the existing notebooks in place until the Flask workflow has feature parity and tests prove equivalence for core transformations.

## Guiding principles

- Preserve the existing algorithms and API behaviors unless a change is required for interactive execution.
- Extract notebook logic into importable service modules before building UI screens.
- Make every expensive or external API action explicit in the UI.
- Never persist OpenAI or PubMed API keys in git or server logs.
- Use progress-saving and resumable jobs for long PubMed/OpenAI operations.
- Keep human review central: the UI should assist decisions, not hide or replace them.
- Prefer local file-based storage for an initial standalone version; add a database only when needed for multi-user persistence.

## Proposed application architecture

```text
ml_review_app/
  app.py                         # Flask factory / entry point
  config.py                      # environment and runtime configuration
  blueprints/
    setup.py                     # search strategy, criteria, API-key session setup
    pubmed.py                    # search/fetch screens and downloads
    imports.py                   # CSV upload, schema mapping, normalization, deduplication
    embeddings.py                # embedding job controls
    clustering.py                # t-SNE, elbow, cluster selection, progressive subsetting
    screening.py                 # AI screening controls, progress, downloads
    extraction.py                # optional structured extraction stage
    exports.py                   # download routes
  services/
    pubmed_service.py            # extracted logic from search_pubmed.ipynb
    import_service.py            # CSV upload/schema mapping and canonicalization
    deduplication_service.py     # extracted logic from deduplicate.ipynb
    embedding_service.py         # extracted logic from extract_pubmed/extract_csv notebooks
    clustering_service.py        # extracted logic from clustering.ipynb
    screening_service.py         # extracted logic from ai_screening.ipynb
    extraction_service.py        # extracted logic from ai_extraction.ipynb
    eval_service.py              # extracted logic from eval.ipynb, optional
  jobs/
    runner.py                    # background-job abstraction
    state.py                     # progress, cancellation, resume metadata
  templates/                     # Jinja templates
  static/                        # CSS, JS, images
  tests/
    fixtures/                    # tiny CSV/XML fixtures, fake embeddings/responses
    unit/
    integration/
Dockerfile
compose.yaml
```

## Storage model

Start with project/session directories under a gitignored runtime folder:

```text
runtime/
  projects/
    <project_id>/
      search_strategy.txt
      inclusion_criteria.txt
      uploaded_source.csv
      column_mapping.json
      pubmed_results_complete.csv
      normalized_records.csv
      deduplicated_records.csv
      pubmed_results_with_embeddings.csv
      clustering_state.json
      selected_records.csv
      ai_screening_full_results.csv
      ai_extraction_full_results.csv
      visualizations/
```

Each project should have a generated ID, human-readable name, creation timestamp, and manifest file. API keys should be held only in memory/session scope or injected through environment variables; if persistent key storage is ever added, it must be encrypted and explicitly opt-in.

## UI workflow

### 1. Project setup

Screen goals:

- Create or load a review project.
- Paste or edit PubMed search strategy.
- Paste or edit inclusion/exclusion criteria.
- Enter PubMed API key and OpenAI API key in password fields.
- Validate required fields before enabling API-running steps.

Human decisions:

- Review name and output location.
- Search strategy text.
- Inclusion/exclusion criteria text.
- Whether keys are entered per session or provided by environment variables.

### 2A. PubMed search and fetch

Screen goals:

- Show query preview and date-range controls.
- Run PubMed count first before full fetch.
- Display number of records found per date window.
- Run full fetch with progress and resumable status.
- Provide downloads for `pubmed_results_complete.csv` and logs.

Core logic to preserve:

- ESearch/EFetch usage.
- Date-window splitting for large result counts.
- PMID cache/deduplication behavior.
- Incremental CSV writes.

Human decisions:

- Date range.
- Whether count is acceptable before fetch.
- Whether to resume or reset previous PubMed results.
- Whether to continue after PubMed warnings/throttling.

### 2B. CSV upload, schema mapping, and deduplication

Screen goals:

- Offer CSV upload as an alternative or supplement to PubMed search/fetch.
- Preview uploaded rows, detected column names, missing values, and duplicate counts.
- Require the reviewer to select the unique ID column.
- Let the reviewer map title, abstract, authors, year/date, journal, DOI, and any custom metadata columns.
- Let the reviewer choose one or more matching columns for deduplication, such as title, DOI, PMID, or title-plus-year.
- Run deduplication using logic extracted from `scripts/deduplicate.ipynb`, starting with normalized-title matching and extending to configured ID/match-column rules.
- Save the original upload, column mapping, normalized CSV, deduplicated CSV, and duplicate-review report.
- Allow the reviewer to download either the original uploaded CSV, normalized CSV, duplicate report, or deduplicated CSV.

Core logic to preserve or extend:

- Preserve the existing title-normalization deduplication behavior as the default for title-based imports.
- Add configurable unique-ID handling so non-PubMed datasets can use PMID, DOI, trial ID, database accession, or a user-specified record ID.
- Add configurable match-column sets for duplicate detection without hard-coding PubMed-only assumptions.
- Keep duplicate removal explainable by writing a duplicate report that shows kept/dropped IDs and matched values.

Human decisions:

- Whether to fetch from PubMed, upload a CSV, or combine both sources.
- Which column is the stable unique identifier.
- Which columns map to title, abstract/text, date/year, journal, authors, and DOI.
- Which columns are used to identify duplicates.
- Whether duplicate groups are automatically collapsed or manually reviewed first.
- Which normalized/deduplicated dataset proceeds to embeddings.

### 3. Embeddings

Screen goals:

- Show detected input CSV schema and row count from PubMed, uploaded, normalized, or deduplicated records.
- Let the reviewer choose title/abstract/text columns when necessary, using the saved column mapping as defaults.
- Display estimated number of API calls.
- Run embeddings in batches with progress, retry status, and resume controls.
- Provide download for `pubmed_results_with_embeddings.csv`.

Core logic to preserve:

- `text-embedding-3-small` default.
- Batch processing.
- JSON serialization of embedding vectors.
- Resume from existing output file.

Human decisions:

- Text column choice.
- Batch size.
- Embedding model if model selection is exposed.
- Whether API cost and runtime are acceptable.

### 4. Initial t-SNE and elbow method

Screen goals:

- Compute t-SNE coordinates from embeddings.
- Compute elbow-method plots over a configurable K range.
- Display Plotly scatterplot and elbow chart side by side.
- Let reviewer choose K from the elbow plot.
- Run K-Means with selected K.
- Show cluster sizes, exemplar titles, and searchable/filterable record tables.

Core logic to preserve:

- Embedding parsing.
- StandardScaler normalization.
- t-SNE projection with cosine-like similarity behavior where currently used.
- K-Means clustering.
- Elbow-method WCSS calculation.

Human decisions:

- t-SNE parameters such as perplexity and random seed, if exposed.
- K-Means cluster count after reviewing the elbow plot.
- Which clusters appear relevant.
- Whether to proceed, refine, or rerun.

### 5. Progressive clustering/subsetting

Screen goals:

- Let the reviewer select one or more included clusters.
- Create a subset from selected clusters.
- Re-run t-SNE, elbow method, and K-Means on the subset.
- Maintain a visible breadcrumb/history of each clustering round.
- Suggest stopping when cluster selection stabilizes, subset size becomes small, or all remaining clusters look relevant.
- Export final selected records to CSV for screening.

Core logic to preserve:

- The current notebook pattern of repeated cluster subsetting and re-clustering.
- Search-by-title inspection aids.
- Export of labeled subset CSVs.

Human decisions:

- Which clusters to include/exclude at every iteration.
- Whether the suggested stopping point is acceptable.
- Whether to manually include/exclude individual records before export.

Suggested stopping criteria:

- Reviewer marks all visible clusters as relevant enough for screening.
- Subset size falls below a configured threshold.
- Two consecutive reclustering rounds produce no meaningful exclusion decision.
- Reviewer explicitly chooses to stop and export.

### 6. AI screening

Screen goals:

- Load final selected records.
- Display criteria used for screening.
- Run a small test batch first.
- Present decision distribution, confidence distribution, and sample reasoning.
- Allow criteria/prompt edits before the full run.
- Run full screening with progress, retries, and resume controls.
- Provide downloadable CSV and report outputs.

Core logic to preserve:

- Pydantic structured `ScreeningDecision` schema.
- `include` / `exclude` / `uncertain` decisions.
- Confidence and reason fields.
- Batch saving and resume behavior.
- Visualization/reporting functions where possible.

Human decisions:

- Criteria changes after test batch.
- Model and reasoning effort.
- Whether to include all selected records or a filtered subset.
- How to adjudicate uncertain/low-confidence records.
- Whether to export all AI decisions or only include/uncertain records.

### 7. Optional AI extraction

Screen goals:

- Load included or included-plus-uncertain records.
- Let users confirm whether full text is available or only abstracts are used.
- Run a small extraction test batch.
- Display extracted fields and confidence/completeness summary.
- Run full extraction with progress/resume.
- Export study characteristics, effect estimates, JSON, and full extraction CSV.

Core logic to preserve:

- Pydantic extraction models.
- Structured OpenAI outputs.
- Incremental writes.
- Export functions for derived tables.

Human decisions:

- Whether uncertain records are extracted.
- Whether abstract-only extraction is acceptable.
- Which model/reasoning effort to use.
- Whether extraction quality is sufficient before full run.

## Refactoring phases

### Phase 0: Baseline capture

- Freeze current notebook behavior with a small fixture dataset.
- Create sample PubMed XML/CSV fixtures that do not require network or paid APIs.
- Record expected outputs for parsing, deduplication, embedding serialization, clustering shape, and screening/extraction schemas.

Deliverable: tests that describe current behavior before app migration begins.

### Phase 1: Extract reusable services

- Move PubMed helper functions into `services/pubmed_service.py`.
- Move CSV upload, column mapping, and canonical schema helpers into `services/import_service.py`.
- Move title normalization/deduplication helpers into `services/deduplication_service.py`.
- Move embedding helpers into `services/embedding_service.py`.
- Move clustering helpers into `services/clustering_service.py`.
- Move screening schema/API wrappers into `services/screening_service.py`.
- Move extraction schemas/API wrappers into `services/extraction_service.py`.
- Keep notebooks as thin examples that call the services, or leave them untouched until parity is confirmed.

Deliverable: importable service modules with unit tests and fixture-based outputs matching notebook behavior.

### Phase 2: Flask scaffold

- Add Flask app factory, blueprints, templates, static assets, and runtime project storage.
- Add simple project creation/loading.
- Add forms for search strategy, inclusion criteria, CSV upload/column mapping, deduplication settings, and API-key entry.
- Add download routes for generated files, normalized uploads, and duplicate reports.

Deliverable: local Flask app runs without calling external APIs and can create/load project state.

### Phase 3: Data import, deduplication, and embedding UI

- Implement PubMed count/fetch screens.
- Implement CSV upload, preview, column mapping, and canonicalization screens.
- Implement deduplication settings, duplicate report preview, manual review option, and deduplicated CSV export.
- Implement progress display and resumable job state.
- Implement embedding screen with batch progress.
- Mock API calls in tests.

Deliverable: UI can fetch or upload records, map/deduplicate them, and embed the selected dataset in a controlled local/dev environment.

### Phase 4: Plotly clustering UI

- Implement t-SNE scatterplots, elbow plots, K selection, cluster summaries, and record tables.
- Implement cluster selection and progressive reclustering history.
- Add export to AI-screening input.

Deliverable: reviewer can iteratively narrow clusters and export selected records.

### Phase 5: AI screening UI

- Implement test-batch workflow.
- Implement full-screening workflow with progress/resume.
- Display decision/confidence charts and sample reasoning.
- Add downloads for screening CSV/report artifacts.

Deliverable: reviewer can run and download AI screening outputs from the app.

### Phase 6: Optional AI extraction UI

- Implement extraction test batch and full extraction.
- Display confidence/completeness summaries.
- Add downloads for extraction CSV/JSON/table exports.

Deliverable: app supports complete search-to-extraction workflow.

### Phase 7: Docker packaging and release hardening

- Add `Dockerfile` and `compose.yaml`.
- Add environment-variable configuration for runtime directories and optional API keys.
- Add production-safe Flask settings and static asset handling.
- Add documentation for local and Docker runs.

Deliverable: standalone Dockerized tool that can run locally with mounted project storage.

## Testing plan

### Unit tests

- PubMed query/date-window splitting.
- PubMed XML parsing from fixtures.
- CSV upload validation, delimiter/encoding handling, and schema detection.
- Column mapping validation for unique ID, title, abstract/text, and optional metadata fields.
- PMID/source-ID deduplication and processed-ID loading.
- Configurable duplicate detection by normalized title, DOI, unique ID, or reviewer-selected match columns.
- Embedding input batching and output serialization with mocked OpenAI responses.
- Clustering array conversion, scaling, t-SNE output shape, WCSS calculation, and K-Means labels.
- Screening Pydantic schema validation and response flattening with mocked OpenAI responses.
- Extraction Pydantic schema validation, flattening, and export-table generation.
- Title normalization and fuzzy matching for evaluation.

### Integration tests

- Create project -> save query/criteria -> run mocked PubMed fetch -> generate mocked embeddings -> cluster -> select clusters -> export selected records.
- Create project -> upload CSV -> select unique ID and match columns -> deduplicate -> generate mocked embeddings -> cluster -> export selected records.
- Selected records -> mocked AI screening -> report/download generation.
- Included records -> mocked AI extraction -> derived exports.
- Resume behavior for interrupted PubMed, embedding, screening, and extraction jobs.

### UI tests

- Form validation for missing search strategy, criteria, CSV unique-ID mapping, match-column choices, and API keys.
- Uploaded CSV preview handles missing/duplicate IDs and warns before destructive deduplication.
- Password/API-key fields are not echoed in logs or page source after submission.
- Plotly t-SNE and elbow charts render with fixture data.
- Cluster selection persists across progressive rounds.
- Downloads return the expected files and content types.

### Docker tests

- Image builds from a clean checkout.
- Container starts with mounted `runtime/` volume.
- Health endpoint succeeds.
- Fixture-based workflow runs inside the container without network or paid API calls.

### Safety and reproducibility tests

- Ensure uploaded CSVs and generated outputs live under `runtime/` or another gitignored directory.
- Ensure API keys are never written to project manifests, CSVs, or logs.
- Ensure model names, prompts/criteria, and run timestamps are recorded with outputs.
- Ensure all external API tests are mocked by default.

## Acceptance criteria

The migration should be considered complete only when:

- A user can complete the workflow from search strategy through downloadable AI screening results in the Flask UI.
- The app can optionally continue through structured AI extraction and export results.
- Core parsing, CSV upload/mapping, deduplication, embedding serialization, clustering, screening schema, and extraction schema behavior are covered by tests.
- The app runs locally and in Docker.
- Long-running stages expose progress, resume, and clear failure messages.
- API keys are handled safely and are not persisted insecurely.
- Human review points are explicit in the UI before every consequential decision.
- Documentation explains both local and Docker usage.
