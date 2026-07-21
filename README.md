# ML Review

## Overview

This repository contains a notebook-first workflow for machine-learning-assisted systematic review work. It retrieves PubMed records, enriches abstracts with embeddings, clusters records for exploratory review, screens abstracts against inclusion criteria with OpenAI structured outputs, extracts structured data from included studies, and optionally compares AI decisions with human review exports.

The project is intentionally organized around manually run notebooks rather than a single automated pipeline. Long-running steps call external APIs, produce large intermediate files, and require human validation.

An accompanying localhost web workflow covers the same review lifecycle with resumable project state, an Evidence Atlas, human screening adjudication, evaluation, structured extraction, and a publication handoff bundle. See [README_FLASK_APP.md](README_FLASK_APP.md) for operation and [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for the validated deployment boundary and scientific safeguards.

## Repository layout

- `environment.yml` — Conda environment definition for the notebook stack.
- `requirements_.txt` — pip-style dependency list mirroring the Python packages in the Conda environment.
- `requirements_app.lock` — exact tested Python dependency set used by the Docker web application.
- `.gitignore` — excludes secrets, logs, caches, and generated PubMed CSVs.
- `scripts/search_strategy.txt` — PubMed query string used by the search notebook.
- `scripts/inclusion_criteria.txt` — criteria used by the AI screening notebook.
- `scripts/search_pubmed.ipynb` — queries PubMed through NCBI E-utilities and writes article metadata.
- `scripts/extract_pubmed.ipynb` — creates OpenAI embeddings for abstracts.
- `scripts/extract_csv.ipynb` — embeds records from an existing CSV source.
- `scripts/deduplicate.ipynb` — helper notebook for deduplicating clustered/exported records.
- `scripts/clustering.ipynb` — runs t-SNE/K-Means exploration and writes cluster visualizations.
- `scripts/ai_screening.ipynb` — screens abstracts with OpenAI structured outputs.
- `scripts/AI_SCREENING_GUIDE.md` — detailed guide for the AI screening workflow.
- `scripts/ai_extraction.ipynb` — extracts structured study data with OpenAI structured outputs and Pydantic models.
- `scripts/eval.ipynb` — compares AI screening output to a human screening CSV using fuzzy title matching.

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate ml_review
```

Optionally register a Jupyter kernel:

```bash
python -m ipykernel install --user --name ml_review --display-name "Python (ml_review)"
```

If you are not using Conda, install the Python dependencies with:

```bash
pip install -r requirements_.txt
```

## Secrets and configuration

Create a local `secret_keys.py` file when running notebooks that call PubMed or OpenAI:

```python
OPENAI_API_KEY = "your-openai-api-key"
PUBMED_API_KEY = "your-ncbi-api-key"
```

`secret_keys.py` is ignored by git and must not be committed. Generated logs, PubMed ID caches, and PubMed result CSVs are also ignored.

## Current dependencies and OpenAI usage

This repository now targets the modern OpenAI Python SDK (`openai>=1.0.0`) with Pydantic (`pydantic>=2.0.0`). New OpenAI code should use the `OpenAI` client and structured-output patterns used by `scripts/ai_screening.ipynb` and `scripts/ai_extraction.ipynb`.

Some older exploratory notebooks may still contain legacy OpenAI patterns or historical local paths. Prefer updating touched cells to relative paths and modern SDK usage rather than copying legacy patterns forward.

## End-to-end workflow

### 1. Define the PubMed search

Edit `scripts/search_strategy.txt` with the PubMed query for the review. Keep shareable examples generic and avoid committing sensitive or project-private details.

### 2. Fetch PubMed records

Run `scripts/search_pubmed.ipynb` from the repository root or adjust paths accordingly. The notebook:

- Reads `scripts/search_strategy.txt`.
- Calls PubMed ESearch/EFetch endpoints with the NCBI API key.
- Splits large searches by date window to stay within API/result constraints.
- Caches PMIDs in `pubmed_ids.json` to avoid duplicate processing across runs.
- Writes `pubmed_results_complete.csv` with fields such as PMID, title, abstract, authors, date, and journal.
- Logs progress to `pubmed_fetch.log`.

### 3. Create embeddings

Run `scripts/extract_pubmed.ipynb` to embed abstracts from `pubmed_results_complete.csv`. It writes `pubmed_results_with_embeddings.csv` with an `Embedding` column containing serialized vectors.

If you already have a CSV source, `scripts/extract_csv.ipynb` provides a related embedding path and writes `csv_results_with_embeddings.csv`.

### 4. Explore clusters

Run `scripts/clustering.ipynb` after embeddings exist. The notebook converts stored embeddings to arrays, scales them, reduces them to two dimensions with t-SNE, clusters with K-Means, and writes cluster CSVs and Plotly visualizations under `outputs/`.

Cluster counts, search terms, and subsets are manual exploratory choices. Treat cluster labels as review aids, not final inclusion/exclusion decisions.

### 5. Screen abstracts with AI

Review `scripts/AI_SCREENING_GUIDE.md`, then run `scripts/ai_screening.ipynb`. The notebook:

- Loads PubMed/embedding records.
- Loads screening criteria from `scripts/inclusion_criteria.txt`.
- Uses OpenAI structured outputs to return validated screening decisions.
- Saves incremental results under `outputs/`, including test and full screening CSVs.
- Produces summary reports and optional visualizations.

Recommended practice is to test on a small sample, refine criteria/prompts, then run a full screen. Human reviewers should manually review all `uncertain` and low-confidence decisions.

### 6. Extract structured data

Run `scripts/ai_extraction.ipynb` after screening. It filters included studies from `outputs/ai_screening_full_results.csv`, calls OpenAI with Pydantic schemas, and writes extraction CSV/JSON exports under `outputs/`.

Start with a small test batch before full extraction because this step can make many paid API calls.

### 7. Evaluate against human screening

Use `scripts/eval.ipynb` when a human screening export is available. It uses fuzzy title matching to align AI and human records, then reports sensitivity/specificity-style metrics and mismatch files.

Before running it, update any historical absolute paths to local relative paths under `scripts/outputs/` or another project-specific output directory.

## Manual steps and review controls

This workflow requires explicit human decisions at several points:

- Confirm the PubMed query and date/search scope.
- Verify API keys and rate-limit behavior before large runs.
- Inspect sample AI screening and extraction outputs before full runs.
- Choose clustering parameters and interpret clusters manually.
- Review all uncertain or low-confidence AI decisions.
- Validate AI screening/extraction against human judgments where available.

## Generated files

Typical generated files include:

- `pubmed_ids.json`
- `pubmed_fetch.log`, `extract_pubmed.log`, `ai_screening.log`, `ai_extraction.log`, `clustering.log`
- `pubmed_results_complete.csv`
- `pubmed_results_with_embeddings.csv`
- `csv_results_with_embeddings.csv`
- `outputs/*.csv`, `outputs/*.html`, `outputs/*.pdf`, and related report files

Do not commit generated outputs unless they are intentionally small fixtures and the user requests them.

## Troubleshooting

- **Missing `secret_keys.py`:** Create the local file with `OPENAI_API_KEY` and/or `PUBMED_API_KEY`.
- **PubMed throttling or incomplete results:** Increase sleep/backoff settings and use smaller date windows.
- **OpenAI cost concerns:** Test on a small batch first, use lower-cost models for development, and rely on resume/progress files to avoid duplicate calls.
- **Notebook path errors:** Run notebooks from the expected working directory or convert hard-coded paths to relative paths.
- **Plotly image export errors:** HTML export usually works with Plotly alone; PDF/static image export may require Kaleido and additional system/browser dependencies.
- **CSV schema errors:** Check that required columns such as `PMID`, `Title`, and `Abstract` are present before downstream embedding, screening, or extraction.

## Reproducibility notes

For each full review run, record:

- Search query and date run.
- Inclusion/exclusion criteria.
- Notebook versions or commit hash.
- OpenAI model names and reasoning/temperature settings.
- Manual screening or extraction validation process.
- Any prompt or schema changes made during calibration.

## Detailed code and notebook behavior

### `scripts/search_pubmed.ipynb`

This notebook is the ingestion step. It imports `requests`, `pandas`, XML parsing utilities, logging, and `PUBMED_API_KEY` from `secret_keys.py`. Its main behavior is split across helper functions that:

- load already-processed PMIDs from an existing result CSV;
- count PubMed records for a query/date range;
- retrieve new PMIDs for a query/date range;
- fetch article details in batches of up to 200 IDs;
- parse PubMed XML into tabular article metadata;
- append records incrementally to `pubmed_results_complete.csv`; and
- recursively split date ranges by year, month, or day when a search window is too large.

The important human decisions are the PubMed query, the overall date range, whether the NCBI key is available, how aggressively to split/date-limit searches, and whether to resume from or delete existing `pubmed_ids.json` / CSV outputs before rerunning.

### `scripts/extract_pubmed.ipynb` and `scripts/extract_csv.ipynb`

These notebooks are embedding steps. They use the OpenAI API key, batch input rows, call the embeddings API with `text-embedding-3-small`, serialize each embedding as JSON, and append results to an output CSV. `extract_pubmed.ipynb` expects `pubmed_results_complete.csv`; `extract_csv.ipynb` supports an existing CSV source and writes `csv_results_with_embeddings.csv`.

The main human decisions are which source CSV to embed, which text column should represent the article content, the embedding batch size, whether to resume from an existing output file, and whether API cost/rate limits are acceptable before running the full file.

### `scripts/clustering.ipynb`

This is an exploratory analysis notebook rather than a deterministic screening step. It reads `csv_results_with_embeddings.csv`, converts serialized embeddings to arrays, standardizes them, reduces them to two dimensions with t-SNE, clusters the projection with K-Means, and writes interactive/static outputs under `outputs/`. Later cells repeatedly subset the data by selected clusters or title search terms, rerun t-SNE/K-Means on those subsets, and save more focused cluster outputs.

The main human decisions are the clustering target, search terms used to inspect clusters, which clusters to keep or discard, t-SNE settings, K-Means cluster counts, and how to interpret exploratory plots. Cluster assignments should be treated as navigation aids for reviewers, not evidence or final inclusion/exclusion labels.

### `scripts/ai_screening.ipynb`

This is the AI-assisted title/abstract screening notebook. It defines a Pydantic `ScreeningDecision` schema and functions to load criteria, load PubMed data, call OpenAI structured outputs, batch-screen records with progress saving, merge screening results with embeddings, analyze decisions, generate visualizations, create a PDF report, and produce PRISMA-style data. The notebook uses GPT-style reasoning settings, defaults to `gpt-5` in its runnable examples, and saves test/full outputs under `outputs/`.

The main human decisions are the exact inclusion criteria, model choice, reasoning effort, whether to run only a test sample or the full screen, batch size, rate-limit delay, confidence thresholds for manual review, and how to handle `uncertain` decisions. The intended workflow is to test a small sample, inspect errors, refine criteria/prompts, run the full screen, and then manually review uncertain or low-confidence records.

### `scripts/ai_extraction.ipynb`

This notebook performs structured data extraction after screening. It defines Pydantic models for geographic locations, exposure metrics, outcome measures, effect estimates, and the full `DataExtraction` payload. It loads included records from `outputs/ai_screening_full_results.csv`, optionally includes uncertain records, calls OpenAI structured outputs, saves incremental extraction results, analyzes extraction completeness/confidence, and exports derived tables such as study characteristics and effect estimates.

The main human decisions are whether to extract from abstracts or add full text, whether to include `uncertain` screened records, which model and reasoning effort to use, the batch size/rate-limit delay, how to validate a sample extraction against manual abstraction, and which exported tables are appropriate for analysis or manuscript preparation.

### `scripts/deduplicate.ipynb`

This helper notebook/script removes duplicate studies by normalized title. It is aimed at outputs such as clustered subsets and keeps a single representative row for each normalized title.

The main human decisions are which file should be deduplicated, whether title-only deduplication is adequate, and whether potential duplicates need manual inspection before deletion.

### `scripts/eval.ipynb`

This notebook evaluates AI screening against a human-screening export. It loads an AI output CSV and a gold/human CSV, normalizes titles, fuzzy-matches records, calculates agreement-style metrics, and writes reports listing matched records and mismatches. It currently contains historical absolute local paths that should be changed before reuse.

The main human decisions are which human file is the reference standard, what fuzzy-match threshold is acceptable, whether ambiguous matches need manual correction, and how to interpret sensitivity/specificity results in light of title-matching errors.

## Human-in-the-loop decisions by stage

| Stage | Human decision | Why it matters |
| --- | --- | --- |
| Search | PubMed query, date range, and whether to use/reset cached PMIDs | Defines the universe of candidate studies and affects reproducibility. |
| Fetch | Retry/backoff behavior and whether results are complete enough | PubMed throttling or large result sets can silently shape coverage. |
| Embedding | Source CSV, text field, batch size, and model | Determines downstream similarity/clustering quality and API cost. |
| Clustering | t-SNE parameters, K-Means cluster count, title search terms, and selected subsets | Clustering is exploratory and requires reviewer interpretation. |
| Screening | Inclusion criteria, model, reasoning effort, test-vs-full run, and confidence handling | These settings directly affect include/exclude/uncertain labels. |
| Manual review | Which AI decisions to audit or override | Required because AI screening is decision support, not the final review decision. |
| Extraction | Abstract vs full-text source, extraction schema, included-vs-uncertain scope, and validation sample | Determines data quality for evidence synthesis. |
| Evaluation | Human reference file and fuzzy-match threshold | Controls the validity of sensitivity/specificity-style metrics. |
| Reporting | Which outputs are appropriate to publish or cite | Prevents overclaiming exploratory/AI-generated artifacts. |

## Current implementation caveats

- The project is notebook-first, so execution order and working directory assumptions matter.
- Several notebooks have one large code cell or manual exploratory cells rather than reusable modules.
- Some historical absolute paths remain in exploratory/evaluation notebooks and should be replaced before reuse.
- API-running notebooks can cost money and should start with small test batches.
- Generated files are not committed by default, so downstream notebooks expect local outputs from prior stages.
- Some Plotly static exports may need additional dependencies such as Kaleido; HTML exports are generally less fragile.
