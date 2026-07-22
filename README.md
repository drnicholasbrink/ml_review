# ML Review

ML Review is a human-in-the-loop systematic-review workspace. It combines a production-like localhost web application with the repository's original Jupyter notebooks for PubMed retrieval, deduplication, embeddings, evidence exploration, AI-assisted screening, full-text review, evaluation, extraction, and publication handoff.

AI output is decision support—not a final clinical or scientific judgment. Human decisions remain separate, override AI decisions when they disagree, and are preserved in the exported audit trail.

## What it supports

- Isolated review projects with guarded project deletion.
- PubMed search and bounded background retrieval, or CSV import and column mapping.
- Duplicate detection with a retained/removed audit.
- Resumable OpenAI embeddings.
- A reproducible Evidence Atlas artifact with precomputed UMAP coordinates and cosine neighbors for Apple's official Embedding Atlas.
- WCSS-first t-SNE/K-Means exploration with immutable root and child branches.
- Structured title-and-abstract screening with broad exclusion categories.
- Focus or list-mode human review, DOI/PubMed links, compound filters, and explicit AI/human/final decision layers.
- An optional full-text stage with per-record PDF upload, resumable PDF-backed AI screening, human overrides, and disagreement filtering.
- Human-reference evaluation using either explicit workflow decisions or a separate included-records CSV.
- Resumable structured extraction from uploaded PDFs, with an explicitly labelled abstract fallback.
- A publication bundle containing protocol inputs, decision audits, evaluation, and extraction artifacts.

## Supported deployment profile

The supplied application is designed for a trusted, single-reviewer workstation. Docker Compose binds it to localhost, runs one Gunicorn worker to serialize filesystem-backed writes, and stores generated project data under the gitignored `runtime/` directory.

Remote or concurrent multi-user deployment is outside this profile. It requires authentication and authorization, TLS, an external durable queue, shared transactional storage, centralized audit logging, backups, retention controls, and an appropriate privacy/security review.

## Quick start with Docker

1. Create the local environment file:

   ```bash
   cp .env.example .env
   ```

2. Add a non-default Flask session secret and any API keys you want the application to use:

   ```dotenv
   ML_REVIEW_SECRET_KEY=replace-with-a-long-random-value
   OPENAI_API_KEY=
   PUBMED_API_KEY=
   ```

   If `scripts/secret_keys.py` already contains the keys, generate `.env` without printing their values:

   ```bash
   python scripts/export_secrets_to_env.py
   ```

3. Build and start the application:

   ```bash
   docker compose up --build
   ```

4. Open <http://localhost:5055>.

To use another host port:

```bash
ML_REVIEW_PORT=5056 docker compose up --build
```

The Docker image installs the exact tested Python set from `requirements_app.lock`. No Node.js frontend build is required to run the application.

## Local application development

```bash
pip install -r requirements_.txt
flask --app wsgi:app run --debug --port 5055
```

Run the tests with:

```bash
pytest -q
```

Or validate inside the release container:

```bash
docker compose run --rm ml-review pytest -q
```

## Notebook environment

The notebooks remain a supported manual workflow. Create the Conda environment with:

```bash
conda env create -f environment.yml
conda activate ml_review
```

Optionally register a kernel:

```bash
python -m ipykernel install --user --name ml_review --display-name "Python (ml_review)"
```

Notebook API keys are read from a local, gitignored `secret_keys.py`:

```python
OPENAI_API_KEY = "your-openai-api-key"
PUBMED_API_KEY = "your-ncbi-api-key"
```

## Web workflow

### 1. Review setup

Save the PubMed query and inclusion/exclusion criteria that define the review. Changing these inputs invalidates stale downstream artifacts while leaving source files available where safe.

### 2. Retrieve or import records

Count and fetch a deliberately bounded PubMed result set, or upload a CSV, inspect its preview, map source columns, and normalize the records.

### 3. Deduplicate

Review duplicate groups before producing the deduplicated source. The retained and removed records remain available as an audit artifact.

### 4. Generate embeddings and explore the Evidence Atlas

Generate resumable OpenAI embeddings. The optional Evidence Atlas step writes a slim Parquet handoff with deterministic UMAP coordinates and nearest neighbors. Local development offers a download-and-drop workflow; a publicly reachable HTTPS deployment can preload the data in Apple's official viewer when `ML_REVIEW_PUBLIC_BASE_URL` is configured.

Atlas exploration does not modify clustering or screening selections.

### 5. Cluster and select records

Inspect WCSS before choosing K. Saved t-SNE/K-Means runs form an immutable branch tree, allowing reviewers to reopen a parent seed and create a new branch without overwriting prior exploration.

Clusters are navigation aids, not eligibility decisions.

### 6. Screen and review

Run structured title-and-abstract screening, then review records one at a time or in a list. Reviewers can filter independently by review status, final decision, AI decision, identifiers, citation text, or keywords.

Decision provenance is explicit:

- **AI decision** is the model output.
- **AI accepted** records that a reviewer accepted the AI result without fabricating a human decision.
- **Human decision** is an explicit reviewer judgment.
- **Final decision** uses the human decision whenever one exists; disagreement remains visible.

Eligible records can proceed to the optional full-text stage. Upload a PDF on the record card, review it manually, or run resumable PDF-backed AI screening. Full-text exclusions require a controlled broad category and a concise record-specific reason. Reviewers may skip full-text screening and continue directly to extraction.

### 7. Evaluate

Evaluation includes locally served Plotly funnel, Sankey, confidence, criterion, exclusion-category, and t-SNE views.

Choose one of two human-reference modes:

- **Workflow decisions:** use only explicit human title/abstract or full-text decisions. Full-text human decisions take precedence, and unreviewed or AI-accepted records are not counted as negatives.
- **Included-records CSV:** upload an independent CSV containing `Title`. Each row is treated as included unless the file also contains a supported decision or status column.

Both modes support configurable fuzzy title matching and uncertain-as-retained behavior. Inspect unmatched or ambiguous titles before interpreting sensitivity, specificity, precision, or F1 metrics.

### 8. Extract and hand off

Run structured extraction on a bounded test sample before the full eligible set. Each record uses its uploaded PDF when present; otherwise the output is marked as an abstract fallback. Validate every extracted field and effect estimate against the source before analysis or publication.

The publication bundle excludes credentials, embeddings, and uploaded copyrighted PDFs.

## Background tasks

PubMed fetching, embedding generation, title/abstract screening, full-text screening, and extraction run in a serial background worker. The **Tasks** view records queued, running, succeeded, failed, and interrupted states with progress and links back to the relevant workflow step.

Task state is persisted within the project directory. Submitted API keys remain in process memory and are never written to task files. After an interruption, resumable CSV-based steps can safely continue completed records.

## Scientific and privacy safeguards

- Validate model and prompt behavior on a human-reviewed sample before a full run.
- Manually resolve uncertain, low-confidence, and AI/human disagreement records.
- Treat “AI accepted” as an audit status, not a human judgment.
- Verify PDF-assisted and abstract-fallback extraction against source documents.
- Record the search, criteria, model names, run date, prompt/schema changes, and application commit hash.
- Review fuzzy title matches and unmatched records before reporting evaluation metrics.
- Full-text uploads must be PDFs no larger than 50 MB. They remain within the owning project's runtime directory, are removed with the project, and are omitted from publication bundles.
- Never commit `.env`, `secret_keys.py`, runtime projects, generated outputs, or API credentials.

## Publication handoff checklist

- Confirm the search query, date range, fetched count, and duplicate audit.
- Freeze the inclusion/exclusion criteria before the final screening run.
- Validate the model and prompts on a representative human-reviewed sample.
- Resolve required title/abstract decisions and any full-text AI/human disagreements.
- Record a broad category and specific rationale for every human full-text exclusion.
- Evaluate against workflow human decisions or an independent included-records CSV and inspect fuzzy-match coverage.
- Confirm each extraction identifies its source as PDF or abstract fallback and manually verify the result.
- Download the publication bundle and retain the application commit hash with the review record.

## Notebook workflow

The original notebooks live under `scripts/` and are intended to be run manually in order:

1. `search_pubmed.ipynb` — fetch PubMed records using `search_strategy.txt`.
2. `extract_pubmed.ipynb` or `extract_csv.ipynb` — generate embeddings.
3. `deduplicate.ipynb` — deduplicate a prepared record source where needed.
4. `clustering.ipynb` — explore t-SNE/K-Means projections and cluster subsets.
5. `ai_screening.ipynb` — run structured title/abstract screening; see `scripts/AI_SCREENING_GUIDE.md`.
6. `ai_extraction.ipynb` — extract structured evidence from retained records.
7. `eval.ipynb` — compare AI output with a human reference using fuzzy title matching.

Some historical exploratory notebook cells may contain legacy paths or SDK patterns. When editing a notebook, prefer repository-relative paths and the modern `OpenAI` client and structured-output APIs.

## Repository layout

| Path | Purpose |
| --- | --- |
| `ml_review_app/` | Flask blueprints, services, templates, and static assets. |
| `scripts/` | Notebook workflow, search/criteria inputs, and helper scripts. |
| `tests/` | Service and end-to-end Flask workflow tests. |
| `runtime/` | Gitignored project state and generated application artifacts. |
| `environment.yml` | Conda notebook environment. |
| `requirements_.txt` | Notebook and local-development Python dependencies. |
| `requirements_app.lock` | Exact Docker application dependency set. |
| `compose.yaml` | Supported localhost application profile. |
| `THIRD_PARTY_NOTICES.md` | Required notices for bundled third-party components. |

## Troubleshooting

- **Application will not start:** confirm `.env` contains a non-default `ML_REVIEW_SECRET_KEY`.
- **`localhost` does not resolve as expected:** open <http://127.0.0.1:5055> and confirm no proxy or hosts-file rule overrides localhost.
- **PubMed throttling:** use an NCBI API key, smaller result/date windows, and the resumable fetch path.
- **OpenAI cost concerns:** start with test samples and retain resume mode to avoid duplicate calls.
- **CSV errors:** verify the file has a header and maps `RecordID` plus at least one of `Title` or `Abstract`.
- **Notebook path errors:** run from the repository root or convert historical absolute paths to repository-relative paths.
- **Static Plotly export errors:** HTML output works without Kaleido; PDF or image export can require additional browser/system packages.

## Licensing

See `THIRD_PARTY_NOTICES.md` for third-party software notices. The locally vendored Lucide icon subset retains its ISC license under `ml_review_app/static/icons/LICENSE`.
