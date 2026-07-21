# Flask Review Application

This application provides an end-to-end web workflow alongside the repository's notebooks. Generated projects and results are stored under `runtime/` and remain uncommitted.

## Configure credentials

Keep credentials out of git. If `scripts/secret_keys.py` already defines `OPENAI_API_KEY` and `PUBMED_API_KEY`, create the Compose environment file with:

```bash
python scripts/export_secrets_to_env.py
```

This creates a gitignored `.env` with mode `0600`, adds a random Flask session secret, and does not print key values. Alternatively, copy `.env.example` to `.env` and populate it manually. A key typed in the interface overrides the configured key for that request and is not persisted.

## Run with Docker

```bash
docker compose up --build
```

Open <http://127.0.0.1:5000>. If that port is occupied:

```bash
ML_REVIEW_PORT=5055 docker compose up --build
```

## Run locally

```bash
pip install -r requirements_.txt
npm ci
npm run build:atlas
flask --app wsgi:app run --debug
```

The local frontend build requires Node.js 22. Docker builds the pinned Atlas bundle automatically.

## Workflow

- Create or load an isolated review project.
- Save a PubMed search strategy and inclusion/exclusion criteria.
- Fetch real PubMed records, or upload, map, normalize, and deduplicate a CSV.
- Generate resumable OpenAI embeddings.
- Optionally build a reproducible Evidence Atlas from every embedded record for local search, cross-filtering, nearest-neighbour inspection, and selection export. Atlas browser state is project/fingerprint scoped and does not change clustering or screening selections.
- Analyze WCSS before choosing K for every root or child projection, then create reproducible t-SNE/K-Means runs in an immutable branch tree with parent navigation and clickable abstract inspection.
- Select clusters and run resumable OpenAI Structured Outputs screening.
- Review screening results in a searchable table and explore interactive Plotly funnel, Sankey, confidence, criterion, exclusion, and t-SNE evaluation views.
- Optionally compare AI screening with a human-reference CSV using one-to-one fuzzy title matching and downloadable metrics/mismatches.
- Download generated CSV artifacts.

AI screening is decision support. Human reviewers must validate prompts and model choices on a sample, review every uncertain and low-confidence result, compare with human screening when available, and record criteria/model/date/prompt changes.

## Operational limitations

API work runs in the Flask request process. Large searches and screening jobs should use a production job queue before multi-user deployment. The application currently exports CSV files; polished extraction and reporting screens remain future work.
