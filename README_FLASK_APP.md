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

Open <http://localhost:5055>. The workstation defaults to port 5055 because macOS may reserve IPv6 port 5000 for AirPlay/AirTunes. To choose another host port:

```bash
ML_REVIEW_PORT=5056 docker compose up --build
```

## Run locally

```bash
pip install -r requirements_.txt
flask --app wsgi:app run --debug --port 5055
```

No Node.js frontend build is required. Docker installs the exact tested Python set from `requirements_app.lock`; notebook/development setup continues to use `environment.yml` or `requirements_.txt`.

## Workflow

- Create or load an isolated review project.
- Save a PubMed search strategy and inclusion/exclusion criteria.
- Fetch real PubMed records, or upload, map, normalize, and deduplicate a CSV.
- Generate resumable OpenAI embeddings.
- Optionally build a reproducible, slim Evidence Atlas artifact with precomputed UMAP coordinates and cosine neighbors, then open it in Apple's official Embedding Atlas with the projection, search text, and neighbors already mapped. A direct Parquet download remains available. Atlas exploration does not change clustering or screening selections.
- Analyze WCSS before choosing K for every root or child projection, then create reproducible t-SNE/K-Means runs in an immutable branch tree with parent navigation and clickable abstract inspection.
- Select clusters and run resumable OpenAI Structured Outputs screening.
- Review source abstracts and AI rationales in a paginated adjudication queue. Human decisions and notes are timestamped, preserve the original AI audit trail, and become the final decisions used downstream.
- Explore locally served Plotly funnel, Sankey, confidence, criterion, exclusion, and t-SNE evaluation views.
- Optionally compare AI screening with a human-reference CSV using one-to-one fuzzy title matching and downloadable metrics/mismatches.
- Run resumable abstract-only structured extraction on a bounded test sample before the full included set, then export nested JSON, study characteristics, effect estimates, and an extraction summary.
- Download individual artifacts or a publication handoff ZIP containing the protocol inputs, decision audit, evaluation, and extraction outputs. Embeddings and credentials are excluded from the bundle.

AI screening and extraction are decision support. Human reviewers must validate prompts and model choices on a sample, review every uncertain and low-confidence screen, validate abstract extraction against source full text, compare with human screening when available, and record criteria/model/date/prompt changes.

## Background tasks

PubMed fetches, embedding generation, AI screening, and structured extraction run in a serial background worker so the browser request returns immediately. Every project has a **Tasks** view with queued, running, succeeded, and failed states; completed/total counts; a progress bar; safe error guidance; and links back to the relevant workflow step. The current task is also visible while navigating elsewhere in the project.

Task status is persisted under the project's runtime directory, while submitted API keys remain in process memory and are never written to task files. Only one task or other write operation may mutate a project at a time. If the app restarts mid-run, the task is marked interrupted and the existing resumable CSV workflow can continue it safely.

## Supported deployment profile

The supplied Compose profile is a production-like, single-reviewer workstation deployment. It binds only to `127.0.0.1`, requires a non-default session secret, uses one Gunicorn worker to serialize filesystem-backed project writes, exposes liveness/readiness checks, serves chart code locally, and applies CSRF and browser security headers.

The official Atlas viewer can fetch the generated Parquet artifact directly from a publicly reachable HTTPS deployment. Secure browser pages cannot fetch from this app's local HTTP address, so local development uses the provided download-and-drop handoff. To enable one-click preload in production, set `ML_REVIEW_PUBLIC_BASE_URL` to the externally reachable HTTPS origin, for example `https://reviews.example.org`.

The bundled background worker is intentionally process-local and sized for the supported single-reviewer workstation deployment. Multi-user or remotely exposed deployment is outside this profile and requires authentication, an external job queue, shared transactional storage, TLS, and an explicit data-governance review.

## Publication handoff checklist

- Confirm the search query, date range, fetched count, and deduplication report.
- Record and freeze the inclusion/exclusion criteria before the final screening run.
- Validate the chosen model and prompt behavior on a human-reviewed sample.
- Resolve every uncertain and low-confidence screening item and sample-check the rest.
- Compare against an independent human reference when one is available; inspect fuzzy-match mismatches.
- Validate every extracted field and effect estimate against the full text. The interface extracts from abstracts only.
- Download the publication bundle and retain the application commit hash alongside it.
