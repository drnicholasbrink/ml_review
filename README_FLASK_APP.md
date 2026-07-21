# Flask App Prototype

This branch contains a Flask prototype that begins implementing the migration plan in `FLASK_APP_MIGRATION_PLAN.md`.

## Run locally

```bash
pip install -r requirements_.txt
flask --app wsgi:app run --debug
```

Open <http://127.0.0.1:5000>.

## Run with Docker

```bash
docker compose up --build
```

The app stores projects under `runtime/`, which is mounted in Docker and should remain uncommitted.

## Implemented workflow

- Create/load review projects.
- Save PubMed search strategy and inclusion criteria text.
- Upload a CSV and preview its columns.
- Map a unique ID plus title/abstract/date/journal/DOI columns.
- Normalize uploaded CSVs to canonical review columns.
- Deduplicate records using normalized title or reviewer-selected match columns.
- Generate deterministic offline embeddings for no-cost local testing.
- Run t-SNE/PCA fallback, elbow scores, K-Means clustering, and Plotly charts.
- Select clusters and export selected records to screening.
- Run deterministic offline screening to exercise the structured-output CSV flow without paid API calls.
- Download generated CSVs.

## Remaining production work

- Harden PubMed ESearch/EFetch for very large searches with full date-window splitting parity and background jobs.
- Wire real OpenAI embeddings, screening, and extraction calls behind explicit API-key prompts; current default service paths include deterministic offline actions for no-cost local tests.
- Add background workers for long-running API jobs.
- Add progressive re-clustering history and individual record include/exclude overrides.
- Add polished reporting exports and full AI extraction screens.
