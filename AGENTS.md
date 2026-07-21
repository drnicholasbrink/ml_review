# AGENTS.md

Guidance for AI agents working in this repository.

## Project purpose

This repository is a notebook-first systematic-review workflow. It retrieves PubMed records, optionally deduplicates and clusters them, uses OpenAI models for abstract screening and structured data extraction, and exports CSV/HTML/PDF artifacts for human review and manuscript preparation.

## Architecture and workflow

The codebase is organized as manually executed Jupyter notebooks under `scripts/`, plus plain-text criteria/query files:

1. `scripts/search_strategy.txt` stores the PubMed query.
2. `scripts/search_pubmed.ipynb` queries NCBI E-utilities, splits date ranges to keep result batches manageable, caches PMIDs in `pubmed_ids.json`, and writes `pubmed_results_complete.csv`.
3. `scripts/extract_pubmed.ipynb` embeds PubMed abstracts and writes `pubmed_results_with_embeddings.csv`.
4. `scripts/clustering.ipynb` explores embeddings with t-SNE/K-Means and writes visualizations/cluster CSVs under `outputs/`.
5. `scripts/inclusion_criteria.txt` stores screening criteria.
6. `scripts/ai_screening.ipynb` uses OpenAI structured outputs to classify articles as `include`, `exclude`, or `uncertain`, with manual review expected for uncertain/low-confidence cases.
7. `scripts/ai_extraction.ipynb` uses OpenAI structured outputs and Pydantic schemas to extract systematic-review data from included articles.
8. `scripts/eval.ipynb` compares AI decisions to human screening exports with fuzzy title matching.

There is no packaged Python module or automated pipeline runner at present; preserve the notebook-first workflow unless the user explicitly requests refactoring.

## Environment and dependencies

- Prefer the Conda environment in `environment.yml` (`conda env create -f environment.yml`, then `conda activate ml_review`).
- The OpenAI dependency is `openai>=1.0.0`; use the modern `OpenAI` client and structured-output APIs for new OpenAI code.
- Keep `numpy<2` unless the notebook stack is tested with newer NumPy.
- Do not add try/except wrappers around imports.

## Secrets, inputs, and outputs

- Never commit secrets. Local API keys are expected in `secret_keys.py`, which is gitignored.
- `secret_keys.py` should define at least `OPENAI_API_KEY` and `PUBMED_API_KEY` when running the relevant notebooks.
- Do not commit generated outputs: logs, `pubmed_ids.json`, PubMed result CSVs, screening/extraction CSVs, figures, reports, or `outputs/` contents unless the user explicitly asks for a small fixture.
- Avoid committing machine-specific absolute paths. Several existing notebooks contain historical local paths; when editing, prefer relative paths from the repository root or from `scripts/` and document the expected working directory.
- Keep sensitive search terms, affiliations, local paths, and identifiers out of shared docs; use neutral examples or placeholders when needed.

## Notebook editing conventions

- Keep notebooks executable top-to-bottom where practical, but be explicit when a cell is intentionally manual or optional.
- Preserve resume/progress-saving behavior for long API workflows.
- Before changing a notebook, inspect its code cells with `python`/`nbformat` or another notebook-aware tool instead of relying only on rendered views.
- When editing notebooks programmatically, avoid changing outputs or metadata unnecessarily.
- For generated plots that require Kaleido/browser/system packages, document the requirement rather than hard-failing in normal analysis paths.

## Manual review expectations

This repo supports systematic review decisions; agents must not represent AI decisions as final clinical or scientific judgments. Preserve documentation that human reviewers should:

- Review all `uncertain` and low-confidence AI screening decisions.
- Validate prompts and model choices on a sample before full screening/extraction.
- Compare against human screening when a gold-standard export is available.
- Record criteria, model name, date, and any prompt changes for reproducibility.

## Testing and checks

- For documentation-only changes, run at least a lightweight file/format check such as `git diff --check`.
- For notebook code changes, run targeted notebook or Python checks when feasible; avoid expensive PubMed/OpenAI calls unless the user explicitly requests them and keys are available.
- Do not run commands that make large API calls or spend money without explicit user intent.

## Git hygiene

- Review `git status --short` before committing.
- Keep commits focused and avoid including generated artifacts.
- Follow repository instructions to commit changes on the current branch and open a PR when changes are made.
