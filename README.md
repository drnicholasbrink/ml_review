**Overview**
- **Goal:** Retrieve PubMed articles for a search strategy, persist metadata to CSV, enrich with OpenAI embeddings, and optionally cluster for exploration.
- **Stages:**
  - **Search/Fetch:** Query PubMed, paginate by date, and write `pubmed_results_complete.csv`.
  - **Embed:** Add OpenAI text embeddings for abstracts to `pubmed_results_with_embeddings.csv`.
  - **Explore:** Run clustering and visualization notebooks.

**Repository Layout**
- `environment.yml`: Conda environment for reproducible setup.
- `.gitignore`: Ignores secrets, logs, large outputs, and caches.
- `secret_keys.example.py`: Template to copy to `secret_keys.py` locally.
- `search_strategy.txt`: Your PubMed query string (use placeholders like `#example`).
- `scripts/search_pubmed.ipynb`: Fetch PubMed IDs and article details into CSV.
- `scripts/extract_pubmed.ipynb`: Generate OpenAI embeddings for abstracts.
- `scripts/clustering.ipynb`: Cluster/visualize embedded records.
- `scripts/embedding.ipynb`: Auxiliary embedding experiments (optional).
- `scripts/pipeline_quick_test.py`: Fast sanity-check that exercises the pipeline on a tiny window.

**Setup**
- **Create env:** `conda env create -f environment.yml`
- **Activate:** `conda activate ml_review`
- **Register kernel (optional):** `python -m ipykernel install --user --name ml_review --display-name "Python (ml_review)"`
- **Secrets:**
  - Copy `secret_keys.example.py` to `secret_keys.py` and fill in values.
  - Do not commit `secret_keys.py`. It’s in `.gitignore` by default.
- **Search query:** Put your PubMed query in `search_strategy.txt` and replace any sensitive specifics with `#example`.

**Secrets Format**
- `secret_keys.py` should define:
  - `OPENAI_API_KEY = "#example"`
  - `PUBMED_API_KEY = "#example"`

**Pipeline**
- **1) Search/Fetch (PubMed)**
  - File: `scripts/search_pubmed.ipynb`
  - Reads the query from `search_strategy.txt` (`#example`).
  - Splits date windows (year/month/day) to respect API limits.
  - Writes article metadata to `pubmed_results_complete.csv` (PMID, Title, Abstract, Authors, Date, Journal).
  - Caches IDs in `pubmed_ids.json` to avoid duplicates across runs.
  - Logs to `pubmed_fetch.log`.

- **2) Embeddings (OpenAI)**
  - File: `scripts/extract_pubmed.ipynb`
  - Loads `pubmed_results_complete.csv` and batches abstracts.
  - Calls OpenAI embeddings (`#example` model) and serializes vectors to JSON strings.
  - Writes `pubmed_results_with_embeddings.csv` with added `Embedding` column.
  - Logs to `extract_pubmed.log`.

- **3) Clustering / Exploration**
  - File: `scripts/clustering.ipynb`
  - Reads `pubmed_results_with_embeddings.csv`.
  - Runs clustering/visualization (`#example` algorithm and parameters).

**Quick Test (Sanity Check)**
- Run a minimal pipeline over a 2–3 day window to validate end-to-end wiring without heavy downloads:
- Command: `python scripts/pipeline_quick_test.py`
- Output: `pubmed_results_complete_quick.csv`, `pubmed_results_with_embeddings_quick.csv`
- Notes:
  - Requires `search_strategy.txt` and valid keys in `secret_keys.py`.
  - If `OPENAI_API_KEY` is missing, the test skips embedding and only checks PubMed fetch.

**Typical Workflow**
- Edit `search_strategy.txt` to your query (use `#example` placeholders when sharing):
  - Example: `("#example heat stress"[title/abstract]) NOT ("#example animals")`
- Execute `scripts/search_pubmed.ipynb` top-to-bottom.
- Execute `scripts/extract_pubmed.ipynb` to append embeddings.
- Optionally run `scripts/clustering.ipynb` for analysis.

**Publishing Guidance**
- Keep `secret_keys.py`, large CSVs, logs, and checkpoints out of git (already in `.gitignore`).
- When sharing notebooks, scrub or replace any identifying values with `#example` (queries, institutions, local paths, etc.).
- If keys were ever committed, rotate them before publishing.

**Troubleshooting**
- PubMed throttling: The notebooks sleep between requests and split by date; consider increasing backoff for larger queries.
- OpenAI SDK: This repo uses the pre-1.0 SDK (`openai<1.0.0`), which exposes `openai.Embedding.create`. If you upgrade to `>=1.0`, update the code accordingly.
- CSV schemas: Ensure `PMID` and `Abstract` columns exist before embedding.

**Reproducibility**
- Update env: `conda env update -f environment.yml --prune`
- Export extra pip deps if you add any: edit `environment.yml` under the `pip:` section with `#example` placeholders where appropriate.

