# Production and publication readiness review

Review date: 2026-07-21

## Supported release profile

This release supports a trusted, single-reviewer workstation running through Docker Compose on localhost. It is designed for an auditable systematic-review workflow, not as a remotely exposed multi-user service. The notebook workflow remains available and is not replaced by the application.

## End-to-end browser validation

The review exercised the complete interface with a real PubMed query bounded to 25 fetched records from a 58-result date window. It generated OpenAI embeddings, built and filtered the Evidence Atlas, created root and child clustering branches, selected 11 records, screened those records with Structured Outputs, compared four human-reference decisions, recorded a human adjudication, and extracted structured data for the three included abstracts. Paid API validation stayed below 200 records.

| Component | Browser checks and release outcome |
| --- | --- |
| Project setup | Criteria and search strategy persist; downstream outputs invalidate when protocol inputs change. |
| PubMed | Count scope and fetch limit remain visible and are recorded in project provenance. |
| CSV import | Column mappings persist across validation and revisit; long fields remain reviewable. |
| Deduplication | Summary counts and a persistent duplicate audit table identify retained and removed records. |
| Embeddings | Model, truncation, dimensions, and progress provenance are visible; runs resume safely. |
| Evidence Atlas | Python computes deterministic UMAP coordinates and cosine neighbors; a responsive browser-native Canvas 2D explorer provides search, filters, pan/zoom, details, neighbors, and CSV export in current Safari and Chrome without WebGL, WebAssembly workers, Node.js, or a Jupyter kernel. |
| Clustering | WCSS-first root/child branches, deterministic settings, source details, DOI/PubMed links, and selection history were exercised. |
| Screening | Original AI fields remain immutable; human decisions, notes, timestamps, final-decision source, paging, filtering, and reviewed export are available. |
| Evaluation | Funnel, Sankey, confidence, criterion, exclusion, t-SNE, human-reference metrics, and downloadable mismatches were checked. |
| Extraction | Included/final decisions feed a resumable Structured Outputs schema; stale extractions are removed when eligibility changes; publication-oriented CSV/JSON exports are available. |
| Handoff | A ZIP assembles protocol inputs, record audit, screening/adjudication, evaluation, and extraction artifacts without credentials or embeddings. |
| Runtime | Local-only binding, required Compose secret, CSRF, security headers, local Plotly delivery, readiness check, and serialized filesystem writes define the supported deployment boundary. |

## Scientific safeguards

- AI screening is decision support, not a final clinical or scientific judgment.
- All uncertain and low-confidence screens require human review; a sample of other decisions should also be checked.
- Abstract-only extraction is a calibration and pre-population aid. Every field and effect estimate must be verified against the full text before analysis or publication.
- Human-reference comparisons depend on fuzzy title matching; ambiguous or unmatched records require manual inspection.
- Retain the publication bundle, protocol version, model names, run date, prompt/schema changes, and application commit hash with the review record.

## Deliberate non-goals and escalation boundary

Remote or concurrent multi-user service is not supported by the supplied Compose profile. That deployment would require authentication/authorization, TLS, a background job system, shared transactional storage, centralized audit logging, backups, retention controls, and a privacy/security review appropriate to the data. These are deployment-architecture requirements rather than hidden limitations of the validated local workflow.
