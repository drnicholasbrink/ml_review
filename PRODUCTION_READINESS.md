# Production and publication readiness review

Review date: 2026-07-21

## Supported release profile

This release supports a trusted, single-reviewer workstation running through Docker Compose on localhost. It is designed for an auditable systematic-review workflow, not as a remotely exposed multi-user service. The notebook workflow remains available and is not replaced by the application.

## End-to-end browser validation

The review exercised the complete interface with a real PubMed query bounded to 25 fetched records from a 58-result date window. It generated OpenAI embeddings, built and filtered the Evidence Atlas, created root and child clustering branches, selected 11 records, screened those records with Structured Outputs, compared four human-reference decisions, recorded a human adjudication, and extracted structured data for the three included abstracts. Paid API validation stayed below 200 records.

The human-review follow-up exercised focus and list modes, title/abstract and full-text stages, source links, PDF controls, and the extraction source library at a 1280-pixel browser viewport. The final pass had no horizontal page overflow or browser-console errors. Responsive layouts use standards supported by current Safari and Chrome without a browser-specific atlas runtime dependency.

| Component | Browser checks and release outcome |
| --- | --- |
| Project setup | Criteria and search strategy persist; downstream outputs invalidate when protocol inputs change. |
| PubMed | Count scope and fetch limit remain visible and are recorded in project provenance. |
| CSV import | Column mappings persist across validation and revisit; long fields remain reviewable. |
| Deduplication | Summary counts and a persistent duplicate audit table identify retained and removed records. |
| Embeddings | Model, truncation, dimensions, and progress provenance are visible; runs resume safely. |
| Evidence Atlas | Python precomputes deterministic UMAP coordinates and cosine neighbors in a slim Parquet artifact. The interface opens Apple's official Embedding Atlas with the data URL and column settings preloaded, and offers a download-and-drop fallback for network-restricted browsers. |
| Clustering | WCSS-first root/child branches, deterministic settings, source details, DOI/PubMed links, and selection history were exercised. |
| Screening | Focus/list modes expose full abstracts, DOI/PubMed links, one-click decisions, progress, compound filters, and separate AI/human/final decision layers. Filtered “AI accepted” actions never create human decisions. The optional full-text stage supports project PDFs, resumable PDF-backed AI screening, explicit disagreement handling, and human overrides. |
| Evaluation | Funnel, Sankey, confidence, criterion, broad exclusion categories, t-SNE, human-reference metrics, and downloadable mismatches were checked. Specific exclusion rationales remain in the screening audit. |
| Extraction | Final full-text decisions feed a resumable Structured Outputs schema. Per-record PDFs are uploaded safely and used when present; abstract fallbacks are explicit; changing a source invalidates only its resumable configuration and stale project extraction references. |
| Background tasks | PubMed fetch, embeddings, title/abstract screening, full-text screening, and extraction run in a serial process-local worker with durable status, live progress polling, safe failures, restart recovery, and project-level write exclusion. |
| Human reference | Evaluation can use explicit human decisions recorded in the staged workflow or a separately uploaded included-records CSV; workflow metrics exclude records without a human decision. |
| Handoff | A ZIP assembles protocol inputs, abstract/full-text decision audit, evaluation, and extraction artifacts without credentials, embeddings, or uploaded copyrighted PDFs. |
| Runtime | Local-only port 5055 binding, required Compose secret, CSRF, security headers, local Plotly delivery, readiness check, and serialized filesystem writes define the supported deployment boundary. |

## Scientific safeguards

- AI screening is decision support, not a final clinical or scientific judgment.
- All uncertain and low-confidence screens and every AI/human disagreement require human review; a sample of other decisions should also be checked. “AI accepted” remains distinguishable from a human judgment throughout the audit.
- PDF-assisted and abstract-fallback extraction are calibration and pre-population aids. Every field and effect estimate must be verified manually against the source document before analysis or publication.
- Human-reference comparisons depend on fuzzy title matching; ambiguous or unmatched records require manual inspection.
- Retain the publication bundle, protocol version, model names, run date, prompt/schema changes, and application commit hash with the review record.

## Deliberate non-goals and escalation boundary

Remote or concurrent multi-user service is not supported by the supplied Compose profile. That deployment would require authentication/authorization, TLS, an external durable queue and worker pool, shared transactional storage, centralized audit logging, backups, retention controls, and a privacy/security review appropriate to the data. These are deployment-architecture requirements rather than hidden limitations of the validated local workflow.
