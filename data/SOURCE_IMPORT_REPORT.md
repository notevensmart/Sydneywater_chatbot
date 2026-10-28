# Source expansion — 21 October 2026

The supplied list contains 48 URLs. The active manifest contains **42 webpages / 342 sections**: 41 supplied pages and the existing Contact us page. Previously: 8 pages / 79 sections. Existing document IDs remain; refreshed page content can change section IDs. Old evaluation results do not certify the new corpus.

Five supplied URLs are not indexed separately: four hub pages redirect to HTTP accounts-billing (the supplied HTTPS destination is indexed), and the duplicate `.html.html` calculator path is represented by the normal path. HTTPS-only safeguards remain intact.

Two pages are excluded: the pension rebate application has insufficient static text after form removal; paperless competition terms have an oversized block that still fails the excerpt limit. The pension information page is indexed. Full dispositions: `source_import_audit.json`.

Large tables are split at whole row boundaries with repeated headers. Retrieval limits, citation validation, approved-host checks and the single-agent architecture are unchanged. No extra application module or model call was added. Form introductions are information only: the chatbot cannot execute forms or calculators. Some pages contain dated maintenance notices, which cannot establish live availability.

Review was agent-assisted extraction/snapshot comparison, not human approval. Extracted paragraphs matched saved HTML; headings, introductory text and size bounds were checked. Human semantic review remains pending.

Validation:

- Readiness passed: 42 documents / 342 sections.
- Five new-topic searches found relevant sources for pensions, settlement, deceased estates, meter responsibility and Blacktown agencies.
- Unchanged full retrieval regression: **20/27 (74.1%)**, versus 27/27 on the smaller corpus. Larger-corpus ranking needs work. Gold labels were not changed. See `retrieval_expanded_sources_all_results.json`.
- Offline tests: **49 passed, 4 failed, 4 live tests deselected**. Two scripted conversation tests failed because their fixed citation was no longer retrieved; two baseline tests reported AttributeError. These failures remain unresolved. No safety gate was weakened.
- New large-table regression test passed. No live-model answer evaluation was run for this import.

Restart Streamlit and start a new conversation. Original manifest/database backups are in `.cache/source-expansion/`.
