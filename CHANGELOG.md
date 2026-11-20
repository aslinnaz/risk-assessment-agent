# Changelog

## Release candidate — independent release audit (2026-11-20)

Status: **RELEASE CANDIDATE, not FINAL VERIFIED.** The real-model evaluation could not run (the API account returned HTTP 429, "no credits remaining"); see `docs/REAL_LLM_EVAL.md`.

### Classification (`classify.py`)
- **Authority expansion now defines L5 in every mode.** Before: only an agent already at L3/L4 became L5, so a recommend-only agent that could grant itself permissions stayed L1, and with an unestablished mode the level stayed undetermined. Now `authority_expansion = yes` gives L5 (rule `POL-L5-EXP`) unless that very fact is in an unresolved conflict.
- **A workflow whose execution is unknown makes the level undetermined** (rule `POL-WF-UNKNOWN`), with the declared ceiling kept as a lower bound for the posture range. Before: unknown workflows were ignored, so `rank (recommend)` + `reject applicants (unknown)` gave a confident L1. The planner now asks once, "What does the agent itself do with it?" (`planner.py`, follow-up `wfx:<id>`).
- **Impact floors from the user's own harm description** (`harm_floors`): safety, fundamental-rights and livelihood harm raise the class to I3 (`POL-I3-SAFETY`, `POL-I3-RIGHTS`, `POL-I3-LIVELIHOOD`); material fairness, privacy, service or rights harm and employment-opportunity loss raise it to I2 (`POL-I2-MATERIAL`, `POL-I2-EMPLOYMENT`). Floors only raise, never lower; a negation ("nobody could die") is not a scenario. Before: only a single employment keyword was recognised, so "patients could die" with "limited" selected stayed I1.
- **Tool permissions have three states**: confirmed, denied, unknown. A later statement replaces an earlier one for the same permission (revocation, correction); a permission both denied and confirmed in the same statement stays confirmed. Tools with no stated permission raise an uncertainty. A tool boundary or unknown-workflow uncertainty keeps a lower bound for the posture range.
- Rule texts no longer print `'None'` for an undetermined level or impact.

### Extraction (`interview.py`)
- **Tool permissions are grounded**: a permission is kept only if a word for it occurs in the user's own words about that tool (the sentences naming it plus the quote); a denial needs a negation or "read-only" cue. Before: a model that wrote `delete` and `transact` for "reads the CRM" was accepted.
- **Execution is never inferred from analysis**: `executes_*` needs an action verb or an autonomy cue in the quote; "ranks candidates" is downgraded to unknown (and then asked about).
- Extraction prompt and schema: `denied_permissions`; actions triggered through other agents count as reachable authority.

### Retrieval (`rag.py`, `catalogs/interview.json`, `report.py`)
- **Support check is stricter** for passages found by search: IDF-weighted proximity of the distinctive terms inside one 450-character window, a higher keyword bar without embeddings, a claim-keyword check when only one curated term exists, and a higher embedding-similarity bar (0.30, **not yet calibrated with real query embeddings**). Irrelevant hits that were shown before (a glossary page for "intended purpose", the registration annex for "Annex III areas", an explainability page for "EU link") are now rejected.
- **Footnotes are removed from recital text.** Before: 27 recital passages carried OJ footnotes (for example Recital 53 began with a motor-vehicle regulation reference), so a citation could show text the recital does not contain.
- **Non-English passages are not indexed** (Turkish pages of the TrustAI playbook), so no source snippet is shown in another language.
- Curated anchors added for TrustAI Governance Playbook sections and NIST items; support terms of `purpose`, `art6_role`, `org_role` tightened; guidance finding in the report uses curated anchors. `tools/verify_catalog.py`: 0 problems.
- **Index caches are written atomically and a damaged cache is discarded and rebuilt** (before: a half-written chunk or embedding file crashed the next start). A damaged embedding file means "keyword-only" with an explicit message until it is rebuilt.
- Embedding batch size 96 → 256 (whole corpus ≈ 6 requests). A valid cached index is reused without any request.

### Source boundary
- The external QA reference (compliance check matrix) is refused **by name and by content** in `rag.build_chunks` (also when it sits in a chunk cache), `rag.custom_entries`, `sources_admin.validate`, and `tools/build_release.py`. The patterns live in code, not in a user-visible catalog.
- `tools/build_release.py` builds the ZIP and fails if the matrix, a Word/ZIP file, `.env`, a cache, local custom sources, ISO/IEC 42001 or a key-like string would be packaged.
- Tests no longer write into the project's `.cache/` (they used to leave test-source index files there).

### Report and PDF (`report.py`, `app.py`)
- PDF: prominent "PRELIMINARY, DECLARATION-BASED ASSESSMENT" banner on page 1; new "Basis and traceability" appendix with the separate conclusions (legal, inherent exposure, posture, declared control coverage, evidence confidence E0, residual risk "Not assessed", operational recommendation), the classification derivation with inputs (answer IDs and verbatim user quotes), the tool and action inventory, risk drivers and the cited sources (name, version, section, page).
- Empty fields read **"Not assessed"** instead of a dash.
- **Crash fixed:** `build_report` raised `KeyError: 'value'` when a tool-boundary trace step was part of the derivation; it now renders.
- The model support check ignores a malformed answer instead of raising `KeyError`.
- UI shows the tool and system inventory (confirmed, denied, not stated).

### Tests and tooling
- `tests/test_audit_release.py` (140 tests): all 24 posture cells, escalation, 21 named scenarios, tool permissions, impact edge cases, extraction attacks, retrieval relevance, fabricated citations, matrix boundary, English-only, end-to-end report/PDF, Streamlit AppTest.
- Tests changed (all others untouched): the four tests that asserted the previous PDF contract ("two pages, no sources or derivation") now assert the new contract; one chunk-count sanity bound was lowered (footnotes and non-English pages are no longer indexed); `test_release` now checks the file list that would be packaged instead of the developer's working tree; `conftest` isolates the runtime cache. The retrieval-diversity tests (every source family used, NIST and TrustAI governance guidance accompany at least five topics) were **not** relaxed: they are satisfied through curated anchors. Reasons in `docs/RELEASE_AUDIT.md`.
- `evals/real_llm_eval.py`: budget-capped real-model evaluation (hard caps on requests and estimated spend).
- Removed the stale `RELEASE_TEST_RESULTS.txt`.
