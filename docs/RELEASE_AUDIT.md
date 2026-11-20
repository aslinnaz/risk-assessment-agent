# Release audit — TrustAI Risk Assessment Agent

**Status: RELEASE CANDIDATE. Not FINAL VERIFIED, not production-ready.** All offline verification passed; the real-model evaluation could not run (HTTP 429, no API credits), so extraction quality with a real model, question quality and the embedding-similarity threshold remain unverified. Nothing here asserts 100 % reliability.

Date: 2026-11-20 · Python 3.14.3, macOS · installed: streamlit 1.67.0, openai 3.57.0, pymupdf 2.1, pypdf 7.1.1, numpy 2.5.5, reportlab 5.1.0, python-dotenv 1.2.4, pytest 9.2.0 (`requirements-tested.txt` pins slightly different versions of some of these; it was not re-run with the pins).
Audited: the supplied `trustai-risk-assessment-agent-final-candidate.zip` (51 files). Baseline before any change: 237 passed, 2 skipped; 1 failed only because a `.env` had just been created in the working copy.

## 1. Architecture and coverage map

```
answer (UI) ─► interview.process ─► llm.structured(extract) ─► interview._sanitize ─► turn["updates"]  (accepted facts only: quote ⊂ answer; grounded permissions/execution)
                                                                       │
              classify.derive (fold of accepted facts: slots, workflows, tools, controls) ─► classify.classify (deterministic L / I / P / label, legal path, blockers, trace)
                                                                       │
 planner.candidates (conflict / follow-up / required / decisive / context; no model) ─► llm.structured(question) ─► interview.make_question ─► rag.retrieve_topic (+ support check)
                                                                       │
 report.build_report ─► findings ─► rag.retrieve_for_claim + support check (+ optional model support check) ─► validate_citations ─► app (UI) / report.pdf_bytes
```

| Module | Role | Model/retrieval may change the classification? |
|---|---|---|
| `classify.py` | derivation, L0–L5, I0–I3, posture table, escalation, label, legal path, blockers, C01–C21 | **No** (no import of llm/rag) |
| `interview.py` | prompts, extraction schema, fact validation, question writing | extraction proposes facts; code accepts only grounded ones |
| `planner.py` | next question, follow-ups (incl. new unknown-workflow follow-up), stop decision | no model |
| `rag.py` | chunking, BM25 + embedding index, support check, excluded-reference guard, cache | retrieval never changes the result |
| `report.py` | findings, citations, validation, client summary, PDF | explains only |
| `sources_admin.py`, `tools/*.py` | approved sources, catalog check, release builder | – |
| `catalogs/*.json` | policy (L/I/P table, labels), controls, interview topics, source manifest | data |

Runtime sources: EU AI Act (OJ L 2024/1689), Commission Implementing Decision (EU) 2026/2274, TrustAI Agentic AI Governance Playbook, TrustAI AI Governance Playbook, NIST AI RMF Playbook. ISO/IEC 42001 is a bibliographic entry only (copyright; `rag=false`, never indexed or sent to an API). The external QA reference is not a source.

## 2. Defects found and fixed (each with code and tests)

| # | Defect (independently reproduced before the fix) | Fix | Tests |
|---|---|---|---|
| D1 | A recommend-only agent that can grant itself permissions stayed **L1**; with an unestablished mode, undetermined | `classify.operational` `POL-L5-EXP` for every mode | `test_named_scenarios[privilege_expansion_*]`, `test_dynamic_delegation_is_flagged_and_expansion_reaches_L5` |
| D2 | A workflow of unknown execution was ignored: `rank (recommend)` + `reject (unknown)` → confident L1 | `POL-WF-UNKNOWN`, posture range from the lower bound, planner follow-up `wfx:` | `test_named_scenarios[unknown_workflow_authority_is_uncertain]`, `test_repeated_question_is_not_asked_twice_and_unknown_workflow_is_asked_once` |
| D3 | Impact I3 boundary: "patients could die", "lose their home", "employees fired" with "limited" selected stayed I1; only one employment regex existed | `harm_floors` (I3 safety / rights / livelihood, I2 material harm and employment), negation-aware, never lowers | `test_impact_uses_the_highest_credible_harm…` (19 cases), `test_read_only_access_does_not_lower_impact` |
| D4 | Invented tool permissions accepted (`reads the CRM` → `delete`, `transact`) | `ground_permissions` | `test_invented_tool_permissions_are_dropped`, `test_denial_without_a_negation_cue_is_not_accepted` |
| D5 | Execution invented from ranking accepted (L4 from "ranks candidates") | execution-verb grounding in `_sanitize` | `test_execution_is_never_inferred_from_ranking_or_scoring`, `test_real_execution_is_accepted` |
| D6 | No "denied" state for a tool permission; revocation and read-only could not be represented | `denied_permissions`, per-permission last-statement-wins, unknown list | `test_read_only_claim_denies…`, `test_revocation_in_a_later_turn…`, `test_later_confirmation_overrides…`, `test_same_statement_contradiction…` |
| D7 | `build_report` crashed (`KeyError: 'value'`) when the tool-boundary rule was in the derivation | `_inputs_view` | `test_tool_boundary_trace_does_not_crash_the_report_and_pdf` |
| D8 | Irrelevant passages shown as support (glossary page for "intended purpose", Annex VIII for Annex III areas, explainability page for "EU link", NIST bias text for "reach") | proximity + IDF + claim check, curated terms/anchors | `test_irrelevant_passages_are_rejected_not_cited`, `test_retrieved_passages_for_every_topic_pass_the_support_check` |
| D9 | OJ footnotes glued into 27 recital passages (citation text the recital does not contain) | `_drop_footnotes` | `test_recital_passages_contain_no_page_footnotes`, `test_real_ai_act_index_has_the_expected_references` |
| D10 | Turkish playbook pages indexed (English-only output; snippets could appear in another language) | `is_non_english` gate | `test_no_non_english_source_passage_is_indexed` |
| D11 | Nothing stopped the QA reference from being registered as a source or from sitting in a cache | name/content guards in four places; release builder | `test_adding_the_matrix_as_a_source_is_refused`, `test_a_registry_entry_for_the_matrix_is_excluded_at_runtime`, `test_build_release_refuses_a_package_containing_the_matrix`, `test_no_matrix_document_in_the_package` |
| D12 | PDF omitted the derivation, tool inventory, separate conclusions and sources; "—" placeholders; arrow glyph rendered as a missing character | new PDF appendix, "Not assessed" fallbacks, `=` instead of an arrow | `test_end_to_end_scripted_interview_report_and_pdf`, `test_report_has_no_dash_placeholders…`, `test_incomplete_assessment_report_has_no_malformed_rule_text_or_missing_glyphs` |
| D13 | Rule text "Declared action authority 'None' maps to None" in the trace | wording | same as D12 |
| D14 | Tests wrote index files for test sources (`CUST-AG1`) into the project `.cache/` | session-scoped temp cache | all |
| D16 | A truncated chunk or embedding cache (crash while writing) crashed the next start | atomic writes; corrupted cache discarded and rebuilt; explicit degraded mode | `test_corrupted_caches_are_discarded_and_rebuilt_not_trusted` |
| D15 | `semantic_check` raised on a malformed model answer | defensive parse | harness dry run |

## 3. Feedback items → code → tests → status

| Brief item | Code | Tests | Status |
|---|---|---|---|
| Source boundary (matrix never an app source; ZIP, manifest, cache, runtime) | `rag.is_excluded_reference/has_excluded_content`, `build_chunks`, `custom_entries`, `sources_admin.validate`, `tools/build_release.py` | `test_no_matrix_*`, `test_matrix_is_not_in_manifest_registry_cache_or_chunks`, `test_no_long_verbatim_overlap_between_the_qa_reference_and_the_indexed_sources` (12-word overlap of the QA reference against every indexed passage: none), 10-word overlap scan of the repository: only `catalogs/policy.json` (the allowed internal policy table) and one generic phrase in `controls.json` | **PASS** |
| 1 Inventory, trace, coverage map | section 1 | AppTest + e2e | **PASS** |
| 2 English-only (UI strings, prompts, catalogs, PDF, sources) | `is_non_english` | `test_catalogs_prompts_and_ui_strings_are_english_only`, `test_no_non_english_source_passage_is_indexed`, `test_user_quotes_stay_verbatim_even_if_not_english` | **PASS** (static scan for Turkish; no other language checked) |
| 3 Question/workflow/tool routing | D1, D2, D4–D6, planner follow-up | scenario and tool tests above | **PASS** offline; live routing **NOT RUN** |
| 4 LLM extraction: validated facts with exact quote and answer ID; injection; uncertainty | `_sanitize`, `clean_answer`, prompt | `test_extraction_rejects_unsupported_or_injected_facts`, `test_instruction_text_is_not_evidence…`, corrections tests | **PASS** with a scripted model; real model **NOT RUN** |
| 5 Retrieval: relevance, metadata, support verdict, startup index, degraded mode, fewer calls | D8–D10, `ensure_embeddings`, batch 256 | `test_missing_embeddings_degrade_visibly…`, `test_fabricated_citation_is_caught_by_validation`, `test_malicious_retrieved_instruction_cannot_change_the_classification` | **PASS** offline in keyword-only mode (the test suite never calls the API); the real corpus embedding index was built and its hybrid startup (`missing = 0`, no request) checked by hand, not by a test; similarity threshold with real query embeddings **NOT VERIFIED** |
| 6 24 cells, labels, escalation, blockers, no control passed on declaration | `posture_matrix`, `escalation_factors`, `control_assessment` | `test_all_24_posture_cells_and_label_mapping`, `test_posture_catalog_equals_reference_table`, `test_escalation_is_one_documented_step…`, existing C01–C21 tests; the policy table also equals the QA reference (`tests/test_decision.py`, run with a local copy, 37 passed) | **PASS** |
| 7 Report/PDF: derivation, inventory, drivers, separate conclusions, citations, "Not assessed", wording | D12, D13 | e2e + PDF text assertions | **PASS** |
| 8 Matrix absent from package, manifest, cache, report sources | see boundary | see boundary | **PASS** |
| Test matrix (scenarios, adversarial, contradictory, missing key/embeddings, repeated questions, placeholders) | `tests/test_audit_release.py` | 140 tests; scenario log in `docs/TEST_RESULTS.md` | **PASS** |
| Real-LLM evaluation | `evals/real_llm_eval.py` | – | **NOT RUN** (HTTP 429) |

## 4. Tests that were changed (and why)

The brief forbids changing expectations to make tests green; each change below follows a requirement of the brief, not the output:
- `test_final_pass` (2 tests), `test_flow::test_result_summary_…`, `test_rag_sources_audit::test_report_pdf_and_state_agree…`: asserted a two-page PDF with no sources or derivation. Brief item 7 requires derivation, inventory and sources in the generated report; the assertions now check those.
- `test_sources::test_real_ai_act_index_has_the_expected_references`: bound `> 1300` → `> 1200`. Footnotes (D9) and Turkish pages (D10) are intentionally no longer indexed (1,293 passages).
- `test_release::_tracked`: checks the files that would be packaged, not the developer's working tree.
- Not changed: retrieval-diversity tests; they pass through curated anchors.

## 5. Remaining limitations and unverified items

1. **Real-model behaviour is unverified** (extraction recall/precision on natural language, question quality, `semantic_check`). The scripted-model tests prove the code paths, not the model.
2. **Embedding similarity threshold (0.30)** is an uncalibrated first value; `retrieval_calibration` in `evals/real_llm_eval.py` measures it with one request. Until then hybrid retrieval may reject or accept borderline passages differently from keyword-only mode. A full corpus embedding index exists locally (`.cache/`, not packaged).
3. **ISO/IEC 42001 is not retrievable** (copyrighted). The brief lists ISO materials as a permitted RAG source; a licensed text can be added with `tools/add_source.py … --indexing-permitted`. Until then reports cite ISO only as a bibliographic entry.
4. The grounding of tool permissions and execution verbs is lexical. It can drop a true permission or downgrade a true execution claim to "unknown" (the safe direction: more uncertainty and one extra question, never a lower level); synonyms outside the word lists are not recognised.
5. Delegation: reachable authority through other agents is counted only if the user's answer describes the delegated action (extraction prompt); the delegator/delegate ceiling is not modelled separately (only an uncertainty note when delegation is declared).
6. Escalation factors are those of internal policy v0.4 (reach, irreversibility, vulnerable groups, sensitive data, policy-only limits). Speed, uncertain likelihood and weak detectability, named in the QA reference as possible escalators, are not asked about or modelled.
7. Impact floors are keyword rules on the user's harm text (English only); they raise, never lower, and cite the rule ID, but an unusual wording is not recognised.
8. Application dates of the AI Act are from the original OJ text; later amendments were not verified against EUR-Lex (stated in the report).
9. The two TrustAI playbooks carry no licence statement (`docs/SOURCES.md`); they are sent to the embedding API as before. Decide before wider distribution.
10. Language and footnote handling are heuristics; the footnote filter was verified on the AI Act file only.
11. UI coverage is Streamlit AppTest (rendering and state), not a browser test.
12. Lint (pyflakes): no undefined names; unused imports/variables remain in older files.
13. Tested on Python 3.14.3 only; the CI file targets 3.12 and was not run.
