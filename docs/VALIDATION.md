# Validation

This file records what was **actually executed**, when and how, and separates it from historical records and from gaps. It does not claim statistical validation, legal validation or production readiness.

## 1. Executed in the handover audit (2026-11-12, macOS)

| Check | Command | Result |
|---|---|---|
| Offline suite, Python 3.14.3 (existing venv), sources present | `python -m pytest -q` | **231 passed, 2 skipped**, 31 live deselected |
| Offline suite, **fresh Python 3.12.7 venv installed from `requirements.txt`** (resolved to openai 3.49.0, pypdf 7.0.1; other packages equal to the tested pins) | `python -m pytest -q` | **231 passed, 2 skipped**; `pip check`: no broken requirements |
| Offline suite with an **empty sources directory** (what CI without source documents sees) | `TRUSTAI_SOURCES_DIR=<empty dir> python -m pytest -q -rs` | **86 passed, 147 skipped** (skips are source-dependent tests) |
| Catalog and anchor verification | `python tools/verify_catalog.py` | 40 control mappings, 76 topic anchors, **0 problems** |
| Syntax and import | `python -m py_compile *.py tools/*.py evals/*.py tests/*.py`; import of every module | OK |
| App start without an API key | `AppTest.from_file("app.py").run()` | No exception; start screen rendered |
| Streamlit server start | `streamlit run app.py --server.headless true` | Health endpoint and index returned HTTP 200 (this shows the server runs, not that the script ran) |
| PDF generation | covered by `tests/test_final_pass.py` and `tests/test_flow.py` | 2-page output, all 21 controls once, no sources or diagnostics, Vera font only |
| Secret scan (working tree and all reachable history of this repository) | pattern scan for common key formats | 0 matches. In the *previous* repository one commit matched only because "ri**sk-**assessment" in a README `git clone` line looks like a key prefix; no credential |

The 2 skipped tests compare the policy table with an optional reference file (`sources/compliance_matrix.docx`) that is not in the repository. Their content is replaced by the frozen-table test `test_decision.py::FROZEN_POSTURE`, which runs.

Not executed: Linux, Python 3.13, a real browser session, the GitHub Actions workflow on GitHub (it was reviewed and its commands were run locally; see section 5).

## 2. Test categories (offline, 233 tests)

| File | Tests | Covers |
|---|---|---|
| `tests/test_app_smoke.py` | 11 | Modules compile; app starts; UI smoke; runs without sources |
| `tests/test_decision.py` | 37 | Frozen 24-cell posture table; `classify.py` imports no model or index; L/I rules; Art. 6 pathways; escalation |
| `tests/test_flow.py` | 63 | Interview flow with a scripted model; quote validation; planner; checklists; report and PDF content |
| `tests/test_procurement_audit.py` | 38 | Conditional approval (not a conflict); single accepted-fact state; context-sensitive gaps; oversight facts; screening |
| `tests/test_rag_sources_audit.py` | 23 | Retrieval and citations; support checks; unknown legal applicability; PDF/report separation |
| `tests/test_final_pass.py` | 47 | One answer interface per question; checklist semantics; C01–C21 mapping and statuses; Art. 6(1); escalation only when real; extraction regressions ("review afterwards", placeholders); C20 legal states; PDF scenarios and font |
| `tests/test_release.py` | 7 | Missing-source error message; manifest/README consistency; no secrets or restricted documents packaged; the key is never printed or put in reports |
| `tests/test_sources.py` | 7 | AI Act structure parsing for every extractor layout; loud failure on mis-parse; real AI Act index |
| `tests/test_live_openai.py` | 31 (live) | Real model: extraction precision/recall, question quality, retrieval relevance, prompt injection, full interviews, PDF contract |

The offline tests use a **scripted fake client**: they test this software's own logic, not model quality. A passing offline suite says nothing about live extraction accuracy.

## 3. Live tests (real OpenAI calls; default `gpt-4.1-mini`)

Live tests cost money and are never run by default. In this handover audit **no live test was run**; the last live runs were during the final development pass (2026-11-12, before this audit) and are summarised here honestly, including failures:

| Run | Result | Cause of failures |
|---|---|---|
| 1 (after the final pass changes) | 26 passed, 5 failed | 4 tests asserted text of the old 9-page PDF (stale tests, updated to the current two-page contract); 1 `cv_screening` interview mismatch that passed on an isolated re-run |
| 2 | 29 passed, 2 failed | FAQ chatbot: the model recorded `max_harm=minimal` that the user never stated; `cv_screening`: the model inferred "executes without approval" from a sentence about scoring, giving L4 instead of L1. Both are extraction-precision errors from model variance |
| 3 | 29 passed, 2 failed | `Connection error` (transient network failure) in `procurement_conditional` and `cv_screening`; those two tests did not complete, so they are **unverified in that run** |

After runs 1 and 2, narrow prompt rules were added (harm only from consequence statements; execution only when the user says the agent itself acts). Targeted repeat measurements of the first extraction pass: FAQ `max_harm` error 2 of 24 runs before, 0 of 24 after (the previous release code showed 0 of 24); `cv_screening` wrong authority 1 of 20 before, 0 of 30 after; the procurement scenario still extracted the expected conditional workflows (8 of 8). These are small samples, not proofs. Two later prompt changes (base-form action text, placeholder conditions) were verified offline only.

**There is no clean, complete live run of the current commit.** One full live interview with the procurement scenario (before those last two changes) gave L4, I2, P4, HIGH, completed in 14 answers, with the "review afterwards" fact recorded and consistent across the review screen, C09 and the PDF (2 pages). It used about 91k prompt and 3k completion tokens for the interview calls, excluding report-time calls (n = 1).

Known live-test fragility: results depend on a non-deterministic model; some assertions are strict on facts the model may attach to a different slot; a transient network error fails a test without indicating a defect.

## 4. Historical records

`evals/RESULTS.md` is a dated log (2026-11-08 to 2026-11-12) of earlier live runs, including figures such as "21/21", "24/24" and "31/31". Those numbers were true for the code and models at that date. They were **not re-executed** in this audit and must not be read as current results. Counts in that log for offline tests (110, 171) are also historical; the current count is in section 1.

## 5. Continuous integration

`.github/workflows/ci.yml` runs on push and pull request with Python 3.12 and **no API key or secrets**:

1. Compile check.
2. **Portable suite:** the offline tests with an empty sources directory.
3. **Full offline suite** with the repository's `sources/`, followed by a job-summary line with the skip count. If a source PDF is later removed from the repository, the source-dependent tests skip instead of failing; the summary then shows the skip count so a green run is not mistaken for full coverage.
4. `tools/verify_catalog.py` only when all required sources are present.

The workflow was validated by parsing the YAML and by running the same commands locally on Python 3.12.7. It has **not** been executed on GitHub's runners.

## 6. Known gaps

- No Linux or Python 3.13 run; PDF font is bundled so output should match, but this is untested.
- No statistical validation: 7 golden scenarios and 11 retrieval topics; support thresholds were tuned on this small set.
- The retrieval support check and the report-time model check have not been audited passage by passage.
- No browser-level UI test; UI behaviour is covered by Streamlit `AppTest` only.
- No test of the extraction under adversarial paraphrase beyond the injection scenarios.
- The legislation text and obligation dates have not been checked against an official source.
- Source-dependent tests do not run in CI unless the PDFs are in the repository (see [SOURCES.md](SOURCES.md)).

## 7. Reproduction

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-tested.txt           # exact tested versions (or requirements.txt for current ones)
python -m pytest -q                              # expect 231 passed, 2 skipped with all five sources present
TRUSTAI_SOURCES_DIR=$(mktemp -d) python -m pytest -q -rs   # expect 86 passed, 147 skipped
python tools/verify_catalog.py                   # expect 0 problems
OPENAI_API_KEY=... python -m pytest -m live -s -q   # optional, paid; results will vary between runs
```
