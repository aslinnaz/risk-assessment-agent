# Developer guide

## Setup

```bash
git clone https://github.com/aslinnaz/risk-assessment-agent.git && cd risk-assessment-agent
python3 -m venv .venv && source .venv/bin/activate     # Python 3.12 or 3.14 verified
pip install -r requirements-tested.txt                  # exact tested versions; requirements.txt allows newer ones
cp .env.example .env                                    # set OPENAI_API_KEY
python -m pytest -q                                     # offline; no key needed
streamlit run app.py
```

The five source PDFs must be in `sources/` (see [SOURCES.md](SOURCES.md)). The first start embeds their passages through the OpenAI API and stores them in `.cache/`; later starts embed only missing passages.

## Configuration

| Setting | Where | Default | Notes |
|---|---|---|---|
| `OPENAI_API_KEY` | environment, `.env`, or Streamlit secrets | none | Required to run the app; never needed for offline tests. Never printed (tested) |
| `OPENAI_MODEL` | same | `gpt-4.1-mini` | Chat model for extraction, questions and the report support check |
| `OPENAI_EMBED_MODEL` | same | `text-embedding-3-small` | Changing it creates a separate embedding cache file |
| `TRUSTAI_SOURCES_DIR` | environment | `./sources` | Used to test the "no sources" case; also lets a deployment keep sources outside the repository |
| `REPORT_FONT`, `REPORT_FONT_BOLD` | environment | bundled Bitstream Vera | Optional override; the font must contain the required glyphs or PDF creation raises an error |
| Question limits | `config.py` | target 12, maximum 20, context topics 3 | Changing them changes interview behaviour and some tests |

`.streamlit/config.toml` holds the theme only. Do not commit `.env` or `.streamlit/secrets.toml` (both are git-ignored).

## Code map

Start with [ARCHITECTURE.md](ARCHITECTURE.md). Practical entry points:

- Add or change a **question**: `catalogs/interview.json` (`topics`).
- Add or change a **fact**: `catalogs/interview.json` (`slots`, `slot_desc`, `equivalences`), then `classify.py` if it affects a decision.
- Change **policy** (tables, labels, blockers, next-step texts): `catalogs/policy.json`.
- Change **control statuses**: `catalogs/controls.json` (`assessment_rules`, `scope_rules`, `result_relevance`).
- Change **prompts**: `interview.py` (`SYSTEM_EXTRACT`, `SYSTEM_QUESTION`).
- Change the **PDF**: `report.py` (`client_summary`, `pdf_bytes`).

## Extending the system

**New interview topic.** Add an object to `topics` with at least `id`, `phase`, `title`, `slots` (and/or `controls`), `priority`, `hint`, `sources`, `anchors` (`"SOURCE-ID|section"`), `default_q`, `options` plus `option_map` (option label → slot values), `evidence_terms` and `concerns`. A checklist topic uses `multi` instead of options. Then run `python tools/verify_catalog.py`: every anchor must exist and be supported by its passage.

**New slot.** Add it to `slots` (value list or `"text"`), describe it in `slot_desc` (this text is shown to the extraction model, so wording matters), and add an `equivalences` rule if other facts follow from it. If a neighbouring slot has a similar meaning, say so in both descriptions; the model confuses near neighbours (see "review afterwards" in [METHODOLOGY.md](METHODOLOGY.md)).

**New control rule.** Add requirements under `assessment_rules["Cnn"]`: each has `id`, `label`, a `slot` (or `decl` for a control declaration), `present`/`absent`/`partial` value lists (`"*"` = any known value, `"set"` = text present), `say` phrases per state and an optional `applies` condition. The aggregation rules are in [METHODOLOGY.md](METHODOLOGY.md), section 8.

**Changing policy thresholds.** Do this only deliberately. The posture table is frozen in `tests/test_decision.py::FROZEN_POSTURE`; update the test and the table together and record why in [HANDOVER.md](HANDOVER.md). `classify.py` must not import `llm` or `rag` (a test enforces this).

**Adding a source after deployment.**
```bash
python tools/add_source.py add path/to/doc.pdf --id NIST-GENAI --title "..." --publisher NIST \
       --version 1.0 --date 2024-07-26 --kind framework --concerns agentic,oversight --indexing-permitted
python tools/add_source.py list                 # active and flagged sources, with reasons
python tools/add_source.py deactivate NIST-GENAI
```
The document is copied to `sources/custom/` (git-ignored), validated (PDF or DOCX, complete metadata, extractable text, no duplicate content, unique id, explicit indexing permission), chunked with page numbers and indexed at the next start. `--supersedes OLDID` replaces an older version. Do not add copyrighted standards or documents whose owner has not agreed to external processing. Adding a source never changes the deterministic classification.

## Testing conventions

- Offline tests use `tests/conftest.py::FakeClient`, a scripted model: one script per answer (`slots`, `workflows`, `evidence`, `controls`). `tests/test_decision.py::st(slots)` builds a state directly from slot values; `tests/test_final_pass.py` and `tests/test_procurement_audit.py` show end-to-end patterns.
- UI tests use `streamlit.testing.v1.AppTest`; pass an absolute path to `app.py`.
- Tests that need the source PDFs use the `index` fixture and skip when they are missing.
- A change that fixes a defect should add a regression test that fails without the fix. Do not edit a test only to make it pass.
- Live tests (`pytest -m live`) cost money and vary between runs; run them deliberately and record the outcome (including failures) in [VALIDATION.md](VALIDATION.md).

## Troubleshooting

| Symptom | Likely cause and action |
|---|---|
| "Required source file(s) not found …" at startup | A PDF is missing from `sources/` or misnamed. Names must match the manifest exactly ([SOURCES.md](SOURCES.md)) |
| `SourceError: … not parsed into the expected structure` | The PDF differs from the expected structure (different edition, scanned file, kerned text). Check the file against the SHA-256 in SOURCES.md; do not edit the validation to pass |
| Setup message about the API key | `OPENAI_API_KEY` missing or invalid; check `.env` or the environment. The key is masked in errors |
| Sidebar says keyword-only search | Embedding call failed (network, quota, key). Retrieval falls back to BM25; restart after fixing. The report states which mode was used |
| Stale or odd retrieval after replacing a source | Delete `.cache/` (it is rebuilt automatically; embeddings are paid again) |
| PDF creation raises "No usable report font" | An override font lacks required glyphs; unset `REPORT_FONT`/`REPORT_FONT_BOLD` |
| Live test fails with `Connection error` | Transient network failure, not a defect; re-run that test |
| Live extraction test fails on a single run | Model variance; repeat the test several times before changing code (see VALIDATION.md, section 3) |
| `ModuleNotFoundError` for `pymupdf` | `pip install -r requirements.txt`; pypdf alone is only a fallback and is stricter about layout |

## Conventions

English only in code, UI and documents. No secrets in commits. Do not commit licensed or restricted documents. Prefer small, tested changes; the layout is intentionally flat.
