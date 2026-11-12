# TrustAI Risk Assessment — prototype

> **Status: prototype for research and feedback.** The result is a **preliminary, declaration-based assessment**. It is not a technical verification, a control test, a compliance finding, a certificate or legal advice.

## Purpose
The application interviews a **human** who knows the AI agent being assessed and produces a preliminary risk assessment from their **declarations** only:
* a TrustAI risk level: **LOW / MEDIUM / HIGH / FORBIDDEN** (or "assessment not completed");
* the internal TrustAI assessment policy result: autonomy **L0-L5**, impact **I0-I3** and assurance posture **P1-P4** (control identifiers C01-C21);
* a PDF report that shows how each value was derived and cites the sources.

It never talks to the target agent, runs tests or attacks, or asks for logs, code or evidence. The EU AI Act is the assessment *reference*; legal applicability is a separate field and defaults to "not determined". Everything in the application (UI, questions, options, rationales, prompts, reports) is in English.

Stack: Python, Streamlit, OpenAI API (chat + embeddings), local BM25 plus a persistent embedding index.

## How it works (question-based, RAG-driven, deterministic result)
1. **Question-based.** `planner.py` decides what to ask next (conflicts, one simple follow-up after "I don't know", missing critical facts, answers that would change the result, a few context topics). Target 10-15 questions, hard limit 20.
   Questions about this specific agent (purpose, authority, harm, oversight) come first. The prohibited-use screening (EU AI Act Art. 5) is **one combined checklist question** (biometric data, behaviour steering, scoring of people) asked after them: tick what applies, choose "None of these", say you are not sure, or answer in your own words. **An unticked box is never recorded as "no" by itself**: "None of these" records an explicit negative, and ticking "This is the complete list: anything I did not tick does not apply" records the unticked items as explicit "no"; otherwise the items that were not ticked stay unknown and are asked once more, only for the open items (the other checklists work the same way). A checklist is the only answer interface of its question (no second set of buttons). Every option question also accepts a free-text answer. The question wording always comes from the catalog (its answer options are mapped to facts by label); a model-written opening sentence is kept only if its content words occur in the conversation so far. Essential scope questions (role, EU link, status) are asked first.
2. **LLM extracts facts only.** Call 1 turns the answer into structured facts (slots). A fact is accepted only if the model supplies a verbatim quote from the user's text; quotes inside instruction-like sentences are ignored, and a negative fact needs a quote about that fact. The quote is stored with the fact.
3. **RAG retrieves evidence from the approved external sources** (EU AI Act, Commission Decision, both TrustAI playbooks, NIST AI RMF Playbook, plus any approved source added after deployment; legislation, voluntary frameworks and guidance are labelled as such and guidance is never presented as a legal obligation). Retrieval is routed by the governance concern of each question (legal classification → legislation; authority, delegation, oversight, reversal → the agentic playbook and NIST; transparency → the AI Act). For each question and each report finding the system retrieves passages (curated anchors, then BM25 + embedding search). A passage is shown only if it passes a **support check** (keywords of the claim and embedding similarity). At report time one batched model call additionally checks the meaning of each claim-passage pair; unsupported passages are dropped and listed; partial support is flagged and never promoted. The **internal assessment policy is not a retrievable source**: it is never indexed, retrieved or cited. Report findings separate three kinds of provenance: user declarations, internal policy rules (rule IDs such as POL-L4 with the policy version) and external sources.
4. **Result is deterministic.** `classify.py` computes the result from the facts only and imports neither the model nor the retrieval index. Retrieved text and the model cannot change it.
5. **Trace.** The report shows answer → extracted fact (with quote) → retrieved evidence → rule → result for L, I, the posture, the legal pathway and the risk level.

### Decision model (internal TrustAI assessment policy; rule IDs POL-...)
* **L** from the declared authority, evaluated **per workflow** (action + condition): the user's answer is turned into workflow facts such as "places purchase orders up to EUR 5,000" (executes without approval) and "orders above EUR 5,000" (executes after approval by a human manager, bound to the action or not, limit technically enforced or not). A conditional approval structure is **not** a contradiction. The maximum declared reachable authority over the material workflows defines L (info only L0, recommends L1, drafts L2, acts after approval **bound to that action** L3 (otherwise L4), acts within a technically enforced envelope without per-action approval L4, **expands its own authority L5**). A genuine contradiction (the same action and condition described two different ways) is asked once, with its scope named.
* **I** from the declared reasonable worst-case harm: I0-I3 (reversibility alone never lowers a class).
* **P** = policy decision table cell (impact × autonomy; the 24-cell table is frozen in a test, and compared with an optional reference file when present). If the posture exposure factors are declared (wide reach, irreversible harm, vulnerable groups, sensitive data, authority limits that are only written in instructions) the posture rises by **one** step; it is never lowered. P4 is an internal "hold for redesign" posture, not a legal prohibition.
* **Risk level** is derived from the final posture: P1 = LOW, P2 = MEDIUM, P3/P4 = HIGH. If L or I is unknown, every possible value is tried to give a posture *range*; a level is issued only if the range maps to one level.
* **Legal pathway** (Art. 6) is reported separately; a supported high-risk pathway makes the level at least HIGH without changing the posture. **FORBIDDEN** is only issued for a supported Art. 5 finding.
* Declared **release blockers** (policy-only limits, approval not bound to the action, no stop mechanism, authority expansion) are listed. **Context-sensitive gaps:** for an agent that executes actions, unknown stop, reversal, delegation, scale, technical enforcement, aggregate limits (split-order circumvention) and approval-bypass facts are critical gaps; for an agent that only recommends they are not applicable. Approval before execution, monitoring during execution, immediate stop, post-action review and reversal are separate facts. A declaration is never treated as proof that a safeguard works.
* **Three separate statuses** in the report: *Interview* (Completed / Incomplete), *Assessment* (Preliminary / Requires further information, e.g. when a control that affects the result is unknown) and *Verification* (always "Not technically verified"). Finishing the interview never implies that a control was tested.
* Report conclusions are separate: legal obligations, inherent exposure, control effectiveness (not assessed), evidence confidence (E0 — declarations only), residual risk (not assessed), operational recommendation (unable to assess).

## Installation
**Python 3.12 or 3.14** (both tested on macOS from a fresh virtual environment; Linux was not available for testing). Exact tested package versions are in `requirements-tested.txt`.
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # includes PyMuPDF (PDF text extraction); pypdf is only a fallback
cp .env.example .env              # put your OPENAI_API_KEY in .env (never committed); or export OPENAI_API_KEY in the shell

```
`OPENAI_MODEL` (default `gpt-4.1-mini`) and `OPENAI_EMBED_MODEL` (default `text-embedding-3-small`) are environment variables; Streamlit secrets are also supported.

## Source files
`sources/` contains the following files (names matter; versions and dates are in `catalogs/sources.json`):

| File name | Source | In the repository |
|---|---|---|
| `EU_AI_Act_OJ_L_2024_1689.pdf` | Regulation (EU) 2024/1689, OJ L, 12.7.2024 (original text) | yes |
| `EU_Commission_Decision.pdf` | Commission Implementing Decision (EU) 2026/2274 (harmonised-standard reference only) | yes |
| `TrustAI_Agentic_Governance_Playbook.pdf`, `TrustAI_Governance_Playbook.pdf` | TrustAI (2026) | yes |
| `AI_RMF_Playbook.pdf` | NIST AI RMF Playbook | yes |
| `compliance_matrix.docx` | **Optional, test-only** reference copy of the internal policy table. The application never reads, indexes or cites it. | no |
| `ISO_42001_2023.pdf` | Optional and **not used**: excluded from the index and prompts for copyright reasons; listed as a bibliographic entry only | no |

At startup every required file is checked (`rag.check_sources`) and the parsed structure is validated (all 113 AI Act articles, 13 annexes, Art. 5(1)(a)-(h), Art. 6(1)-(4) and every anchor the catalogs cite, `Index.validate`). A missing file or a mis-parsed document stops the application with a message naming it; the search index is built automatically (no indexing script). Source texts are treated as data; instructions inside them are never applied. See `sources/README.md`.

## Running
```bash
source .venv/bin/activate
streamlit run app.py      # opens http://localhost:8501
```
On startup the passages are extracted locally (cached in `.cache/`) and the **embedding index is prepared automatically**: the saved index is loaded and only missing passages are embedded (a small one-off API cost on the first run). If this fails the sidebar and the report say that keyword-only search was used. User answers and reports stay in the session.

## Tests
```bash
python -m pytest -q                      # offline: scripted fake client, no API calls; syntax/app-start/UI smoke tests also run without sources/
python tools/verify_catalog.py           # catalog citations and interview anchors vs. the real passages
python -m pytest -m live -s -q           # REAL model: extraction, question quality, retrieval, injection (costs money)
```
`evals/scenarios.py` holds the golden scenarios; `evals/RESULTS.md` records the latest live results and the defects they uncovered.

## Structure
`app.py` UI · `interview.py` interview (extraction call + question call) · `planner.py` question selection and finishing (no model) · `classify.py` deterministic classification and trace · `rag.py` extraction / index / retrieval / support checks · `report.py` report and PDF · `catalogs/` source manifest, C01-C21 coverage map, interview topics and decision policy · `evals/` live evaluation.

## Current limitations
* **Small live evaluation.** 6 scenarios and 11 retrieval topics (see `evals/RESULTS.md`); it found and fixed real defects but is not a statistical validation. Support thresholds were tuned on this small set.
* **Declarations only.** Nothing is verified. A quote shows the words came from the user, not that they are true or that they mean the extracted fact.
* **FORBIDDEN** depends on facts extracted by the model (each needs its own user quote); it is a preliminary finding that needs human legal review.
* **Project choices beyond the internal policy's table:** the mapping P → LOW/MEDIUM/HIGH; irreversible harm and sensitive data as posture-escalation factors; the maximum authority over workflows; policy-only limits kept at L4 with an escalation and a blocker. The policy is internal and has not been externally validated.
* **Currency of the legislation is not verified.** The loaded AI Act text is the original 2024 text; secondary sources report postponed high-risk dates (not confirmed from an official source). The report states this.
* ISO/IEC 42001 may be referenced bibliographically; its text is never indexed or retrieved.
* C01-C21 is a coverage map of declarations, not a safety score; most controls are not asked.
* No license has been added yet; the code and catalogs are for review before sharing.


## Approved sources added after deployment
Legislation, voluntary frameworks, guidance or licensed standards can be added without changing the application:
```bash
python tools/add_source.py add path/to/document.pdf --id NIST-GENAI --title "..." --publisher NIST --version 1.0 --date 2024-07-26 \
       --kind framework --concerns agentic,oversight --indexing-permitted
python tools/add_source.py list            # active, excluded (with the reason)
python tools/add_source.py deactivate NIST-GENAI
```
The document is copied to `sources/custom/`, validated (PDF or DOCX; complete metadata; extractable text; licence confirmation that indexing is permitted; no duplicate content; unique id), chunked by section with page numbers, and indexed at the next start (only new passages are embedded). `--supersedes OLDID` replaces an older version. Retrieval is routed by `--concerns` (legal, agentic, oversight, transparency, data, all). Incomplete, scanned, duplicate or unlicensed documents are rejected or flagged; copyrighted standards must not be added. Adding a source never changes the deterministic classification.

## Report
The PDF is a concise client report of two pages (three at most): page 1 = assessed agent, risk result (level, L, I, P), why this result, up to three next steps and the limitation statement; page 2 = the C01–C21 table (ID, control, assessment, finding). Control statuses are Declared present, Declared absent, Partially assessed, Unknown, Not assessed and Not applicable; each comes from an explicit fact-to-control mapping in `catalogs/controls.json` (`assessment_rules`) and is shown identically in the application. Sources, retrieval diagnostics, the answer trail and the decision trace stay in the application and in the stored records; they are not in the PDF. The PDF uses the Unicode font bundled with ReportLab (identical on macOS and Linux; `REPORT_FONT` / `REPORT_FONT_BOLD` override it).
