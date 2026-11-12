# Handover

Audience: the developer or researcher taking over responsibility for this prototype. Status as of the handover audit on 2026-11-12. Read [VALIDATION.md](VALIDATION.md) for exactly what was verified and [SOURCES.md](SOURCES.md) before sharing the repository.

## 1. What the project is, and is not

A working research prototype that produces a **preliminary, declaration-based** risk assessment of an AI agent from an interview. It has been exercised with scripted tests and with a small number of live runs. It is **not** production software, **not** a compliance or legal tool, and its internal policy (L0–L5, I0–I3, P1–P4, C01–C21, the LOW/MEDIUM/HIGH mapping, the escalation factors) has not been externally validated.

## 2. Completed work (implemented)

- Interview with planner, per-workflow authority facts, quote-validated extraction, one accepted-fact state, 20-question limit.
- Deterministic classification with a rule-ID trace; legal pathway kept separate from the TrustAI level; Art. 6(1) requires both product and third-party assessment declarations.
- Checklist semantics that never turn an unticked item into a "no" by themselves; review screen with confirmed / negative / unknown / not collected.
- Explicit fact-to-control mapping for C01–C21 with six statuses, identical in the application and the PDF.
- Local RAG over approved sources with support checks; startup validation of source structure.
- Two-page client PDF with an embedded Unicode font and no silent font fallback.
- 233 offline tests, 31 live tests, a catalog verifier, a CI workflow and documentation (this `docs/` folder).

Earlier development passes (release stabilisation, a procurement-agent audit, a final development pass, a live smoke test and a corrective pass) are recorded in `evals/RESULTS.md` and VALIDATION.md. The Git history of the original working repository (`risk-assessment-agentt`) holds the detailed commits; the public repository was created as a single clean commit.

## 3. Decisions required from the project owner

These are not engineering tasks; they need an authorised person. None has been decided in this audit.

| # | Decision | Why it matters |
|---|---|---|
| 1 | **Redistribution of source PDFs.** `TrustAI_Governance_Playbook.pdf`, `TrustAI_Agentic_Governance_Playbook.pdf` and `AI_RMF_Playbook.pdf` have no established redistribution permission, and all three are tracked in the public repository | Possible rights violation. Options: obtain written permission, or remove them from the index and follow the setup process in SOURCES.md. Setting the repository to private in the meantime is reversible |
| 2 | **History.** Removing a file from the index does not remove it from Git history, and the previous repository (`risk-assessment-agentt`, 7 commits) also contains these PDFs | Purging history requires rewriting it and force-pushing; this was not done and needs explicit approval. If that repository is public, it needs the same decision |
| 3 | **Licence.** No licence file exists. A licence must not be added until code ownership and authorisation are confirmed | Without a licence, others have no right to reuse the code. The TrustAI policy content may belong to an organisation, not the individual author |
| 4 | **PyMuPDF is AGPL-3.0** (or a commercial licence from Artifex). The application imports it | If the app is distributed together with PyMuPDF or offered as a network service, AGPL obligations may apply to the combined work. Decide whether to accept AGPL, buy a commercial licence, or switch to the permissively licensed pypdf (stricter on layout; the code supports it as a fallback but the AI Act parse was validated with PyMuPDF) |
| 5 | **Policy ownership and validation.** The L/I/P tables and controls are an internal policy | Someone accountable should own changes and arrange external review before results are relied on |
| 6 | **Commit author data.** The public repository's commit carries the author's name and personal e-mail address | Replace with a project or noreply identity if that is not intended (requires rewriting the single commit) |
| 7 | **Data protection.** User answers, and source passages, are sent to the OpenAI API | Confirm this is acceptable for the intended users and sources; the application has no consent screen or data-processing agreement |

## 4. Known issues and limitations

| Severity | Issue |
|---|---|
| High (publication) | Source redistribution rights not established for three tracked PDFs (decision 1) |
| Medium | **No clean full live run of the current commit.** Live runs show occasional extraction-precision errors from model variance (a fact attached to the wrong slot, an unstated value, or an action inferred as autonomous). Narrow prompt rules reduced these in small repeat tests; they are not eliminated. Re-run `pytest -m live` several times and record the results before relying on the live behaviour |
| Medium | Quote validation proves origin, not meaning (stem-level match). A wrong fact with a plausible quote is accepted. FORBIDDEN and the legal pathway depend on such facts and must be reviewed by a person |
| Medium | Legislation currency is not verified. The loaded AI Act is the 2024 text; obligation dates in `policy.json` and the manifest `currency_note` are unverified secondary information |
| Medium | Report-time support check and retrieved passages have not been audited passage by passage; thresholds were tuned on a small set |
| Low | The L4 wording ("technically enforced operating envelope") appears in the result even when the user declared policy-only limits. The escalation and blocker explain it, but the text can read as contradictory |
| Low | Most controls (C03, C06–C08, C10, C12–C15, C17, C21 and others) are rarely asked within 20 questions, so many show "Not assessed". This is intended, not a defect |
| Low | Two tests that compare the policy table with a reference Word file skip because the file is not in the repository; the frozen-table test covers the table itself |
| Low | CI has not run on GitHub yet; Linux and Python 3.13 are untested |
| Low | `evals/RESULTS.md` is historical; its counts are not current |
| Info | The PDF font (Bitstream Vera) has no arrow glyph; arrows are removed from PDF text |

## 5. Operational requirements

- Python 3.12 or 3.14; the packages in `requirements-tested.txt`.
- An OpenAI API key with access to `gpt-4.1-mini` and `text-embedding-3-small` (or other models set through the environment variables). Cost is per assessment plus a one-off source embedding; no price estimate is made by the application. One observed interview used about 91k prompt tokens (n = 1).
- Outbound HTTPS to the OpenAI API. No other network access.
- The five source PDFs in `sources/` with exact names; a writable `.cache/` directory.
- No authentication, authorisation, persistence or multi-user separation: sessions live in the Streamlit process. Run it locally or behind access control provided elsewhere.
- Streamlit and OpenAI SDK versions move quickly; use `requirements-tested.txt` for a reproducible environment and re-run the tests after any upgrade.

## 6. Responsibility boundaries

| Area | Owner after handover |
|---|---|
| Code, tests, CI, documentation | New developer |
| Meaning and thresholds of L/I/P, labels and C01–C21 | The accountable policy owner (decision 5); engineering changes them only on instruction and with the frozen-table test updated |
| Interpretation of the EU AI Act, Art. 5/6/50 findings, any statement about legal applicability | Qualified legal counsel; the application only flags pathways for review |
| Redistribution and processing rights of source documents | The rights holders and the project owner |
| Truth of an assessment's inputs | The person answering the interview; nothing is verified |
| Use of an assessment for a decision | The user of the result; the report states it is preliminary and declaration-based |

Material that existed only in the original working copy and is **not** in the repository: the licensed ISO/IEC 42001 PDF (must never be committed), `compliance_matrix.docx` (reference copy of the policy table, used by two optional tests) and a working-notes Word file. Ask the previous owner for the matrix if those two tests should run.

## 7. Recommended next steps (not implemented)

1. Resolve decisions 1 and 2 before wider sharing; consider making the repository private until then.
2. Confirm ownership and add a licence (decision 3); review PyMuPDF (decision 4).
3. Run `pytest -m live` several times on the current commit, record the outcomes in VALIDATION.md, and track extraction-precision failures.
4. Push a change and confirm the CI workflow passes on GitHub; consider a Linux run of the PDF tests.
5. Verify the AI Act text, amendments and application dates against EUR-Lex and update `policy.json` and `sources.json`.
6. Arrange an external review of the internal policy and the C01–C21 mapping.
7. If the tool will be used beyond demonstrations: persistence of assessments, authentication, a consent and data-processing notice, and a human legal-review step for FORBIDDEN and high-risk pathways.
8. Improve extraction robustness (for example a second validation pass for enumerated facts) only with a regression test per defect.
