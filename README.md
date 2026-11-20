# TrustAI Risk Assessment Agent

A question-based **research prototype** that conducts a structured interview about an AI agent and generates a **preliminary, declaration-based risk assessment**. It does not access or test the assessed agent, verify permissions, certify compliance, or provide legal advice.

## How it works

1. **Interview:** The user describes the agent’s purpose, accessible tools and permissions (`read`, `write`, `modify`, `delete`, `send`, `execute`, `transact`), approvals, potential harms and safeguards.
2. **Fact extraction:** An LLM extracts structured facts from the answers. The application checks their grounding in user-provided text and preserves uncertainty.
3. **Evidence retrieval:** RAG retrieves relevant passages from the approved regulatory and governance documents, with source citations and relevance checks.
4. **Rule-based assessment:** Deterministic Python rules calculate autonomy **L0–L5**, impact **I0–I3**, assurance posture **P1–P4**, and the TrustAI risk label. **P1/P2 = LOW; P3/P4 = HIGH**. Supported prohibited-use findings are handled separately. Retrieved documents and the LLM cannot directly override the classifier.
5. **Report:** The application and PDF explain the result, the user-answer-to-fact-to-rule trace, tool/action inventory, relevant citations and C01–C21 control coverage.

**Important distinctions:** Analysis and recommendations are not external actions. The classification considers the maximum declared technically reachable authority, including self-expansion of permissions. Missing or contradictory facts are not assumed safe. The TrustAI L/I/P policy is an internal assessment framework, not an EU AI Act risk classification or a validated predictive model.

## Architecture

| Component | Responsibility |
|---|---|
| `app.py`, `planner.py`, `interview.py` | Streamlit interview, question selection and fact extraction |
| `rag.py`, `sources_admin.py` | Source ingestion, automatic index preparation and evidence retrieval |
| `classify.py`, `catalogs/` | Deterministic classification and control rules |
| `report.py` | On-screen findings and preliminary-assessment PDF |
| `tests/`, `evals/` | Offline regression tests and optional live-model evaluation |

## Setup

Requires Python 3.12 (the main tested target), an OpenAI API key and the approved source PDFs listed in `docs/SOURCES.md`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set OPENAI_API_KEY; never commit this file
streamlit run app.py
```

The first startup creates the local embedding index automatically. **This uses the OpenAI API and may incur costs.** The interview also requires API access. Do not run live evaluation without explicitly approving its cost.

## Testing

```bash
python -m pytest -q              # offline tests; no live model calls
python tools/verify_catalog.py   # catalog/source verification
```

The previous audit reported **378 passed, 2 skipped** offline; the real-LLM end-to-end evaluation could not be completed because API credits were unavailable. These figures are historical audit results, not a new verification of this cleaned package. See `docs/RELEASE_AUDIT.md` and `docs/VALIDATION.md` for details and limitations.

## Scope and limitations

- Assessments are based on **user declarations**, not technical inspections, penetration tests or verified controls.
- LLM extraction and retrieval relevance have not been fully validated against real-model evaluation; residual risk and control effectiveness remain unverified.
- Legal applicability, prohibited-use screening and regulatory references require qualified human review.
- The professor’s Compliance Check Matrix is **not bundled, indexed or used as a RAG source**. Internal policy rules are retained in `catalogs/`.
- Source PDF redistribution rights require review before making the repository public; see `docs/SOURCES.md`.

Additional documentation: [Architecture](docs/ARCHITECTURE.md) · [Methodology](docs/METHODOLOGY.md) · [Validation](docs/VALIDATION.md) · [Source manifest](docs/SOURCES.md) · [Release audit](docs/RELEASE_AUDIT.md).
