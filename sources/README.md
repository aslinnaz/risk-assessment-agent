# Source files

This folder holds the external documents used for retrieval. At startup the application checks that **every required file below exists** and that the legal structure and mandatory references were parsed (`rag.check_sources`, `Index.validate`). A missing file or a mis-parsed document stops the application with a message that names the file or reference; nothing is skipped silently.

| File (exact name) | Source | In the git repository |
|---|---|---|
| `EU_AI_Act_OJ_L_2024_1689.pdf` | Regulation (EU) 2024/1689, OJ L, 12.7.2024 | yes |
| `EU_Commission_Decision.pdf` | Commission Implementing Decision (EU) 2026/2274 | yes |
| `TrustAI_Agentic_Governance_Playbook.pdf` | TrustAI Agentic AI Governance Playbook | yes |
| `TrustAI_Governance_Playbook.pdf` | TrustAI AI Governance Playbook | yes |
| `AI_RMF_Playbook.pdf` | NIST AI RMF Playbook | yes |
| `ISO_42001_2023.pdf` | ISO/IEC 42001:2023 | **no, optional and never indexed or sent to an API** (copyrighted) |
| `compliance_matrix.docx` | Optional, test-only reference copy of the internal policy table | **no** |

The internal assessment policy (autonomy, impact, posture, control identifiers) is **not a documentary source**: the application never reads, indexes, retrieves or cites a document for it, and `compliance_matrix.docx` is not required. Without it, only the single test that compares the policy table with it is skipped.

**Adding approved sources later:** `python tools/add_source.py add ...` (see the README). Registered documents live in `sources/custom/` together with `registry.json`; they are local deployment data and are not committed. Any other file in this folder is ignored by the application.
