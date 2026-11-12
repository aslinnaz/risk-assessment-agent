# Assessment methodology

Everything here describes the implemented behaviour (`classify.py`, `catalogs/policy.json`, `catalogs/controls.json`, `interview.py`). Sections marked **Not implemented** are recommendations only.

**Scope of the result.** The result is a preliminary assessment from one person's declarations. Nothing is tested or verified, and the evidence state is always "E0: declarations only". The risk level is a **TrustAI label**, not an EU AI Act category, and the legal pathway is reported separately.

## 1. From answers to facts

1. Each answer goes to an extraction call (`SYSTEM_EXTRACT`) that is instructed to return slot values plus a short **verbatim quote** from the user's text.
2. `interview._sanitize` accepts a value only if the quote matches the user's own answer (`evidence_ok`, after removing instruction-like sentences). Matching is on word stems, not exact text: a positive value needs at least 80% of the quote's stems to occur in the answer; a **negative** value (`no`, `none`) needs an explicit negation word in the answer and at least 50% stem overlap, and additionally a quote that mentions the fact itself (`evidence_terms` in the catalog), so an unrelated negative sentence does not count. The check proves where the words came from, **not** that the quote supports the chosen value. Rejected extractions are kept with their reason and are never used.
3. Values that come from a selected option or a ticked checklist item are recorded as exact choices, with the option label as evidence.
4. Slot-confusion guard: a "review afterwards" statement extracted into `human_review` (decision maker can see the basis and object) is re-routed to `post_action_review`; the two are different safeguards.
5. Placeholder text such as "unspecified threshold" in a workflow condition is dropped; it is not something the user said.
6. `classify.derive` folds all turns, in order, into **one** accepted-fact state. Equivalences in `interview.json` propagate facts that follow logically (for example "not a regulated product" implies "no third-party assessment"). A conflict exists only for the same workflow and field described two ways; it is asked once with its scope named.

A quote shows that the words came from the user. It does not show that they are true or that they mean the extracted fact.

## 2. Autonomy L

The user's description is turned into **per-workflow** facts (action, condition, threshold, execution mode, approval authority, whether approval is bound to the action, whether limits are technically enforced). The agent's scalar authority is the maximum over the material workflows. Mapping (`autonomy_map`):

| Declared authority | L | Rule |
|---|---|---|
| Information only | L0 | |
| Recommends; a human decides and acts | L1 | |
| Prepares a draft; a human executes | L2 | |
| Acts after approval **bound to that action** | L3 | If approval is not declared as bound to the action it is treated as **L4** (`POL-L3-BIND`) |
| Acts within an envelope without per-action approval | L4 | The policy text says "technically enforced"; policy-only limits stay L4 and are handled by escalation and a release blocker (section 5) |
| Can expand its own authority or access | L5 | A declared ability to expand authority raises L3 or L4 to L5 (`POL-L5-EXP`) |

Conditional approval (different rules for different amounts, suppliers or action types) is normal and is **not** a contradiction. Delegation to other agents is recorded as a note: a delegate must never hold more authority than the delegator.

## 3. Impact I

Declared reasonable worst-case harm maps directly: minimal I0, limited I1, serious I2, critical I3. Reversibility alone never lowers a class. I is never inferred from a description of what the agent does.

## 4. Posture P and risk level

**Decision table** (impact × autonomy; frozen in `tests/test_decision.py::FROZEN_POSTURE`, 24 cells):

| | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| I0 | P1 | P1 | P1 | P2 | P2 | P3 |
| I1 | P1 | P2 | P2 | P2 | P3 | P4 |
| I2 | P3 | P3 | P3 | P3 | P4 | P4 |
| I3 | P3 | P3 | P3 | P3 | P4 | P4 |

**Escalation (`POL-ESC`).** From I1 upwards the posture rises by **one** step if at least one exposure factor is declared: wide reach, irreversible harm, vulnerable groups affected, sensitive personal data, or (for L4/L5) limits that are policy-only or absent. Several factors still cause one step; reasons are recorded; the posture is never lowered. An escalation is reported **only if the final posture is higher than the table value**; if the table already gives P4 the declared factors are recorded but no escalation is claimed.

**Risk level** (`label_from_posture`): P1 LOW, P2 MEDIUM, P3 HIGH, P4 HIGH. P4 means "hold for redesign": an internal release decision, not a legal prohibition, and not escalated to FORBIDDEN. If L or I is unknown, every possible value is tried to give a posture **range**; a level is issued only if the range maps to a single level, otherwise the result is "not completed" and the missing critical facts are named.

## 5. Legal pathway, prohibitions and transparency (EU AI Act reference)

The AI Act is the reference text. EU applicability defaults to "not determined" and is set only by an explicit user declaration. "No EU link declared" produces a general TrustAI assessment, not an assessment of EU obligations.

| Legal code | Condition |
|---|---|
| `annex1_high` | Product (or safety component) under Annex I legislation **and** a third-party conformity assessment, both declared. A "safety component = yes" answer alone is **not** enough (Art. 6(1) needs both) |
| `annex3_high_profiling` | Annex III area declared and profiling of people declared (Art. 6(3): the exception never applies) |
| `annex3_high` | Annex III area declared and the Art. 6(3) exception not supported; also the cautious default when the exception facts are unknown |
| `annex3_exception_claimed` | Annex III area declared, exception claimed by declaration (narrow or preparatory task, does not influence the decision, no profiling); documentation and registration duties still apply |
| `not_high_by_declaration` | Annex III declared as not involved **and** the Annex I route explicitly excluded |
| `partly_assessed` | Annex III declared as not involved but the Annex I route is not established. Not described as "no legal pathway" |
| `out_of_scope_declared` | No EU link declared (the Regulation may be out of scope; not verified as a legal analysis) |
| `undetermined` | Annex III answer missing or unknown |

A supported high-risk pathway makes the level at least HIGH without changing the posture; human approval or low autonomy does not lower it.

**FORBIDDEN** is issued only for a supported Art. 5(1)(a)–(h) finding: all conditions of the prohibition supported by user-quoted facts, exception, scope and date considered. A keyword is not enough. It is a preliminary finding that needs human legal review. The interview stops; the agent itself is not stopped.

**Transparency (Art. 50)** features are recorded as separate notes; on their own they do not raise the TrustAI level.

## 6. Release blockers, gaps and recommendations

- **Release blockers** are listed when declared: prohibited use, policy-only limits, approval not bound to the action, no immediate stop, authority expansion. They are findings, not part of the L/I/P formula.
- **Context-sensitive gaps.** For an agent that executes actions, unknown stop, reversal, delegation, scale, technical enforcement, aggregate limits (split-order circumvention) and approval-bypass facts are critical gaps; for an agent that only recommends they are not applicable. Approval before execution, monitoring during execution, immediate stop, post-action review and reversal are separate facts.
- **Next steps** in the PDF (at most three) come from `improvements` rules in `policy.json` that match the facts. They are generic recommendations, not findings.

## 7. Uncertainty handling

| Situation | Treatment |
|---|---|
| User says "I don't know" | Recorded as **unknown**; one simpler follow-up is asked at most once; a critical unknown blocks a level or is listed as a gap |
| Checklist item not ticked | **Unknown**, asked once more for the open items only |
| Checklist with "This is the complete list" ticked | Unticked items become explicit "no" |
| "None of these" | All items explicit "no" |
| Mixed certainty ("not sure about some") | Ticked = yes, the rest unknown |
| Missing information | Never converted to a negative or to "not applicable" |
| Model proposes a value without a valid quote | Rejected and recorded; not used |

The review screen shows every fact as confirmed, explicit negative, unknown or not collected, and the user may correct it before the result is produced.

## 8. Controls C01–C21

`controls.json["assessment_rules"]` maps accepted facts to the 21 controls. Each control has one or more requirements; each requirement reads a slot (or a control declaration), lists the values that count as present, absent or partial, and has an applicability condition.

| Status | Rule |
|---|---|
| Declared present | Every applicable requirement is declared present |
| Declared absent | Every applicable requirement is declared absent |
| Partially assessed | Mixed results, or some requirements missing (the missing ones are named) |
| Unknown | Only "I don't know" answers exist |
| Not assessed | Nothing was collected; the finding states exactly what is missing |
| Not applicable | An explicit scope rule applies with a stated reason. C16 needs declared "no interaction, no generated content, no emotion/biometric features"; C20 needs every high-risk route explicitly excluded (or no EU link declared). Missing information never yields "not applicable" |

"Present" always means **declared**; nothing is verified. Most controls are not asked within the question limit, so many show "Not assessed". C01–C21 is a coverage map of declarations, not a safety score. The same rows appear in the application and the PDF.

## 9. Rule provenance

| Element | Origin | Status |
|---|---|---|
| L0–L5 and I0–I3 definitions, the 24-cell P table | Project's internal TrustAI policy; checked against a reference Word file by tests when it is present locally (it is not in the repository, so those two tests skip) | Internal; not externally validated |
| P to LOW/MEDIUM/HIGH mapping | Project choice | Not externally validated |
| Escalation factors (irreversible harm, sensitive data, policy-only limits, etc.) and "one step" | Project choice | Not externally validated |
| Maximum authority over workflows; policy-only limits kept at L4 | Project choice | Not externally validated |
| Control IDs C01–C21 and their requirement mappings | Project catalog (`controls.json`); anchors checked against the source passages by `tools/verify_catalog.py` | Internal |
| Art. 5, 6, 50 screening | EU AI Act text loaded from `sources/`; interpretation by the project | Needs legal review |
| Obligation application dates in `policy.json` | Project data | **Not verified against an official source** |

Rule IDs (`POL-…`, `CTL-Cnn`) in the trace identify the policy rule applied; the policy version is `TrustAI internal assessment policy v0.3 (2026-11-12)`.

## 10. Known methodological limits

- Declarations may be wrong, incomplete or optimistic; the method cannot detect this.
- The same L can describe very different real systems; "technically enforced" is not verified for L4.
- The text of L4 ("technically enforced operating envelope") appears in the report even when the user declared policy-only limits; the escalation and blocker explain the difference. **Not implemented:** a separate label for policy-only L4.
- Model extraction varies between runs (see [VALIDATION.md](VALIDATION.md)).
- **Not implemented:** weighting, probability or residual-risk estimation, evidence-based confidence beyond E0.
