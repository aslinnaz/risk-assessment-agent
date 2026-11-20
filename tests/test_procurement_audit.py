"""Regression tests for the procurement-agent end-to-end audit: conditional approvals, a single authoritative fact state, deterministic classification,
context-sensitive gaps, oversight separation, neutral explanations and screening. The model is scripted (FakeClient): these tests concern the assessor's own software."""
import pytest

import classify as C
import interview as iv
import planner as P
import rag
import report as rp
from conftest import FakeClient

ANSWER = ("The agent helps our procurement team review requests, check approved suppliers and create purchase orders. It can place orders up to EUR 5,000 without approval for each order. "
          "Orders above EUR 5,000 require approval from a human manager.")
SMALL = {"id": "po_small", "action": "place purchase orders", "condition": "up to EUR 5,000", "threshold": "EUR 5,000", "recipient": "approved suppliers", "recipient_restricted": "yes",
         "execution": "executes_without_approval", "evidence": "place orders up to EUR 5,000 without approval for each order"}
LARGE = {"id": "po_large", "action": "place purchase orders", "condition": "above EUR 5,000", "threshold": "EUR 5,000", "execution": "executes_after_approval",
         "approval_authority": "human manager", "approval_bound": "yes", "evidence": "Orders above EUR 5,000 require approval from a human manager"}
GATES_NO = {"g_biometric": "no", "g_influence": "no", "g_person_eval": "no"}


def start(index, answer=ANSWER, workflows=(SMALL, LARGE), slots=None):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, answer, index, FakeClient([{"slots": {"purpose": "Procurement agent: reviews requests, checks approved suppliers, creates purchase orders (up to EUR 5,000 on its own)",
                                                           "max_harm": "serious", "annex3_area": "none", **GATES_NO, **(slots or {})}, "workflows": list(workflows)}]))
    return s


def answer(s, index, text, **upd):
    iv.process(s, len(s["turns"]) - 1, text, index, FakeClient([upd]))
    return s


# ---------------------------------------------------------------- 1, 2, 5: conditional approval is not a contradiction; L4 x I2 = P4
def test_conditional_approval_is_not_a_contradiction_and_gives_l4_i2_p4(index):
    s = start(index)
    s = answer(s, index, "The limit is enforced by the purchasing system itself; larger orders are blocked until approved.",
               workflows=[{"id": "po_small", "limit_enforcement": "technical", "evidence": "enforced by the purchasing system itself"}])
    d = C.derive(s)
    assert d["conflicts"] == {} and d["unresolved_conflicts"] == {}
    assert [c for c in P.candidates(s) if c["kind"] == "conflict"] == []
    cls = C.classify(s)
    assert (cls["L"], cls["I"], cls["posture"]) == ("L4", "I2", "P4") and cls["risk_label"] == "HIGH"
    assert cls["escalation"] == []  # the limit is declared as technically enforced: no exposure escalation, P4 comes from the table itself
    assert d["slots"]["action_mode"] == "envelope_action" and d["slots"]["approval_exists"] == "partial"
    assert C.trace_integrity(s, cls) == []


def test_restating_the_same_conditional_structure_does_not_conflict(index):
    s = start(index)
    s = answer(s, index, "As I said: up to EUR 5,000 on its own, above that a manager approves.", workflows=[SMALL, LARGE])
    assert C.derive(s)["conflicts"] == {}
    assert C.classify(s)["L"] == "L4"


def test_l4_is_supported_by_the_autonomous_workflow_and_the_enforcement_declaration_not_by_an_ambiguous_choice(index):
    s = start(index)
    s = answer(s, index, "The limit is enforced by the purchasing system itself.", workflows=[{"id": "po_small", "limit_enforcement": "technical", "evidence": "enforced by the purchasing system itself"}])
    steps = {t["step"]: t for t in C.classify(s)["trace"]}
    quotes = " | ".join(i.get("quote") or "" for i in steps["Autonomy level (L)"]["inputs"])
    assert "up to EUR 5,000 without approval" in quotes and "enforced by the purchasing system" in quotes
    assert "Some actions are approved" not in quotes
    assert steps["Autonomy level (L)"]["rule_id"] == "POL-L4" and steps["Posture (P)"]["rule_id"] == "POL-P-I2xL4"
    assert any(i.get("role") == "ceiling" for i in steps["Autonomy level (L)"]["inputs"])


# ---------------------------------------------------------------- 3: genuine contradictions use the real scope of the statements
def test_genuine_contradiction_about_the_same_workflow_is_asked_with_its_scope(index):
    s = start(index, workflows=[{**SMALL, "execution": "executes_after_approval", "approval_bound": "yes", "evidence": "place orders up to EUR 5,000 without approval for each order"}])
    s = answer(s, index, "Actually it places those small orders without anyone approving them.", workflows=[{"id": "po_small", "execution": "executes_without_approval", "evidence": "without anyone approving them"}])
    d = C.derive(s)
    assert "wf:po_small:execution" in d["conflicts"]
    cand = next(c for c in P.candidates(s) if c["kind"] == "conflict")
    q = iv.make_question(cand["topic"], index, s, kind="conflict", cand=cand)
    assert "place purchase orders (up to EUR 5,000)" in q["text"] and "I don't know" in q["options"]
    assert C.classify(s)["L"] is None  # no guessing while the contradiction is open
    # resolving it with the option the person picks settles the authority
    pick = next(o for o in q["options"] if o.startswith("It runs without approval"))
    s["turns"][-1]["question"] = q
    iv.process(s, len(s["turns"]) - 1, pick, index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["conflicts"] == {} and C.classify(s)["L"] == "L4"


def test_unclarified_contradiction_stays_unresolved(index):
    s = start(index, workflows=[{**SMALL, "execution": "executes_after_approval", "approval_bound": "yes"}])
    s = answer(s, index, "No, it does it without approval.", workflows=[{"id": "po_small", "execution": "executes_without_approval", "evidence": "it does it without approval"}])
    cand = next(c for c in P.candidates(s) if c["kind"] == "conflict")
    s["turns"][-1]["question"] = iv.make_question(cand["topic"], index, s, kind="conflict", cand=cand)
    iv.process(s, len(s["turns"]) - 1, "I don't know", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert "wf:po_small:execution" in d["unresolved_conflicts"] and C.classify(s)["L"] is None


# ---------------------------------------------------------------- 4: rejected facts never enter the fact state or an explanation
def test_rejected_extraction_never_supports_a_classification(index):
    s = iv.new_state(); iv.add_first(s, index)
    fc = FakeClient([{"slots": {"purpose": "p", "action_mode": "envelope_action"}, "evidence": {"action_mode": "it flies to the moon"}}])  # quote is not in the user's text
    iv.process(s, 0, "It answers questions about opening hours.", index, fc)
    u = s["turns"][0]["updates"]
    assert "action_mode" not in u["slots"] and any(k == "action_mode" for k, _ in u["rejected"])
    d = C.derive(s)
    assert d["slots"].get("action_mode") in (None, "unknown")
    cls = C.classify(s)
    assert cls["L"] is None and C.trace_integrity(s, cls) == []


def test_trace_integrity_flags_a_rejected_extraction_cited_as_support(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, "It answers questions.", index, FakeClient([{"slots": {"purpose": "p", "action_mode": "envelope_action"}, "evidence": {"action_mode": "unrelated words"}}]))
    cls = C.classify(s)
    cls["trace"].append({"step": "forged", "inputs": [{"slot": "action_mode", "value": "envelope_action", "answer_id": "A1", "quote": "x"}], "rule": "", "result": ""})
    assert any("rejected extraction" in v or "not an accepted fact" in v for v in C.trace_integrity(s, cls))


def test_selected_option_replaces_the_stale_rejection_and_is_its_own_evidence(index):
    s = iv.new_state(); iv.add_first(s, index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["action_mode"], index, s)
    label = "Acts without approval"
    s["turns"][0]["question"]["option_sets"] = {label: {"action_mode": "envelope_action"}}
    iv.process(s, 0, label, index, FakeClient([{"slots": {"purpose": "p", "action_mode": "envelope_action"}, "evidence": {"action_mode": "invented quote"}}]))
    u = s["turns"][0]["updates"]
    assert u["slots"]["action_mode"] == "envelope_action" and not [r for r in u["rejected"] if r[0] == "action_mode"]
    assert C.derive(s)["evidence"]["action_mode"] == label


def test_correcting_an_answer_invalidates_downstream_facts_and_regenerates_the_result(index):
    s = start(index)
    assert C.classify(s)["L"] == "L4"
    iv.process(s, 0, "Sorry, it only recommends orders; a buyer places them.", index,
               FakeClient([{"slots": {"purpose": "Recommends purchase orders", "max_harm": "serious", "annex3_area": "none", **GATES_NO}, "workflows":
                            [{"id": "po_rec", "action": "recommend purchase orders", "condition": "all orders", "execution": "recommend", "evidence": "it only recommends orders"}]}]))
    d = C.derive(s)
    assert "po_small" not in d["workflows"] and d["slots"]["action_mode"] == "recommend" and C.classify(s)["L"] == "L1"


# ---------------------------------------------------------------- 6, 7: L3 and L5
def test_l3_requires_approval_bound_to_the_specific_action(index):
    ans = "Every order needs approval from a human manager before it is placed. The system blocks anything else."
    wf = {"id": "po_all", "action": "place purchase orders", "condition": "all orders", "execution": "executes_after_approval", "approval_authority": "human manager", "approval_bound": "yes",
          "limit_enforcement": "technical", "evidence": "Every order needs approval from a human manager"}
    s = start(index, answer=ans, workflows=[wf], slots={"max_harm": "limited"})
    cls = C.classify(s)
    assert (cls["L"], cls["I"], cls["posture"]) == ("L3", "I1", "P2") and cls["risk_label"] == "LOW"
    s2 = start(index, answer=ans, workflows=[{**wf, "approval_bound": "no"}], slots={"max_harm": "limited"})
    c2 = C.classify(s2)
    assert c2["L"] is None and any("Specific-action approval was not established" in n for n in c2["level_notes"])


def test_l5_when_the_agent_can_expand_its_own_authority(index):
    s = start(index, slots={"authority_expansion": "yes"})
    cls = C.classify(s)
    assert cls["L"] == "L5" and (cls["I"], cls["posture"]) == ("I2", "P4")
    assert any(b["id"] == "authority_expansion" for b in cls["blockers"])


# ---------------------------------------------------------------- 8, 9, 10: context-sensitive critical gaps
def _gaps(cls):
    return {g["slot"]: g for g in cls["context_gaps"]}


def test_missing_technical_enforcement_is_a_critical_gap_and_not_invented(index):
    cls = C.classify(start(index))
    g = _gaps(cls)["limits_enforced"]
    assert g["critical"] and g["state"] == "not_assessed"
    assert cls["escalation"] == []  # unknown enforcement is neither assumed technical nor assumed policy-only
    assert any("Technical enforcement" in n or "technical enforcement" in n for n in cls["level_notes"]) or any("technical enforcement" in u.lower() or "limit" in u.lower() for u in cls["uncertainties"])


def test_missing_stop_rollback_delegation_scale_aggregate_and_bypass_are_critical_for_an_acting_agent(index):
    g = _gaps(C.classify(start(index)))
    for slot in ("can_stop", "can_rollback", "delegates_to_agents", "reach", "aggregate_limit", "approval_bypass", "limits_enforced"):
        assert g[slot]["critical"] and g[slot]["state"] == "not_assessed", slot


def test_gaps_are_context_sensitive_not_everything_is_critical(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, "It only recommends a refund to an employee.", index, FakeClient([{"slots": {"purpose": "refund recommender", "action_mode": "recommend", "max_harm": "limited", "annex3_area": "none", **GATES_NO}}]))
    g = _gaps(C.classify(s))
    assert all(not v["critical"] for v in g.values())
    assert g["can_stop"]["state"] == "not_applicable" and g["can_stop"]["reason"]


def test_declared_values_close_a_gap_but_are_never_proof(index):
    s = start(index)
    s = answer(s, index, "A person can stop it at once; wrong orders can be cancelled.", slots={"can_stop": "yes", "can_rollback": "yes"})
    g = _gaps(C.classify(s))
    assert g["can_stop"]["state"] == "declared" and g["can_rollback"]["state"] == "declared"
    rep = rp.build_report(s, index)
    assert rep["verification_status"] == "Not technically verified" and "never treated as proof" in " ".join(v for _, v in rep["conclusions"])


def test_critical_context_topics_are_asked_for_an_acting_agent(index):
    s = start(index)
    ids = [c["topic"]["id"] for c in P.candidates(s)]
    assert "oversight" in ids and "boundaries" in ids and "aggregate_cap" in ids and "limits_enforced" in ids


# ---------------------------------------------------------------- 13: approval before execution, monitoring, intervention, post-action review and reversal are separate facts
def test_post_action_review_is_not_approval_before_execution(index):
    s = start(index)
    s = answer(s, index, "A person reviews the orders afterwards and can watch it run.", slots={"post_action_review": "yes", "monitoring_live": "yes"})
    d = C.derive(s)
    cls = C.classify(s)
    assert d["slots"]["post_action_review"] == "yes" and d["slots"]["action_mode"] == "envelope_action" and cls["L"] == "L4"  # reviewing afterwards does not create an approval requirement
    assert d["controls"].get("C09") is None or d["controls"]["C09"]["status"] != "declared_present"


def test_oversight_checklist_keeps_the_concepts_apart():
    m = C.TOPICS()["oversight"]["multi"]
    assert {i["slot"] for i in m["items"]} == {"can_stop", "monitoring_live", "post_action_review", "can_rollback"}
    text, sets = iv.multi_answer(m, ["post_action_review"], "confirm")
    assert sets["post_action_review"] == "yes" and sets["can_stop"] == "unknown" and "approval" not in " ".join(i["label"].lower() for i in m["items"])


# ---------------------------------------------------------------- 10: no risk leakage while the interview is running
def test_why_explanations_never_disclose_a_risk_level(index, monkeypatch):
    s = start(index)
    for c in P.candidates(s):
        q = iv.make_question(c["topic"], index, s, kind=c["kind"], cand=c)
        blob = " ".join(str(v) for v in q["why"].values()) + " " + str(q.get("impact", ""))
        assert not iv.leaks_result(blob), (c["topic"]["id"], blob)
    assert not iv.leaks_result(P.NEUTRAL_IMPACT)


def test_model_written_rationale_that_leaks_the_result_is_replaced(index):
    s = start(index)
    cand = next(c for c in P.candidates(s) if c["kind"] == "question")
    nxt = {"topic_id": cand["topic"]["id"], "question": "A neutral question about the agent?", "options": ["Yes", "No"], "missing": "The risk level will be HIGH if you say no.",
           "clarifies": "It decides whether the posture becomes P4.", "source_note": "", "passage_ids": [], "adapted": True}
    q = iv.make_question(cand["topic"], index, s, text=nxt["question"], nxt=nxt, cand=cand)
    assert not iv.leaks_result(" ".join(q["why"].values()))


# ---------------------------------------------------------------- 12: Article 5 / 50 screening
def test_article_5_checklist_multi_selection_and_unticked_items(index):
    m = C.TOPICS()["screening"]["multi"]
    _, sets = iv.multi_answer(m, ["g_biometric", "g_person_eval"], "confirm")
    assert sets == {"g_biometric": "yes", "g_influence": "unknown", "g_person_eval": "yes"}  # an unticked box is not a negative declaration
    _, none = iv.multi_answer(m, [], "none")
    assert set(none.values()) == {"no"}


def test_none_of_these_is_mutually_exclusive_in_the_ui(index):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    s = iv.new_state(); iv.add_first(s, index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["screening"], index, s)
    at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=60)
    at.session_state["state"] = s
    at.run()
    none_btn = next(b for b in at.button if b.label == "None of these")
    assert not none_btn.disabled
    at.checkbox[0].check().run()
    assert next(b for b in at.button if b.label == "None of these").disabled
    assert not next(b for b in at.button if b.label == "Confirm my selection").disabled


def test_a_positive_screening_answer_asks_for_clarification_and_does_not_conclude_forbidden(index):
    s = iv.new_state(); iv.add_first(s, index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["screening"], index, s)
    text, sets = iv.multi_answer(s["turns"][0]["question"]["multi"], ["g_person_eval"], "confirm")
    s["turns"][0]["question"]["option_sets"] = {text: sets}
    iv.process(s, 0, text, index, FakeClient([{"slots": {"purpose": "scores candidates"}}]))
    cls = C.classify(s)
    assert cls["risk_label"] != "FORBIDDEN"
    assert any(c["kind"] in ("followup", "question") and c["topic"]["id"] in ("eval_criteria", "prohibited_detail") for c in P.candidates(s))


def test_unsettled_prohibition_is_flagged_for_legal_review_not_cleared(index):
    s = start(index)
    s["turns"][0]["updates"]["slots"].update({"g_person_eval": "yes", "eval_uses_social_traits": "yes", "p_social_scoring": "possible"})
    rep = rp.build_report(s, index)
    assert rep["legal_review"] and "legal review" in rep["legal_review"][0] and rep["risk_label"] != "FORBIDDEN"


def test_transparency_questions_use_the_article_50_criteria_not_a_generic_shortcut(index):
    t = C.TOPICS()
    q = t["t_synthetic"]["default_q"].lower()
    assert "free-form text" in q and "structured fields" in q and "generates content" not in q
    assert t["t_synthetic"]["option_map"]["No, it only takes actions or fills in structured fields"] == {"t_synthetic_content": "no"}
    assert "published to inform the public" in " ".join(t["t_deepfake"]["options"]) and any("imitate" in o for o in t["t_deepfake"]["options"])
    assert t["t_deepfake"]["condition"] == "synthetic_yes"  # detailed follow-ups only when relevant to the declared use


def test_internal_structured_use_has_no_transparency_finding_but_public_text_does(index):
    base = {"purpose": "p", "action_mode": "recommend", "max_harm": "limited", "annex3_area": "none", **GATES_NO}
    s = start(index, workflows=[], slots={**base, "t_synthetic_content": "no", "t_interacts_people": "no"})
    assert C.classify(s)["transparency"] == []
    s2 = start(index, workflows=[], slots={**base, "t_synthetic_content": "yes", "t_deepfake_public_text": "yes"})
    names = [t["slot"] for t in C.classify(s2)["transparency"]]
    assert "t_synthetic_content" in names and "t_deepfake_public_text" in names
    rep = rp.build_report(s2, index)
    f = next(f for f in rep["findings"] if f["role"] == "transparency")
    assert "depends on the role" in f["text"] and "does not raise the TrustAI risk level" in f["text"]


# ---------------------------------------------------------------- interview sequencing
def test_scope_questions_come_right_after_the_purpose(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, "It handles purchase requests.", index, FakeClient([{"slots": {"purpose": "handles purchase requests"}}]))
    ids = [c["topic"]["id"] for c in P.candidates(s)]
    assert ids.index("org_role") < ids.index("action_mode") and ids.index("eu_link") < ids.index("annex3")


def test_approval_binding_is_only_recorded_for_workflows_that_execute_after_approval(index):
    s = start(index, workflows=[{**SMALL, "approval_bound": "no", "approval_authority": "nobody"}, LARGE])
    wfs = C.derive(s)["workflows"]
    assert wfs["po_small"].get("approval_bound") is None and wfs["po_large"]["approval_bound"] == "yes"
    rep = rp.build_report(s, index)
    row = next(w for w in rep["workflows"] if w["id"] == "po_small")
    assert row["approval_bound"] == "not applicable"


# ---------------------------------------------------------------- Article 5 screening: unticked is not 'no'; uncertainty is preserved and asked once more
def _screening_state(index, ticked, mode="confirm"):
    s = iv.new_state(); iv.add_first(s, index)
    q = iv.make_question(C.TOPICS()["screening"], index, s)
    s["turns"][0]["question"] = q
    text, sets = iv.multi_answer(q["multi"], ticked, mode)
    q["option_sets"] = {text: sets}
    iv.process(s, 0, text, index, FakeClient([{"slots": {"purpose": "an agent"}}]))
    return s


def test_unticked_screening_items_stay_unknown_until_none_of_these_is_confirmed(index):
    s = _screening_state(index, ["g_person_eval"])
    d = C.derive(s)
    assert d["slots"]["g_person_eval"] == "yes" and d["slots"]["g_biometric"] == "unknown" and d["slots"]["g_influence"] == "unknown"
    cand = [c for c in P.candidates(s) if c["topic"]["id"] == "screening" and c["kind"] == "followup"]
    assert cand, "the remaining items must be asked once more"
    q = iv.make_question(cand[0]["topic"], index, s, kind="followup", cand=cand[0])
    assert [i["slot"] for i in q["multi"]["items"]] == ["g_biometric", "g_influence"] and "None of these" in q["text"]
    # confirming "None of these" for the remaining items is the explicit negative declaration
    text, sets = iv.multi_answer(q["multi"], [], "none")
    q["option_sets"] = {text: sets}
    s["turns"].append({"id": "A2", "question": q, "answer": None, "updates": {}})
    iv.process(s, len(s["turns"]) - 1, text, index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == d["slots"]["g_influence"] == "no" and d["slots"]["g_person_eval"] == "yes"


def test_uncertainty_in_the_screening_blocks_a_final_level(index):
    s = _screening_state(index, [], mode="unsure")
    cls = C.classify(s)
    assert C.derive(s)["slots"]["g_biometric"] == "unknown" and cls["risk_label"] is None
    assert any(k.startswith("p_") for k in cls["unresolved"])


def test_none_of_these_is_the_only_way_to_record_an_all_negative_screening(index):
    s = _screening_state(index, [], mode="none")
    d = C.derive(s)
    assert {d["slots"][k] for k in ("g_biometric", "g_influence", "g_person_eval")} == {"no"}
    assert not [c for c in P.candidates(s) if c["topic"]["id"] == "screening"]


# ---------------------------------------------------------------- no repeated question about reversal
def test_reversal_is_asked_once(index):
    rev = C.TOPICS()["reversibility"]
    s_open = start(index)                                                    # nothing about reversal declared yet: the separate question may be asked
    assert C.topic_active(rev, C.derive(s_open)["slots"])
    for declared in ("yes", "no"):                                           # the checklist already answered it: the separate question is skipped
        s = answer(start(index), index, "about reversal", slots={"can_rollback": declared})
        assert not C.topic_active(rev, C.derive(s)["slots"]) and "reversibility" not in [c["topic"]["id"] for c in P.candidates(s)]


def test_aggregate_cap_wording_is_not_specific_to_orders():
    q = C.TOPICS()["aggregate_cap"]["default_q"].lower()
    assert "each order or payment" in q and "per recipient" in q


def test_citation_lines_label_legislation_and_guidance(index):
    rep = rp.build_report(start(index), index)
    kinds = {c["kind_label"] for c in rep["cited"]}
    assert "Legislation" in kinds and "Guidance" in kinds
    assert all(c["kind_label"] in rp.cite_text(c) for c in rep["cited"])


def test_question_limit_is_stated_when_it_ends_the_interview_with_missing_facts(index):
    import config
    turns = [{"id": f"A{i}", "question": {"topic": "purpose", "kind": "question", "text": "q", "phase": "Scope"}, "answer": "I don't know",
              "updates": {"slots": {"purpose": "an agent"} if i == 1 else {}, "controls": {}, "basis": [], "evidence": {}}} for i in range(1, config.MAX_QUESTIONS + 1)]
    s = {"settings": {}, "turns": turns, "phase": "summary", "summary_confirmed": True, "report": None, "error": None, "usage": []}
    assert C.answered_count(s) == config.MAX_QUESTIONS and P.candidates(s) == []      # the interview stops at the limit
    rep = rp.build_report(s, index)
    assert rep["risk_label"] is None and rep["interview_status"] == "Incomplete" and rep["assessment_status"] == "Requires further information"
    assert any("question limit" in g for g in rep["critical_gaps"])
    assert C.derive(s)["slots"].get("action_mode") in (None, "unknown") and C.derive(s)["slots"].get("max_harm") in (None, "unknown")   # nothing was invented


# ---------------------------------------------------------------- question wording: catalog sentence + grounded bridge only
def test_model_paraphrase_that_changes_the_meaning_is_replaced_by_the_catalog_question(index):
    s = start(index)
    top = C.TOPICS()["authority_expansion"]
    drifted = "You mentioned the agent operates within a group scope. Can this agent delegate tasks or escalate authority to other agents or systems?"
    q = iv.make_question(top, index, s, text=drifted, options=["Yes", "No"], nxt={"adapted": True}, cand={"fact": "authority_expansion", "impact": ""})
    assert q["text"] == top["default_q"] and "delegate" not in q["text"] and q["options"] == top["options"]


def test_bridge_is_kept_only_when_it_is_grounded_in_the_conversation(index):
    s = start(index)
    top = C.TOPICS()["max_harm"]
    good = "You mentioned the agent places purchase orders for the procurement team. What is the worst reasonable outcome?"
    bad = "You mentioned the agent controls a nuclear reactor cooling loop. What is the worst reasonable outcome?"
    q1 = iv.make_question(top, index, s, text=good, nxt={"adapted": True}, cand={"fact": "max_harm", "impact": ""})
    q2 = iv.make_question(top, index, s, text=bad, nxt={"adapted": True}, cand={"fact": "max_harm", "impact": ""})
    assert q1["text"].startswith("You mentioned the agent places purchase orders") and q1["text"].endswith(top["default_q"])
    assert q2["text"] == top["default_q"]


def test_catalog_questions_avoid_legal_jargon_for_non_specialists():
    import re
    jargon = re.compile(r"\b(Annex|Article|Art\.)\s*[IVX\d]|natural persons?|synthetic content|harmoni[sz]ation|deployer|\bprovider\b|conformity assessment", re.I)
    for t in C.interview_cfg()["topics"]:  # every topic that can be shown, including the control questions that surface only when a gap needs them
        for txt in [t["default_q"], t.get("simple_q", "")] + list(t.get("options", [])):
            assert not jargon.search(txt or ""), (t["id"], txt)
