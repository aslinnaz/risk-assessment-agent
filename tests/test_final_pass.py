"""Final stabilisation pass: one answer interface per question, checklist semantics, C01-C21 fact-to-control mapping, Art. 6(1) handling and the two-page client report.
The model is scripted (FakeClient) or absent: these tests concern the assessor's own software, not live model behaviour."""
import re
from pathlib import Path

import pytest

import classify as C
import interview as iv
import planner as P
import report as rp
from conftest import FakeClient
from test_decision import st as mkstate
from test_procurement_audit import start, answer

CIDS = [f"C{n:02d}" for n in range(1, 22)]
STATUSES = {"declared_present", "declared_absent", "partial", "unknown", "not_assessed", "not_applicable"}
ACTING = {"purpose": "Procurement agent", "action_mode": "envelope_action", "approval_exists": "partial", "limits_enforced": "policy_only", "authority_expansion": "no",
          "max_harm": "serious", "eu_link": "yes"}
OVERSIGHT = {"can_stop": "no", "monitoring_live": "no", "can_rollback": "no", "post_action_review": "yes"}


def _pdf_text(rep):
    import pymupdf
    with pymupdf.open(stream=rp.pdf_bytes(rep), filetype="pdf") as doc:
        return [re.sub(r"\s+", " ", p.get_text()) for p in doc]


# ------------------------------------------------------------------ 2: exactly one answer interface per question
def test_checklist_questions_never_carry_answer_buttons(index):
    s = iv.new_state(); iv.add_first(s, index)
    for tid in (t["id"] for t in C.interview_cfg()["topics"] if t.get("multi")):
        top = C.TOPICS()[tid]
        q = iv.make_question(top, index, s)
        assert q["multi"] and q["options"] == [] and q["option_sets"] == {} and q["control_sets"] == {}, tid
        fu = iv.make_question(top, index, s, kind="followup", cand={"text": top.get("simple_q", "x"), "options": ["Yes", "No", "I don't know"], "sets": {"Yes": {}}, "fact": tid})
        assert fu["options"] == [] and fu["option_sets"] == {}, tid  # the planner's simple options must not create a second set of buttons


def test_checklist_ui_has_one_set_of_controls(index):
    from streamlit.testing.v1 import AppTest
    import rag
    s = iv.new_state(); iv.add_first(s, index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["oversight"], index, s)
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=60)
    at.session_state["state"] = s
    at.run()
    assert not at.exception
    labels = [b.label for b in at.button]
    assert labels.count("Confirm my selection") == 1 and labels.count("None of these") == 1 and not {"Yes", "No", "I don't know"} & set(labels)
    assert len(at.checkbox) == 4 + 1 and len(at.text_area) == 1  # four safeguards + 'complete list', one optional free-text field


# ------------------------------------------------------------------ 2/3: confirmed, denied, unknown and unasked are different
MULTI = {"items": [{"slot": "can_stop", "label": "a"}, {"slot": "monitoring_live", "label": "b"}, {"slot": "post_action_review", "label": "c"}, {"slot": "can_rollback", "label": "d"}],
         "none_label": "None of these", "unsure_label": "I'm not sure about some of these"}


def test_checklist_semantics():
    _, sets = iv.multi_answer(MULTI, ["post_action_review"], "confirm")
    assert sets == {"post_action_review": "yes", "can_stop": "unknown", "monitoring_live": "unknown", "can_rollback": "unknown"}  # silence is never a 'no'
    _, sets = iv.multi_answer(MULTI, ["post_action_review"], "exhaustive")
    assert sets == {"post_action_review": "yes", "can_stop": "no", "monitoring_live": "no", "can_rollback": "no"}  # explicit 'complete list' confirmation
    _, sets = iv.multi_answer(MULTI, [], "none")
    assert set(sets.values()) == {"no"} and len(sets) == 4
    text, sets = iv.multi_answer(MULTI, ["can_stop"], "unsure")
    assert sets["can_stop"] == "yes" and sets["can_rollback"] == "unknown" and "not sure" in text  # mixed certainty is preserved
    assert "complete list" in iv.multi_answer(MULTI, ["can_stop"], "exhaustive")[0]


def _answer_oversight(index, mode):
    s = start(index)
    s = answer(s, index, "Limits are only in the instructions.", slots={"limits_enforced": "policy_only", "authority_expansion": "no"})
    top = C.TOPICS()["oversight"]
    q = iv.make_question(top, index, s)
    text, sets = iv.multi_answer(q["multi"], ["post_action_review"], mode)
    q["option_sets"] = {text: sets}
    s["turns"][-1]["question"] = {**q, "topic": "oversight"}
    iv.process(s, len(s["turns"]) - 1, text, index, FakeClient([{"slots": {}}]))
    return s


def test_questions_8_and_9_exhaustive_list_is_not_asked_again(index):
    s = _answer_oversight(index, "exhaustive")
    d = C.derive(s)
    assert d["slots"]["post_action_review"] == "yes" and d["slots"]["can_stop"] == d["slots"]["monitoring_live"] == d["slots"]["can_rollback"] == "no"
    assert not [c for c in P.candidates(s) if c["topic"]["id"] in ("oversight", "stop_ability", "rollback_ability")]
    assert C.trace_integrity(s, C.classify(s)) == []


def test_questions_8_and_9_non_exhaustive_list_asks_only_the_open_items_once(index):
    s = _answer_oversight(index, "confirm")
    d = C.derive(s)
    assert d["slots"]["post_action_review"] == "yes" and d["slots"]["can_stop"] == "unknown"
    fu = [c for c in P.candidates(s) if c["topic"]["id"] == "oversight"]
    assert fu and fu[0]["kind"] == "followup"
    q = iv.make_question(fu[0]["topic"], index, s, kind="followup", cand=fu[0])
    assert [it["slot"] for it in q["multi"]["items"]] == ["can_stop", "monitoring_live", "can_rollback"] and q["options"] == []  # the answered item is not repeated


# ------------------------------------------------------------------ 4: C01-C21 mapping
def test_every_control_has_a_rule_status_and_explanation():
    cls = C.classify(mkstate({**ACTING, **OVERSIGHT}))
    ca = cls["control_assessment"]
    assert list(ca) == CIDS
    for cid, a in ca.items():
        assert a["status"] in STATUSES and a["rule"] == f"CTL-{cid}" and a["finding"], cid
        if a["status"] == "not_assessed":
            assert a["finding"].startswith(("Not collected", "Not assessed")) and "Needed:" in a["finding"], cid
        if a["status"] in ("declared_present", "declared_absent", "partial"):
            assert a["basis"], cid  # an assessed control names the accepted facts behind it
    assert set(C.controls_cfg()["assessment_rules"]) == set(CIDS)


def test_procurement_controls_follow_the_declarations():
    ca = C.classify(mkstate({**ACTING, **OVERSIGHT, "wf_threshold": "yes"}))["control_assessment"]
    assert ca["C09"]["status"] == "partial"  # post-action review alone is not effective oversight when stop and live watching are unavailable
    assert "post-action review declared" in ca["C09"]["finding"].lower() and "immediate intervention unavailable" in ca["C09"]["finding"].lower()
    assert ca["C11"]["status"] == "partial" and "instructions or policy" in ca["C11"]["finding"]
    assert ca["C18"]["status"] == "partial" and "unavailable" in ca["C18"]["finding"]
    assert ca["C19"]["status"] == "partial" and "Rollback unavailable" in ca["C19"]["finding"]
    assert ca["C14"]["status"] == "not_assessed" and "total limit" in ca["C14"]["finding"]
    assert ca["C03"]["status"] == "not_assessed"


def test_status_semantics():
    base = {**ACTING, "approval_exists": "yes", "limits_enforced": "technical", "approval_bypass": "no", **{k: "yes" for k in OVERSIGHT}}
    assert C.classify(mkstate(base))["control_assessment"]["C09"]["status"] == "declared_present"
    assert C.classify(mkstate({**base, "can_stop": "no", "monitoring_live": "no", "post_action_review": "no", "approval_exists": "no"}))["control_assessment"]["C09"]["status"] == "declared_absent"
    unknown = C.classify(mkstate({**base, "can_stop": "unknown", "monitoring_live": "unknown", "post_action_review": "unknown", "approval_exists": "unknown"}))["control_assessment"]["C09"]
    assert unknown["status"] == "unknown"  # uncertainty is not a 'no' and not 'not assessed'
    assert C.classify(mkstate({**base, "limits_enforced": "none"}))["control_assessment"]["C11"]["status"] == "partial"  # one absent requirement among present ones
    absent = C.classify(mkstate({**ACTING, "limits_enforced": "none", "authority_expansion": "yes", "approval_bypass": "yes"}))["control_assessment"]["C11"]
    assert absent["status"] == "declared_absent"  # a missing safeguard is a finding, never 'not applicable'


def test_not_applicable_needs_a_reason_and_is_never_a_missing_safeguard():
    ca = C.classify(mkstate({**ACTING, "t_interacts_people": "no", "t_synthetic_content": "no", "t_emotion_biometric": "no"}))["control_assessment"]
    assert ca["C16"]["status"] == "not_applicable" and ca["C16"]["finding"]
    for cid, a in ca.items():
        if a["status"] == "not_applicable":
            assert len(a["finding"]) > 20, cid
    ca = C.classify(mkstate({**ACTING, "limits_enforced": "none"}))["control_assessment"]
    assert ca["C11"]["status"] != "not_applicable"


def test_controls_do_not_count_as_open_gaps_when_declarations_exist():
    cls = C.classify(mkstate({**ACTING, **OVERSIGHT}))
    assert all(g["state"] == "addressed" for cid, g in cls["control_gaps"].items() if cid in ("C09", "C11"))


# ------------------------------------------------------------------ 5: Art. 6(1)
@pytest.mark.parametrize("over,code,high", [
    ({"annex1_product": "yes", "annex1_third_party": "yes"}, "annex1_high", True),
    ({"annex1_product": "yes"}, "partly_assessed", False),
    ({"annex1_product": "yes", "annex1_third_party": "unknown"}, "partly_assessed", False),
    ({"annex1_product": "yes", "annex1_third_party": "no"}, "not_high_by_declaration", False)])
def test_product_safety_component_alone_is_not_legal_high_risk(over, code, high):
    cls = C.classify(mkstate({"action_mode": "recommend", "max_harm": "minimal", "eu_link": "yes", **over}))
    assert cls["legal"]["code"] == code and cls["legal"]["high"] == high
    if over.get("annex1_third_party") != "yes":
        assert any("Art. 6(1)" in u for u in cls["uncertainties"])  # reported as requiring review, not silently dropped


def test_annex1_option_sets_record_both_facts(index):
    q = iv.make_question(C.TOPICS()["annex1"], index, iv.new_state())
    sets = {k: v for k, v in q["option_sets"].items()}
    assert any(v == {"annex1_product": "yes", "annex1_third_party": "yes"} for v in sets.values())
    assert any(v == {"annex1_product": "no", "annex1_third_party": "no"} for v in sets.values())


# ------------------------------------------------------------------ 6/7: the report
def _state(slots):
    s = mkstate(slots)
    s["turns"][0]["question"]["text"] = "Describe the agent."
    return s


def _rep(slots, index):
    return rp.build_report(_state(slots), index, None, semantic=False)


SCENARIOS = {
    "read_only": {"action_mode": "info_only", "max_harm": "minimal", "eu_link": "yes"},
    "after_approval": {"action_mode": "approved_action", "approval_exists": "yes", "approval_bound": "yes", "limits_enforced": "technical", "authority_expansion": "no", "max_harm": "limited", "eu_link": "yes"},
    "no_approval": {**ACTING, "approval_exists": "no"},
    "technical_limits": {**ACTING, "limits_enforced": "technical"},
    "policy_limits": {**ACTING, **OVERSIGHT},
    "missing_critical": {"purpose": "p", "eu_link": "yes"},
    "annex3": {"action_mode": "recommend", "max_harm": "serious", "annex3_area": "employment", "influences_decision": "yes", "profiling": "no", "art6_3_exception": "none", "eu_link": "yes"},
    "uncertain_legal": {"action_mode": "recommend", "max_harm": "limited", "eu_link": "unknown", "annex3_area": "unknown"},
    "partial_safeguards": {**ACTING, "can_stop": "unknown", "post_action_review": "yes"},
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_have_consistent_ui_and_pdf(name, index):
    rep = _rep(SCENARIOS[name], index)
    again = _rep(SCENARIOS[name], index)
    assert rep["risk_label"] == again["risk_label"] and [c["status"] for c in rep["controls"]] == [c["status"] for c in again["controls"]]  # deterministic
    pages = _pdf_text(rep)
    assert 3 <= len(pages) <= 6, len(pages)
    txt = " ".join(pages)
    for cid in CIDS:
        assert " ".join(pages[1:]).replace("C01\u2013C21", "").count(cid) == 1, (name, cid)  # every control exactly once in the table (the heading names the range only)
    for c in rep["controls"]:
        assert c["detail"] in txt and c["title"].replace("  ", " ")[:12] in txt
    for banned in ("Citation validation", "Answer trail", "Technical appendix", "OpenAI"):
        assert banned not in txt, banned
    assert "based on user declarations only" in txt and "official category of the EU AI Act" in txt
    if rep["risk_label"]:
        assert {"LOW": "Low", "MEDIUM": "Medium", "HIGH": "High"}[rep["risk_label"]] in pages[0]
    else:
        assert "Not issued" in pages[0] and "critical information is missing" in pages[0].lower()


def test_forbidden_and_incomplete_reports(index):
    f = _state({"g_person_eval": "yes", "eval_uses_social_traits": "yes", "p_social_scoring": "supported", "action_mode": "recommend", "max_harm": "minimal"})
    f["turns"][0]["updates"]["basis"] = [{"key": "p_social_scoring", "conditions_confirmed": True, "exception_considered": True, "scope_date_considered": True}]
    rep = rp.build_report(f, index, None, semantic=False)
    pages = _pdf_text(rep)
    assert rep["risk_label"] == "FORBIDDEN" and "Unacceptable" in pages[0] and "prohibited" in pages[0].lower() and "Stop the prohibited use" in pages[0]
    assert all(c in " ".join(pages) for c in CIDS)
    inc = _rep({"purpose": "p"}, index)
    assert inc["risk_label"] is None and "Not issued" in _pdf_text(inc)[0]


def test_pdf_reports_the_real_values_not_examples(index):
    rep = _rep(SCENARIOS["annex3"], index)
    first = _pdf_text(rep)[0]
    assert rep["cls"]["L"] in first and rep["cls"]["I"] in first and rep["cls"]["posture"] in first and "purchase orders" not in first


def test_pdf_font_renders_required_characters_and_long_text(index):
    import pymupdf
    s = _state({**ACTING, **OVERSIGHT, "purpose": "Satın alma ajanı – “ğüşiöç İĞÜŞÖÇ” costs > €5,000 and < €20,000 — " + "very long description " * 20})
    rep = rp.build_report(s, index, None, semantic=False)
    rep["controls"][0]["finding"] = "Long finding " * 40
    with pymupdf.open(stream=rp.pdf_bytes(rep), filetype="pdf") as doc:
        txt = " ".join(p.get_text() for p in doc)
        fonts = {sp["font"] for p in doc for b in p.get_text("dict")["blocks"] for l in b.get("lines", []) for sp in l["spans"]}  # fonts that actually drew text
    for ch in "–“”€ğüşöçİĞÜŞÖÇ<>":
        assert ch in txt, ch
    assert all("Vera" in f or "RF" in f or "Bitstream" in f for f in fonts), fonts  # an embedded Unicode font, never a silent fallback
    assert "Long finding" in txt


def test_font_loader_uses_the_bundled_unicode_font_and_never_a_font_without_the_glyphs(monkeypatch, tmp_path):
    monkeypatch.setenv("REPORT_FONT", str(tmp_path / "missing.ttf"))
    monkeypatch.setenv("REPORT_FONT_BOLD", str(tmp_path / "missing-bold.ttf"))
    assert rp._fonts() == ("RF", "RF-B")  # a missing custom font falls through to the font bundled with ReportLab (same on macOS and Linux)
    monkeypatch.setattr(rp, "REQUIRED_GLYPHS", rp.REQUIRED_GLYPHS + "\u4e2d")  # a character the bundled font cannot draw
    with pytest.raises(RuntimeError, match="No usable report font"):
        rp._fonts()


def test_review_screen_distinguishes_confirmed_denied_unknown_and_not_asked(index):
    s = mkstate({**ACTING, **OVERSIGHT, "can_stop": "unknown", "g_biometric": "no", "g_influence": "unknown"})
    secs = dict(iv.summary_sections(s))
    by = {t: stt for items in secs.values() for stt, t in items}
    assert by["Immediate stop"] == "unknown" and by["Steering people's behaviour"] == "unknown"
    assert by["A person reviews what it did afterwards"] == "confirmed" and by["A person cannot watch it while it runs"] == "negative"
    assert by["It does not use biometric data about people"] == "negative"
    assert by["Hand-off to other agents"] == "not_collected"  # never asked: not shown as 'no'
    flat = " ".join(t for items in secs.values() for _, t in items)
    assert "\u2014" not in flat and "None" not in flat and "_" not in flat  # no placeholder dashes or raw field names


# ================================================================== corrective pass after the live smoke test
# ------------------------------------------------------------------ 1: an escalation is only reported when the posture really rose above the matrix baseline
def _why(cls, d, rows=()):
    return rp.client_summary(cls, d, list(rows), [], "legal")["why"]


@pytest.mark.parametrize("harm,base,final,escalated", [("limited", "P3", "P4", True),   # L4 x I1 = P3; policy-only limits raise it one step
                                                       ("serious", "P4", "P4", False)])  # L4 x I2 is already P4: nothing to escalate, nothing to claim
def test_escalation_is_described_only_when_the_posture_rose(harm, base, final, escalated):
    s = mkstate({**ACTING, "max_harm": harm})
    cls = C.classify(s)
    assert (cls["L"], cls["posture_base"], cls["posture"]) == ("L4", base, final)
    assert bool(cls["escalation"]) == escalated
    text = " ".join(_why(cls, C.derive(s)))
    assert ("raised from" in text) == escalated
    if escalated:
        assert f"raised from {base} to {final}" in text  # taken from the classification, not hardcoded
    assert "from P4 to P4" not in text
    trace = " ".join(t["rule"] for t in cls["trace"] if t.get("rule_id") == "POL-ESC")
    assert ("already" not in trace) == escalated and (not trace or "no escalation step applies" in trace or escalated)


def test_every_posture_statement_in_the_pdf_matches_the_classification():
    for harm in ("minimal", "limited", "serious", "critical"):
        for mode in ("recommend", "approved_action", "envelope_action"):
            s = mkstate({**ACTING, "action_mode": mode, "max_harm": harm})
            cls = C.classify(s)
            if cls["posture"] is None:
                continue
            claims = re.findall(r"raised from (P\d) to (P\d)", " ".join(_why(cls, C.derive(s))))
            assert [(cls["posture_base"], cls["posture"])] * len(claims) == claims and all(a != b for a, b in claims), (harm, mode, claims)


# ------------------------------------------------------------------ 2: explicitly stated facts survive extraction (exact live procurement scenario)
LIVE_FIRST = ("A procurement assistant that reviews purchase requests, compares supplier quotations, and prepares purchase orders. It can execute some actions without per-action "
              "human approval. Limits exist only as policy instructions. A human reviews actions afterwards, but cannot immediately stop or reverse them. The agent does not perform "
              "prohibited AI practices.")


LIVE_WF = {"id": "po_auto", "action": "execute purchase orders", "condition": "within policy limits", "execution": "executes_without_approval", "limit_enforcement": "policy_only",
           "evidence": "execute some actions without per-action human approval"}


def _extract(index, slots, evidence, answer=LIVE_FIRST):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, answer, index, FakeClient([{"slots": slots, "evidence": evidence, "workflows": [LIVE_WF]}]))
    return s


def test_review_afterwards_is_recorded_as_post_action_review_even_if_the_model_picks_the_neighbouring_slot(index):
    """The live model filled 'human_review' (decision maker sees the basis) for 'a human reviews actions afterwards'. That is a different safeguard."""
    s = _extract(index, {"human_review": "yes", "can_stop": "no", "can_rollback": "no"},
                 {"human_review": "A human reviews actions afterwards", "can_stop": "cannot immediately stop", "can_rollback": "cannot ... reverse them"})
    d = C.derive(s)
    assert d["slots"].get("post_action_review") == "yes" and "human_review" not in d["slots"]
    assert d["slots"]["can_stop"] == "no" and d["slots"]["can_rollback"] == "no"  # the explicit negatives are kept as negatives
    assert "afterwards" in d["evidence"]["post_action_review"]
    sections = dict(iv.summary_sections(s))
    safeguards = {t: st for st, t in sections["Safeguards and limitations"]}
    assert safeguards["A person reviews what it did afterwards"] == "confirmed"
    assert safeguards["A person cannot stop it immediately"] == "negative"
    assert not any("Review afterwards" in t for st, t in sections["Safeguards and limitations"] if st == "not_collected")
    ca = C.classify(s)["control_assessment"]
    assert "review afterwards" not in ca["C09"]["missing"] and ca["C09"]["status"] == "partial"  # review exists, immediate stop does not: partial, never 'present'


def test_a_basis_review_stays_human_review_and_a_denied_review_stays_denied(index):
    s = _extract(index, {"human_review": "yes"}, {"human_review": "the manager can see the reasons and object"}, answer="The manager can see the reasons and object to a recommendation.")
    d = C.derive(s)
    assert d["slots"].get("human_review") == "yes" and "post_action_review" not in d["slots"]
    s = _extract(index, {"human_review": "no"}, {"human_review": "Nobody reviews the orders afterwards"}, answer=LIVE_FIRST.replace("A human reviews actions afterwards", "Nobody reviews the orders afterwards"))
    d = C.derive(s)
    assert d["slots"].get("post_action_review") == "no"  # an explicit denial is not 'not collected'
    assert dict((t, st) for st, t in dict(iv.summary_sections(s))["Safeguards and limitations"])["Nobody reviews what it did afterwards"] == "negative"


def test_stated_facts_are_not_overwritten_by_later_unrelated_answers(index):
    s = _extract(index, {"human_review": "yes", "can_stop": "no"}, {"human_review": "A human reviews actions afterwards", "can_stop": "cannot immediately stop"})
    s = answer(s, index, "The worst outcome would be serious.", slots={"max_harm": "serious"})
    s = answer(s, index, "I don't know.", slots={})
    d = C.derive(s)
    assert d["slots"]["post_action_review"] == "yes" and d["slots"]["can_stop"] == "no" and d["slots"]["max_harm"] == "serious"


# ------------------------------------------------------------------ 3: legal uncertainty never turns into 'not applicable' or 'no legal pathway'
def _c20(over):
    s = mkstate({**ACTING, **over})
    cls = C.classify(s)
    return cls, cls["control_assessment"]["C20"]


@pytest.mark.parametrize("over,code,c20_status", [
    ({"annex3_area": "unknown"}, "undetermined", {"not_assessed", "unknown"}),                               # Annex III not answered
    ({"annex3_area": "none"}, "partly_assessed", {"not_assessed", "unknown"}),                               # Annex I route not asked
    ({"annex3_area": "none", "annex1_product": "unknown"}, "partly_assessed", {"not_assessed", "unknown"}),
    ({"annex3_area": "none", "annex1_product": "yes", "annex1_third_party": "unknown"}, "partly_assessed", {"not_assessed", "unknown"}),
    ({"annex3_area": "none", "annex1_product": "yes", "annex1_third_party": "no"}, "not_high_by_declaration", {"not_applicable"}),   # both routes explicitly excluded
    ({"annex3_area": "none", "annex1_product": "no"}, "not_high_by_declaration", {"not_applicable"}),
    ({"eu_link": "no"}, "out_of_scope_declared", {"not_applicable"}),                                                                  # explicit: no EU link declared
    ({"annex3_area": "employment"}, "annex3_high", {"not_assessed", "unknown", "partial", "declared_present", "declared_absent"})])      # applicable: anything but 'not applicable'
def test_c20_is_not_applicable_only_when_every_route_is_explicitly_excluded(over, code, c20_status):
    cls, c20 = _c20(over)
    assert cls["legal"]["code"] == code, cls["legal"]
    assert c20["status"] in c20_status, c20
    if c20["status"] == "not_applicable":
        assert c20["finding"] and "declar" in c20["finding"].lower()
    else:
        assert c20["finding"]  # an explanation is always given; 'not assessed' names what is missing
        if c20["status"] == "not_assessed":
            assert c20["missing"]


def test_partly_assessed_legal_pathway_is_not_worded_as_no_legal_pathway(index):
    rep = rp.build_report(_state({**ACTING, "annex3_area": "none"}), index, None, semantic=False)
    line = rep["summary"]["legal_line"]
    assert "partly assessed" in line and "no legal high-risk pathway" not in line.lower()
    assert "TrustAI internal label, not an official category of the EU AI Act" in " ".join(_pdf_text(rep))
    assert "no legal high-risk pathway by declaration" in rp.build_report(_state({**ACTING, "annex3_area": "none", "annex1_product": "no"}), index, None, semantic=False)["summary"]["legal_line"]


def test_live_scenario_report_has_no_contradictions_between_screen_pdf_and_controls(index):
    """Exact live procurement scenario with the facts the live run established (stop/rollback negative, review afterwards present, policy-only limits)."""
    s = _extract(index, {"human_review": "yes", "can_stop": "no", "can_rollback": "no", "max_harm": "serious", "annex3_area": "none", **GATES_NO_},
                 {"human_review": "A human reviews actions afterwards", "can_stop": "cannot immediately stop", "can_rollback": "cannot ... reverse them", "max_harm": "serious"})
    rep = rp.build_report(s, index)
    cls = rep["cls"]
    pages = _pdf_text(rep)
    assert 3 <= len(pages) <= 6
    full = " ".join(pages)
    assert "from P4 to P4" not in full and "kill switch" not in full and "monitor runs" not in full
    assert full.count("P4 is an internal release decision") <= 1
    for cid, a in cls["control_assessment"].items():
        assert STATUS_WORD[a["status"]] in full
    assert "none was technically verified" in pages[1] and not re.search(r"\b(compliant|certified)\b", full, re.I)  # nothing is presented as verified or compliant
    assert cls["control_assessment"]["C20"]["status"] != "not_applicable"
    review = {t: st for st, t in dict(iv.summary_sections(s))["Safeguards and limitations"]}
    assert review["A person reviews what it did afterwards"] == "confirmed" and review["A wrong action cannot be reversed"] == "negative"


GATES_NO_ = {"g_biometric": "no", "g_influence": "no", "g_person_eval": "no"}
STATUS_WORD = {"declared_present": "Declared present", "declared_absent": "Declared absent", "partial": "Partially assessed", "unknown": "Unknown", "not_assessed": "Not assessed", "not_applicable": "Not applicable"}


def test_c20_not_assessed_names_the_open_legal_facts():
    _, c = _c20({"annex3_area": "none"})
    assert c["status"] == "not_assessed" and c["missing"] == ["regulated-product route (Annex I)"] and "Annex I" in c["finding"]
    _, c = _c20({"annex3_area": "unknown"})
    assert c["missing"] == ["sensitive-use area (Annex III)", "regulated-product route (Annex I)"]


def test_placeholder_conditions_are_not_shown_as_something_the_user_said(index):
    wf = {**LIVE_WF, "condition": "below unspecified threshold", "threshold": "unspecified"}
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, LIVE_FIRST, index, FakeClient([{"slots": {"purpose": "x"}, "workflows": [wf]}]))
    w = next(iter(C.derive(s)["workflows"].values()))
    assert not w.get("condition") and not w.get("threshold")
    assert "unspecified" not in " ".join(t for _, items in iv.summary_sections(s) for _, t in items)
