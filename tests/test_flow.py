import re
import pytest
import classify as C, interview as iv, planner as P, report as rp, rag, llm
from conftest import FakeClient

GATES_NO = {"g_biometric": "no", "g_influence": "no", "g_person_eval": "no"}
EXAMPLE = ("The agent reviews customers' refund requests at an e-commerce company and gives a recommendation to the support employee. The employee makes the final decision; "
           "the agent does not make payments or refunds and does not talk to customers directly.")
EX_SLOTS = {"purpose": "Review customer refund requests and give a recommendation to the support employee", "affected_decisions": "refund decision",
            "action_mode": "recommend", "t_interacts_people": "no"}
TRUTH = {"eu_link": "unknown", "org_role": "deployer", "use_status": "live", **GATES_NO, "annex3_area": "none", "annex1_product": "no", "max_harm": "limited",
         "personal_data": "ordinary", "vulnerable_groups": "no", "t_synthetic_content": "no", "reversibility": "reversible", "reach": "single",
         "human_review": "yes", "art6_3_exception": "none"}


def start(index):
    s = iv.new_state(); iv.add_first(s, index); return s


def run_truth(s, index, truth, limit=40):
    """Answers each question with 'truth' for the slots of the asked topic (a scripted loop; it only tests our own flow)."""
    topics = []
    while s["phase"] == "interview" and C.answered_count(s) < limit:
        q = s["turns"][-1]["question"]
        topics.append((q["kind"], q["topic"], q.get("fact")))
        top = C.TOPICS()[q["topic"]]
        upd = {sl: truth[sl] for sl in C.required_slots(top) if sl in truth}
        if q["kind"] == "followup" and q.get("fact") in truth:
            upd = {q["fact"]: truth[q["fact"]]}
        iv.process(s, len(s["turns"]) - 1, "answer", index, FakeClient([{"slots": upd}]))
    return topics


def _turn(i, topic, slots, ans="x", kind="question", ctl=None, evidence=None):
    return {"id": i, "question": {"topic": topic, "kind": kind}, "answer": ans, "updates": {"slots": slots, "controls": ctl or {}, "basis": [], "evidence": evidence or {}}}


def state_of(slots, **kw):
    return {"settings": {}, "turns": [_turn("A1", "purpose", slots, **kw)]}


BASE = {"purpose": "p", "annex3_area": "none", "max_harm": "limited", **GATES_NO}


def test_sources_metadata(index):
    man = rag.load_manifest()["sources"]
    assert {m["source_id"] for m in man if not m["api_transfer_allowed"]} == {"ISO42001"}
    assert all(c["source_id"] != "ISO42001" for c in index.chunks)
    assert all(c["version"] for c in index.chunks)
    assert all(c["page"] for c in index.chunks)  # every indexed passage is a PDF passage with a page
    assert {"AIA", "TAI-AG", "TAI-GOV", "NIST-PB"} <= {c["source_id"] for c in index.chunks}
    assert "TAI-MX" not in {c["source_id"] for c in index.chunks} and "TAI-MX" not in {m["source_id"] for m in man}  # the internal policy is not a source
    assert index.lookup("AIA", "Art. 6(3)") and index.lookup("AIA", "Art. 113")


@pytest.mark.needs_sources
def test_catalog_citations():
    import tools.verify_catalog as v
    assert v.main() == 0


# ---------------------------------------------------------------- the application is English
def test_everything_user_facing_is_english():
    import json, re
    from pathlib import Path
    tr = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
    for f in Path("catalogs").glob("*.json"):
        txt = json.dumps(json.loads(f.read_text(encoding="utf-8")), ensure_ascii=False)
        assert not tr.findall(txt), f
    for f in ("app.py", "interview.py", "report.py", "classify.py", "planner.py", "llm.py", "rag.py"):
        src = Path(f).read_text(encoding="utf-8")
        assert not tr.findall(src), f


# ---------------------------------------------------------------- causes of repetition
def test_unknown_never_overwrites_definite(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", "g_biometric": "no"}}]))
    iv.process(s, 1, "y", index, FakeClient([{"slots": {"annex3_area": "none", "g_biometric": "unknown"}}]))
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "no" and d["slots"]["p_rbi_law_enforcement"] == "none"
    assert not [c for c in P.candidates(s) if c["topic"]["id"] in ("g_biometric", "prohibited_detail") or c.get("gate") == "g_biometric"]


def test_conflict_asks_only_the_conflicting_fact_once(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", "g_biometric": "no"}}]))
    iv.process(s, 1, "y", index, FakeClient([{"slots": {"annex3_area": "biometrics"}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "conflict" and q["fact"] == "annex3_area" and not iv.is_vague(q["text"])
    assert "biometric" in q["text"].lower() and "sensitive use area" in q["text"] and len(q["options"]) == 3
    iv.process(s, 2, "I don't know", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["annex3_area"] == "unknown" and not d["conflicts"]  # could not be resolved → unknown, not 'no'
    assert all(t["question"]["kind"] != "conflict" for t in s["turns"][3:])
    assert C.answered_count(s) == 3


def test_equivalent_facts_close_each_other(index):
    s = {"turns": [{"id": "A1", "question": {"topic": "annex3", "kind": "question"}, "answer": "x",
                    "updates": {"slots": {"annex3_area": "biometrics"}, "controls": {}, "basis": []}}]}
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "yes" and "g_biometric" in d["derived"]
    assert "g_biometric" not in [t["id"] for t in C.open_topics(s)]


def test_dont_know_gets_one_simple_followup_then_moves_on(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "g_influence": "no", "g_person_eval": "no", "annex3_area": "none"}}]))
    s["turns"][-1]["question"] = iv.make_question(C.TOPICS()["g_biometric"], index, s)
    iv.process(s, len(s["turns"]) - 1, "I don't know", index, FakeClient([{"slots": {"g_biometric": "unknown"}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "g_biometric" and "For example" in q["text"]
    iv.process(s, len(s["turns"]) - 1, "still don't know", index, FakeClient([{"slots": {"g_biometric": "unknown"}}]))
    assert all(t_["question"].get("fact") != "g_biometric" for t_ in s["turns"][len(s["turns"]) - 1:])
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "unknown" and d["slots"].get("p_facial_scraping") is None
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    cls = C.classify(s)
    assert cls["risk_label"] is None and any("prohibit" in x.lower() for x in cls["uncertainties"])


def test_no_unrequested_defaults(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    d = C.derive(s)
    for k in ("g_biometric", "g_influence", "g_person_eval", "annex3_area", "max_harm", "personal_data", "approval_bound", "human_review", "eu_link"):
        assert d["slots"].get(k) is None, k


# ---------------------------------------------------------------- refund example
def test_refund_example_flow(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    d = C.derive(s)
    assert d["slots"]["action_mode"] == "recommend" and d["slots"]["t_interacts_people"] == "no" and d["slots_src"]["action_mode"] == "A1"
    ids = [c["topic"]["id"] for c in P.candidates(s)]
    for known in ("purpose", "action_mode", "t_interacts", "approval_bound", "limits_enforced", "authority_expansion"):
        assert known not in ids, known
    assert {"annex3", "max_harm", "screening"} <= set(ids) and "g_biometric" not in ids and "eu_link" in ids and "org_role" in ids  # scope questions come early
    topics = run_truth(s, index, TRUTH)
    n = C.answered_count(s)
    assert s["phase"] == "summary" and 8 <= n <= 15, (n, topics)
    asked = [t for _, t, _ in topics]
    assert len(asked) == len(set(asked)), asked
    cls = C.classify(s)
    assert cls["L"] == "L1" and cls["I"] == "I1" and cls["posture"] == "P2" and cls["risk_label"] == "MEDIUM"  # matrix: I1 x L1 = P2 (Supervised) → MEDIUM
    rep = rp.build_report(s, index)
    st = {c["id"]: c["status"] for c in rep["controls"]}
    assert st["C16"] == "not_applicable" and st["C20"] == "not_applicable"
    assert st["C09"] == "declared_present" and cls["assessment_status"] == "complete"
    assert st["C08"] == "not_assessed" and st["C11"] == "not_assessed" and st["C12"] == "not_assessed"
    assert all(c["note"] for c in rep["controls"] if c["status"] == "not_applicable")
    assert rep["verification"]["all_ok"] and rp.pdf_bytes(rep)[:4] == b"%PDF"


def test_ends_when_enough_and_context_is_limited(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {**EX_SLOTS, **TRUTH}}]))
    assert s["phase"] == "summary" or all(c["topic"].get("context") for c in P.candidates(s))
    ctx = 0
    while s["phase"] == "interview":
        ctx += 1
        iv.process(s, len(s["turns"]) - 1, "x", index, FakeClient([{"slots": {}}]))
    assert ctx <= 3 and C.answered_count(s) <= 4


def test_hard_limit_20_and_incomplete(index):
    s = start(index)
    while s["phase"] == "interview" and C.answered_count(s) < 40:
        iv.process(s, len(s["turns"]) - 1, "I don't know", index, FakeClient([{"slots": {}}]))
    assert C.answered_count(s) <= 20 and s["phase"] == "summary"
    facts = [t["question"].get("fact") for t in s["turns"] if t["question"].get("kind") == "followup"]
    assert len(facts) == len(set(facts))
    cls = C.classify(s)
    assert cls["risk_label"] is None and cls["assessment_status"] == "incomplete" and cls["unresolved"]
    rep = rp.build_report(s, index)
    assert any("Critical information missing" in g or "could not be completed" in g for g in rep["critical_gaps"])


def test_progress_summary_from_state_only(index):
    s = start(index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    pg = P.progress(s)
    assert "intended purpose" in pg["known"] and "agent authority" in pg["known"]
    assert "prohibited-use screening" in pg["need"]


# ---------------------------------------------------------------- question clarity / options
def test_mixed_topic_model_question_is_replaced(index):
    s = start(index)
    mixed = "Does the agent use biometric data, and does it work in a sensitive area such as recruitment?"
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes"}, "topic": "g_biometric", "question": mixed}]))
    q = s["turns"][-1]["question"]
    if q["topic"] == "g_biometric":
        assert q["text"] == C.TOPICS()["g_biometric"]["default_q"] and "recruitment" not in q["text"] and q["options"] == ["Yes", "No", "I don't know"]
    assert not iv.mixes_topics(C.TOPICS()["g_biometric"]["default_q"], C.TOPICS()["g_biometric"])
    assert "biometric" not in C.TOPICS()["personal_data"]["default_q"].lower()
    assert "biometric" not in C.TOPICS()["annex3"]["default_q"].lower() and "Biometric" not in " ".join(C.TOPICS()["annex3"]["options"])


def test_oversight_retrieves_playbook(index):
    for tid in ("review_basis", "approval_bound", "stop_ability", "limits_enforced", "authority_expansion"):
        ps = rag.retrieve_topic(index, C.TOPICS()[tid], max_total=3)
        assert {p["source_id"] for p in ps} & {"TAI-AG", "NIST-PB"}, tid
    assert rag.retrieve_topic(index, C.TOPICS()["annex3"], max_total=2)[0]["source_id"] == "AIA"  # the legal text comes first; the playbook may follow
    assert all(p["source_id"] != "ISO42001" for t in C.TOPICS().values() for p in rag.retrieve_topic(index, t))


def test_legal_passage_keeps_conditions(index):
    ps = index.lookup("AIA", "Art. 6(3)")[0]["text"]
    for needle in ("narrow procedural task", "profiling of natural persons"):
        assert needle in ps
    ids = [p["chunk_id"] for p in rag.retrieve_topic(index, C.TOPICS()["art6_role"], max_total=6)]
    assert "AIA:Art-6-3" in ids and "AIA:Art-6-4" in ids


def test_humanize_and_why_storage(index):
    assert "importer_distributor" not in iv.humanize("importer_distributor role approved_action").lower()
    s = start(index)
    fc = FakeClient([{"slots": {"purpose": "p"}, "topic": "eu_link", "question": "Is your role approved_action or envelope_action?"}])
    iv.process(s, 0, "x", index, fc)
    q = s["turns"][-1]["question"]
    assert "approved_action" not in q["text"] and isinstance(q["why"], dict) and q["why"]["missing"] and "passage_ids" in q
    assert fc.calls == 2 and fc.extract_calls == 1  # one extraction call + one question call


def test_index_status_reflects_disk(index):
    st = index.status()
    assert st["chunks"] > 1000 and st["embeddings"] == index.refresh_embeddings()


def test_out_of_scope_is_general(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "no"}, "topic": "g_biometric"}]))
    q = s["turns"][-1]["question"]
    assert q["framing"] == "general" or q["topic"] not in iv.LEGAL_ONLY
    assert C.classify(s)["legal"]["code"] == "out_of_scope_declared"


def test_edit_invalidates_later_turns(index):
    s = start(index)
    fc = FakeClient([{"slots": {"purpose": "p", "eu_link": "yes"}}, {"slots": {"g_influence": "no"}}])
    iv.process(s, 0, "a", index, fc)
    iv.process(s, 1, "b", index, fc)
    assert len(s["turns"]) == 3 and C.classify(s)["assessment_status"] == "incomplete"
    iv.process(s, 0, "corrected", index, FakeClient([{"slots": {"purpose": "q"}}]))
    assert len(s["turns"]) == 2 and C.derive(s)["slots"]["purpose"] == "q" and "g_influence" not in C.derive(s)["slots"]


def test_first_question_has_source_fields(index):
    q = iv.first_question(index)
    assert q["passage_ids"] and not q["no_passage"] and q["why"]["source_note"]
    c = index.by_id[q["passage_ids"][0]]
    assert c["ref"] == "Art. 3(12)" and c["page"] and c["version"] and c["url"]


def test_no_unsupported_source_note(index):
    q = iv.make_question(C.TOPICS()["ctl_owner"], index, {"turns": []}, nxt={"missing": "m", "clarifies": "c", "source_note": "The source says this", "passage_ids": ["none:none"]}, text="Is there an owner?")
    assert q["no_passage"] and q["why"]["source_note"] == "" or q["generic_why"]


def test_technical_error_keeps_answer_without_extra_question(index):
    s = start(index)

    class Boom(FakeClient):
        def __init__(self):
            super().__init__([])

            class _C:
                def create(self, **kw): raise RuntimeError("network error")

            class _Chat: completions = _C()
            self.chat = _Chat()
    try:
        iv.process(s, 0, EXAMPLE, index, Boom())
    except RuntimeError:
        pass
    assert s["turns"][0]["answer"] == EXAMPLE and len(s["turns"]) == 1 and C.answered_count(s) == 1
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    assert C.answered_count(s) == 1 and len(s["turns"]) == 2


def test_transparency_and_approval_do_not_downgrade():
    base = {"purpose": "p", "eu_link": "yes", "annex3_area": "employment", "profiling": "no", "art6_3_exception": "none", "influences_decision": "yes",
            "action_mode": "recommend", "max_harm": "minimal", "t_interacts_people": "yes", "human_review": "yes", **GATES_NO}
    s = {"turns": [{"id": "A1", "question": {"topic": "annex3", "kind": "question"}, "answer": "x", "updates": {"slots": dict(base), "controls": {}, "basis": []}}]}
    c = C.classify(s)
    assert c["risk_label"] == "HIGH" and c["legal"]["high"] and c["posture"] == "P1"  # the legal pathway raises the level; the posture stays P1
    assert any("legal high-risk pathway" in b["rule"] for b in c["label_basis"])
    s["turns"][0]["updates"]["slots"]["annex3_area"] = "none"
    c = C.classify(s)
    assert c["risk_label"] == "LOW" and c["transparency"]


def test_low_with_unaddressed_relevant_control_is_not_presented_as_complete(index):
    s = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
         "updates": {"slots": {"purpose": "p", "action_mode": "recommend", "annex3_area": "none", "max_harm": "minimal", **GATES_NO}, "controls": {}, "basis": []}}]}
    c = C.classify(s)
    assert c["risk_label"] == "LOW" and c["assessment_status"] == "partial" and "C09" in c["control_gaps"]
    assert any("C09" in u for u in c["uncertainties"])
    assert "review_basis" in [c["topic"]["id"] for c in P.candidates(s)]
    s["turns"][0]["updates"]["slots"]["human_review"] = "yes"
    assert C.classify(s)["assessment_status"] == "complete"


def test_scope_settings_default_and_eu_scope_question_only_when_undetermined(index):
    s = iv.new_state()
    assert s["settings"]["scope_country"] == "Not specified" and s["settings"]["eu_link"] == "unknown"
    iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    c = C.classify(s)
    assert c["applicability"] == "unknown" and c["legal"]["code"] != "out_of_scope_declared"
    assert "eu_link" in [x["topic"]["id"] for x in P.candidates(s)]  # undetermined scope is asked early, in plain words
    s2 = iv.new_state({"eu_link": "no"})
    assert C.classify(s2)["applicability"] == "no" and C.classify(s2)["legal"]["code"] == "out_of_scope_declared"
    s3 = iv.new_state({"eu_link": "yes"})
    assert C.derive(s3)["slots"]["eu_link"] == "yes" and C.derive(s3)["slots_src"]["eu_link"] == "SETTINGS"
    iv.add_first(s3, index)
    assert "eu_link" not in [x["topic"]["id"] for x in P.candidates(s3)]  # a declared setting is never asked again


def test_result_summary_is_sourced_and_declaration_based(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    run_truth(s, index, TRUTH)
    rep = rp.build_report(s, index)
    assert rep["interview_status"] == "Completed" and rep["verification_status"] == "Not technically verified" and rep["risk_label"] == "MEDIUM" and rep["posture"] == "P2"
    assert len(rep["key_findings"]) <= 3 and len(rep["improvements"]) <= 3
    assert "declarations" in rep["note"] and "preliminary" in rep["note"].lower() and "not a technical verification" in rep["note"]
    assert rep["context"]["scope_country"] == "Not specified" and rep["context"]["applicability"] == "Not determined"
    assert any("legal applicability was not determined" in g for g in rep["critical_gaps"])
    by = {c["no"] for c in rep["cited"]}
    for k in rep["key_findings"]:
        assert set(k["cite_nos"]) <= by and (k["cite_nos"] or k["method"] or k["unsupported"])
    for c in rep["cited"]:  # source name, version, article/section and location are always available
        assert c["source"] and c["version"] and c["ref"] and c["locator"]
    groups = {c["group"] for c in rep["controls"]}
    assert groups <= set(C.STATUS_LABEL.values())
    pdf = rp.pdf_bytes(rep)
    pages = [re.sub(r"\s+", "", x) for x in pdf_pages(pdf)]  # whitespace-free: PDF text extractors differ in spacing, not in content
    first = pages[0]
    for needle in ("Preliminary Assessment Report", "A. Assessed Agent", "B. Risk Assessment Result", "C. Why This Result?", "D. Recommended Next Steps", "E. Assessment Limitations",
                   "based on user declarations only", "not an official category of the EU AI Act"):
        assert re.sub(r"\s+", "", needle) in first, needle
    assert len(pages) <= 3 and all(f"C{n:02d}" in "".join(pages[1:]) for n in range(1, 22))  # the control table (all 21 controls) follows the result page
    for banned in ("Sourcepassages", "Citationvalidation", "Answertrail", "Methodanddetails", "[K1]"):
        assert not any(banned in x for x in pages), banned


def pdf_pages(pdf: bytes) -> list[str]:
    """Text of every PDF page with whichever extractor is installed (PyMuPDF preferred)."""
    try:
        import pymupdf
        with pymupdf.open(stream=pdf, filetype="pdf") as doc:
            return [pg.get_text() for pg in doc]
    except ImportError:
        import io, pypdf
        return [pg.extract_text() or "" for pg in pypdf.PdfReader(io.BytesIO(pdf)).pages]


def groups():
    return C.interview_cfg()["prohibition_groups"]


def test_no_vague_followups_in_catalog():
    for tp in C.interview_cfg()["topics"]:
        assert not iv.is_vague(tp["default_q"]) or tp["id"] == "prohibited_detail", tp["id"]
    for g in groups().values():
        assert not iv.is_vague(g["detail_q"]) and g["detail_options"] and g["requires"]
    assert iv.is_vague("Could you please provide more details regarding your previous answer?") and iv.is_vague("Can you elaborate?")


def test_options_are_never_truncated():
    src = open("app.py").read()
    assert "text-overflow: clip" in src and "white-space: normal" in src and "st.columns(min(len(opts)" not in src
    for tp in C.interview_cfg()["topics"]:
        for o in tp.get("options", []):
            assert "…" not in o and "..." not in o and iv.humanize(o) == o
    for g in groups().values():
        assert all("…" not in o for o in g["detail_options"])


def test_gate_no_skips_detail_branches(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "eu_link": "yes", **GATES_NO}}]))
    d = C.derive(s)
    assert d["slots"]["p_facial_scraping"] == "none" and d["slots_src"]["p_facial_scraping"] == "A1" and d["derived"]["p_rbi_law_enforcement"]
    assert d["slots"]["p_manipulation"] == "none"
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    s["turns"][0]["updates"]["slots"]["g_biometric"] = "yes"
    fu = [c for c in P.candidates(s) if c["kind"] == "followup"]
    assert len(fu) == 1 and fu[0]["fact"] == "detail:biometric" and "purpose" in fu[0]["text"]


# ---------------------------------------------------------------- source scenario ≠ user fact
HIRING = "A recruitment agent; it gives candidates a suitability score and supports the hiring decision."
CREDIT = "An agent that gives credit applications a risk score; an analyst makes the decision."


def first_turn(index, text, slots, **kw):
    s = start(index)
    iv.process(s, 0, text, index, FakeClient([{"slots": slots, **kw}]))
    return s


def test_scoring_does_not_open_social_scoring_or_crime(index):
    for text, area, ev in ((HIRING, "employment", "gives candidates a suitability score"), (CREDIT, "essential_services", "gives credit applications a risk score")):
        s = first_turn(index, text, {"purpose": text, "g_person_eval": "yes", "annex3_area": area}, evidence={"g_person_eval": ev, "annex3_area": text.split(";")[0]})
        d = C.derive(s)
        assert d["slots"]["g_person_eval"] == "yes" and d["slots"].get("annex3_area") == area
        assert d["slots"].get("p_social_scoring") is None and d["slots"].get("p_crime_profiling") is None
        assert not C.detail_groups(d["slots"]) and not [c for c in P.candidates(s) if c["kind"] == "followup"]
        assert s["turns"][-1]["question"]["topic"] != "annex3"
        cand = {c["topic"]["id"]: c for c in P.candidates(s)}
        assert "eval_criteria" in cand and cand["eval_criteria"]["rank"][0] == 2
    q = C.TOPICS()["eval_criteria"]["default_q"]
    assert "criteria" in q and "social" not in q.lower() and "crime" not in q.lower()


def test_influencing_a_decision_is_not_behaviour_manipulation(index):
    s0 = first_turn(index, HIRING, {"purpose": "p", "g_influence": "no"}, evidence={"g_influence": "supports the hiring decision"})
    assert C.derive(s0)["slots"].get("g_influence") in (None, "unknown")  # a decision-support statement contains no negation: not evidence for 'no'
    s = first_turn(index, "It supports the decision; it does not steer users towards a behaviour, no.", {"purpose": "p", "g_influence": "no"}, evidence={"g_influence": "does not steer users"})
    assert C.derive(s)["slots"]["g_influence"] == "no" and C.derive(s)["slots"]["p_manipulation"] == "none"
    assert "merely giving a recommendation to a decision maker does not count" in C.TOPICS()["g_influence"]["default_q"].lower()
    s2 = first_turn(index, HIRING, {"purpose": "p", "g_influence": "yes", "p_manipulation": "possible"}, evidence={"g_influence": "supports the hiring decision"})
    assert C.derive(s2)["slots"].get("p_manipulation") is None
    q = s2["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "detail:influence" and q["options"][0].startswith("With a clear")


def test_evidence_must_come_from_user_text(index):
    s = first_turn(index, HIRING, {"purpose": "p", "g_person_eval": "yes", "eval_uses_social_traits": "yes"},
                   evidence={"g_person_eval": "gives candidates a suitability score", "eval_uses_social_traits": "scores them on social behaviour"})
    d = C.derive(s)
    assert d["slots"]["g_person_eval"] == "yes"
    assert d["slots"].get("eval_uses_social_traits") in (None, "unknown")
    assert s["turns"][0]["updates"]["rejected"] and not C.detail_groups(d["slots"])
    assert iv.evidence_ok("gives candidates a suitability score", HIRING) and not iv.evidence_ok("social media data", HIRING)


def test_extracted_facts_keep_the_user_quote(index):
    """Trace: answer -> extracted fact -> (quote). The quote is stored with the fact and reported."""
    s = first_turn(index, HIRING, {"purpose": "p", "g_person_eval": "yes"}, evidence={"g_person_eval": "gives candidates a suitability score"})
    d = C.derive(s)
    assert d["evidence"]["g_person_eval"] == "gives candidates a suitability score" and d["slots_src"]["g_person_eval"] == "A1"


def _ask_criteria(index, text="A recruitment agent; it gives candidates a score."):
    s = first_turn(index, text, {"purpose": "p", "g_person_eval": "yes"}, evidence={"g_person_eval": "gives candidates a score"})
    s["turns"][-1]["question"] = iv.make_question(C.TOPICS()["eval_criteria"], index, s)
    return s


def test_criteria_answers_drive_preconditions(index):
    s = _ask_criteria(index)
    iv.process(s, len(s["turns"]) - 1, "Only experience and test results; no social data and no crime risk", index,
               FakeClient([{"slots": {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"}}]))
    d = C.derive(s)
    assert d["slots"]["p_social_scoring"] == "none" and d["slots"]["p_crime_profiling"] == "none" and d["derived"]["p_social_scoring"]
    assert not [c for c in P.candidates(s) if c["kind"] == "followup"]
    s2 = _ask_criteria(index)
    iv.process(s2, len(s2["turns"]) - 1, "Social media behaviour is also used, no crime prediction", index,
               FakeClient([{"slots": {"eval_uses_social_traits": "yes", "eval_predicts_crime": "no"}}]))
    q = s2["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "detail:social" and "context" in q["text"] and q["passage_ids"]
    assert all(index.by_id[p]["ref"].startswith("Art. 5(1)(c)") for p in q["passage_ids"])
    assert C.derive(s2)["slots"]["p_crime_profiling"] == "none"


def test_dont_know_in_criteria_does_not_confirm_or_clear(index):
    s = _ask_criteria(index)
    iv.process(s, len(s["turns"]) - 1, "I don't know", index, FakeClient([{"slots": {}}]))
    q = s["turns"][-1]["question"]
    assert q["kind"] == "followup" and q["fact"] == "eval_criteria"
    iv.process(s, len(s["turns"]) - 1, "still don't know", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "unknown" and d["slots"].get("p_social_scoring") is None
    assert not [c for c in P.candidates(s) if c["fact"] == "eval_criteria"]
    cls = C.classify(s)
    assert cls["risk_label"] is None and any("prohibit" in u.lower() for u in cls["uncertainties"])


def _to_followup(index, group, slots):
    s = first_turn(index, "x", {"purpose": "p", **slots})
    assert s["turns"][-1]["question"]["fact"] == "detail:" + group, s["turns"][-1]["question"]
    return s


B_OK = [{"key": "p_social_scoring", "conditions_confirmed": True, "exception_considered": True, "scope_date_considered": True}]
PRE = {"g_person_eval": "yes", "eval_uses_social_traits": "yes"}


def test_forbidden_requires_basis_and_preconditions(index):
    s = _to_followup(index, "social", PRE)
    iv.process(s, len(s["turns"]) - 1, "Yes, it is used in a different context", index, FakeClient([{"slots": {"p_social_scoring": "supported"}}]))  # no basis
    assert s["phase"] != "forbidden" and C.derive(s)["slots"]["p_social_scoring"] == "possible"
    s2 = _to_followup(index, "social", PRE)
    iv.process(s2, len(s2["turns"]) - 1, "Yes, it is used in a different context", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": B_OK}]))
    assert s2["phase"] == "forbidden" and C.classify(s2)["risk_label"] == "FORBIDDEN"
    rep = rp.build_report(s2, index)
    assert rep["verification"]["all_ok"] and rp.pdf_bytes(rep)[:4] == b"%PDF"
    s3 = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
          "updates": {"slots": {"purpose": "p", "g_person_eval": "no", "p_social_scoring": "supported"}, "controls": {}, "basis": B_OK}}]}
    assert C.classify(s3)["risk_label"] != "FORBIDDEN"


def test_forbidden_basis_needs_a_user_quote_for_every_part(index):
    """The model cannot assert 'conditions/exception/scope considered' without verbatim quotes from the user's answer."""
    s = _to_followup(index, "social", PRE)
    b = [{**B_OK[0], "exception_evidence": "the exception was clearly considered by the provider"}]
    iv.process(s, len(s["turns"]) - 1, "Yes, it is used in a different context", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": b}]))
    assert s["phase"] != "forbidden" and C.derive(s)["slots"]["p_social_scoring"] == "possible"
    assert any("exception_considered" in k for t in s["turns"] for k, _ in t["updates"].get("rejected", []))


def test_forbidden_with_undetermined_applicability_but_not_when_declared_out(index):
    s = _to_followup(index, "social", PRE)
    iv.process(s, len(s["turns"]) - 1, "Yes, in a different context", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": B_OK}]))
    assert s["phase"] == "forbidden" and C.classify(s)["applicability"] == "unknown"
    s2 = iv.new_state({"eu_link": "no"}); iv.add_first(s2, index)
    iv.process(s2, 0, "x", index, FakeClient([{"slots": {"purpose": "p", **PRE}}]))
    iv.process(s2, len(s2["turns"]) - 1, "yes", index, FakeClient([{"slots": {"p_social_scoring": "supported"}, "basis": B_OK}]))
    assert s2["phase"] != "forbidden"


# ---------------------------------------------------------------- interview flow
def test_first_question_and_not_vague():
    assert C.TOPICS()["purpose"]["default_q"] == "Could you briefly describe what the agent we will assess does?"
    assert not iv.is_vague(C.TOPICS()["purpose"]["default_q"]) and not iv.is_vague("Could you describe which criteria the score is calculated from?")
    assert iv.is_vague("Could you describe it?") and iv.is_vague("Could you please provide more details?")


def test_context_mention_is_not_two_questions():
    top = C.TOPICS()["g_biometric"]
    assert not iv.mixes_topics("You said candidates are scored. Does the agent use face or voice data to recognise people?", top)
    assert iv.mixes_topics("Does the agent use biometric data? Does it work in a sensitive area such as recruitment?", top)
    assert iv.mixes_topics("Does the agent use biometric data, and does it work in a sensitive area such as recruitment?", top)


def test_options_keep_all_and_record_value(index):
    q = iv.make_question(C.TOPICS()["annex3"], index, {"turns": []})
    assert "I don't know" in q["options"] and "Another area" in q["options"] and len(q["options"]) == 6
    assert q["option_sets"]["Recruitment / employee management"] == {"annex3_area": "employment"}
    s = start(index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["g_biometric"], index, s)
    iv.process(s, 0, "No", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == "no" and s["turns"][0]["updates"]["choice"] == {"label": "No", "set": {"g_biometric": "no"}, "controls": {}}
    assert d["slots"]["p_rbi_law_enforcement"] == "none" and d["evidence"]["g_biometric"] == "No"  # the selected option is its own evidence


def test_open_questions_have_no_forced_options(index):
    assert iv.first_question(index)["options"] == []
    q = iv.make_question(C.TOPICS()["eval_criteria"], index, {"turns": []})
    assert q["options"] == [] and "criteria" in q["text"]


def test_approval_exists_before_approval_bound():
    bound = C.TOPICS()["approval_bound"]
    assert not C.topic_active(bound, {"action_mode": "unknown"})
    assert not C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "no"})
    assert not C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "unknown"})
    assert C.topic_active(bound, {"action_mode": "unknown", "approval_exists": "yes"})
    s = {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
         "updates": {"slots": {"purpose": "p", "action_mode": "approved_action"}, "controls": {}, "basis": []}}]}
    d = C.derive(s)
    assert d["slots"]["approval_exists"] == "yes" and C.topic_active(bound, d["slots"])
    assert "approval_exists" not in [c["topic"]["id"] for c in P.candidates(s)]
    assert "approval_bound" in [c["topic"]["id"] for c in P.candidates(s)]


def test_stop_and_rollback_are_separate_facts():
    assert "human_intervention" not in C.interview_cfg()["slots"]
    a, b = C.TOPICS()["stop_ability"], C.TOPICS()["rollback_ability"]
    assert a["slots"] == ["can_stop"] and b["slots"] == ["can_rollback"] and "undone" not in a["default_q"] and "stop" not in b["default_q"]


def test_valid_model_choice_is_kept_among_the_allowed_candidates_and_the_catalog_wording_is_used(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p"}, "topic": "eu_link", "question": "Which region does the agent work in?"}]))
    q = s["turns"][-1]["question"]
    assert q["topic"] == "eu_link" and q["text"] == C.TOPICS()["eu_link"]["default_q"]  # the options belong to the catalog sentence; a paraphrase could change their meaning


def test_scope_questions_cannot_be_skipped_by_the_model(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p"}, "topic": "annex3", "question": "Which area does the agent work in?"}]))
    assert s["turns"][-1]["question"]["topic"] in ("org_role", "eu_link", "use_status")  # the model's pick of a later topic is overridden while scope questions are open


def test_invalid_model_choice_uses_neutral_catalog_question_of_a_seen_topic(index):
    s = start(index)
    iv.process(s, 0, "x", index, FakeClient([{"slots": {"purpose": "p", "annex3_area": "none"}, "topic": "annex3", "question": "Shall we ask about this area perhaps?"}]))
    q = s["turns"][-1]["question"]
    assert q["topic"] != "annex3"
    assert not iv.is_vague(q["text"]) and q["text"] == iv.humanize(C.TOPICS()[q["topic"]]["default_q"]) or q["kind"] != "question"


def test_ui_has_no_usage_counters_and_no_manual_embedding_step():
    import re
    src = open("app.py").read()
    for bad in ("usage_caption", "usage_summary", "setup_usage", "planner.progress", "(25)", "Build embedding index", "_build_index"):
        assert bad not in src, bad
    assert "ensure_embeddings" in src and "startup_index" in src  # the index is prepared as part of normal startup
    assert not re.search(r"\b25\b", open("README.md").read()) and "25 questions" not in src
    assert not re.search(r"h1,h2,h3,p,li,label,span,div", src)
    assert "primaryColor" in open(".streamlit/config.toml").read()


def test_semantic_check_runs_only_when_requested_or_client_given(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    fc = FakeClient([])
    rep = rp.build_report(s, index, fc, semantic=False)
    assert fc.calls == 0 and rep["verification"]["semantic_done"] is False
    rep2 = rp.build_report(s, index)  # no client: deterministic support checks only
    assert rep2["verification"]["semantic_done"] is False


# ---------------------------------------------------------------- data consistency
def test_explicit_rejection_survives_paraphrased_evidence(index):
    s = _ask_criteria(index)
    ans = "No, there is no social media or personality analysis; no crime prediction either."
    iv.process(s, len(s["turns"]) - 1, ans, index, FakeClient([{"slots": {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"},
               "evidence": {"eval_uses_social_traits": "personality analysis is not done", "eval_predicts_crime": "crime prediction is not done"}}]))
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "no" and d["slots"]["eval_predicts_crime"] == "no"
    assert d["slots"]["p_social_scoring"] == "none" and d["slots"]["p_crime_profiling"] == "none"
    assert not [c for c in P.candidates(s) if c["fact"] == "eval_criteria"]
    assert not iv.evidence_ok("social media data", "Experience and test results", "yes")
    assert not iv.evidence_ok("personality analysis is not done", "Personality analysis is done", "no")
    assert iv.evidence_ok("personality analysis is not done", ans, "no")
    assert iv.evidence_ok("the agent doesn't read e-mails", "It doesn't read any e-mails", "no")


def test_compound_followup_answer_maps_to_every_slot(index):
    def followup(index):
        s = _ask_criteria(index)
        iv.process(s, len(s["turns"]) - 1, "I don't know", index, FakeClient([{"slots": {}}]))
        assert s["turns"][-1]["question"]["kind"] == "followup"
        return s
    s = followup(index)
    q = s["turns"][-1]["question"]
    assert q["option_sets"]["No, neither"] == {"eval_uses_social_traits": "no", "eval_predicts_crime": "no"}
    iv.process(s, len(s["turns"]) - 1, "No, neither", index, FakeClient([{"slots": {"eval_predicts_crime": "no"}}]))
    d = C.derive(s)
    assert d["slots"]["eval_uses_social_traits"] == "no" and d["slots"]["eval_predicts_crime"] == "no" and d["slots"]["p_social_scoring"] == "none"
    s2 = followup(index)
    iv.process(s2, len(s2["turns"]) - 1, "Yes, social behaviour or personality data is used", index, FakeClient([{"slots": {}}]))
    d2 = C.derive(s2)
    assert d2["slots"]["eval_uses_social_traits"] == "yes" and d2["slots"]["eval_predicts_crime"] == "unknown"


def test_declaration_answers_are_not_turned_into_unknown(index):
    s = start(index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s)
    assert s["turns"][0]["question"]["control_sets"]["No"]["C20"]["status"] == "declared_absent"
    iv.process(s, 0, "No conformity assessment was done.", index, FakeClient([{"slots": {}}]))
    assert C.derive(s)["controls"]["C20"]["status"] == "declared_absent"
    s2 = start(index)
    s2["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s2)
    iv.process(s2, 0, "No", index, FakeClient([{"slots": {}}]))
    assert C.derive(s2)["controls"]["C20"]["status"] == "declared_absent"
    for txt in ("It was not done but we plan it next month", "I don't know"):
        s3 = start(index)
        s3["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_conformity"], index, s3)
        iv.process(s3, 0, txt, index, FakeClient([{"slots": {}}]))
        assert C.derive(s3)["controls"]["C20"]["status"] == "unknown"
    st = {"settings": {}, "turns": [_turn("A1", "ctl_conformity", {}, ctl={"C20": {"status": "declared_absent", "note": ""}}),
                                    _turn("A2", "ctl_conformity", {}, ctl={"C20": {"status": "unknown", "note": ""}})]}
    assert C.derive(st)["controls"]["C20"]["status"] == "declared_absent"


def test_control_declaration_needs_a_user_quote(index):
    """A control status the model invents without a quote from the user is dropped (it would otherwise change the coverage)."""
    s = start(index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["ctl_owner"], index, s)
    fc = FakeClient([{"slots": {}, "controls": [{"control_id": "C03", "status": "declared_present", "note": "", "evidence": "a named owner exists and is trained"}]}])
    iv.process(s, 0, "We are a small team and just started", index, fc)
    assert "C03" not in C.derive(s)["controls"] or C.derive(s)["controls"]["C03"]["status"] == "unknown"
    assert any(k == "C03" for k, _ in s["turns"][0]["updates"]["rejected"])


def _conflict_state():
    return {"settings": {}, "turns": [_turn("A1", "purpose", {**BASE, "action_mode": "envelope_action"}, "The agent rejects candidates without approval"),
                                      _turn("A2", "approval_bound", {"approval_bound": "yes"}, "Yes, approval specific to the action")]}


def test_cross_slot_approval_conflict_blocks_certain_autonomy(index):
    s = _conflict_state()
    d = C.derive(s)
    assert "approval_bound" in d["conflicts"] and C.autonomy_conflict(d)
    cls = C.classify(s)
    assert cls["L"] is None and cls["posture"] is None and "action_mode" in cls["unresolved"]  # never a guessed L
    assert any("conflict" in u for u in cls["uncertainties"])
    top = P.candidates(s)[0]
    assert top["kind"] == "conflict" and top["fact"] == "approval_bound"
    q = iv.make_question(top["topic"], index, s, kind="conflict", cand=top)
    assert "Which of these applies" in q["text"] and "All actions are done with action-specific approval" in q["options"] and q["no_passage"]
    s["turns"].append({"id": "A3", "question": q, "answer": "All actions are done with action-specific approval", "updates": {}})
    iv.process(s, 2, "All actions are done with action-specific approval", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert not d["conflicts"] and not d["unresolved_conflicts"] and d["slots"]["action_mode"] == "approved_action" and C.classify(s)["L"] == "L3"


def test_unresolved_approval_conflict_stays_uncertain_and_is_asked_once(index):
    s = _conflict_state()
    top = P.candidates(s)[0]
    q = iv.make_question(top["topic"], index, s, kind="conflict", cand=top)
    s["turns"].append({"id": "A3", "question": q, "answer": "I don't know", "updates": {}})
    iv.process(s, 2, "I don't know", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert "approval_bound" in d["unresolved_conflicts"] and d["slots"]["approval_bound"] == "unknown"
    cls = C.classify(s)
    assert cls["L"] is None and cls["posture"] is None
    assert not [c for c in P.candidates(s) if c["kind"] == "conflict"]


def test_mixed_approval_is_classified_by_the_highest_authority_exercised_without_approval(index):
    """Matrix section 2: 'assess the maximum technically reachable authority'. If some actions are approved and some are not, the unapproved ones set the level."""
    s = _conflict_state()
    top = P.candidates(s)[0]
    q = iv.make_question(top["topic"], index, s, kind="conflict", cand=top)
    s["turns"].append({"id": "A3", "question": q, "answer": "Some actions are approved, some are not", "updates": {}})
    iv.process(s, 2, "Some actions are approved, some are not", index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert not d["conflicts"] and not d["unresolved_conflicts"] and d["slots"]["action_mode"] == "envelope_action"
    assert C.classify(s)["L"] == "L4"


# ---------------------------------------------------------------- combined prohibited-use check (checklist question)
def test_screening_is_one_checklist_question_and_comes_after_agent_specific_questions(index):
    s = iv.new_state()
    iv.add_first(s, index)
    iv.process(s, 0, "It recommends refunds to a support employee, who decides.", index,
               FakeClient([{"slots": {"purpose": "refund recommender", "action_mode": "recommend"}, "evidence": {"action_mode": "recommends refunds"}}]))
    ids = [c["topic"]["id"] for c in P.candidates(s)]
    assert "screening" in ids and not any(i in ids for i in ("g_biometric", "g_influence", "g_person_eval"))
    top = C.TOPICS()["screening"]
    q = iv.make_question(top, index, s, text="Does it use faces? Does it score people?")  # model wording is ignored for checklists
    assert q["multi"] and q["text"] == top["default_q"] and q["options"] == []


def test_multi_answer_only_none_of_these_is_an_explicit_negative():
    m = C.TOPICS()["screening"]["multi"]
    text, sets = iv.multi_answer(m, [], "none")
    assert sets == {"g_biometric": "no", "g_influence": "no", "g_person_eval": "no"} and text == "None of these."
    text, sets = iv.multi_answer(m, ["g_person_eval"], "confirm")
    assert sets == {"g_person_eval": "yes", "g_biometric": "unknown", "g_influence": "unknown"} and text.startswith("Selected:") and "Not answered:" in text  # unticked is NOT a 'no'
    _, sets = iv.multi_answer(m, ["g_biometric"], "unsure")
    assert sets == {"g_biometric": "yes", "g_influence": "unknown", "g_person_eval": "unknown"}


def test_screening_checklist_answer_is_recorded_with_its_own_evidence(index):
    s = iv.new_state()
    iv.add_first(s, index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["screening"], index, s)
    text, sets = iv.multi_answer(C.TOPICS()["screening"]["multi"], [], "none")
    s["turns"][0]["question"]["option_sets"] = {text: sets}
    iv.process(s, 0, text, index, FakeClient([{"slots": {}}]))
    d = C.derive(s)
    assert d["slots"]["g_biometric"] == d["slots"]["g_influence"] == d["slots"]["g_person_eval"] == "no"
    assert d["slots"]["p_rbi_law_enforcement"] == "none" and d["evidence"]["g_influence"] == text


def test_every_source_family_is_used_by_the_topic_catalog(index):
    """RAG must not be AI-Act-only: the playbooks, the TrustAI matrix and the Commission decision are wired to topics and return supported passages."""
    seen = set()
    for t in C.interview_cfg()["topics"]:
        seen |= {p["source_id"] for p in rag.retrieve_topic(index, t, max_total=6)}
    assert {"AIA", "TAI-GOV", "TAI-AG", "NIST-PB", "DEC2274"} <= seen and "TAI-MX" not in seen, seen


def test_checklist_skips_items_already_declared(index):
    s = iv.new_state()
    iv.add_first(s, index)
    iv.process(s, 0, "It scores candidates.", index, FakeClient([{"slots": {"purpose": "p", "g_person_eval": "yes"}, "evidence": {"g_person_eval": "scores candidates"}}]))
    q = iv.make_question(C.TOPICS()["screening"], index, s)
    assert [it["slot"] for it in q["multi"]["items"]] == ["g_biometric", "g_influence"]


def test_checklist_topic_does_not_widen_the_terms_that_identify_a_negative_fact():
    """'Recruiters can't see how the score is calculated' is a negation about something else: it must not count as evidence that nothing steers users."""
    assert not (iv.slot_terms("g_influence") & rag.keywords("score ranking rank scoring"))
    assert not iv.negation_is_about("g_influence", "Recruiters can see the score but not how it was calculated", "x " * 30)


def test_statuses_are_separate_and_completion_does_not_imply_verification(index):
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, EXAMPLE, index, FakeClient([{"slots": EX_SLOTS}]))
    run_truth(s, index, TRUTH)
    rep = rp.build_report(s, index)
    assert rep["verification_status"] == "Not technically verified"
    assert rep["interview_status"] in ("Completed", "Incomplete") and rep["assessment_status"] in ("Preliminary", "Requires further information")
    # an unknown control that affects the result must flip the assessment status, whatever the interview status is
    s2 = iv.new_state(); iv.add_first(s2, index)
    rep2 = rp.build_report(s2, index)
    assert rep2["interview_status"] == "Incomplete" and rep2["assessment_status"] == "Requires further information" and rep2["assessment_reasons"]
    assert rep2["verification_status"] == "Not technically verified"
    for r in (rep, rep2):
        assert "Complete" not in (r["interview_status"], r["assessment_status"])
