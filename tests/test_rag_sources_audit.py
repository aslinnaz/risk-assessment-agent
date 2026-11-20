"""Source-aware retrieval, citation validation, source ingestion after deployment, failure handling, report/state consistency and the removal of the internal
policy document from every retrievable or visible surface."""
import json
import re
from pathlib import Path

import pytest

import classify as C
import config
import interview as iv
import planner as P
import rag
import report as rp
import sources_admin as sa
from conftest import FakeClient
from test_procurement_audit import ANSWER, GATES_NO, LARGE, SMALL, answer, start

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.needs_sources


# ---------------------------------------------------------------- 14: retrieval follows the question, not the AI Act by default
def _srcs(index, tid, n=8):
    return [p["source_id"] for p in rag.retrieve_topic(index, C.TOPICS()[tid], max_total=n)]


def test_retrieval_is_routed_by_the_governance_concern(index):
    legal = _srcs(index, "annex3")
    assert legal[0] == "AIA" and "TAI-GOV" in legal                                   # legal classification: legislation first, guidance beside it
    agentic = _srcs(index, "boundaries")
    assert agentic[0] in ("TAI-AG", "NIST-PB") and "AIA" not in agentic[:2]           # tool permissions and delegation: agentic playbook and NIST first
    oversight = _srcs(index, "oversight")
    assert {"TAI-AG", "NIST-PB"} <= set(oversight) and oversight[0] in ("TAI-AG", "NIST-PB")
    transp = _srcs(index, "t_synthetic")
    assert transp[0] == "AIA"                                                         # transparency obligations: the applicable legislation
    allsrc = set()
    for tid in C.TOPICS():
        allsrc |= set(_srcs(index, tid))
    assert {"AIA", "NIST-PB", "TAI-AG", "TAI-GOV"} <= allsrc


def test_not_every_topic_leads_with_the_ai_act(index):
    topics = [t["id"] for t in C.interview_cfg()["topics"] if t.get("asked") is not False]
    leading = [(_srcs(index, t, 3) or [None])[0] for t in topics]
    assert len({x for x in leading if x}) >= 2 and leading.count("AIA") < len(leading) * 0.8       # authority and oversight questions lead with the agentic playbook
    assert sum(1 for t in topics if "NIST-PB" in _srcs(index, t, 4)) >= 5                          # NIST guidance accompanies many questions
    assert sum(1 for t in topics if "TAI-GOV" in _srcs(index, t, 4)) >= 5


# ---------------------------------------------------------------- 15: citations are validated, unsupported passages are rejected
def test_unsupported_passage_is_rejected_by_the_support_check(index):
    art5 = index.lookup("AIA", "Art. 5(1)(c)")[0]
    unrelated = index.lookup("NIST-PB", "GOVERN 3.2")[0]
    claim, terms = "social scoring of natural persons", [["social behaviour", "social scoring"]]
    assert rag.support(art5, claim, terms, None)["ok"] and not rag.support(unrelated, claim, terms, None)["ok"]


def test_citation_validation_catches_tampered_or_invented_citations(index):
    s = start(index)
    rep = rp.build_report(s, index)
    assert rep["citation_check"]["ok"] and rep["citation_check"]["checked"] == len(rep["cited"])
    good = dict(rep["cited"][0])
    assert rp.validate_citations([good], index)["ok"]
    for tamper in ({"chunk_id": "AIA:Art-999"}, {"ref": "Art. 999"}, {"version": "2099"}, {"snippet": "an invented quotation that is not in the source"}):
        bad = rp.validate_citations([{**good, **tamper}], index)
        assert not bad["ok"] and bad["problems"], tamper


def test_partial_support_is_never_silently_promoted(index):
    s = start(index)

    class Partial(FakeClient):
        pass
    fc = FakeClient([])
    orig = rp.semantic_check

    def fake_semantic(findings, index_, client, usage):
        for f in findings:
            if f["passages"]:
                f["semantic"] = "partial"
        return True
    rp.semantic_check, rp.llm_structured = fake_semantic, None
    try:
        rep = rp.build_report(s, index, client=fc, semantic=True)
    finally:
        rp.semantic_check = orig
    flagged = [f for f in rep["findings"] if f["semantic"] == "partial"]
    assert flagged and set(rep["verification"]["flagged"]) >= {f["id"] for f in flagged} and not rep["verification"]["all_ok"]


def test_internal_rule_findings_need_no_external_citation_and_are_not_flagged_partial(index):
    rep = rp.build_report(start(index), index)
    op = next(f for f in rep["findings"] if f["role"] == "operational")
    assert op["kind"] == "internal_rule" and op["sources"] == [] and op["semantic"] == "not checked" and op["id"] not in rep["verification"]["flagged"]
    assert "POL-L4" in op["rules"]


# ---------------------------------------------------------------- 17: failures are visible, never silent
def test_retrieval_failure_modes_are_explicit(index, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SOURCES_DIR", tmp_path)
    rag.get_index.cache_clear()
    with pytest.raises(rag.SourceError):
        rag.get_index()
    rag.get_index.cache_clear()
    empty = type("E", (), {"chunks": [], "retrieve": rag.Index.retrieve})()
    assert rag.Index.retrieve(empty, "anything") == []
    r = rag.ensure_embeddings(index, None)
    assert r["ok"] or r["error"]  # no client: the keyword-only mode is reported, not hidden


# ---------------------------------------------------------------- 18, 19: the report agrees with the accepted fact state; unknown applicability stays undetermined
def test_report_pdf_and_state_agree_for_the_procurement_scenario(index):
    import pymupdf
    s = start(index)
    s = answer(s, index, "The limit is enforced by the purchasing system itself.", workflows=[{"id": "po_small", "limit_enforcement": "technical", "evidence": "enforced by the purchasing system itself"}])
    cls = C.classify(s)
    rep = rp.build_report(s, index)
    assert rep["integrity"]["ok"] and rep["cls"]["L"] == cls["L"] == "L4" and rep["posture"] == cls["posture"] == "P4"
    with pymupdf.open(stream=rp.pdf_bytes(rep), filetype="pdf") as doc:
        txt = re.sub(r"\s+", " ", " ".join(p.get_text() for p in doc))
    assert "L4" in txt and "I2" in txt and "P4" in txt and "Hold for redesign" in txt and "up to EUR 5,000" in txt
    assert "Some actions are approved" not in txt
    assert "Technical appendix" not in txt and "Citation validation" not in txt and "Basis and traceability" in txt and "Classification derivation" in txt and "Tool and action inventory" in txt
    assert rep["summary"]["steps"] and "Redesign" in rep["summary"]["steps"][0]
    assert rep["use_text"] and "EUR 5,000" in " ".join(rep["capabilities"]) and "EUR 5,000" in rep["use_text"]


def test_unknown_legal_applicability_is_not_turned_into_a_finding(index):
    rep = rp.build_report(start(index), index)
    cls = rep["cls"]
    assert cls["applicability"] == "unknown" and cls["legal"]["code"] != "out_of_scope_declared"
    assert rep["context"]["applicability"] == "Not determined"
    lg = " ".join(v for k, v in rep["conclusions"] if k.startswith("A."))
    assert "partly assessed" in lg and "does not mean that no obligation applies" in lg
    assert any("EU legal applicability was not determined" in g for g in rep["critical_gaps"])
    assert "only partly assessed" in next(f["text"] for f in rep["findings"] if f["role"] == "legal")


def test_operational_posture_is_not_a_legal_prohibition(index):
    rep = rp.build_report(start(index), index)
    assert rep["posture"] == "P4" and rep["risk_label"] == "HIGH" and rep["cls"]["legal"]["code"] == "partly_assessed"  # Annex III excluded by declaration, Annex I route not established
    assert "not a legal category" in " ".join(v for k, v in rep["conclusions"] if k.startswith("C."))
    assert "internal" in rep["posture_meaning"].lower() and "not a legal" in rep["posture_meaning"].lower()


# ---------------------------------------------------------------- 20: the internal policy document is not a source anywhere
FORBIDDEN_NAMES = ("Compliance Check Matrix", "compliance_matrix", "TAI-MX")


def test_internal_policy_document_is_absent_from_the_index_retrieval_citations_and_reports(index):
    assert all(c["source_id"] != "TAI-MX" for c in index.chunks)
    assert not [c for c in index.chunks if any(n in (c["text"][:200] + c["title"]) for n in ("Compliance Check Matrix",))]
    for tid in C.TOPICS():
        for p in rag.retrieve_topic(index, C.TOPICS()[tid], max_total=8):
            assert p["source_id"] != "TAI-MX"
    s = start(index)
    rep = rp.build_report(s, index)
    blob = json.dumps({k: v for k, v in rep.items() if k not in ("cls",)}, default=str)
    for n in FORBIDDEN_NAMES:
        assert n not in blob, n
    import pymupdf
    with pymupdf.open(stream=rp.pdf_bytes(rep), filetype="pdf") as doc:
        txt = " ".join(p.get_text() for p in doc)
    for n in FORBIDDEN_NAMES:
        assert n not in txt, n


def test_no_visible_application_text_names_the_internal_policy_document():
    files = [ROOT / f for f in ("app.py", "interview.py", "report.py", "classify.py", "planner.py", "rag.py")] + list((ROOT / "catalogs").glob("*.json"))
    for f in files:
        txt = f.read_text(encoding="utf-8")
        for n in ("Compliance Check Matrix", "compliance_matrix", "matrix §"):
            assert n not in txt, (f.name, n)
        assert "TAI-MX" not in txt.replace("TAI-MX not", ""), f.name


# ---------------------------------------------------------------- risk pathways: LOW, MEDIUM, HIGH, FORBIDDEN and the insufficient / ambiguous cases
BASE = {"purpose": "p", "annex3_area": "none", **GATES_NO}


def _state(slots, basis=None, wf=None):
    return {"settings": {}, "turns": [{"id": "A1", "question": {"topic": "purpose", "kind": "question"}, "answer": "x",
                                       "updates": {"slots": {**BASE, **slots}, "controls": {}, "basis": basis or [], "evidence": {}, "workflows": wf or []}}]}


@pytest.mark.parametrize("slots,label,posture", [
    ({"action_mode": "info_only", "max_harm": "minimal"}, "LOW", "P1"),
    ({"action_mode": "recommend", "max_harm": "limited"}, "LOW", "P2"),
    ({"action_mode": "envelope_action", "max_harm": "serious", "limits_enforced": "technical"}, "HIGH", "P4"),
])
def test_low_medium_high_pathways(slots, label, posture):
    c = C.classify(_state(slots))
    assert (c["risk_label"], c["posture"]) == (label, posture)


def test_forbidden_pathway_needs_a_supported_article_5_finding():
    s = _state({"g_person_eval": "yes", "eval_uses_social_traits": "yes", "p_social_scoring": "supported", "action_mode": "recommend", "max_harm": "limited"},
               basis=[{"key": "p_social_scoring", "conditions_confirmed": True, "exception_considered": True, "scope_date_considered": True}])
    assert C.classify(s)["risk_label"] == "FORBIDDEN"
    assert C.classify(_state({"g_person_eval": "yes", "eval_uses_social_traits": "yes", "p_social_scoring": "possible", "action_mode": "recommend", "max_harm": "limited"}))["risk_label"] is None


def test_insufficient_information_gives_a_range_or_no_label_never_invented_certainty():
    c = C.classify(_state({"action_mode": "envelope_action"}))  # impact unknown
    assert c["I"] is None and c["posture"] is None and c["posture_range"]
    c2 = C.classify(_state({"max_harm": "limited"}))            # autonomy unknown: P1-P4 are all possible, so no level is issued
    assert c2["L"] is None and c2["risk_label"] is None and c2["posture_range"] == "P1–P4"
    c3 = C.classify(_state({"max_harm": "serious"}))            # autonomy unknown, but every possibility maps to HIGH: a range, and the level is decidable
    assert c3["posture"] is None and c3["posture_range"] == "P3–P4" and c3["risk_label"] == "HIGH"


# ---------------------------------------------------------------- 16: new approved sources can be added after deployment
def _pdf(path: Path, title: str, paragraphs: list[str]):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(str(path), pagesize=A4)
    y = 800
    for line in [title] + paragraphs:
        for chunk in re.findall(r".{1,95}(?:\s|$)", line):
            c.drawString(40, y, chunk.strip())
            y -= 14
            if y < 60:
                c.showPage(); y = 800
        y -= 8
    c.save()


AGENT_TEXT = ["1. Delegated authority and tool permissions",
              "Agents should receive the least privilege needed. Authority limits must be enforced by the system, not only written in instructions. Delegation to other agents or tools must not "
              "exceed the authority of the delegating agent. Privileged routes must not bypass approval boundaries. Human approval should be bound to the specific action, amount and recipient.",
              "2. Stopping and reversal",
              "Operators must be able to stop an agent immediately and to reverse actions it has taken. Aggregate limits protect against split transactions that circumvent a single-action threshold. "
              "Every executed action should be recorded in a log that identifies who or what approved it. Monitoring during execution is different from review after the action."]
META = {"source_id": "CUST-AG1", "title": "Example Agent Safety Guidance", "publisher": "Example Institute", "version": "1.0", "date": "2026-10-01", "kind": "guidance",
        "concerns": ["agentic", "oversight"], "license": {"indexing_permitted": True}}


@pytest.fixture
def custom_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "custom_dir", lambda: tmp_path / "custom")
    rag.get_index.cache_clear()
    yield tmp_path
    rag.get_index.cache_clear()


def test_new_source_is_ingested_indexed_retrieved_and_cited_with_metadata(custom_dir, index):
    src = custom_dir / "guidance.pdf"
    _pdf(src, "Example Agent Safety Guidance", AGENT_TEXT)
    before = C.classify(start(index))
    r = sa.add_source(src, META)
    assert r["ok"], r["problems"]
    man = {m["source_id"]: m for m in rag.load_manifest()["sources"]}
    assert man["CUST-AG1"]["version"] == "1.0" and man["CUST-AG1"]["kind"] == "guidance" and man["CUST-AG1"]["publisher"] == "Example Institute"
    ix = rag.Index()
    mine = [c for c in ix.chunks if c["source_id"] == "CUST-AG1"]
    assert mine and all(c["version"] == "1.0" and c["page"] for c in mine) and any("least privilege" in c["text"] for c in mine)
    assert "CUST-AG1" in [p["source_id"] for p in rag.retrieve_topic(ix, C.TOPICS()["boundaries"], max_total=8)]   # routed by concern, no code change
    assert "CUST-AG1" not in [p["source_id"] for p in rag.retrieve_topic(ix, C.TOPICS()["annex3"], max_total=8)]  # not a legal-classification source
    after = C.classify(start(index))                       # adding a source never changes the deterministic classification
    assert (after["L"], after["I"], after["posture"], after["risk_label"], after["trace"]) == (before["L"], before["I"], before["posture"], before["risk_label"], before["trace"])
    assert rag.source_meta("CUST-AG1")["short"]
    rep = rp.build_report(start(index), ix)
    assert rep["citation_check"]["ok"]


def test_unsupported_or_incomplete_documents_are_rejected_not_trusted(custom_dir):
    txt = custom_dir / "notes.txt"; txt.write_text("x" * 2000)
    assert not sa.add_source(txt, META)["ok"]
    thin = custom_dir / "thin.pdf"; _pdf(thin, "Thin", ["Too short."])
    assert any("no extractable text" in p for p in sa.add_source(thin, META)["problems"])
    ok = custom_dir / "ok.pdf"; _pdf(ok, "T", AGENT_TEXT)
    assert any("missing metadata: publisher" in p for p in sa.add_source(ok, {k: v for k, v in META.items() if k != "publisher"})["problems"])
    assert any("licence" in p for p in sa.add_source(ok, {**META, "license": {}})["problems"])
    assert any("kind must be" in p for p in sa.add_source(ok, {**META, "kind": "internal_policy"})["problems"])
    assert any("already exists" in p for p in sa.add_source(ok, {**META, "source_id": "AIA"})["problems"])
    assert rag.custom_registry()["sources"] == []     # nothing was registered


def test_duplicates_are_detected_by_content(custom_dir):
    a = custom_dir / "a.pdf"; _pdf(a, "T", AGENT_TEXT)
    assert sa.add_source(a, META)["ok"]
    b = custom_dir / "b.pdf"; b.write_bytes(a.read_bytes())
    dup = sa.add_source(b, {**META, "source_id": "CUST-AG2"})
    assert not dup["ok"] and any("duplicate" in p for p in dup["problems"])


def test_deactivation_and_superseded_versions_remove_a_source_from_the_index(custom_dir, index):
    a = custom_dir / "a.pdf"; _pdf(a, "T", AGENT_TEXT)
    assert sa.add_source(a, META)["ok"]
    assert any(c["source_id"] == "CUST-AG1" for c in rag.Index().chunks)
    assert sa.set_active("CUST-AG1", False)
    assert not any(c["source_id"] == "CUST-AG1" for c in rag.Index().chunks) and sa.status()[0]["reason"] == "deactivated"
    assert sa.set_active("CUST-AG1", True)
    v2 = custom_dir / "v2.pdf"; _pdf(v2, "T", [AGENT_TEXT[0], AGENT_TEXT[1] + " Version two adds a revised section on monitoring during execution.", AGENT_TEXT[2], AGENT_TEXT[3]])
    assert sa.add_source(v2, {**META, "source_id": "CUST-AG2", "version": "2.0", "supersedes": "CUST-AG1"})["ok"]
    st = {s["source_id"]: s for s in sa.status()}
    assert st["CUST-AG2"]["state"] == "active" and "superseded" in st["CUST-AG1"]["reason"]


def test_restricted_or_unlicensed_standards_are_not_indexed(custom_dir):
    a = custom_dir / "std.pdf"; _pdf(a, "T", AGENT_TEXT)
    r = sa.add_source(a, {**META, "kind": "standard", "license": {"indexing_permitted": False}})
    assert not r["ok"] and any("copyrighted standards must not be added" in p for p in r["problems"])


def test_a_changed_document_triggers_reindexing(custom_dir, index):
    a = custom_dir / "a.pdf"; _pdf(a, "T", AGENT_TEXT)
    assert sa.add_source(a, META)["ok"]
    n1 = len([c for c in rag.Index().chunks if c["source_id"] == "CUST-AG1"])
    dest = custom_dir / "custom" / "CUST-AG1.pdf"
    _pdf(dest, "T", AGENT_TEXT + ["3. Logging", "Trajectory logs must record prompts, tool calls, results and actions so that a transaction can be reconstructed afterwards by an independent reviewer."])
    ix = rag.Index()
    assert any("Trajectory logs" in c["text"] for c in ix.chunks if c["source_id"] == "CUST-AG1")
