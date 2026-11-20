"""Release regressions for the professor's final scope and classification feedback."""
import classify as C
import interview as I
import report as R


def state(slots, workflows=None, tools=None):
    return {"settings": {}, "turns": [{"id": "A1", "question": {"kind": "question", "topic": "action_mode", "text": "Describe the agent"},
            "answer": "The agent reads CVs and ranks candidates. Some qualified candidates may lose the job opportunity.",
            "updates": {"slots": slots, "workflows": workflows or [], "tools": tools or [], "evidence": {}, "controls": {}, "basis": []}}]}


def test_read_only_ranking_is_not_external_execution():
    s = state({"max_harm": "limited", "harm_desc": "Some qualified candidates may lose the job opportunity."},
              workflows=[{"id": "rank", "action": "rank candidates", "execution": "recommend", "evidence": "ranks candidates"}],
              tools=[{"name": "ATS", "kind": "system", "permissions": ["read"], "evidence": "reads CVs"}])
    d = C.derive(s)
    assert d["tools"][0]["permissions"] == ["read"]
    assert d["slots"]["tools_summary"] == "ATS"
    result = C.classify(s)
    assert result["L"] == "L1"
    assert result["I"] == "I2"
    assert result["posture"] == "P3"
    assert any(x.get("rule_id") == "POL-I2-EMPLOYMENT" for x in result["trace"])


def test_automatic_rejection_is_execution():
    s = state({"max_harm": "serious"}, workflows=[{"id": "reject", "action": "reject applicants", "execution": "executes_without_approval", "evidence": "rejects applicants"}])
    assert C.classify(s)["L"] == "L4"


def test_unknown_specific_approval_does_not_imply_l4():
    assert C.classify(state({"action_mode": "approved_action", "approval_bound": "unknown", "max_harm": "limited"}))["L"] is None


def test_posture_labels_follow_matrix():
    assert C.policy()["label_from_posture"] == {"P1": "LOW", "P2": "LOW", "P3": "HIGH", "P4": "HIGH"}


def test_inventory_extraction_rejects_unsupported_tools():
    raw = {"tool_updates": [{"name": "ATS", "kind": "system", "scope": "", "permissions": ["read"], "evidence": "reads CVs"},
                            {"name": "Payment API", "kind": "tool", "scope": "", "permissions": ["transact"], "evidence": "can transact"}]}
    turn = {"answer": "The agent reads CVs.", "question": {"kind": "question", "topic": "tool_inventory"}}
    out = I._sanitize(raw, turn)
    assert [t["name"] for t in out["tools"]] == ["ATS"]


def test_unassessed_area_has_no_empty_dashes():
    src = __import__('pathlib').Path('report.py').read_text()
    assert 'else "—"' not in src


def test_write_permission_does_not_silently_pass_as_read_only():
    s = state({"max_harm": "limited"}, workflows=[{"id": "rank", "action": "rank candidates", "execution": "recommend", "evidence": "ranks candidates"}], tools=[{"name": "ATS", "permissions": ["read", "modify"], "evidence": "reads CVs"}])
    c = C.classify(s)
    assert c["L"] is None
    assert any(x.get("rule_id") == "POL-TOOL-BOUNDARY" for x in c["trace"])
