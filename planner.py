"""Question selection and finishing decision (no model call).

A topic can be asked ONLY in these cases:
  1. conflict: two earlier declarations conflict (only that item, at most once),
  2. follow-up: after an 'unknown' or 'yes' answer, at most ONE simple/detail question for the same item,
  3. required: critical information needed to complete the result is missing (blocking),
  4. decisive: the possible values of the answer change the result in simulation (label, legal path, L/I, transparency),
  5. context: does not change the result but is a limited number of topics chosen for the operational picture in the report (at most CONTEXT_MAX).
'It exists in the catalog' is not a reason to ask a question. The C01–C21 controls are a coverage map, not a question list.
"""
from __future__ import annotations

import itertools

import classify as C
import config

NEUTRAL_IMPACT = "This clarifies a factor that can affect how the agent is assessed."
LABEL_NAMES = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "FORBIDDEN": "FORBIDDEN", None: "incomplete"}


def _fp(cls: dict) -> tuple:
    return (cls["risk_label"], cls["assessment_status"], cls["legal"]["code"],
            tuple(sorted(t["slot"] for t in cls["transparency"])), cls["L"], cls["I"], cls["posture"])


def _sim(state: dict, assign: dict) -> dict:
    t = {"id": "SIM", "question": {"kind": "sim"}, "answer": "x", "updates": {"slots": assign, "controls": {}, "basis": []}}
    return C.classify({**state, "turns": state["turns"] + [t]})


def _combos(top: dict) -> list[dict]:
    spec = C.interview_cfg()["slots"]
    req = C.required_slots(top)
    if not req:
        return []
    pools = [[v for v in spec[sl] if v != "unknown"] for sl in req]
    ex = C.interview_cfg().get("exclusions", [])
    out = [dict(zip(req, vals)) for vals in itertools.product(*pools)]
    return out[:60]


def impact(state: dict, top: dict, base_cls: dict) -> dict:
    """What do the possible answers to this topic change in the assessment? (simulation; no numeric score)"""
    base = _fp(base_cls)
    outs, labels = {base}, set()
    for a in _combos(top):
        cls = _sim(state, a)
        fp = _fp(cls)
        if fp != base:
            outs.add(fp)
            labels.add(cls["risk_label"])
    changed = len(outs) > 1
    if not changed:
        text = ""
    else:
        names = sorted({LABEL_NAMES[l] for l in labels} - {"incomplete"})
        text = NEUTRAL_IMPACT  # never discloses or predicts a risk level before the person has answered
    return {"changes": changed, "n": len(outs), "text": text}


def blocking_topic_ids(d: dict, cls: dict) -> set[str]:
    """Topics that own the critical slots blocking the label/completion."""
    owner = {}
    for t in C.interview_cfg()["topics"]:
        for sl in C.required_slots(t):
            owner.setdefault(sl, t["id"])
    ids = set()
    for sl in cls["unresolved"]:
        if sl.startswith("p_"):
            gid = C.group_of(sl)
            if gid:  # the first information whose precondition has not been asked yet (is empty) must be learned; if all are 'yes', the detail follow-up is asked anyway
                miss = next((k for k in C.interview_cfg()["prohibition_groups"][gid]["requires"] if d["slots"].get(k) is None), None)
                if miss and miss in owner:
                    ids.add(owner[miss])
        elif sl in owner:
            ids.add(owner[sl])
    return ids


def context_topic_ids(d: dict, cls: dict) -> set[str]:
    """Topics owning a CRITICAL context-sensitive gap that was never asked (see classify.context_gaps): they are asked, but do not block the label."""
    owner = {}
    for t in C.interview_cfg()["topics"]:
        for sl in C.required_slots(t):
            owner.setdefault(sl, t["id"])
    return {owner[g["slot"]] for g in cls.get("context_gaps", []) if g["critical"] and g["state"] == "not_assessed" and g["slot"] in owner}


def candidates(state: dict, exclude_topic: str | None = None) -> list[dict]:
    """Ordered list of candidate questions. An empty list = the interview should end."""
    d = C.derive(state)
    cls = C.classify(state)
    s, cfg = d["slots"], C.interview_cfg()
    count = C.answered_count(state)
    if cls["risk_label"] == "FORBIDDEN" or count >= config.MAX_QUESTIONS:
        return []
    eu_out = s.get("eu_link") == "no"
    topics = {t["id"]: t for t in cfg["topics"]}
    out = []
    # 1) conflict
    for slot, c in d["conflicts"].items():
        if d["conflict_asked"].get(slot, 0) == 0:
            own = topics["action_mode"] if slot.startswith("wf:") else next((t for t in cfg["topics"] if slot in t["slots"]), None)
            if own:
                out.append({"kind": "conflict", "topic": own, "fact": slot, "conflict": c, "impact": "Two earlier declarations conflict.", "rank": (0, 0)})
    # 2) follow-up
    for gid in C.detail_groups(s):  
        if not eu_out and d["followups"].get("detail:" + gid, 0) == 0:
            g = cfg["prohibition_groups"][gid]
            out.append({"kind": "followup", "topic": topics["prohibited_detail"], "fact": "detail:" + gid, "gate": gid, "text": g["detail_q"], "options": g.get("detail_options", []),
                        "impact": "Clarifies whether the declared feature meets the conditions of a prohibited use.", "rank": (1, 0)})
    for t in cfg["topics"]:
        if t["id"] in ("prohibited_detail",) or not t.get("simple_q") or not (t["critical"] or t.get("multi")):
            continue
        if eu_out and (t["id"].startswith("g_") or t["id"] in ("screening", "annex3", "profiling", "art6_role", "art6_influence", "eval_criteria")):
            continue
        if d["asked"].get(t["id"], 0) >= 1 and d["followups"].get(t["id"], 0) == 0 and C.topic_active(t, s) \
                and any(s.get(sl) == "unknown" for sl in C.required_slots(t)):
            out.append({"kind": "followup", "topic": t, "fact": t["id"], "text": t["simple_q"], "options": t.get("simple_options", t.get("options", [])),
                        "sets": t.get("simple_option_map", {}),
                        "impact": "Asks the unknown critical information once more with a plain example.", "rank": (1, -t["priority"])})
    # 3-5) topics
    blocking = blocking_topic_ids(d, cls) | context_topic_ids(d, cls)
    gap_topics = set()
    for cid, g in cls.get("control_gaps", {}).items():
        if g["state"] == "unasked":  
            own = [t for t in cfg["topics"] if cid in t["controls"] and C.topic_active(t, s) and not C.topic_resolved(t, d)]
            own.sort(key=lambda t: (bool(t["slots"]), -t["priority"]))  # declaration topics first
            if own:
                gap_topics.add(own[0]["id"])
    ctx_asked = sum(1 for tt in state["turns"] if C.is_counted(tt) and C.TOPICS().get(tt["question"].get("topic"), {}).get("context"))
    for t in cfg["topics"]:
        if t["id"] == exclude_topic or C.topic_resolved(t, d) or not C.topic_active(t, s):
            continue
        if t.get("asked") is False and t["id"] not in gap_topics:
            continue
        if eu_out and t["id"] in ("profiling", "art6_role", "art6_influence"):
            continue
        if t.get("early"):
            out.append({"kind": "question", "topic": t, "fact": t["id"], "impact": "Establishes the scope of the assessment.", "rank": (2, -t["priority"])})
            continue
        if t["id"] in gap_topics:
            out.append({"kind": "question", "topic": t, "fact": t["id"], "impact": "This covers a safeguard that has not been described yet and that bears on the assessment.", "rank": (2, -t["priority"])})
            continue
        if t["id"] in blocking:
            out.append({"kind": "question", "topic": t, "fact": t["id"], "impact": "Information needed to complete the assessment.", "rank": (2, -t["priority"])})
            continue
        im = impact(state, t, cls)
        if im["changes"]:
            out.append({"kind": "question", "topic": t, "fact": t["id"], "impact": im["text"], "rank": (3, -im["n"], -t["priority"])})
        elif t.get("context") and count < config.TARGET_QUESTIONS and ctx_asked < config.CONTEXT_MAX:
            out.append({"kind": "question", "topic": t, "fact": t["id"], "impact": "Adds context for the description of the agent in the report.", "rank": (4, -t["priority"])})
    return sorted(out, key=lambda c: c["rank"])


def progress(state: dict) -> dict:
    """UI summary: derived from the current state (no model call)."""
    d = C.derive(state)
    cls = C.classify(state)
    s, P = d["slots"], C.policy()
    known = []
    def ok(k): return s.get(k) not in (None, "unknown")
    if ok("purpose"): known.append("intended purpose")
    pk = list(P["prohibition_refs"])
    if all(ok(k) for k in pk): known.append("prohibited-use screening")
    if ok("annex3_area"): known.append("sensitive use area")
    if ok("action_mode"): known.append("agent authority")
    if ok("max_harm"): known.append("possible harm")
    need = []
    labels = P["critical_labels"]
    seen = set()
    for sl in cls["unresolved"]:
        lab = "prohibited-use screening" if sl.startswith("p_") else labels.get(sl, sl)
        if lab not in seen:
            seen.add(lab)
            need.append(lab)
    for c in d["conflicts"]:
        need.append("conflicting information: " + C.interview_cfg()["slot_desc"].get(c, c))
    return {"known": known, "need": need}
