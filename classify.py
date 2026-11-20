"""Derivation from the interview state and DETERMINISTIC classification.

Nothing in this module calls a model or the retrieval index: the risk result is computed only from the user's accepted facts (slots and per-workflow
authority facts) with the internal TrustAI assessment policy (autonomy L0-L5, impact I0-I3, posture P1-P4; control identifiers C01-C21) and a versioned
decision table. Retrieved passages and the LLM cannot change it. Every step is recorded in the result `trace` with an internal rule ID (POL-...) and the
policy version, so that answer -> accepted fact -> rule -> result can be shown in the report. The policy is internal: it is never a retrievable source.

Fact state (single source of truth): `derive()` folds only ACCEPTED facts (a verbatim user quote, or a selected option). Rejected extractions live in the
turn record for diagnostics and never enter this state. Derived facts (for example the authority ceiling over several workflows) carry `derived_from`
with the accepted input facts they were computed from."""
from __future__ import annotations

import json
import re
from functools import lru_cache

import config


@lru_cache(maxsize=1)
def policy() -> dict:
    return json.loads((config.CATALOG_DIR / "policy.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def interview_cfg() -> dict:
    return json.loads((config.CATALOG_DIR / "interview.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def controls_cfg() -> dict:
    return json.loads((config.CATALOG_DIR / "controls.json").read_text(encoding="utf-8"))


TOPICS = lambda: {t["id"]: t for t in interview_cfg()["topics"]}  # noqa: E731
STATUS_LABEL = {"declared_present": "Declared present", "declared_absent": "Declared absent", "partial": "Partially assessed",
                "unknown": "Unknown", "not_assessed": "Not assessed", "not_applicable": "Not applicable"}
ORDER = ["LOW", "HIGH"]


COUNTED_KINDS = ("question", "followup", "conflict")  # answered assessment questions


def is_counted(t: dict) -> bool:
    return t.get("answer") is not None and t["question"].get("kind") in COUNTED_KINDS


def answered_count(state: dict) -> int:
    return sum(1 for t in state["turns"] if is_counted(t))


def _definite(v) -> bool:
    return v not in (None, "unknown")


# ----------------------------------------------------------------------- per-workflow authority facts
TOOL_PERMISSIONS = ("read", "write", "modify", "delete", "send", "execute", "transact")
WF_FIELDS = ("action", "condition", "threshold", "recipient", "execution", "approval_authority", "approval_bound", "limit_enforcement", "recipient_restricted")
WF_RANK = {"no_action": 0, "recommend": 1, "draft": 2, "executes_after_approval": 3, "executes_without_approval": 4}
WF_MODE = {"no_action": "info_only", "recommend": "recommend", "draft": "draft", "executes_after_approval": "approved_action", "executes_without_approval": "envelope_action"}
WF_EXEC_LABEL = {"no_action": "only provides information", "recommend": "recommends (a person decides and acts)", "draft": "prepares a draft (a person executes it)",
                 "executes_after_approval": "executes after approval", "executes_without_approval": "executes without approval for each action"}
WF_CONFLICT_FIELDS = ("execution", "limit_enforcement")
WF_AUTHORITY_SLOTS = ("action_mode", "approval_exists", "approval_bound", "limits_enforced")


def wf_definite(v) -> bool:
    return v not in (None, "", "unknown")


def wf_label(w: dict) -> str:
    """Short human label of a workflow: action (condition)."""
    cond = (w.get("condition") or "").strip()
    return f"{(w.get('action') or w['id']).strip()}" + (f" ({cond})" if cond else "")


WF_OPTION = {"execution": {"executes_without_approval": "It runs without approval for each action", "executes_after_approval": "It runs only after approval",
                           "recommend": "It only recommends; a person decides and acts", "draft": "It only prepares drafts; a person executes them",
                           "no_action": "It only provides information"},
             "limit_enforcement": {"technical": "The system itself enforces the limit", "policy_only": "The limit exists only in instructions or policy", "none": "There is no limit"}}


def wf_conflict(cur: dict, wid: str, field: str, old: str, new: str, old_src, new_src) -> dict:
    """A GENUINE contradiction: the same workflow (same action and condition) was described two different ways. The question names the workflow and its scope."""
    lo, ln = WF_OPTION[field].get(old, old), WF_OPTION[field].get(new, new)
    what = "what it can do" if field == "execution" else "how its limit is enforced"
    return {"old": old, "new": new, "old_src": old_src, "new_src": new_src, "old_slot": f"wf:{wid}:{field}", "workflow": wid, "field": field,
            "question": f"You described {what} for \u201c{wf_label(cur)}\u201d in two different ways. Which one is correct?",
            "resolve": {lo: {f"wf:{wid}:{field}": old}, ln: {f"wf:{wid}:{field}": new}, "I don't know": {}}}


def _derive_authority(wfs: dict, slots: dict, src: dict, ev: dict, derived: dict, derived_from: dict) -> None:
    """Authority is evaluated per workflow; the maximum declared reachable authority over the material workflows defines the autonomy input (rule POL-WF-MAX).
    The scalar facts (action_mode, approval_exists, approval_bound, limits_enforced) become DERIVED facts that list the accepted workflow facts they come from."""
    known = [w for w in wfs.values() if wf_definite(w.get("execution"))]
    if not known:
        return
    ceiling = max(known, key=lambda w: WF_RANK[w["execution"]])
    mode = WF_MODE[ceiling["execution"]]
    inputs = [{"workflow": w["id"], "label": wf_label(w), "answer_id": w["fields_src"].get("execution"), "quote": (w.get("quotes") or {}).get("execution", ""),
               "execution": w["execution"], "role": "ceiling" if w is ceiling else "considered"} for w in sorted(known, key=lambda w: -WF_RANK[w["execution"]])]

    def put(slot, value, why):
        slots[slot], src[slot], derived[slot] = value, ceiling["fields_src"].get("execution"), why
        ev[slot] = (ceiling.get("quotes") or {}).get("execution", "")
        derived_from[slot] = {"rule": "POL-WF-MAX", "why": why, "inputs": inputs}
    put("action_mode", mode, f"Maximum declared reachable authority over {len(known)} workflow(s): {wf_label(ceiling)} {WF_EXEC_LABEL[ceiling['execution']]}.")
    approvals = [w for w in known if w["execution"] == "executes_after_approval"]
    if mode == "envelope_action":
        put("approval_exists", "partial" if approvals else "no", "Derived from the workflow facts: " + ("some workflows need approval, the ceiling workflow does not." if approvals else "no workflow needs approval."))
        bound = {w.get("approval_bound") for w in approvals}
        if approvals and "no" in bound:
            put("approval_bound", "no", "An approval workflow was declared not to be bound to the specific action.")
        elif approvals and bound == {"yes"}:
            put("approval_bound", "yes", "Every approval workflow was declared to be bound to the specific action.")
    elif mode == "approved_action":
        put("approval_exists", "yes", "Every material workflow executes only after approval.")
        if wf_definite(ceiling.get("approval_bound")):
            put("approval_bound", ceiling["approval_bound"], f"Declared for {wf_label(ceiling)}.")
    enforced = ceiling.get("limit_enforcement")
    if mode in ("envelope_action", "approved_action") and wf_definite(enforced):
        slots["limits_enforced"], src["limits_enforced"] = enforced, ceiling["fields_src"].get("limit_enforcement")
        ev["limits_enforced"] = (ceiling.get("quotes") or {}).get("limit_enforcement", "")
        derived_from["limits_enforced"] = {"rule": "POL-WF-MAX", "why": f"Declared for {wf_label(ceiling)}.",
                                           "inputs": [{"workflow": ceiling["id"], "label": wf_label(ceiling), "answer_id": src["limits_enforced"], "quote": ev["limits_enforced"], "execution": ceiling["execution"], "role": "ceiling"}]}


# ----------------------------------------------------------------------- derivation
def derive(state: dict) -> dict:
    """Sequential fold over the turns.
    - 'unknown' NEVER overwrites a definite statement.
    - If a different definite value arrives (except p_* refinements), a conflict is recorded and only that item is asked.
    - Equivalent/linked fields are closed by a single answer (equivalences); skipped branches are kept in derived together with their reasons."""
    cfg = interview_cfg()
    spec = cfg["slots"]
    slots, src, controls, csrc, basis, ev = {}, {}, {}, {}, {}, {}
    asked, followups, conflict_asked, conflicts, unresolved_conflicts = {}, {}, {}, {}, {}
    wfs, wf_conflicts, derived_from, tools = {}, {}, {}, {}
    st_eu = (state.get("settings") or {}).get("eu_link", "unknown")
    if st_eu in ("yes", "no"):  # assessment setting
        slots["eu_link"], src["eu_link"] = st_eu, "SETTINGS"
    for t in state["turns"]:
        if t.get("answer") is None:
            continue
        aid, q = t["id"], t["question"]
        kind = q.get("kind")
        u = t.get("updates") or {}
        for k, v in (u.get("slots") or {}).items():
            old = slots.get(k)
            if v == "unknown" and _definite(old):
                continue
            if (kind != "conflict" and spec.get(k) != "text" and not k.startswith("p_")
                    and _definite(old) and _definite(v) and v != old):
                conflicts[k] = {"old": old, "new": v, "old_src": src[k], "new_src": aid, "old_slot": k}
            slots[k], src[k] = v, aid
            quote = (u.get("evidence") or {}).get(k) or ((u.get("choice") or {}).get("label") if k in ((u.get("choice") or {}).get("set") or {}) else None)
            if quote:
                ev[k] = quote
            elif v != "unknown":
                ev.pop(k, None)
        for item in u.get("tools") or []:
            name = (item.get("name") or "").strip()
            if not name:
                continue
            old = tools.setdefault(name.casefold(), {"name": name, "kind": "unknown", "permissions": [], "denied_permissions": [], "scope": "", "evidence": [], "answer_ids": []})
            if item.get("kind") and item["kind"] != "unknown":
                old["kind"] = item["kind"]
            if item.get("scope"):
                old["scope"] = item["scope"]
            # Per permission: confirmed (declared), denied (explicitly excluded, e.g. 'read-only' or 'revoked') or unknown (not stated). A later statement replaces an
            # earlier one for the same permission (correction, revocation). Within ONE statement a permission that is both denied and confirmed stays confirmed
            # (the cautious reading: the contradiction is not resolved in favour of the lower authority).
            confirmed = set(item.get("permissions") or []) & set(TOOL_PERMISSIONS)
            denied = (set(item.get("denied_permissions") or []) & set(TOOL_PERMISSIONS)) - confirmed
            old["permissions"] = sorted((set(old["permissions"]) - denied) | confirmed)
            old["denied_permissions"] = sorted((set(old["denied_permissions"]) - confirmed) | denied)
            if item.get("evidence") and item["evidence"] not in old["evidence"]:
                old["evidence"].append(item["evidence"])
            if aid not in old["answer_ids"]:
                old["answer_ids"].append(aid)
        for w in u.get("workflows") or []:
            wid, cur = w["id"], wfs.get(w["id"])
            if cur is None:
                cur = wfs[wid] = {"id": wid, "fields_src": {}, "quotes": {}}
            for f in WF_FIELDS:
                nv = w.get(f)
                if not wf_definite(nv):
                    continue
                ov = cur.get(f)
                if kind != "conflict" and f in WF_CONFLICT_FIELDS and wf_definite(ov) and ov != nv:
                    wf_conflicts[f"wf:{wid}:{f}"] = wf_conflict(cur, wid, f, ov, nv, cur["fields_src"].get(f), aid)
                cur[f] = nv
                cur["fields_src"][f] = aid
                if w.get("evidence"):
                    cur["quotes"][f] = w["evidence"]
        if kind == "conflict" and str(q.get("fact", "")).startswith("wf:"):
            f = q["fact"]
            conflict_asked[f] = conflict_asked.get(f, 0) + 1
            wf_conflicts.pop(f, None)
            _, wid, field = f.split(":", 2)
            if any(w["id"] == wid and wf_definite(w.get(field)) for w in (u.get("workflows") or [])):
                unresolved_conflicts.pop(f, None)
            else:
                unresolved_conflicts[f] = aid  # not clarified: the authority stays undetermined instead of being guessed
        elif kind == "conflict" and q.get("fact"):
            f = q["fact"]
            conflict_asked[f] = conflict_asked.get(f, 0) + 1
            conflicts.pop(f, None)
            if not _definite((u.get("slots") or {}).get(f)):
                slots[f], src[f] = "unknown", aid  # conflict could not be resolved: recorded as unknown
                unresolved_conflicts[f] = aid     # unresolved conflict: no certain finding is produced
            else:
                unresolved_conflicts.pop(f, None)
        for cid, cv in (u.get("controls") or {}).items():
            old = controls.get(cid)
            if cv.get("status") == "unknown" and old and old.get("status") in ("declared_present", "declared_absent"):
                continue  # 'unknown' does not overwrite a definite statement
            controls[cid], csrc[cid] = cv, aid
        for b in u.get("basis") or []:
            basis[b["key"]] = {**b, "answer_id": aid}
        if kind == "followup" and q.get("fact"):
            followups[q["fact"]] = followups.get(q["fact"], 0) + 1
        if kind in ("question", "followup") and q.get("topic") in TOPICS():
            asked[q["topic"]] = asked.get(q["topic"], 0) + 1
            top = TOPICS()[q["topic"]]
            for sl in required_slots(top):
                if sl not in slots:  # asked but nothing could be extracted: unknown
                    slots[sl], src[sl] = "unknown", aid
            for cid in ([] if top["slots"] else top["controls"]):
                if cid not in controls and cid not in ("C01", "C02"):
                    controls[cid], csrc[cid] = {"status": "unknown", "note": ""}, aid
    derived = {}
    conflicts.update(wf_conflicts)
    if wfs:
        for sl in WF_AUTHORITY_SLOTS:  # authority is evaluated per workflow: unscoped scalar statements neither conflict with nor override scoped facts
            conflicts.pop(sl, None)
            unresolved_conflicts.pop(sl, None)
        if not any(k.startswith("wf:") for k in (set(conflicts) | set(unresolved_conflicts))):
            _derive_authority(wfs, slots, src, ev, derived, derived_from)
    if any(w.get("threshold") for w in wfs.values()):
        first_wf = next(w for w in wfs.values() if w.get("threshold"))
        slots["wf_threshold"], src["wf_threshold"], derived["wf_threshold"] = "yes", first_wf["fields_src"].get("execution") or "A1", "A workflow with an amount or limit was declared."
    for r in cfg.get("equivalences", []):
        if all(slots.get(k) == v for k, v in r["if"].items()):
            for k, v in r["then"].items():
                if slots.get(k) is None:
                    first = next(iter(r["if"]))
                    slots[k], src[k], derived[k] = v, src[first], r["reason"]
    for gid, g in cfg.get("prohibition_groups", {}).items():
        # if one precondition is explicitly 'no', the sub-prohibition screens are skipped; the reason is recorded.
        if any(slots.get(k) == "no" for k in g["requires"]):
            for dep in g["deps"]:
                if slots.get(dep) is None:
                    slots[dep], src[dep], derived[dep] = "none", src[next(k for k in g["requires"] if slots.get(k) == "no")], g["skip"]
    for r in cfg.get("exclusions", []):
        if wfs and (set(r["if"]) | set(r["excludes"])) & set(WF_AUTHORITY_SLOTS):
            continue  # scoped (per-workflow) facts exist: a conditional approval structure is not a contradiction
        if all(slots.get(k) == v for k, v in r["if"].items()):
            for k, bad in r["excludes"].items():
                if slots.get(k) in bad:
                    ifk = next(iter(r["if"]))
                    if conflict_asked.get(k, 0) == 0:
                        conflicts[k] = {"old": slots[ifk], "new": slots[k], "old_src": src[ifk], "new_src": src[k], "old_slot": ifk,
                                        "question": r.get("question"), "resolve": r.get("resolve")}
                    else:  # asked and still conflicting
                        unresolved_conflicts[k] = src.get(k)
    for x in tools.values():  # confirmed / denied / unknown are three different states; 'unknown' is never read as 'denied'
        x["unknown_permissions"] = [p for p in TOOL_PERMISSIONS if p not in x["permissions"] and p not in x["denied_permissions"]]
    if tools:
        names = ", ".join(x["name"] for x in tools.values())
        slots["tools_summary"] = names
        src["tools_summary"] = list(tools.values())[-1]["answer_ids"][-1]
        derived["tools_summary"] = "Named tools and systems recorded with per-item permissions; unspecified permissions remain unknown."
    def _decl(cid, slot_names, yes, no):
        vals = [(k, slots.get(k)) for k in slot_names if slots.get(k) is not None]
        if cid in controls or not vals:
            return
        k, v = vals[0]
        if v in yes:
            controls[cid], csrc[cid] = {"status": "declared_present", "note": "declared"}, src.get(k)
        elif v in no:
            controls[cid], csrc[cid] = {"status": "declared_absent", "note": "declared absent"}, src.get(k)
        else:  # partial / unknown: recorded as unknown
            controls[cid], csrc[cid] = {"status": "unknown", "note": "partial or unknown"}, src.get(k)
    if not (derived_from.get("action_mode") and slots.get("action_mode") == "envelope_action"):  # approvals on some workflows do not evidence oversight of the autonomous ones
        _decl("C09", ["approval_bound", "human_review"], ("yes",), ("no",))
    _decl("C11", ["limits_enforced"], ("technical",), ("policy_only", "none"))
    return {"slots": slots, "slots_src": src, "controls": controls, "controls_src": csrc, "basis": basis, "asked": asked,
            "derived": derived, "derived_from": derived_from, "workflows": wfs, "tools": list(tools.values()), "evidence": ev, "followups": followups, "conflicts": conflicts,
            "conflict_asked": conflict_asked, "unresolved_conflicts": unresolved_conflicts}


def control_gaps(d: dict, legal_code: str) -> dict:
    """Controls relevant to the result and their states: {cid: {"why","state"}}; state: addressed | unknown | unasked.
    An unasked control is not treated as 'not relevant'; 'unasked' lowers result sufficiency, 'unknown' is listed as a critical gap."""
    out, s = {}, d["slots"]
    ca = control_assessment(d, legal_code)  # the same fact-to-control mapping the report shows
    for r in controls_cfg().get("result_relevance", []):
        if "when_legal" in r:
            ok = legal_code in r["when_legal"]
        elif "when_any" in r:
            ok = any(s.get(k) in v for k, v in r["when_any"].items())
        else:
            ok = all(s.get(k) in v for k, v in r["when"].items())
        if not ok:
            continue
        for cid in r["controls"]:
            st = ca[cid]["status"]
            state = "unasked" if st == "not_assessed" else "unknown" if st == "unknown" else "addressed"  # partial = information was collected, it is not an open gap
            out[cid] = {"why": r["why"], "state": state}
    return out


def control_scope(d: dict, legal_code: str) -> dict:
    """Reasoned 'not relevant for this use' scope rules (from the catalog) for controls that have not been declared."""
    out, s = {}, d["slots"]
    for r in controls_cfg().get("scope_rules", []):
        if "when_legal" in r:
            ok = legal_code in r["when_legal"]
        else:
            tests = [s.get(k) in v for k, v in r["when"].items()]
            ok = all(tests) if r.get("all") else any(tests) if len(tests) == 1 else all(tests)
        if ok:
            for cid in r["controls"]:
                if d["controls"].get(cid, {}).get("status") in (None, "unknown"):
                    out[cid] = {"status": "not_relevant", "note": r["reason"]}
    return out


# ----------------------------------------------------------------------- C01-C21 assessment: explicit fact -> control mapping (catalogs/controls.json "assessment_rules")
def _req_state(r: dict, d: dict) -> tuple[str, list]:
    """State of ONE requirement of a control: present | absent | partial | unknown | missing, plus the accepted facts it rests on."""
    s = d["slots"]
    if r.get("decl"):
        st = d["controls"].get(r["decl"], {}).get("status")
        basis = [{"slot": "declaration " + r["decl"], "value": st, "answer_id": d["controls_src"].get(r["decl"])}] if st else []
        return {"declared_present": "present", "declared_absent": "absent", "unknown": "unknown"}.get(st, "missing"), basis
    names = policy()["prohibition_refs"] if r.get("slots") == "PROHIBITION_REFS" else [r["slot"]]
    vals = [s.get(n) for n in names]
    basis = [{"slot": n, "value": v, "answer_id": d["slots_src"].get(n)} for n, v in zip(names, vals) if v is not None]
    if any(v is None for v in vals):
        return "missing", basis
    if r.get("slots"):  # prohibited-use screening: every prohibition screen must be 'none'
        if any(v == "supported" for v in vals):
            return "absent", basis
        if any(v in ("possible", "excepted") for v in vals):
            return "partial", basis
        return ("unknown" if any(v == "unknown" for v in vals) else "present"), basis
    v = vals[0]
    pres = r.get("present")
    if pres == "set":
        return ("present" if v else "missing"), basis
    if v == "unknown":
        return "unknown", basis
    if pres == "*" or v in (pres or []):
        return "present", basis
    if v in r.get("absent", []):
        return "absent", basis
    if v in r.get("partial", []):
        return "partial", basis
    return "missing", basis


def _applies(r: dict, d: dict) -> bool:
    return all(d["slots"].get(k) in v for k, v in (r.get("applies") or {}).items())


def control_assessment(d: dict, legal_code: str) -> dict:
    """Declaration-based status of every control C01-C21 from the accepted facts: {cid: {status, finding, missing, basis, rule, parts}}.
    status: declared_present | declared_absent | partial | unknown | not_assessed | not_applicable. A control with several requirements is evaluated on all of them: it is
    'declared present' only when every applicable requirement is present, and mixed results are 'partial' (e.g. a review after the fact does not establish oversight when an
    immediate stop is unavailable). Nothing here verifies a safeguard; 'present' means 'declared'."""
    cfg = controls_cfg()
    scope = control_scope(d, legal_code)
    out = {}
    for c in cfg["controls"]:
        cid = c["id"]
        parts, basis = [], []
        for r in cfg["assessment_rules"].get(cid, []):
            if not _applies(r, d):
                continue
            st, b = _req_state(r, d)
            parts.append({"req": r["id"], "label": r["label"], "state": st, "say": r["say"].get(st, "")})
            basis += b
        known = [p for p in parts if p["state"] != "missing"]
        states = {p["state"] for p in known}
        missing = [p["label"] for p in parts if p["state"] == "missing"]
        if cid in scope:  # the scope rule's own reason (explicit declarations) applies; it only exists when no explicit control declaration was made
            status, finding = "not_applicable", scope[cid]["note"]
            short = next((r.get("short") for r in cfg["scope_rules"] if cid in r["controls"]), "")
            finding = short or finding
        elif not known:
            status = "not_assessed"
            finding = "Not collected in this interview. Needed: " + (", ".join(missing) if missing else c["info_needed"].rstrip(".").lower()) + "."
            if cid == "C20" and legal_code in ("partly_assessed", "undetermined"):  # name the legal facts that are open, not the downstream activity
                s_ = d["slots"]
                missing = ([] if s_.get("annex3_area") not in (None, "unknown") else ["sensitive-use area (Annex III)"]) + \
                          ([] if s_.get("annex1_product") == "no" or (s_.get("annex1_product") == "yes" and s_.get("annex1_third_party") == "no") else ["regulated-product route (Annex I)"])
                finding = "Not assessed: the legal high-risk pathway is not fully established" + (". Needed: " + ", ".join(missing) if missing else "") + "."
        elif states == {"unknown"}:
            status = "unknown"
            finding = "You were not sure: " + "; ".join(p["label"] for p in known) + "."
        elif states == {"present"} and not missing:
            status, finding = "declared_present", "; ".join(p["say"] for p in known if p["say"])
        elif states == {"absent"} and not missing:
            status, finding = "declared_absent", "; ".join(p["say"] for p in known if p["say"])
        else:
            status = "partial"
            finding = "; ".join(p["say"] for p in known if p["say"] and p["state"] != "unknown")
            unk = [p["label"] for p in known if p["state"] == "unknown"]
            if unk:
                finding += ("; " if finding else "") + "unsure: " + ", ".join(unk)
        if finding and status != "not_assessed":
            finding = finding[0].upper() + finding[1:]
            if status == "partial" and missing:
                finding += ". Not yet described: " + ", ".join(missing)
            if not finding.endswith("."):
                finding += "."
        out[cid] = {"status": status, "finding": finding, "missing": missing if status in ("partial", "not_assessed", "unknown") else [],
                    "basis": basis, "rule": f"CTL-{cid}", "parts": parts}
    return out


def group_of(dep: str) -> str | None:
    for gid, g in interview_cfg().get("prohibition_groups", {}).items():
        if dep in g["deps"]:
            return gid
    return None


def preconditions_met(slots: dict, gid: str) -> bool:
    """For a detail branch, ALL preconditions must be 'yes' by user declaration."""
    return all(slots.get(k) == v for k, v in interview_cfg()["prohibition_groups"][gid]["requires"].items())


def detail_groups(slots: dict) -> list[str]:
    """Prohibition groups that need a detail question: all preconditions are 'yes' and the sub-field is unresolved."""
    return [gid for gid, g in interview_cfg().get("prohibition_groups", {}).items()
            if preconditions_met(slots, gid) and any(slots.get(x) in (None, "possible") for x in g["deps"])]


def topic_active(top: dict, slots: dict) -> bool:
    c = top.get("condition")
    if not c:
        return True
    eu_out = slots.get("eu_link") == "no"
    mode = slots.get("action_mode")
    if c == "harm_known_no_rollback":  # the oversight checklist already answers whether a wrong action can be reversed
        return slots.get("max_harm") in ("limited", "serious", "critical", "unknown") and slots.get("can_rollback") not in ("yes", "no")
    if c == "has_threshold":
        return slots.get("wf_threshold") == "yes"
    if c == "approval_exists_yes":
        return slots.get("approval_exists") == "yes"
    if c == "person_eval_yes":
        return not eu_out and slots.get("g_person_eval") == "yes"
    if c == "annex3_area_set":
        return not eu_out and slots.get("annex3_area") not in (None, "none", "unknown")
    if c == "acts":
        return mode in ("approved_action", "envelope_action", "self_expanding", "unknown", None)
    if c == "acts_or_reads":
        return mode != "info_only"
    if c == "recommends":
        return mode in ("recommend", "draft")
    if c == "harm_known":
        return slots.get("max_harm") in ("limited", "serious", "critical", "unknown")
    if c == "personal_data_set":
        return slots.get("personal_data") in ("ordinary", "special", "unknown")
    if c == "synthetic_yes":
        return slots.get("t_synthetic_content") == "yes"
    if c == "legal_high_or_unknown":
        return not eu_out and slots.get("annex3_area") not in (None, "none")
    return True


def required_slots(top: dict) -> list[str]:
    """Slots required to resolve a topic: free-text fields (summary/description) are optional, except 'purpose'."""
    spec = interview_cfg()["slots"]
    return [sl for sl in top["slots"] if spec[sl] != "text" or sl == "purpose"]


def option_sets(top: dict) -> dict:
    """Option label → {slot: value}. A choice is stored together with the answered field and value (the label is not left to re-interpretation).
    Without an explicit mapping (option_map), yes/no/unknown questions with a single required slot are mapped automatically."""
    if top.get("option_map"):
        return top["option_map"]
    req = required_slots(top)
    spec = interview_cfg()["slots"]
    if len(req) == 1 and spec[req[0]] == ["yes", "no", "unknown"]:
        return {"Yes": {req[0]: "yes"}, "No": {req[0]: "no"}, "I don't know": {req[0]: "unknown"}}
    return {}


def control_option_sets(top: dict) -> dict:
    """Option label → {control: status} for topics without slots. Labels starting with 'Yes/No/I don\'t know/Partly' are recorded deterministically."""
    if top["slots"] or not top.get("options"):
        return {}
    out = {}
    for o in top["options"]:
        low = o.strip().lower()
        st, note = (("declared_present", "declared") if low.startswith("yes") and not low.startswith("yes, but") else ("declared_absent", "declared absent") if low.startswith("no") and not low.startswith("no,")
                    else ("unknown", "") if low.startswith("i don't know") else ("unknown", "partly / unclear") if low.startswith("partly") else (None, None))
        if st:
            out[o] = {cid: {"status": st, "note": note} for cid in top["controls"] if cid not in ("C01", "C02")}
    return out


def topic_resolved(top: dict, d: dict) -> bool:
    n = d["asked"].get(top["id"], 0)
    if top["id"] == "prohibited_detail":
        return (not topic_active(top, d["slots"])) or n >= 2
    req = required_slots(top)
    if req:
        return all(d["slots"].get(sl) is not None for sl in req)
    return n >= 1


OPTIMISTIC = {"approval_exists_yes", "person_eval_yes", "annex3_area_set", "acts", "acts_or_reads", "recommends", "harm_known", "personal_data_set",
              "synthetic_yes", "legal_high_or_unknown"}


def open_topics(state: dict, optimistic: bool = False) -> list[dict]:
    """Open and valid topics in order of priority. optimistic=True: topics whose condition may change with this answer
    before it is processed are also treated as candidates."""
    d = derive(state)
    out = []
    for top in interview_cfg()["topics"]:
        if topic_resolved(top, d):
            continue
        if topic_active(top, d["slots"]) or (optimistic and top.get("condition") in OPTIMISTIC and d["slots"].get("eu_link") != "no"):
            out.append(top)
    return sorted(out, key=lambda t: -t.get("priority", 0))


# ----------------------------------------------------------------------- classification
POSTURES = ["P1", "P2", "P3", "P4"]
LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5"]
IMPACTS = ["I0", "I1", "I2", "I3"]
ACTING_MODES = ("approved_action", "envelope_action", "self_expanding")


def forbidden_findings(d: dict) -> list[dict]:
    """Only if the conditions of the prohibition are supported (with user-quoted basis) and the exception/scope/date assessment is supported."""
    pol = policy()["prohibition_refs"]
    out = []
    if d["slots"].get("eu_link") == "no":
        return out  # it is declared that there is no EU link
    for k, (ref, name) in pol.items():
        gid = group_of(k)
        if d["slots"].get(k) == "supported" and gid and preconditions_met(d["slots"], gid):
            b = d["basis"].get(k, {})
            if b.get("conditions_confirmed") and b.get("exception_considered") and b.get("scope_date_considered"):
                out.append({"key": k, "ref": ref, "name": name, "answer_id": b["answer_id"],
                            "slot_answer_id": d["slots_src"].get(k)})
    return out


AUTONOMY_SLOTS = {"action_mode", "approval_exists", "approval_bound", "limits_enforced", "authority_expansion"}


def autonomy_conflict(d: dict) -> bool:
    """Is there an unresolved (unasked or unanswered) conflict in the approval/autonomy fields?"""
    keys = set(d["conflicts"]) | set(d["unresolved_conflicts"])
    return bool(keys & AUTONOMY_SLOTS) or any(k.startswith("wf:") for k in keys)


def _inp(d: dict, *slots) -> list[dict]:
    """Inputs of a rule: slot, value, answer id and the user's quote. A DERIVED fact is expanded into the accepted workflow facts it was computed from."""
    out = []
    for k in slots:
        if d["slots"].get(k) is None:
            continue
        df = d.get("derived_from", {}).get(k)
        if df:
            out += [{"slot": k, "value": d["slots"].get(k), "answer_id": i["answer_id"], "quote": i["quote"], "workflow": i["label"], "role": i["role"],
                     "execution": WF_EXEC_LABEL[i["execution"]]} for i in df["inputs"]]
        else:
            out.append({"slot": k, "value": d["slots"].get(k), "answer_id": d["slots_src"].get(k), "quote": d["evidence"].get(k)})
    return out


def escalation_factors(d: dict, L: str | None, I: str | None) -> list[dict]:
    """Exposure factors that raise the posture by ONE step (internal policy rule POL-ESC: record the reason; no additive weights)."""
    s, out = d["slots"], []
    if I in (None, "I0"):
        return out
    if s.get("reach") == "wide":
        out.append({"slot": "reach", "reason": "wide reach"})
    if s.get("reversibility") == "irreversible":
        out.append({"slot": "reversibility", "reason": "harm cannot be reversed"})
    if s.get("vulnerable_groups") == "yes":
        out.append({"slot": "vulnerable_groups", "reason": "vulnerable groups are affected"})
    if s.get("personal_data") == "special":
        out.append({"slot": "personal_data", "reason": "sensitive personal data is used"})
    if L in ("L4", "L5") and s.get("limits_enforced") in ("policy_only", "none"):
        out.append({"slot": "limits_enforced", "reason": "authority limits are not technically enforced (policy-only or absent)"})
    return out


def posture_for(I: str, L: str, escalate: bool) -> tuple[str, str]:
    """(base posture from the policy decision table, final posture after at most one escalation step). The posture is never lowered."""
    base = policy()["posture_matrix"][I][L]
    final = POSTURES[min(POSTURES.index(base) + 1, 3)] if escalate else base
    return base, final


# ----------------------------------------------------------------------- impact floors from the user's own description of the harm
_NEG = re.compile(r"\b(no|not|never|none|cannot|can't|cant|without|unlikely|nobody|no one|neither|nor|impossible|rule[sd]? out)\b[^.;]{0,30}$", re.I)
_I3_PATTERNS = (
    (r"\b(death|deaths|die|dies|died|dying|fatal\w*|kill\w*|life[- ]threatening|serious(?:ly)? (?:injur\w+|harm\w*|hurt)|physical (?:harm|injur\w+)|injur(?:y|ies|ed)|endanger\w*)\b",
     "POL-I3-SAFETY", "The declared harm concerns the safety of people (death, injury or danger to life), which is severe harm that reversibility alone cannot lower."),
    (r"\b(wrongful(?:ly)? (?:arrest\w*|detain\w*|detention|convict\w*|deport\w*)|deportation|imprison\w*|denied (?:asylum|bail)|loss of (?:liberty|freedom))\b",
     "POL-I3-RIGHTS", "The declared harm concerns fundamental rights (liberty, asylum, due process), which is severe harm."),
    (r"\b(evict\w*|homeless\w*|lose (?:their |his |her )?(?:home|housing|livelihood)|loss of (?:livelihood|housing|income)|"
     r"denied (?:essential|emergency|medical|health ?care|treatment|social benefits?|benefits?|food|water|electricity|heating)|"
     r"(?:fired|dismissed|laid off|terminated|lose (?:their |his |her )?(?:current )?jobs?)(?!.{0,40}(?:opportunit|application|candidate|applicant|hiring))|bankrupt\w*|insolven\w*)\b",
     "POL-I3-LIVELIHOOD", "The declared harm concerns livelihood or essential services (home, income, existing employment, essential care or benefits), which is severe harm."),
)
_I2_PATTERNS = (
    (r"(?:lose|lost|denied|miss|depriv\w*|rejected|excluded|exclusion|loss of).{0,75}(?:job|employment|hiring|work opportunity|career opportunity)|(?:qualified candidates?|applicants?).{0,65}(?:lose|lost|denied|rejected|excluded|miss)",
     "POL-I2-EMPLOYMENT", "Credible loss of an employment opportunity is material livelihood/fairness harm, not limited operational inconvenience."),
    (r"\b(discriminat\w*|biased?|unfair(?:ly)?|unjust(?:ly)?|privacy (?:breach\w*|violation\w*|leak\w*)|data (?:breach\w*|leak\w*)|(?:leak|expos|disclos)\w* (?:of )?(?:personal|confidential|private|sensitive) (?:data|information)|"
     r"wrong(?:ly)? (?:charged|billed|denied|rejected|refused)|denied (?:a )?(?:loan|credit|mortgage|insurance|service|access)|loss of (?:service|access)|legal (?:claims?|action|liability))\b",
     "POL-I2-MATERIAL", "The declared harm concerns material fairness, privacy, service or rights consequences for people."),
)


def _hits(patterns, text: str):
    out = []
    for pat, rid, why in patterns:
        for m in re.finditer(pat, text, re.I):
            if not _NEG.search(text[max(0, m.start() - 40):m.start()]):  # 'nobody could die' is not a death scenario
                out.append((rid, why))
                break
    return out


def harm_floors(desc: str) -> list[tuple[str, str, str]]:
    """Minimum impact classes implied by what the USER wrote about the harm: [(class, rule id, reason)], highest first. The user's own selected class can be higher; it is never lowered."""
    text = (desc or "").lower()
    out = [("I3", rid, why) for rid, why in _hits(_I3_PATTERNS, text)] + [("I2", rid, why) for rid, why in _hits(_I2_PATTERNS, text)]
    return out


def operational(d: dict) -> dict:
    """Autonomy level L, impact level I, posture P and risk level from the declared facts, with a step-by-step trace."""
    s, P = d["slots"], policy()
    trace, notes = [], []
    mode = s.get("action_mode")
    L = None
    floor_L = None  # lower bound of the autonomy level when the level is undetermined only because part of the authority is not established
    if autonomy_conflict(d):
        notes.append("Approval/autonomy declarations conflict; no certain autonomy level was produced.")
        trace.append({"step": "Autonomy level (L)", "inputs": _inp(d, "action_mode", "approval_exists", "approval_bound"),
                      "rule": "Conflicting approval/autonomy declarations are not resolved by guessing.", "result": "undetermined (conflict)"})
    else:
        L = P["autonomy_map"].get(mode)
        enf = s.get("limits_enforced")
        trace.append({"step": "Autonomy level (L)", "inputs": _inp(d, "action_mode", *(["limits_enforced"] if mode in ("envelope_action", "approved_action") else [])),
                      "rule_id": f"POL-{L}" if L else "POL-L", "rule": (f"Declared action authority '{mode}' maps to {L} (" + P["autonomy_notes"].get(L, "") + ")" if L else "No action authority was declared, so no autonomy level is derived (undetermined, not assumed)")
                      + (" The authority is the maximum declared reachable authority over the material workflows." if d.get("derived_from", {}).get("action_mode") else "")
                      + (" The limit was declared to be technically enforced." if mode in ("envelope_action", "approved_action") and enf == "technical" else
                         " The limit was declared to exist only in instructions or policy; this escalates the posture (POL-ESC)." if mode in ("envelope_action", "approved_action") and enf in ("policy_only", "none") else
                         " Technical enforcement of the limit has not been established." if mode == "envelope_action" else ""), "result": L or "undetermined"})
        if L == "L3" and s.get("approval_bound") != "yes":
            L = None
            msg = "Specific-action approval was not established. L3 cannot be confirmed and L4 cannot be inferred from missing evidence."
            notes.append(msg)
            trace.append({"step": "Autonomy uncertainty", "rule_id": "POL-L3-BIND", "inputs": _inp(d, "approval_bound"), "rule": msg, "result": "undetermined"})
        unknown_wf = [w for w in d.get("workflows", {}).values() if (w.get("action") or "").strip() and not wf_definite(w.get("execution"))]
        if L in ("L0", "L1", "L2", "L3") and unknown_wf:
            floor_L = L
            msg = ("What the agent itself does was not established for: " + "; ".join(wf_label(w) for w in unknown_wf)
                   + f". The declared ceiling ({L}) is a lower bound only; the authority level is undetermined, not assumed to be {L}.")
            L = None
            notes.append(msg)
            trace.append({"step": "Autonomy uncertainty", "rule_id": "POL-WF-UNKNOWN", "inputs": [], "rule": msg, "result": "undetermined"})
    if s.get("authority_expansion") == "yes" and "authority_expansion" not in (set(d["conflicts"]) | set(d["unresolved_conflicts"])) and L != "L5":
        # L5 is defined by the ability to select or expand execution scope or delegated authority without approval for each expansion. The maximum technically
        # reachable authority counts, whatever the current mode is (a recommend-only agent that can grant itself new permissions can reach L5).
        L, floor_L = "L5", None
        msg = "The agent can expand its own authority, which defines L5 in the internal policy (maximum reachable authority, whatever its current mode)."
        notes.append(msg)
        trace.append({"step": "Autonomy adjustment", "rule_id": "POL-L5-EXP", "inputs": _inp(d, "authority_expansion"), "rule": msg, "result": "L5"})
    # Accessible action permissions contradict a claim of recommendation-only
    # authority unless the technical execution boundary has been established.
    # Permissions alone do not prove that an action was executed or that L4 applies.
    action_permissions = {"write", "modify", "delete", "send", "execute", "transact"}
    privileged_tools = [x["name"] for x in d.get("tools", []) if action_permissions.intersection(x.get("permissions", []))]
    if L in ("L0", "L1", "L2") and privileged_tools:
        floor_L = L
        notes.append("Action-capable tool permissions were declared (" + ", ".join(privileged_tools) + "). The technical boundary preventing external action has not been established; autonomy is undetermined.")
        trace.append({"step": "Tool/action authority discrepancy", "rule_id": "POL-TOOL-BOUNDARY", "inputs": [{"tool": x["name"], "permissions": x["permissions"], "answer_ids": x["answer_ids"]} for x in d.get("tools", []) if x["name"] in privileged_tools], "rule": "Action-capable access conflicts with a recommendation-only authority claim until technical restrictions are established.", "result": "undetermined"})
        L = None
    open_tools = [x["name"] for x in d.get("tools", []) if len(x.get("unknown_permissions", [])) == len(TOOL_PERMISSIONS)]
    if open_tools:
        notes.append("The permissions of these tools or systems were not stated (neither confirmed nor denied): " + ", ".join(open_tools) + ". Reachable authority through them is not established.")
    if s.get("delegates_to_agents") == "yes":
        notes.append("Delegation to other agents/tools was declared: the delegation ceiling must be tested (a delegate must never hold more authority than the delegator).")
    I = P["impact_map"].get(s.get("max_harm"))
    for floor, rid, why in harm_floors(s.get("harm_desc") or ""):
        if I is None or IMPACTS.index(I) < IMPACTS.index(floor):  # a floor only ever RAISES the class stated by the user; it never lowers it
            I = floor
            trace.append({"step": "Impact adjustment", "rule_id": rid, "inputs": _inp(d, "harm_desc", "max_harm"), "rule": why, "result": floor})
    trace.append({"step": "Impact level (I)", "rule_id": f"POL-{I}" if I else "POL-I", "inputs": _inp(d, "max_harm"),
                  "rule": (f"Declared reasonable worst-case harm '{s.get('max_harm')}' maps to {I} ({P['impact_notes'][I]}); reversibility alone cannot lower the class." if I
                           else "No worst-case harm was declared, so no impact level is derived (undetermined, not assumed)."), "result": I or "undetermined"})
    res = {"L": L, "I": I, "level_notes": notes, "trace": trace}
    cand_L = [L] if L else ((LEVELS[LEVELS.index(floor_L):] if floor_L else LEVELS) if I else [])
    cand_I = [I] if I else (IMPACTS if L else [])
    posts, labs, esc_all, base_set = set(), set(), [], set()
    for li in cand_L:
        for ii in cand_I:
            esc = escalation_factors(d, li, ii)
            base, final = posture_for(ii, li, bool(esc))
            base_set.add(base)
            posts.add(final)
            labs.add(P["label_from_posture"][final])
            if L and I:
                esc_all = esc
    res["escalation_declared"] = esc_all  # exposure factors that were declared, whether or not they changed the posture
    res["escalation"] = []  # factors that actually raised the posture above the matrix baseline (set below, from the real base/final values)
    order = sorted(posts, key=POSTURES.index)
    res["posture_set"] = order
    res["posture_range"] = order[0] if len(order) == 1 else (f"{order[0]}–{order[-1]}" if order else None)
    res["label_set"] = sorted(labs)
    if L and I:
        base, final = posture_for(I, L, bool(esc_all))
        res.update(posture_base=base, posture=final)
        trace.append({"step": "Posture (P)", "rule_id": f"POL-P-{I}x{L}", "inputs": _inp(d, "action_mode", "max_harm"),
                      "rule": f"Policy decision table: impact {I} × autonomy {L} = {base}.", "result": base})
        if esc_all and final != base:
            res["escalation"] = esc_all
            trace.append({"step": "Posture escalation", "rule_id": "POL-ESC", "inputs": _inp(d, *[e["slot"] for e in esc_all]),
                          "rule": "Exposure factors declared: " + "; ".join(e["reason"] for e in esc_all) + ". The policy escalates the posture when exposure increases; one step is applied (no additive weights).",
                          "result": final})
        elif esc_all:
            trace.append({"step": "Posture escalation", "rule_id": "POL-ESC", "inputs": _inp(d, *[e["slot"] for e in esc_all]),
                          "rule": "Exposure factors declared: " + "; ".join(e["reason"] for e in esc_all) + f". The matrix posture is already {base}, the highest level, so no escalation step applies.",
                          "result": final})
        res["trace_posture_final"] = final
    else:
        res.update(posture_base=None, posture=None)
        if order:
            trace.append({"step": "Posture range", "rule_id": "POL-RANGE", "inputs": [],
                          "rule": "The policy needs both L and I. With the missing dimension unknown, every possible value was tried to give a classification range instead of inventing certainty.",
                          "result": res["posture_range"]})
    res["label"] = (sorted(labs)[0] if len(labs) == 1 else None)
    return res


def release_blockers(d: dict, forb: list) -> list[dict]:
    """Release blockers of the internal policy, recorded when the declarations indicate them. They are findings, not verified failures."""
    s, B = d["slots"], policy()["release_blockers"]
    out = []

    def add(key, *slots):
        out.append({"id": key, "text": B[key], "inputs": _inp(d, *slots)})
    if forb:
        add("forbidden")
    acting = s.get("action_mode") in ACTING_MODES
    if acting and s.get("limits_enforced") in ("policy_only", "none"):
        add("limits_policy_only", "limits_enforced", "action_mode")
    if s.get("action_mode") == "approved_action" and s.get("approval_bound") in ("no", "partial"):
        add("approval_not_bound", "approval_bound", "action_mode")
    elif any(w.get("execution") == "executes_after_approval" and w.get("approval_bound") == "no" for w in d.get("workflows", {}).values()):
        add("approval_not_bound")
    if acting and s.get("can_stop") == "no":
        add("no_stop", "can_stop")
    if s.get("authority_expansion") == "yes":
        add("authority_expansion", "authority_expansion")
    return out


def is_acting(d: dict) -> bool:
    return d["slots"].get("action_mode") in ACTING_MODES


def context_gaps(d: dict, L: str | None, I: str | None) -> list[dict]:
    """Context-sensitive gap prioritisation. A gap is CRITICAL when its absence materially affects the classification, a serious harm scenario or a critical
    control boundary for THIS agent; otherwise it is listed as not critical (or not applicable, with the reason). 'not_assessed' = never asked, 'unknown' = asked
    and not known. A declared value (present or absent) is never a gap here, and never proof that a safeguard works."""
    s = d["slots"]
    wfs = list(d.get("workflows", {}).values())
    acting = is_acting(d)
    without = s.get("action_mode") in ("envelope_action", "self_expanding") or any(w.get("execution") == "executes_without_approval" for w in wfs)
    approvals = any(w.get("execution") == "executes_after_approval" for w in wfs) or s.get("action_mode") == "approved_action"
    thresholds = any(w.get("threshold") for w in wfs)
    serious = I in (None, "I2", "I3")
    rules = [
        ("can_stop", "Immediate stop or pause", acting, acting, "An agent that executes actions must be stoppable at once; without a stop mechanism a harmful run cannot be contained."),
        ("can_rollback", "Reversal of a wrong action", acting, acting and serious, "If a wrong transaction cannot be reversed, harm becomes permanent; this bears directly on the impact level."),
        ("delegates_to_agents", "Delegation to other agents or tools", acting, acting, "A delegate must never hold more authority than the delegator; delegation can widen the reachable authority."),
        ("reach", "Scale of possible impact (transactions, people, units)", acting, acting and serious, "The number of transactions, people or units affected bounds the worst reasonable outcome."),
        ("limits_enforced", "Technical enforcement of the authority limit", without, without, "Authority that is limited only in instructions can be exceeded; this raises the posture."),
        ("aggregate_limit", "Aggregate limit (protection against split orders)", without and thresholds, without and thresholds, "A per-action threshold can be circumvented by splitting one large action into many small ones."),
        ("approval_bypass", "Privileged actions that bypass the approval boundary", approvals or thresholds, approvals or thresholds, "An approval boundary is only meaningful if privileged paths cannot skip it."),
    ]
    out = []
    for slot, label, relevant, critical, why in rules:
        if not relevant:
            reason = "The agent was not declared to execute actions." if slot in ("can_stop", "can_rollback", "delegates_to_agents", "reach") and not acting else \
                "No workflow executes without approval, so this boundary is not in play." if slot in ("limits_enforced", "aggregate_limit") else "No approval boundary was declared."
            out.append({"slot": slot, "label": label, "critical": False, "state": "not_applicable", "why": why, "reason": reason})
            continue
        v = s.get(slot)
        state = "not_assessed" if v is None else "unknown" if v == "unknown" else "declared"
        out.append({"slot": slot, "label": label, "critical": bool(critical), "state": state, "why": why, "value": v})
    return out


def legal_path(d: dict) -> dict:
    s, P = d["slots"], policy()
    area, a1, eu = s.get("annex3_area"), s.get("annex1_product"), s.get("eu_link")
    res = {"code": "undetermined", "high": False, "uncertain": [], "refs": [("AIA", "Art. 6")], "answers": []}
    if eu == "no":
        res.update(code="out_of_scope_declared", refs=[("AIA", "Art. 2(1)")], answers=[d["slots_src"].get("eu_link")])
        res["scope_note"] = ("No EU link was declared; the Regulation may be out of scope (Art. 2). This declaration was not verified as a legal scope analysis. "
                             "The rest of the interview is a general TrustAI risk assessment, not an assessment of EU obligations.")
        res["scope_code"] = "out_of_scope_declared"
        return res
    elif eu in (None, "unknown"):
        res["applicability"] = "unknown"
    tp = s.get("annex1_third_party")
    annex1_excluded = a1 == "no" or (a1 == "yes" and tp == "no")  # the Annex I (product safety) route is explicitly excluded
    if a1 == "yes" and tp == "yes":  # Art. 6(1): product (or safety component) under Annex I legislation AND third-party conformity assessment
        res.update(code="annex1_high", high=True)
        res["refs"] = [("AIA", "Art. 6(1)"), ("AIA", "Annex I (intro)")]
        res["answers"] = [d["slots_src"].get("annex1_product"), d["slots_src"].get("annex1_third_party")]
        res["scope_code"] = "annex1_high"
        res["uncertain"].append("Art. 6(1) was applied on the declaration that the product needs a third-party conformity assessment; whether it is covered by the Annex I legislation was not verified.")
        return res
    if a1 == "yes":  # a product safety component alone does not establish the legal high-risk pathway
        res["uncertain"].append("A product safety component was declared, but Art. 6(1) also requires a third-party conformity assessment under Annex I legislation; "
                                + ("it was declared that none is needed." if tp == "no" else "this was not established.") + " "
                                + ("" if tp == "no" else "The pathway needs legal review; it was not treated as definitively high-risk."))
        res["annex1_review"] = tp != "no"
    if area in P["annex3_areas"]:
        res["refs"] = [("AIA", "Art. 6(2)"), ("AIA", P["annex3_refs"][area])]
        res["answers"] = [d["slots_src"].get("annex3_area")]
        prof, exc, infl = s.get("profiling"), s.get("art6_3_exception"), s.get("influences_decision")
        res["refs"] += [("AIA", "Art. 6(3)")]
        if prof == "yes":
            res.update(code="annex3_high_profiling", high=True)
        elif exc in ("narrow_procedural", "improve_prior_human_work", "detect_patterns", "preparatory") and infl == "no" and prof == "no":
            res.update(code="annex3_exception_claimed", high=False)
            res["refs"].append(("AIA", "Art. 6(4)"))
            res["uncertain"].append("The Art. 6(3) exception rests on the declaration; the provider must document the assessment and the registration obligation (Art. 6(4), 49(2)) still applies.")
        else:
            res.update(code="annex3_high", high=True)
            if exc in (None, "unknown") or prof in (None, "unknown"):
                res["uncertain"].append("Whether the Art. 6(3) exception and profiling apply was not settled; the high-risk pathway was assumed as the cautious option.")
        res["answers"] += [d["slots_src"].get(k) for k in ("profiling", "art6_3_exception", "influences_decision") if d["slots_src"].get(k)]
    elif area == "none":
        # Annex III was declared as not involved. Only if the Annex I (product) route is excluded as well is "not high-risk by declaration" established.
        res.update(code="not_high_by_declaration" if annex1_excluded else "partly_assessed")
        res["answers"] = [d["slots_src"].get("annex3_area")]
        res["refs"].append(("AIA", "Annex III (intro)"))
    res["annex1_excluded"] = annex1_excluded
    # code used by the control layer: a legal exclusion (C20 not applicable) needs every high-risk route excluded, not just the one that was asked about
    res["scope_code"] = res["code"] if (annex1_excluded or res["code"] not in ("not_high_by_declaration", "annex3_exception_claimed")) else "partly_assessed"
    return res


def classify(state: dict) -> dict:
    d = derive(state)
    s, P = d["slots"], policy()
    forb = forbidden_findings(d)
    op = operational(d)
    L, I, lnotes = op["L"], op["I"], op["level_notes"]
    legal = legal_path(d)
    transp = [{"slot": k, "ref": ref, "name": name, "answer_id": d["slots_src"].get(k)}
              for k, (ref, name) in P["transparency_refs"].items() if s.get(k) == "yes"]
    unresolved = []
    eu_out = s.get("eu_link") == "no"
    for k in P["critical_slots"]:
        if eu_out and (k.startswith("p_") or k == "annex3_area"):
            continue  # with a declaration of no EU link, the legal screening is not counted as a critical gap
        v = s.get(k)
        if v in (None, "unknown", "") or (k.startswith("p_") and v == "possible"):
            unresolved.append(k)
    unc = list(legal["uncertain"])
    cgaps = context_gaps(d, L, I)
    unc += [f"Critical information not established ({g['state'].replace('_', ' ')}): {g['label']}. {g['why']}" for g in cgaps if g["critical"] and g["state"] in ("unknown", "not_assessed")]
    if autonomy_conflict(d):
        unc.append("Approval/autonomy declarations conflict; no certain approval/autonomy finding was produced.")
        if "action_mode" not in unresolved:
            unresolved.append("action_mode")
    for k in unresolved:
        unc.append(f"Critical information missing: {P['critical_labels'].get(k, k)}.")
    unc += lnotes
    eu_s = s.get("eu_link") if s.get("eu_link") in ("yes", "no") else "unknown"
    out = {"applicability": eu_s, "legal": legal, "transparency": transp, "L": L, "I": I, "level_notes": lnotes, "trace": op["trace"],
           "posture": op["posture"], "posture_base": op["posture_base"], "posture_range": op["posture_range"], "escalation": [e["reason"] for e in op["escalation"]],
           "uncertainties": unc, "forbidden": forb, "unresolved": unresolved, "context_gaps": cgaps, "workflows": [{k: v for k, v in w.items() if k not in ("quotes",)} for w in d["workflows"].values()], "risk_label": None, "operational_label": None, "assessment_status": "incomplete",
           "label_basis": [], "policy_version": P["policy_version"], "blockers": release_blockers(d, forb)}
    out["control_gaps"] = control_gaps(d, legal["scope_code"])
    out["control_assessment"] = control_assessment(d, legal["scope_code"])
    if forb:
        out.update(risk_label="FORBIDDEN", assessment_status="complete")
        out["label_basis"] = [{"rule_id": "POL-ART5", "rule": "A prohibited-use finding (Art. 5) is supported by the user's declarations, with conditions, exception and scope/date considered.", "result": "FORBIDDEN",
                               "inputs": [{"slot": f["key"], "value": "supported", "answer_id": f["slot_answer_id"], "quote": d["evidence"].get(f["key"])} for f in forb]}]
        return out
    if [k for k in unresolved if k.startswith("p_")]:
        return out  # no label is given before the prohibition screening is complete
    out["operational_label"] = op["label"]
    label = None
    basis = []
    if op["label"]:
        rng = "" if op["posture"] else f" (posture range {op['posture_range']})"
        basis.append({"rule_id": f"POL-LABEL-{op['posture'] or op['posture_range']}", "rule": f"Risk level derived from the posture{rng}: P1/P2 = LOW, P3/P4 = HIGH.", "result": op["label"],
                      "inputs": []})
    if legal["high"]:
        label = "HIGH"
        basis.append({"rule_id": "POL-LEGAL-HIGH", "rule": "The legal high-risk pathway (Art. 6) is supported by the declarations, so the risk level is at least HIGH regardless of the operational posture. "
                              "Human approval or low autonomy does not lower it.", "result": "HIGH", "inputs": _inp(d, "annex3_area", "annex1_product", "profiling", "art6_3_exception", "influences_decision")})
    elif op["label"] == "HIGH":
        label = "HIGH"
    elif op["label"] and not unresolved:
        label = op["label"]
    if label:
        out.update(risk_label=label, assessment_status="complete", label_basis=basis)
        titles = {c["id"]: c["title"] for c in controls_cfg()["controls"]}
        for cid, g in out["control_gaps"].items():
            if g["state"] == "unknown":
                out["uncertainties"].append(f"A control that affects the result is unknown: {cid} {titles[cid]}.")
            elif g["state"] == "unasked" and label in ("LOW", "MEDIUM"):
                out["assessment_status"] = "partial"  # a label exists but a relevant control affecting the result has not been assessed
                out["uncertainties"].append(f"A control that affects the result has not been assessed yet: {cid} {titles[cid]} ({g['why']})")
        if unresolved and label == "HIGH":
            out["uncertainties"].append("The level is HIGH with the information available; missing fields do not lower it but may change the detail.")
    return out


# ----------------------------------------------------------------------- trace integrity (single authoritative fact state)
def accepted_fact_keys(state: dict) -> set[tuple[str, str]]:
    """(slot, answer_id) pairs that were ACCEPTED in a turn (quote or selected option). Rejected extractions are never in this set."""
    out = set()
    for t in state["turns"]:
        if t.get("answer") is None:
            continue
        for k in ((t.get("updates") or {}).get("slots") or {}):
            out.add((k, t["id"]))
    return out


def trace_integrity(state: dict, cls: dict) -> list[str]:
    """Violations of 'every decision input is an accepted fact (or a derived fact with accepted inputs)'. An empty list means the explanation is consistent
    with the classification state. A rejected extraction that appears as a decision input is reported here."""
    d = derive(state)
    acc = accepted_fact_keys(state)
    wf_src = {(i["answer_id"]) for df in d["derived_from"].values() for i in df["inputs"]}
    bad = []
    steps = list(cls.get("trace", [])) + list(cls.get("label_basis", []))
    for st in steps:
        for i in st.get("inputs", []):
            slot, aid = i.get("slot"), i.get("answer_id")
            if slot in d["derived_from"]:
                if aid not in wf_src:
                    bad.append(f"{slot}: derived input {aid} is not an accepted workflow fact")
            elif aid in (None, "SETTINGS") or str(aid).startswith("SIM"):
                continue
            elif (slot, aid) not in acc and not any(slot in (dd or {}) for dd in [d["derived"]]):
                bad.append(f"{slot}: {aid} is not an accepted fact")
    rejected = {(k, t["id"]) for t in state["turns"] for k, _ in ((t.get("updates") or {}).get("rejected") or [])}
    for st in steps:
        for i in st.get("inputs", []):
            if (i.get("slot"), i.get("answer_id")) in rejected and (i.get("slot"), i.get("answer_id")) not in acc:
                bad.append(f"{i.get('slot')}: a rejected extraction from {i.get('answer_id')} is cited as support")
    return bad
