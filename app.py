"""TrustAI Risk Assessment — Streamlit interface """
import html

import streamlit as st

import classify as C
import config
import interview as iv
import llm
import rag
import report as rp
import sources_admin as sa

st.set_page_config(page_title="TrustAI Risk Assessment", page_icon="🛡️", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
/* Colors come from the .streamlit/config.toml theme (navy, white, a single teal accent). Only narrowly scoped component styles live here. */
.block-container { max-width: 760px; padding-top: 2.2rem; }
.ta-steps { display:flex; gap:6px; margin:4px 0 18px 0; }
.ta-step { flex:1; text-align:center; padding:6px 4px; border-radius:8px; font-size:.9rem; border:1px solid #CBD5E1; }
.ta-step.on { background:#0F9D94; color:#fff; border-color:#0F9D94; font-weight:600; }
.ta-step.done { border-color:#0F9D94; }
.ta-card { border:1px solid #CBD5E1; border-left:6px solid #0F9D94; border-radius:10px; padding:14px 16px; margin:10px 0 14px 0; background:#fff; }
.ta-card.forbidden { border-left-color:#0B1F3A; border-left-width:10px; }
.ta-muted { opacity:.75; }
/* Option buttons: full text, line wrapping, height fitted to content (Streamlit's default is nowrap + ellipsis) */
.stButton > button, .stButton > button *, .stFormSubmitButton > button, .stFormSubmitButton > button * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; word-break: break-word; }
.stButton > button { height:auto !important; min-height:2.5rem; text-align:left; justify-content:flex-start; padding:.5rem .9rem; line-height:1.35; }
.stButton > button p { text-align:left; margin:0; }
</style>
""", unsafe_allow_html=True)

PHASES = ["Scope", "Use", "Authority and impact", "Result"]
LABEL_NAMES = {"LOW": "LOW", "MEDIUM": "MEDIUM", "HIGH": "HIGH", "FORBIDDEN": "FORBIDDEN (prohibited-use finding)"}


@st.cache_resource(show_spinner="Loading the source index…")
def index():
    try:
        return rag.get_index()
    except rag.SourceError as e:  # a mis-parsed source must be visible, never silently replaced by a smaller index
        st.error("Source documents problem: " + str(e))
        st.stop()


@st.cache_resource
def _client(key: str):
    return llm.get_client()


def client_or_none():
    try:
        key = config.api_key()
        if not key:
            raise llm.SetupError("OPENAI_API_KEY is not set. Put the key in the .env file (see .env.example) and restart the application.")
        return _client(key), None
    except llm.SetupError as e:
        return None, str(e)


def startup_index(idx, client):
    """Part of normal startup: loads the saved embedding index and embeds only the missing passages (once per session, no button needed)."""
    if "emb_status" not in st.session_state:
        with st.spinner("Preparing the semantic search index (only missing passages are embedded)…"):
            st.session_state.emb_status = rag.ensure_embeddings(idx, client)
    return st.session_state.emb_status


def S():
    if "state" not in st.session_state:
        st.session_state.state = None
    if "pending" not in st.session_state:
        st.session_state.pending = None
    return st.session_state


def queue(kind, idx=None, key=None, text=""):
    if key:
        text = st.session_state.get(key, "")
    if text.strip() or kind in ("confirm", "reset", "retry"):
        st.session_state.pending = {"kind": kind, "idx": idx, "text": text}


def progress(state):
    cur = 0
    if state:
        if state["phase"] in ("summary", "forbidden") or state.get("report"):
            cur = 3
        else:
            last = state["turns"][-1]["question"]
            cur = PHASES.index(last.get("phase")) if last.get("phase") in PHASES else 0
    cells = "".join(f'<div class="ta-step {"on" if i == cur else "done" if i < cur else ""}">{"✓ " if i < cur else ""}{p}</div>' for i, p in enumerate(PHASES))
    st.markdown(f'<div class="ta-steps">{cells}</div>', unsafe_allow_html=True)


def why_box(idx, q, key):
    """Saved short rationale + source; closed by default, opening it makes no API call."""
    with st.expander("Why this question?"):
        w = q.get("why")
        reason = (w.get("clarifies") or w.get("missing")) if isinstance(w, dict) else w
        if reason:
            st.write(reason)
        if q.get("framing") == "general":
            st.write("No EU link was declared; this question belongs to the general TrustAI assessment, not to an AI Act obligation.")
        if q.get("kind") == "conflict":
            st.write("This question is based on the consistency of your earlier answers, not on a source.")
        elif q.get("no_passage"):
            st.write("No source passage passed the support check for this question; this is recorded as an uncertainty.")
        if q.get("adapted") and q.get("passage_ids"):
            st.write("This question is adapted from the principle; the source does not contain it verbatim and it is not presented as a legal obligation.")
        for n, pid in enumerate(q.get("passage_ids", [])):
            c = idx.by_id.get(pid)
            if not c:
                continue
            m = rag.source_meta(c["source_id"])
            sec = f" · {c['section']}" if c.get("section") and c["section"] not in c["ref"] else ""
            st.markdown(f"**Source:** {m['short']} — {rp.KIND_LABEL.get(m['kind'], m['kind'])} ({c['version']}) · {c['ref']}{sec} · {rag.locator(c)}" + (f" · [link]({c['url']})" if c.get("url") else ""))
            if st.toggle("Show the passage", key=f"tg_{key}_{n}"):
                st.caption("“" + c["text"] + "”")


def queue_multi(i, tid, q, mode):
    """Checklist answer: the ticked items become exact slot values, stored with the answer text like a selected option."""
    if mode == "confirm" and st.session_state.get(f"mcx_{tid}"):
        mode = "exhaustive"  # the person confirmed the list as complete: unticked items are explicit 'no'
    checked = [it["slot"] for it in q["multi"]["items"] if st.session_state.get(f"mc_{tid}_{it['slot']}")]
    text, sets = iv.multi_answer(q["multi"], checked, mode)
    q.setdefault("option_sets", {})[text] = sets
    queue("answer", i, None, text)


def render_multi(i, tid, q):
    m = q["multi"]
    for it in m["items"]:
        st.checkbox(it["label"], key=f"mc_{tid}_{it['slot']}")
    any_ticked = any(st.session_state.get(f"mc_{tid}_{it['slot']}") for it in m["items"])
    st.checkbox("This is the complete list: anything I did not tick does not apply", key=f"mcx_{tid}", disabled=not any_ticked)
    c1, c2, c3 = st.columns(3)
    c1.button("Confirm my selection", key=f"mcc_{tid}", type="primary", disabled=not any_ticked, on_click=queue_multi, args=(i, tid, q, "confirm"), use_container_width=True)
    # 'None of these' excludes every other selection: it is only available while nothing is ticked
    c2.button(m["none_label"], key=f"mcn_{tid}", disabled=any_ticked, on_click=queue_multi, args=(i, tid, q, "none"), use_container_width=True)
    c3.button(m["unsure_label"], key=f"mcu_{tid}", on_click=queue_multi, args=(i, tid, q, "unsure"), use_container_width=True)
    st.caption("Select all that apply. Items you leave unticked stay unanswered unless you tick \u201cThis is the complete list\u201d. \u201c" + m["none_label"] + "\u201d is only available while nothing is ticked. You can also describe it in your own words below.")


def render_question(state, idx):
    """Current question only: text, options (optional), answer field and rationale link."""
    i = len(state["turns"]) - 1
    t = state["turns"][i]
    if state["phase"] != "interview" or t.get("answer") is not None:
        return
    q = t["question"]
    n_done = C.answered_count(state)
    st.caption(f"Question {min(n_done + 1, config.MAX_QUESTIONS)} of up to {config.MAX_QUESTIONS} · " + ("select all that apply" if q.get("multi") else "choose one option or answer in your own words" if q.get("options") else "answer in your own words"))
    st.markdown(f"#### {q['text']}")
    why_box(idx, q, t["id"])
    if q.get("multi"):
        render_multi(i, t["id"], q)
    for j, o in enumerate([] if q.get("multi") else q.get("options") or []):  # stacked one per row, full text; the selected label is stored together with its field/value
        st.button(o, key=f"op_{t['id']}_{j}", on_click=queue, args=("answer", i, None, o), use_container_width=True)
    with st.form(f"f_{t['id']}", clear_on_submit=True):
        st.text_area("Or answer in your own words" if q.get("options") or q.get("multi") else "Your answer", height=100, key=f"in_{t['id']}", placeholder="Prefer your own words? Describe it here. “I don't know” is also a valid answer.")
        st.form_submit_button("Send", type="primary", on_click=queue, args=("answer", i, f"in_{t['id']}"))


def render_history(state):
    """Closed 'Interview history': previous questions/answers and answer editing (no nested expanders)."""
    done = [(i, t) for i, t in enumerate(state["turns"]) if t.get("answer") is not None]
    if not done:
        return
    with st.expander("Interview history", expanded=False):
        for i, t in done:
            if t["question"].get("kind") == "correction":
                st.markdown(f"*Correction:* {t['answer']}")
                continue
            st.markdown(f"**{t['question']['text']}**")
            st.write(t["answer"])
            if st.toggle("Edit this answer", key=f"edt_{t['id']}"):
                st.write("If you change it, the questions and results after this answer are discarded; your question allowance does not increase.")
                st.text_area("New answer", value=t["answer"], key=f"ed_{t['id']}")
                st.button("Save", key=f"sv_{t['id']}", on_click=queue, args=("edit", i, f"ed_{t['id']}"))
            st.divider()


def render_summary(state):
    cls = C.classify(state)
    st.subheader("This is how I understood you")
    for title, items in iv.summary_sections(state):
        st.markdown(f"**{title}**")
        for stt, text in items:
            st.markdown(f"- {text}" + ("" if stt == "confirmed" else f" *({iv.STATE_LABEL[stt]})*"))
    if cls["unresolved"] and C.answered_count(state) >= config.MAX_QUESTIONS:
        st.warning(f"The question limit ({config.MAX_QUESTIONS}) was reached. Missing fields: " + ", ".join(C.policy()["critical_labels"].get(k, k) for k in cls["unresolved"]) + ". No new question round is opened; you can write a correction below if needed.")
    st.write("If something was misunderstood, write it below; a correction does not increase your question count.")
    with st.form("corr", clear_on_submit=True):
        st.text_area("Correction (optional)", height=80, key="corr_in")
        st.form_submit_button("Apply the correction", on_click=queue, args=("correction", None, "corr_in"))
    st.button("Correct, show the result", type="primary", on_click=queue, args=("confirm",))


def cite_md(c):
    """Source name, version, article/section and location together, as a visible line."""
    sec = f" · {c['section']}" if c["section"] and c["section"] not in c["ref"] else ""
    return f"**[K{c['no']}] {c['source']}** — {c['kind_label']} ({c['version']}) · {c['ref']}{sec} · {c['locator']}" + (f" · [link]({c['url']})" if c["url"] else "")


def render_result(state, idx):
    rep = state["report"]
    lab = rep["risk_label"]
    title = LABEL_NAMES[lab] if lab else "Assessment not completed"
    ctx = rep["context"]
    sub = rep["label_note"] or ("Not a fifth risk level; no level was issued because critical information is missing." if not lab else "TrustAI preliminary risk level; not an official category of the AI Act.")
    st.warning("**Preliminary, declaration-based assessment.** " + rep["banner"] + " It is not a technical verification, a control test or a compliance finding.")
    post = ""
    if rep["posture_text"]:
        post = f'<br><b>Assurance posture (policy P1–P4): {html.escape(rep["posture_text"])}</b> &nbsp;·&nbsp; Autonomy {rep["cls"]["L"] or "?"} &nbsp;·&nbsp; Impact {rep["cls"]["I"] or "?"}'
    stat = (f'<br><b>Interview:</b> {html.escape(rep["interview_status"])} &nbsp;·&nbsp; <b>Assessment:</b> {html.escape(rep["assessment_status"])} &nbsp;·&nbsp; <b>Verification:</b> {html.escape(rep["verification_status"])}')
    why = "Requires further information because: " + "; ".join(rep["assessment_reasons"]) + "." if rep["assessment_reasons"] else "Finishing the interview does not mean any control was tested or verified."
    st.markdown(f'<div class="ta-card {"forbidden" if lab == "FORBIDDEN" else ""}"><b>Risk level: {html.escape(title)}</b> <small>(TrustAI label, not an EU AI Act category)</small>{post}{stat}<br><span class="ta-muted">{html.escape(why)}</span>'
                f'<br><span class="ta-muted">{html.escape(sub)}</span></div>', unsafe_allow_html=True)
    if rep["posture_meaning"]:
        st.caption(rep["posture_meaning"])
    st.caption("Evidence confidence: " + rep["evidence_state"])
    st.write(f"Agent operating scope: {ctx['scope_country']} · Legal applicability: {ctx['applicability']} · Reference framework: {ctx['reference']}")
    if lab == "FORBIDDEN":
        st.info("The interview was stopped; this does not mean the assessed agent was technically stopped.")
    st.subheader("Assessed use")
    st.write(rep["use_text"])
    if rep["capabilities"]:
        st.markdown("**Material capabilities and limits (as declared)**")
        for c in rep["capabilities"]:
            st.markdown(f"**•** {c}")
    st.subheader("How the result was derived")
    st.caption("Answer → extracted fact → rule → result. The result is computed by fixed rules from your declarations; retrieved passages and the language model cannot change it. " + rep["label_rule"])
    for n, stp in enumerate(rep["derivation"], 1):
        st.markdown(f"**{n}. {stp['step']} → {stp['result']}**" + (f"  `{stp['rule_id']}`" if stp.get("rule_id") else ""))
        st.write(stp["rule"])
        for i in stp["inputs"]:
            q = f" — “{i['quote']}”" if i.get("quote") else ""
            wf = f" ({i['workflow']}: {i['execution']}; {i['role']})" if i.get("workflow") else ""
            st.caption(f"from answer {i['answer_id']}: {i['slot']} = {i['value_label']}{wf}{q}")
    st.caption(f"Rule IDs refer to the {rep['policy_version']}. {rep['policy_notice']}")
    if rep["drivers"]:
        st.subheader("Key risk drivers")
        for dr in rep["drivers"]:
            st.markdown(f"**•** {dr}")
    st.subheader("Key findings")
    for k in rep["key_findings"]:
        st.markdown(f"**•** {k['text']}")
        if k["decl"]:
            st.write(k["decl"])  # the user's statement
        for c in k["cites"]:
            st.markdown("Source: " + cite_md(c))
            st.caption("“" + c["snippet"] + "…”")
        if k["unsupported"]:
            st.caption("No retrieved passage passed the support check for this finding, so it is not presented as source-backed.")
        elif k["method"] and not k["cites"]:
            st.write("Provenance: internal assessment policy (not an external source)" + (f": {', '.join(k['rules'])}" if k.get("rules") else "."))
    if rep["blockers"]:
        st.subheader("Declared release blockers (internal policy)")
        for b in rep["blockers"]:
            st.markdown(f"**•** {b['text']}")
    if rep["improvements"]:
        st.subheader("Required improvements")
        for im in rep["improvements"]:
            st.markdown(f"**•** {im['text']}")
            for c in im["cites"]:
                st.markdown("Basis: " + cite_md(c))
    if rep["critical_gaps"]:
        st.subheader("Critical gaps affecting the result")
        for g in rep["critical_gaps"]:
            st.markdown(f"**•** {g}")
    for lr in rep["legal_review"]:
        st.markdown(f"**•** {lr}")
    st.write(rep["note"])
    st.download_button("Download the PDF report", rp.pdf_bytes(rep), file_name="trustai_risk_assessment.pdf", mime="application/pdf", type="primary")
    with st.expander("Method and details"):
        st.markdown("**Separate conclusions**")
        for k, v in rep["conclusions"]:
            st.write(f"• **{k}:** {v}")
        st.markdown("**Scope and framework.** The reference framework and legal applicability are separate fields. " + rep["disclaimer"])
        if rep["legal_scope_note"]:
            st.write(rep["legal_scope_note"])
        for a in rep["dates"]:
            st.write(f"{a['what']} — {a['date']} ({a['ref']}). {a['status']}")
        st.write(rep["currency"])
        for n in rep["level_notes"]:
            st.write("• " + n)
        st.markdown("**Findings, user declarations and sources**")
        for f in rep["findings"]:
            ans = ", ".join(f"answer {a}" for a in f["answers"])
            nos = " ".join(f"[K{n}]" for n in f["cite_nos"])
            st.write(f"{f['id']}. {f['text']}" + (f" — Declared in: {ans}" if ans else "") + (f" — Sources: {nos}" if nos else "") + (f" — model support check: {f['semantic']}" if f["semantic"] != "not checked" else ""))
            for r in f["rejected"]:
                st.caption(f"Not used as support — {r['chunk_id']}: {r.get('reason') or 'support check failed'}")
        v = rep["verification"]
        st.write("Every retrieved passage must pass a keyword and embedding-similarity support check" + ("; a model also checked the meaning of each claim-passage pair." if v["semantic_done"] else "; the additional model check did not run.")
                 + (f" Flagged findings: {', '.join(v['flagged'])}." if v["flagged"] else "") + " " + rep["retrieval_note"])
        for w in rep["warnings"]:
            st.warning(w)
        if rep["workflows"]:
            st.markdown("**Workflow facts** (authority is evaluated per workflow)")
            for w in rep["workflows"]:
                st.caption(f"{w['label']}: {w['execution']}; threshold: {w['threshold']}; approval authority: {w['approval_authority']}; approval bound: {w['approval_bound']}; limit enforcement: {w['limit_enforcement']} — answer {w['answer_id']}")
        ig, cc = rep["integrity"], rep["citation_check"]
        st.caption("Fact-state integrity: " + ("every decision input is an accepted fact or a derived fact with accepted inputs." if ig["ok"] else "VIOLATIONS: " + "; ".join(ig["violations"])))
        st.caption(f"Citation validation: {cc['checked']} checked" + ("; all valid." if cc["ok"] else "; problems: " + "; ".join(cc["problems"])))
        st.markdown("**Answer trail** (answer → extracted facts → retrieved evidence)")
        for a in rep["trail"]:
            st.write(f"Answer {a['id']} — {a['q']} → {a['a']}")
            for fct in a["facts"]:
                st.caption(f"accepted fact: {fct['slot']} = {fct['value_label']}" + (f" (quote: “{fct['quote']}”)" if fct["quote"] else ""))
            for wf in a["workflows"]:
                st.caption(f"accepted workflow fact: {wf['label']} — {wf['execution']}" + (f" (quote: “{wf['quote']}”)" if wf["quote"] else ""))
            for r in a["superseded"]:
                st.caption("superseded (not used): " + r)
            for r in a["rejected"]:
                st.caption("rejected extraction (never used as support): " + r)
            for c in a["passages"]:
                st.markdown("&nbsp;&nbsp;evidence for the question: " + cite_md(c))
        st.markdown("**C01–C21 coverage map** (declaration status; not a safety score)")
        st.write(", ".join(f"{k}: {n}" for k, n in rep["control_counts"].items()))
        st.table([{"ID": c["id"], "Control": c["title"], "Assessment": c["detail"], "Finding": c["finding"]} for c in rep["controls"]])  # the same rows the PDF shows
        if rep["noncritical_gaps"]:
            st.markdown("**Non-critical gaps**")
            for g in rep["noncritical_gaps"]:
                st.write("• " + g)
        st.markdown("**Source passages**")
        for c in rep["cited"]:
            st.markdown(cite_md(c))
        st.markdown("**Source list and versions**")
        for b in rep["bibliography"]:
            st.write(f"{b['title']} — {b['kind']}; version: {b['version']}; date: {b['date']}" + (f"; {b['url']}" if b["url"] else "") + f". {b['note']}")


def run_pending(idx, client):
    p = st.session_state.pending
    st.session_state.pending = None  
    state = st.session_state.state
    try:
        if p["kind"] == "reset":
            st.session_state.state = None
        elif p["kind"] == "retry":
            with st.spinner("Retrying…"):
                iv.process(state, len(state["turns"]) - 1, state["turns"][-1]["answer"], idx, client)
        elif p["kind"] in ("answer", "edit"):
            with st.spinner("Evaluating your answer…"):
                iv.process(state, p["idx"], p["text"], idx, client)
        elif p["kind"] == "correction":
            with st.spinner("Applying the correction…"):
                iv.process(state, len(state["turns"]) - 1, p["text"], idx, client, mode="correction")
        elif p["kind"] == "confirm":
            state["summary_confirmed"] = True
        if state and ((state["phase"] == "forbidden") or (state["summary_confirmed"] and not state.get("report"))):
            with st.spinner("Preparing the report…"):
                state["report"] = rp.build_report(state, idx, client)
    except llm.SetupError as e:
        if st.session_state.state:
            st.session_state.state["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        if st.session_state.state:
            st.session_state.state["error"] = "The operation could not be completed: " + llm._clean(e) + " Your answer is saved; you can try again."


def sidebar(idx, client, setup_msg, started, emb):
    """Settings and setup status live in the sidebar. The embedding index is prepared automatically at startup (no manual step)."""
    with st.sidebar:
        if not started:
            st.markdown("**Assessment settings**")
            dflt = C.policy()["context_defaults"]
            st.caption("Operating scope of the assessed agent (not the location of this application). “Not specified” does not mean outside the EU; the AI Act is the assessment reference.")
            st.text_input("Agent operating scope", value=dflt["scope_country"], key="cfg_country")
            st.radio("Link with the EU", ["Not determined", "Yes — used in the EU or affects people/organisations in the EU", "No"], key="cfg_eu")
            st.divider()
        st.markdown("**Setup status**")
        stt = idx.status()
        st.write(f"API key: {'found' if not setup_msg else 'missing'}")
        st.write(f"Source passages: {stt['chunks']}")
        for s_ in sa.status():
            st.caption(f"Added source {s_['source_id']}: {s_['state']}" + (f" — {s_['reason']}" if s_.get("reason") else ""))
        mode = idx.mode()
        st.write(f"Search index: {'semantic + keyword (ready)' if mode == 'hybrid' else 'semantic + keyword (' + str(stt['missing']) + ' passages keyword-only)' if mode == 'partial' else 'keyword-only'}")
        if emb and emb.get("built"):
            st.caption(f"{emb['built']} passages were embedded automatically at startup.")
        if emb and emb.get("error"):
            st.warning(emb["error"])
            if client is not None and st.button("Retry index setup"):
                st.session_state.pop("emb_status", None)
                st.rerun()


def main():
    st.markdown("### TrustAI Risk Assessment")
    st.write("A short interview with someone who knows the AI agent, producing a preliminary risk assessment based on your declarations.")
    st.info("**Preliminary, declaration-based assessment.** " + C.policy()["banner"])
    client, setup_msg = client_or_none()
    idx = index()
    emb = startup_index(idx, client)
    state = S().state
    sidebar(idx, client, setup_msg, started=state is not None, emb=emb)
    progress(state)
    if state is None:
        if setup_msg:
            st.warning("Setup required: " + setup_msg)
        if idx.status()["chunks"] == 0:
            st.warning("Source files were not found. Follow the source preparation step in the README.")
        if st.button("Start the assessment", type="primary", disabled=bool(setup_msg) or idx.status()["chunks"] == 0):
            dflt = C.policy()["context_defaults"]
            eu_map = {"Not determined": "unknown", "No": "no"}
            s_ = iv.new_state({"scope_country": (st.session_state.get("cfg_country") or dflt["scope_country"]).strip() or dflt["scope_country"],
                               "eu_link": eu_map.get(st.session_state.get("cfg_eu", "Not determined"), "yes")})
            iv.add_first(s_, idx)
            st.session_state.state = s_
            st.rerun()
        return
    if st.session_state.pending:
        if client is None:
            state["error"] = setup_msg
            st.session_state.pending = None
        else:
            run_pending(idx, client)
            st.rerun()
    state = st.session_state.state
    if state is None:
        st.rerun()
    if state.get("error"):
        st.error(state["error"])
        if state["phase"] == "interview" and state["turns"][-1].get("answer") is not None:
            st.button("Try again", on_click=queue, args=("retry",))
    n = C.answered_count(state)
    if state.get("report"):
        render_result(state, idx)
    elif state["phase"] == "summary":
        render_summary(state)
    elif state["phase"] == "interview" and state["turns"][-1].get("answer") is None:
        st.caption(f"Question {n + 1} · at most {config.MAX_QUESTIONS}")
        render_question(state, idx)
    render_history(state)
    st.divider()
    st.button("New assessment", on_click=queue, args=("reset",))


main()
