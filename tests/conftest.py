import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import rag  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the real OpenAI API (costs money); run with `pytest -m live`")
    config.addinivalue_line("markers", "needs_sources: requires the source documents in sources/")


def sources_present() -> bool:
    import config as appcfg
    return all((appcfg.SOURCES_DIR / s["file"]).exists() for s in rag.load_manifest()["sources"] if s["rag"] and s["api_transfer_allowed"] and not s.get("custom"))


def pytest_collection_modifyitems(config, items):
    """Live tests need an explicit `-m live` and an API key. Source-dependent tests are skipped (by the `index` fixture and the `needs_sources` marker) when
    the source files are missing; syntax, app-start and UI smoke tests never depend on the sources."""
    import config as appcfg
    if not sources_present():
        skip_src = pytest.mark.skip(reason="source files are missing: copy them into sources/ (see sources/README.md)")
        names = {"index", "live_index"}
        for it in items:
            if names & set(getattr(it, "fixturenames", [])) or "needs_sources" in it.keywords:
                it.add_marker(skip_src)
    if "live" in (config.getoption("-m") or "") and appcfg.api_key() and sources_present():
        return
    skip = pytest.mark.skip(reason="live test: run with `pytest -m live` and a valid OPENAI_API_KEY")
    for it in items:
        if "live" in it.keywords:
            it.add_marker(skip)


@pytest.fixture(scope="session")
def index():
    return rag.get_index()


class FakeClient:
    """Scripted fake client used only to test the application's own flow (produces no real result)."""
    def __init__(self, scripts):
        self.scripts, self.calls, self.last_user, self.extract_calls = list(scripts), 0, '', 0
        outer = self

        class _C:
            def create(self, **kw):
                outer.calls += 1
                user = kw["messages"][1]["content"]
                topics = re.findall(r"CANDIDATE topic (\w+)", user)
                outer.last_user = user
                name = kw["response_format"]["json_schema"]["name"]
                if name == "extract":  # one script per answer; the question call reuses it
                    outer.last_upd = outer.scripts.pop(0) if outer.scripts else {}
                    outer.extract_calls += 1
                upd = getattr(outer, "last_upd", {})
                m = re.search(r"LAST ANSWER \(\w+\): (.*?)\n\n", user, re.S)
                ans = m.group(1) if m else ""
                ev = upd.get("evidence", {})
                res = {"workflow_updates": [{"id": "", "action": "", "condition": "", "threshold": "", "recipient": "", "recipient_restricted": "unknown", "execution": "unknown", "approval_authority": "", "approval_bound": "unknown", "limit_enforcement": "unknown", "evidence": ans, **w} for w in upd.get("workflows", [])],
                       "slot_updates": [{"slot": k, "value": v, "evidence": ev.get(k, ans)} for k, v in upd.get("slots", {}).items()],
                       "control_updates": [{"evidence": ans, **c} for c in upd.get("controls", [])], "basis": [{"conditions_evidence": ans, "exception_evidence": ans, "scope_date_evidence": ans, **b} for b in upd.get("basis", [])],
                       "next": {"topic_id": upd.get("topic") or (topics[0] if topics else "purpose"),
                                "question": upd.get("question", "Test question goes here?"), "options": ["Yes", "No"], "missing": "Missing test information.",
                                "clarifies": "Test clarification.", "source_note": "", "passage_ids": [], "adapted": False}}

                class M:
                    content = json.dumps(res)

                class Ch:
                    message = M

                class R:
                    choices = [Ch]
                return R

        class _Chat:
            completions = _C()
        self.chat = _Chat()

        class _E:
            def create(self, **kw):
                raise RuntimeError("no embed")
        self.embeddings = _E()
