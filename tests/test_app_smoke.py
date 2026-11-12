"""Source-independent smoke tests: they must run (and pass) even when sources/ is empty."""
import py_compile
from pathlib import Path

import pytest

import classify as C
import interview as iv
import rag

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.mark.parametrize("name", ["app", "classify", "config", "interview", "llm", "planner", "rag", "report"])
def test_module_compiles(name):
    py_compile.compile(str(ROOT / f"{name}.py"), doraise=True)


def test_app_starts_without_exception():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception


@pytest.fixture
def empty_index(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "build_chunks", lambda: [])
    monkeypatch.setattr(rag.Index, "_emb_path", lambda self: tmp_path / "emb.npz")
    monkeypatch.setattr(rag.Index, "_migrate_legacy", lambda self: False)
    return rag.Index()


def test_checklist_question_renders_without_sources(empty_index, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(rag, "get_index", lambda: empty_index)  # the app itself refuses to start without sources; here only the UI code is exercised
    s = iv.new_state()
    iv.add_first(s, empty_index)
    s["turns"][0]["question"] = iv.make_question(C.TOPICS()["screening"], empty_index, s)
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["state"] = s
    at.run()
    assert not at.exception
    assert len(at.checkbox) == 4 and "Confirm my selection" in [b.label for b in at.button]  # three items + the 'complete list' confirmation
    labels = [b.label for b in at.button]
    assert not any(l in labels for l in ("Yes", "No", "I don't know")) and sum(l == "None of these" for l in labels) == 1  # no second set of answer buttons


def test_app_stops_with_a_clear_message_when_sources_are_missing(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    import config
    monkeypatch.setattr(config, "SOURCES_DIR", tmp_path)
    import streamlit as st
    rag.get_index.cache_clear()
    st.cache_resource.clear()  # the app caches the index per process; start from a clean cache
    at = AppTest.from_file(APP, default_timeout=60).run()
    rag.get_index.cache_clear()
    st.cache_resource.clear()
    assert not at.exception
    assert at.error and "EU_AI_Act_OJ_L_2024_1689.pdf" in at.error[0].value and "sources/README.md" in at.error[0].value
