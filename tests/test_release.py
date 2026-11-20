"""Release hygiene: source packaging, startup checks, secrets. No model calls."""
import json
import re
import sys
from pathlib import Path

import pytest

import config
import llm
import rag

ROOT = Path(__file__).resolve().parent.parent
FAKE_KEY = "sk-" + "TESTSECRET0123456789abcdefghij"  # built at runtime so this file itself holds no key-like literal


def test_missing_required_source_gives_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SOURCES_DIR", tmp_path)
    rag.get_index.cache_clear()
    with pytest.raises(rag.SourceError) as e:
        rag.get_index()
    msg = str(e.value)
    assert "EU_AI_Act_OJ_L_2024_1689.pdf" in msg and "AI_RMF_Playbook.pdf" in msg and "sources/README.md" in msg
    assert "compliance_matrix.docx" not in msg  # the internal policy is not a documentary source and is not required
    assert "ISO_42001_2023.pdf" not in msg.split("Restricted documents")[0]  # the optional, restricted standard is never demanded
    rag.get_index.cache_clear()


def test_manifest_filenames_are_documented_in_sources_readme():
    readme = (ROOT / "sources" / "README.md").read_text(encoding="utf-8")
    for s in rag.load_manifest()["sources"]:
        assert s["file"] in readme, s["file"]


@pytest.mark.needs_sources
def test_every_required_source_file_exists_and_is_indexed(index):
    assert not [s["file"] for s in rag.load_manifest()["sources"] if s["rag"] and not (config.SOURCES_DIR / s["file"]).exists()]
    indexed = {c["source_id"] for c in index.chunks}
    assert {s["source_id"] for s in rag.load_manifest()["sources"] if s["rag"]} <= indexed
    assert "ISO42001" not in indexed  # restricted standard: never indexed
    assert "TAI-MX" not in indexed  # the internal assessment policy is never indexed or retrievable


def _tracked() -> list[str]:
    """The files that would be packaged (tools/build_release.py): a developer's local .env or QA reference copy is never part of the release."""
    sys.path.insert(0, str(ROOT / "tools"))
    import build_release
    return [str(p.relative_to(ROOT)) for p in build_release.package_files()]


def test_no_secrets_or_restricted_documents_are_packaged():
    files = _tracked()
    assert ".env" not in files and not [f for f in files if f.startswith(".env") and f != ".env.example"]
    assert not [f for f in files if "iso_42001" in f.lower() or f.lower().endswith((".zip", ".docx"))], "restricted or private files must not be packaged"
    key = re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_\-]{30,}")
    for f in files:
        p = ROOT / f
        if p.suffix in (".py", ".md", ".json", ".toml", ".txt", ".ini", ".example", "") and p.is_file():
            assert not key.search(p.read_text(encoding="utf-8", errors="ignore")), f"key-like string in {f}"


def test_key_is_read_from_the_environment_and_masked_in_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    assert config.api_key() == FAKE_KEY
    assert FAKE_KEY not in llm._clean(RuntimeError(f"401 invalid key {FAKE_KEY}"))
    assert FAKE_KEY not in rag.re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", f"embed failed {FAKE_KEY}")


def test_app_never_prints_the_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    shown = " ".join(str(getattr(el, "value", "")) for grp in (at.markdown, at.text, at.caption, at.warning, at.error, at.info) for el in grp)
    assert FAKE_KEY not in shown and FAKE_KEY[:12] not in shown


@pytest.mark.needs_sources
def test_report_and_pdf_do_not_contain_the_key(index, monkeypatch):
    import interview as iv
    import report as rp
    from conftest import FakeClient
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    s = iv.new_state(); iv.add_first(s, index)
    iv.process(s, 0, "It only answers opening-hours questions on our website.", index, FakeClient([{"slots": {"purpose": "FAQ", "action_mode": "info_only"}}]))
    rep = rp.build_report(s, index)
    assert FAKE_KEY not in json.dumps(rep, default=str)
    assert FAKE_KEY.encode() not in rp.pdf_bytes(rep)
