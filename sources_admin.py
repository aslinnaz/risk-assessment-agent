"""Approved-source management: add, validate, activate/deactivate and supersede documents AFTER deployment, without touching the application code.

A source is an external document (legislation, voluntary framework, guidance or a licensed standard). Adding one only extends retrieval and citations: it never
changes the deterministic classification (classify.py does not import this module or rag.py). Nothing is trusted silently: a document without complete
metadata, without extractable text, without licence information that permits indexing, or that duplicates an indexed document is rejected or flagged.

    python tools/add_source.py add path/to/doc.pdf --id NIST-GENAI --title "..." --publisher NIST --version 1.0 --date 2024-07-26 --kind framework --concerns agentic,oversight --indexing-permitted
    python tools/add_source.py list | deactivate ID | activate ID
Restart the application afterwards (the index is rebuilt automatically at startup; only new passages are embedded).
"""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
from pathlib import Path

import config
import rag

ALLOWED_SUFFIXES = (".pdf", ".docx")
MIN_TEXT_CHARS = 500
REQUIRED = ("source_id", "title", "publisher", "version", "date", "kind")


def _registry_path() -> Path:
    return config.custom_dir() / "registry.json"


def _save(reg: dict) -> None:
    config.custom_dir().mkdir(parents=True, exist_ok=True)
    _registry_path().write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _extract_text(path: Path) -> str:
    if path.suffix == ".pdf":
        return " ".join(rag.extract_pages(path))
    return " ".join(v if k == "p" else " ".join(c for row in v for c in row) for k, v in rag._docx_blocks(path))


def validate(path: Path, meta: dict) -> list[str]:
    """Problems that stop a document from being registered (empty list = acceptable)."""
    problems = []
    if rag.is_excluded_reference(path.name, str(meta.get("title") or ""), str(meta.get("source_id") or ""), str(meta.get("short") or "")):
        problems.append("this document is an external development/QA reference (for example the compliance check matrix); it is never an application source and cannot be registered")
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        problems.append(f"unsupported file type '{path.suffix}' (use PDF or DOCX)")
    for k in REQUIRED:
        if not str(meta.get(k) or "").strip():
            problems.append(f"missing metadata: {k}")
    if meta.get("kind") and meta["kind"] not in rag.CUSTOM_KINDS:
        problems.append(f"kind must be one of {sorted(rag.CUSTOM_KINDS)}")
    if meta.get("source_id") and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_\-]{1,30}", meta["source_id"]):
        problems.append("source_id must be 2-31 characters: letters, digits, '-' or '_'")
    if meta.get("date") and not re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", str(meta["date"])):
        problems.append("date must be YYYY, YYYY-MM or YYYY-MM-DD")
    bad = [c for c in (meta.get("concerns") or []) if c not in rag.CUSTOM_CONCERNS]
    if bad:
        problems.append(f"unknown concerns {bad}; use {list(rag.CUSTOM_CONCERNS)}")
    if not (meta.get("license") or {}).get("indexing_permitted"):
        problems.append("licence: indexing is not confirmed as permitted (pass --indexing-permitted only if the document may be indexed and sent to the embedding API); "
                        "copyrighted standards must not be added")
    builtin = {s["source_id"] for s in json.loads((config.CATALOG_DIR / "sources.json").read_text(encoding="utf-8"))["sources"]}
    existing = {e["source_id"] for e in rag.custom_registry()["sources"]}
    if meta.get("source_id") in builtin | existing:
        problems.append(f"source_id '{meta.get('source_id')}' already exists")
    if not path.exists():
        problems.append(f"file not found: {path}")
        return problems
    if not problems:
        try:
            text_ = _extract_text(path)
            if rag.has_excluded_content(text_):
                problems.append("the document text identifies it as an external development/QA reference (compliance check matrix); it cannot be registered")
            if len(text_.strip()) < MIN_TEXT_CHARS:
                problems.append(f"no extractable text (fewer than {MIN_TEXT_CHARS} characters); scanned documents need OCR first")
        except Exception as e:  # noqa: BLE001
            problems.append(f"the document could not be read: {str(e)[:120]}")
        h = rag.file_hash(path)
        for s in rag.load_manifest()["sources"]:
            sp = rag.source_path(s)
            if s["rag"] and sp.exists() and sp.resolve() != path.resolve() and rag.file_hash(sp) == h:
                problems.append(f"duplicate of the already indexed source {s['source_id']}")
                break
    return problems


def add_source(path: str | Path, meta: dict) -> dict:
    """Validates and registers a document (copied into sources/custom/). Returns {"ok", "problems", "entry"}."""
    path = Path(path)
    problems = validate(path, meta)
    if problems:
        return {"ok": False, "problems": problems, "entry": None}
    config.custom_dir().mkdir(parents=True, exist_ok=True)
    dest = config.custom_dir() / f"{meta['source_id']}{path.suffix.lower()}"
    if path.resolve() != dest.resolve():
        shutil.copy2(path, dest)
    entry = {"source_id": meta["source_id"], "short": meta.get("short") or meta["title"][:60], "title": meta["title"], "publisher": meta["publisher"], "file": dest.name,
             "version": str(meta["version"]), "date": str(meta["date"]), "kind": meta["kind"], "url": meta.get("url"), "concerns": meta.get("concerns") or ["all"],
             "license": {"indexing_permitted": True, "redistribution": bool((meta.get("license") or {}).get("redistribution"))}, "supersedes": meta.get("supersedes"),
             "note": meta.get("note"), "active": True, "sha256": rag.file_hash(dest), "added": dt.date.today().isoformat()}
    reg = rag.custom_registry()
    reg["sources"].append(entry)
    _save(reg)
    rag.get_index.cache_clear()
    return {"ok": True, "problems": [], "entry": entry}


def set_active(source_id: str, active: bool) -> bool:
    reg = rag.custom_registry()
    hit = False
    for e in reg["sources"]:
        if e["source_id"] == source_id:
            e["active"], hit = active, True
    if hit:
        _save(reg)
        rag.get_index.cache_clear()
    return hit


def status() -> list[dict]:
    """Every registered custom source with its state: active (indexed) or flagged with the reason."""
    active, flagged = rag.custom_entries()
    out = [{"source_id": s["source_id"], "title": s["title"], "version": s["version"], "kind": s["kind"], "state": "active"} for s in active]
    out += [{"source_id": f["source_id"], "title": f["title"], "state": "excluded", "reason": f["reason"]} for f in flagged]
    return out
