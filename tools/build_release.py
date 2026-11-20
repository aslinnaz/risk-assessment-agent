"""Builds the release ZIP and verifies it. Run from the project root:  python tools/build_release.py --out /path/to/name.zip

What is packaged: the application, catalogs, permitted source PDFs, tests, docs. What is NEVER packaged: .env and other secrets, runtime caches (.cache, embeddings, chunk files),
Git history, virtual environments, bytecode, generated reports, local custom sources (sources/custom), ISO/IEC 42001, Word/ZIP files, and any external development/QA reference
document (the compliance check matrix). `verify()` fails the build if one of them slipped in or if a key-like string is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import rag  # noqa: E402

EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".cache", "node_modules", "reports", "outputs", "custom"}
EXCLUDE_NAMES = {".DS_Store", ".env", "RELEASE_TEST_RESULTS.txt"}
EXCLUDE_SUFFIX = {".pyc", ".pyo", ".docx", ".doc", ".zip", ".npz", ".npy", ".log", ".bak"}
KEY = re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_\-]{30,}")  # not preceded by a letter: "risk-assessment-agent-…" is not a key


def package_files(root: Path = ROOT) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if not p.is_file() or set(rel.parts) & EXCLUDE_DIRS or p.name in EXCLUDE_NAMES or (p.name.startswith(".env") and p.name != ".env.example") or p.suffix.lower() in EXCLUDE_SUFFIX:
            continue
        if "iso_42001" in p.name.lower():
            continue
        out.append(p)
    return out


def verify(files: list[Path], root: Path = ROOT) -> list[str]:
    problems = []
    for p in files:
        rel = str(p.relative_to(root))
        if rag.is_excluded_reference(p.name):
            problems.append(f"{rel}: external development/QA reference document in the package")
        if p.suffix.lower() in {".docx", ".doc", ".zip"} or p.name == ".env" or "iso_42001" in p.name.lower():
            problems.append(f"{rel}: restricted or private file in the package")
        if p.suffix in {".py", ".md", ".json", ".toml", ".txt", ".ini", ".example", ".yml", ""}:
            text = p.read_text(encoding="utf-8", errors="ignore")
            if KEY.search(text):
                problems.append(f"{rel}: key-like string")
            if p.suffix == ".json" and rag.has_excluded_content(text):
                problems.append(f"{rel}: contains text of an external development/QA reference document")
    return problems


def build(out: Path, prefix: str = "trustai-risk-assessment-agent") -> dict:
    files = package_files()
    problems = verify(files)
    if problems:
        raise SystemExit("REFUSED:\n  - " + "\n  - ".join(problems))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"{prefix}/{p.relative_to(ROOT)}")
    return {"zip": str(out), "files": len(files)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    print(json.dumps(build(Path(ap.parse_args().out))))
