"""Local RAG: PDF/Word extraction, article/section-bounded chunking, BM25 + persistent embedding index.

- Cache by file hash: the same file is not re-read / re-embedded.
- Embeddings are built automatically at startup for missing passages (ensure_embeddings); keyword search is the visible fallback.
- Sources with api_transfer_allowed=false (ISO) do not enter the index.
- Source texts are data; instructions inside them are not applied.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

import config

_HDR = re.compile(r"^(OJ L, .*EN|EN OJ L, .*|ELI: http\S+.*)$")


# --------------------------------------------------------------------------- manifest
CUSTOM_KINDS = {"legislation": ("legal", 1), "framework": ("framework", 2), "standard": ("standard", 2), "guidance": ("guidance", 3)}  # registry kind -> (manifest kind, authority)
CUSTOM_CONCERNS = ("legal", "agentic", "oversight", "transparency", "data", "all")


# External development/QA reference documents are NEVER application sources (not indexed, retrieved, cited, packaged or sent to an API). They are recognised by file/title pattern and by
# a content marker. The patterns are assembled here, in code, so that no user-visible catalog or string has to name the document.
_EXCLUDED_NAME = re.compile(r"compliance[\W_]*(?:check[\W_]*)?matri|agentic[\W_]*ai[\W_]*compliance|matrix[\W_]*v?2\b", re.I)
_EXCLUDED_MARKERS = (" ".join(("agentic", "ai", "compliance", "check", "matrix")), " ".join(("compliance", "check", "matrix")))


def is_excluded_reference(*names: str) -> bool:
    """Is a file name, title or source id one of the external development/QA reference documents (for example the compliance check matrix)? Those are never application sources."""
    return any(_EXCLUDED_NAME.search(n or "") for n in names)


def has_excluded_content(text: str) -> bool:
    low = re.sub(r"\s+", " ", (text or "").lower())
    return any(mk in low for mk in _EXCLUDED_MARKERS)


def custom_registry() -> dict:
    path = config.custom_dir() / "registry.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"sources": []}


def custom_entries() -> tuple[list[dict], list[dict]]:
    """(active manifest entries, flagged entries with the reason). Sources added after deployment are never silently treated as authoritative:
    deactivated, superseded, duplicate, missing or not licensed for indexing entries are excluded and reported."""
    base = json.loads((config.CATALOG_DIR / "sources.json").read_text(encoding="utf-8"))["sources"]
    entries = custom_registry()["sources"]
    superseded = {e["supersedes"] for e in entries if e.get("active") and e.get("supersedes")}
    seen = {file_hash(config.SOURCES_DIR / s["file"]) for s in base if s["rag"] and (config.SOURCES_DIR / s["file"]).exists()}
    active, flagged = [], []
    for e in entries:
        f = config.custom_dir() / e["file"]
        reason = None
        if is_excluded_reference(e.get("file", ""), e.get("title", ""), e.get("source_id", ""), e.get("short", "")):
            reason = "excluded reference document (development/QA reference, never an application source)"
        elif not e.get("active"):
            reason = "deactivated"
        elif e["source_id"] in superseded:
            reason = "superseded by a newer active source"
        elif not f.exists():
            reason = f"file not found: {f.name}"
        elif not (e.get("license") or {}).get("indexing_permitted"):
            reason = "indexing is not permitted by the licence information; listed as a bibliographic entry only"
        elif e["kind"] not in CUSTOM_KINDS:
            reason = f"unsupported kind '{e['kind']}'"
        else:
            h = file_hash(f)
            if h in seen:
                reason = "duplicate of a document that is already indexed"
            seen.add(h)
        if reason:
            flagged.append({"source_id": e["source_id"], "title": e.get("title"), "reason": reason})
            continue
        kind, authority = CUSTOM_KINDS[e["kind"]]
        active.append({"source_id": e["source_id"], "short": e.get("short") or e["title"][:60], "title": e["title"], "publisher": e.get("publisher", ""), "file": f"custom/{e['file']}",
                       "path": str(f), "version": e["version"], "date": e["date"], "kind": kind, "authority": authority, "url": e.get("url"), "rag": True, "api_transfer_allowed": True,
                       "custom": True, "concerns": e.get("concerns") or ["all"],
                       "currency_note": e.get("note") or "Added after deployment as an approved source. It supports retrieval and citations only; it never changes the deterministic classification."})
    return active, flagged


def load_manifest() -> dict:
    man = json.loads((config.CATALOG_DIR / "sources.json").read_text(encoding="utf-8"))
    man["sources"] = man["sources"] + custom_entries()[0]
    return man


def source_path(s: dict) -> Path:
    return Path(s["path"]) if s.get("path") else config.SOURCES_DIR / s["file"]


def custom_sources_for(concerns) -> list[str]:
    """Active approved sources added after deployment that address one of the given concerns (retrieval is routed by concern, not hard-wired per source)."""
    want = set(concerns or [])
    return [s["source_id"] for s in custom_entries()[0] if "all" in s["concerns"] or want & set(s["concerns"])]


def source_meta(source_id: str) -> dict:
    for s in load_manifest()["sources"]:
        if s["source_id"] == source_id:
            return s
    raise KeyError(source_id)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- extraction
class SourceError(RuntimeError):
    """A source document could not be parsed into the expected legal structure. Raised loudly; missing references are never hidden."""


_LONE_MARK = re.compile(r"^(\d{1,3}\.|\([a-z]{1,4}\)|\(\d{1,3}\))$")


def _canon_page(text: str) -> str:
    """Extractor-independent layout: a paragraph/point marker that an extractor put on its own line ('1.', '(a)', '(12)') is joined with the text that follows,
    so '1.\\nThe purpose' and '1. The purpose' both parse as numbered paragraph 1."""
    lines = (text or "").split("\n")
    out, i = [], 0
    while i < len(lines):
        cur = lines[i].strip()
        if _LONE_MARK.match(cur) and i + 1 < len(lines) and lines[i + 1].strip():
            out.append(cur + " " + lines[i + 1].lstrip())
            i += 2
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


def extract_pages(path: Path) -> list[str]:
    """PyMuPDF is preferred (no kerning artefacts such as 'Ar ticle'); pypdf is the fallback. Both are normalised by _canon_page."""
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            pages = [p.get_text() for p in doc]
    except ImportError:
        import logging
        logging.getLogger("pypdf").setLevel(logging.ERROR)
        from pypdf import PdfReader
        pages = [(p.extract_text() or "") for p in PdfReader(str(path)).pages]
    return [_canon_page(t) for t in pages]


def _sq(s: str) -> str:
    """Whitespace-free form, used to match structural headings even when an extractor splits letters ('Ar ticle')."""
    return re.sub(r"\s+", "", s or "")


def _flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- chunking
def locator(c: dict) -> str:
    """Where the passage sits in the source file (PDF page, or the Word document)."""
    return f"PDF p. {c['page']}" if c.get("page") else "Word document"


def _mk(src, ref, title, page, text, kind, section=""):
    return {"source_id": src["source_id"], "ref": ref, "title": title, "section": section,
            "version": src["version"], "page": page, "text": text.strip(), "kind": kind,
            "url": src.get("url"), "authority": src["authority"]}


def _split_long(text: str, limit: int = 1800) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;:])\s+", text):
        if len(cur) + len(sent) > limit and cur:
            parts.append(cur)
            cur = ""
        cur += (" " if cur else "") + sent
    if cur:
        parts.append(cur)
    return parts


_FOOTNOTE_START = re.compile(r"^\((\d{1,3})\) (?:Regulations?|Directives?|Council|Commission|Decisions?|Treaty|Judgment|Proposal|Opinion|Report|Communication|Recommendation|Joint|Resolution|Position|Charter|Case|Agreement|Convention|Delegated|Implementing|European|Declaration|Framework)\b")


def _drop_footnotes(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Removes page-bottom footnotes (e.g. '(31) Regulation (EU) 2019/2144 … (OJ L 325, 16.12.2019, p. 1).') from the recital text: they are not part of the recital, and
    a footnote that is glued into a recital would be cited as if the recital said it. A block is dropped only if it starts like a footnote and ends with an '(OJ …' reference on the same page."""
    out, i = [], 0
    while i < len(lines):
        pg, ln = lines[i]
        if _FOOTNOTE_START.match(ln.strip()):
            end = next((k for k in range(i, min(i + 10, len(lines))) if lines[k][0] == pg and re.search(r"\(OJ [^)]*\)\.?\s*$|\bp\. \d+\)\.?\s*$", lines[k][1].strip())), None)
            if end is not None:
                i = end + 1
                continue
        out.append(lines[i])
        i += 1
    return out


def chunk_aia(src, pages):
    if any(re.search(r"\bAr ticle \d|\bHA VE ADOPTED|REGUL A TION", pg) for pg in pages):
        raise SourceError(f"{src['file']}: the PDF text extractor split words into letters ('Ar ticle', 'pur pose'), which makes legal retrieval unreliable. "
                          "Install PyMuPDF (pip install pymupdf, listed in requirements.txt) or upgrade pypdf.")
    lines = []
    for i, t in enumerate(pages, 1):
        for ln in t.split("\n"):
            ln = ln.rstrip()
            if ln and not _HDR.match(ln.strip()):
                lines.append((i, ln))
    chunks = []
    # --- recitals / body split
    idx = next((i for i, (_, l) in enumerate(lines) if _sq(l).startswith("HAVEADOPTEDTHISREGULATION")), None)
    if idx is None:
        raise SourceError(f"{src['file']}: the enacting formula 'HAVE ADOPTED THIS REGULATION' was not found, so recitals and articles cannot be separated. "
                          "Install PyMuPDF (pip install pymupdf) or check that the PDF is the OJ text.")
    rec, body = _drop_footnotes(lines[:idx]), lines[idx + 1:]
    cur, n = None, 1
    for pg, ln in rec:
        m = re.match(r"^\((\d{1,3})\) ", ln)
        if m and int(m.group(1)) == n:
            if cur:
                chunks.append(cur)
            cur = [n, pg, [ln]]
            n += 1
        elif cur:
            cur[2].append(ln)
    if cur:
        chunks.append(cur)
    out = []
    for n_, pg, ls in chunks:
        for j, part in enumerate(_split_long(_flat(" ".join(ls)), 2200)):
            out.append(_mk(src, f"Recital ({n_})" + (f" #{j+1}" if j else ""), "Recital", pg, part, "recital"))
    # --- articles and annexes
    blocks, cur, art_n = [], None, 1
    skip_next = False
    for pg, ln in body:
        s = ln.strip()
        if skip_next:
            skip_next = False
            if s.isupper():
                continue
        if re.match(r"^(CHAPTER|SECTION)[IVX\d]+$", _sq(s)):
            skip_next = True
            continue
        ma = re.match(r"^Article(\d+)$", _sq(s))
        mx = re.match(r"^ANNEX([IVX]+)$", _sq(s))
        if cur is None or cur["type"] == "art":
            if ma and int(ma.group(1)) == art_n:
                cur = {"type": "art", "id": art_n, "page": pg, "lines": []}
                blocks.append(cur)
                art_n += 1
                continue
        if mx and (cur is None or cur["id"] != mx.group(1)):
            cur = {"type": "annex", "id": mx.group(1), "page": pg, "lines": []}
            blocks.append(cur)
            continue
        if cur:
            cur["lines"].append((pg, ln))
    for b in blocks:
        label = f"Art. {b['id']}" if b["type"] == "art" else f"Annex {b['id']}"
        ls = b["lines"]
        title = ls[0][1].strip() if ls else ""
        ls = ls[1:] if b["type"] == "art" else ls
        if b["type"] == "annex":
            title = ls[0][1].strip() if ls else ""
            ls = ls[1:]
        if b["type"] == "art" and b["id"] == 3:  # definitions: (1) ‘AI system’ … (N) — one passage per numbered definition
            defs, curd, nd = [[None, ls[0][0] if ls else b["page"], []]], None, 1
            for pg, ln in ls:
                m = re.match(r"^\((\d{1,2})\) ", ln)
                if m and int(m.group(1)) == nd:
                    defs.append([str(nd), pg, [ln]])
                    nd += 1
                else:
                    defs[-1][2].append(ln)
            for lab, pg, pl in defs:
                txt = _flat(" ".join(pl))
                if not txt:
                    continue
                rf = "Art. 3 (intro)" if lab is None else f"Art. 3({lab})"
                for j, part in enumerate(_split_long(txt, 2200)):
                    out.append(_mk(src, rf + (f" #{j+1}" if j else ""), f"Art. 3 — {title} (definition {lab})" if lab else f"Art. 3 — {title}", pg, part, "article", title))
            continue
        # paragraph split
        paras, curp, nxt = [], [None, ls[0][0] if ls else b["page"], []], 1
        for pg, ln in ls:
            m = re.match(r"^(\d+)\. ", ln)
            if m and int(m.group(1)) == nxt:
                paras.append(curp)
                curp = [str(nxt), pg, [ln]]
                nxt += 1
            else:
                curp[2].append(ln)
        paras.append(curp)
        for lab, pg, pl in paras:
            if not pl:
                continue
            txt = _flat(" ".join(pl))
            ref = label if lab is None else f"{label}({lab})" if b["type"] == "art" else f"{label}, {lab}."
            if lab is None and len(paras) > 1:
                ref = label + " (intro)"
            # sub-point split
            if len(txt) > 1800 and b["type"] == "art":
                items, curi, nl = [], [None, pg, []], 0
                for l2 in pl:
                    m2 = re.match(r"^\(([a-z])\) ", l2)
                    if m2 and ord(m2.group(1)) - 96 == nl + 1:
                        items.append(curi)
                        curi = [m2.group(1), pg, [l2]]
                        nl += 1
                    else:
                        curi[2].append(l2)
                items.append(curi)
                if len(items) > 2:
                    head = _flat(" ".join(items[0][2]))
                    if len(head) >= 300:  # a long introduction is not discarded
                        for j, part in enumerate(_split_long(head, 2200)):
                            out.append(_mk(src, ref + " (intro)" + (f" #{j+1}" if j else ""), f"{label} — {title}", items[0][1], part, "article", title))
                    for il, ipg, ill in items:
                        if il is None:
                            continue
                        pg = ipg
                        t2 = _flat(" ".join(ill))
                        t2 = (head[:300] + " … " + t2) if len(head) < 300 and head else t2
                        for j, part in enumerate(_split_long(t2, 2200)):
                            out.append(_mk(src, f"{ref}({il})" + (f" #{j+1}" if j else ""), f"{label} — {title}", pg, part, "article" if b["type"] == "art" else "annex", title))
                    continue
            for j, part in enumerate(_split_long(txt, 2200)):
                out.append(_mk(src, ref + (f" #{j+1}" if j else ""), f"{label} — {title}", pg, part, "article" if b["type"] == "art" else "annex", title))
    return out


def chunk_decision(src, pages):
    out = []
    for i, t in enumerate(pages, 1):
        t = _flat("\n".join(l for l in t.split("\n") if not _HDR.match(l.strip())))
        if i == 1:
            out.append(_mk(src, "Recitals", "Commission Implementing Decision (EU) 2026/2274", i, t, "recital"))
        elif i == 2:
            out.append(_mk(src, "Art. 1–2", "Articles", i, t, "article"))
        else:
            out.append(_mk(src, "Annex", "Annex — harmonised standard", i, t, "annex"))
    return out


PLAYBOOK_SECTIONS = {
    "TAI-AG": [(3, "01 What Makes Agents Different"), (5, "02 The Anatomy of an Agent"), (7, "03 Where Things Go Wrong"),
               (10, "04 Designing for Oversight"), (12, "05 The Logging Problem"), (14, "Multi-Agent Systems"),
               (19, "Deploying Responsibly"), (22, None), (28, "Glossary"), (32, "References")],
    "TAI-GOV": [(3, "01 The Case for Governance"), (4, "02 Transparency in AI Governance"), (6, "03 EU AI Act Compliance"),
                (7, "04 Agentic AI"), (9, "05 Explainability in Practice"), (11, None)],
}


def _section(source_id, page):
    cur = None
    for start, name in PLAYBOOK_SECTIONS.get(source_id, []):
        if page >= start:
            cur = name
    return cur


def chunk_playbook(src, pages):
    out = []
    for i, t in enumerate(pages, 1):
        t = _flat(t)
        t = re.sub(r"^TrustAI (Agentic )?AI (Governance )?Playbook trustai\.com\.tr ", "", t)
        t = re.sub(r" 2026 \d+$", "", t)
        if len(t) < 80:
            continue
        for j, part in enumerate(_split_long(t, 1500)):
            sec = _section(src["source_id"], i)
            out.append(_mk(src, f"s. {i}" + (f".{j+1}" if j else ""), src["short"] + (f" — {sec}" if sec else ""), i, part, "guidance", sec or ""))
    return out


def chunk_nist(src, pages):
    lines = []
    for i, t in enumerate(pages, 1):
        if i < 6:
            continue
        for ln in t.split("\n"):
            if ln.strip():
                lines.append((i, ln.strip()))
    out, cur = [], None
    pat = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) \d+\.\d+$")
    for pg, ln in lines:
        if pat.match(ln):
            cur = [ln, pg, []]
            out.append(cur)
        elif cur:
            cur[2].append(ln)
    res = []
    for ref, pg, ls in out:
        txt = _flat(" ".join(ls))
        for j, part in enumerate(_split_long(txt, 2000)):
            res.append(_mk(src, ref + (f" #{j+1}" if j else ""), f"NIST AI RMF Playbook — {ref}", pg, part, "guidance"))
    return res


# --------------------------------------------------------------------------- Word (.docx) reading, shared by generic source ingestion
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_blocks(path: Path) -> list[tuple]:
    """Reads the body of a .docx in order: ('p', text) paragraphs and ('t', [[cell, ...], ...]) tables."""
    import xml.etree.ElementTree as ET
    import zipfile
    root = ET.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    body = root.find(_W + "body")

    def text(el):
        return _flat("".join(t.text or "" for t in el.iter(_W + "t")))
    out = []
    for el in body:
        if el.tag == _W + "p":
            t = text(el)
            if t:
                out.append(("p", t))
        elif el.tag == _W + "tbl":
            rows = [[text(tc) for tc in tr.findall(_W + "tc")] for tr in el.findall(_W + "tr")]
            out.append(("t", [r for r in rows if any(r)]))
    return out


_HEADING = re.compile(r"^(\d+(\.\d+){0,3}\.?\s+\S.{0,80}|(Article|Section|Chapter|Annex|Appendix|Part)\s+[\dIVXA-Z]+\b.{0,70}|[A-Z][A-Z0-9 ,&/\-]{4,70})$")


def _heading(line: str) -> bool:
    return len(line) <= 90 and not line.endswith((".", ",", ";", ":")) and bool(_HEADING.match(line))


def chunk_generic(src, content):
    """Generic chunker for approved sources added after deployment (PDF pages or a .docx path): keeps section headings and page numbers, ~1500 characters per passage.
    Passages carry the registry metadata (title, publisher, version, date); nothing is inferred about legal authority beyond the registered kind."""
    kind = "article" if src["kind"] == "legal" else "guidance"
    if isinstance(content, Path):
        units = [(None, p) for k, p in _docx_blocks(content) if k == "p"] + [(None, " | ".join(c for c in row if c)) for k, rows in _docx_blocks(content) if k == "t" for row in rows]
    else:
        units = [(i, ln) for i, pg in enumerate(content, 1) for ln in pg.split("\n")]
    out, buf, section, page, n = [], [], "", None, 0

    def flush():
        nonlocal buf, page, n
        txt = _flat(" ".join(buf))
        if len(txt) >= 60:
            n += 1
            same = sum(1 for o in out if o["page"] == page) if page else 0
            ref = (f"s. {page}" + (f".{same + 1}" if same else "")) if page else f"part {n}"
            out.append(_mk(src, ref, src["short"] + (f" — {section}" if section else ""), page, txt, kind, section))
        buf, page = [], None
    for pg, ln in units:
        s = _flat(ln)
        if not s:
            continue
        if _heading(s):
            flush()
            section = s[:90]
            continue
        buf.append(s)
        page = page or pg
        if sum(len(x) for x in buf) > 1500:
            flush()
    flush()
    return out


CHUNKERS = {"AIA": chunk_aia, "DEC2274": chunk_decision, "TAI-AG": chunk_playbook,
            "TAI-GOV": chunk_playbook, "NIST-PB": chunk_nist}


_TR_CHARS = re.compile(r"[\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc]")
_TR_WORDS = {"ve", "bir", "i\u00e7in", "ile", "olan", "bu", "da", "veya", "gibi", "daha", "tarafindan", "taraf\u0131ndan", "yapay", "zeka", "yasas\u0131", "ajan", "sistem", "kim", "nas\u0131l", "ise", "ancak", "olarak"}


def is_non_english(text: str) -> bool:
    """Heuristic language gate: a passage that is mostly Turkish is not indexed, so no source snippet or citation shown to the user is in another language. (User quotes are never translated.)"""
    words = re.findall(r"[^\W\d_]+", (text or "").lower())
    if len(words) < 12:
        return False
    letters = sum(len(w) for w in words)
    return len(_TR_CHARS.findall(text)) / max(1, letters) > 0.02 or sum(w in _TR_WORDS for w in words) / len(words) > 0.06


def build_chunks() -> list[dict]:
    """Chunks the permitted sources; caches by file hash."""
    man = load_manifest()
    key = hashlib.sha256()
    todo = []
    for s in man["sources"]:
        if not s["rag"] or not s["api_transfer_allowed"]:
            continue
        p = source_path(s)
        if not p.exists():
            continue
        if is_excluded_reference(p.name, s.get("title", ""), s.get("source_id", ""), s.get("file", "")):
            raise SourceError(f"{p.name}: this is an external development/QA reference document, not an application source; it must not be indexed.")
        key.update(file_hash(p).encode())
        todo.append((s, p))
    key.update(Path(__file__).read_bytes())
    cache = config.CACHE_DIR / f"chunks_{key.hexdigest()[:16]}.json"
    if cache.exists():
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if isinstance(cached, list) and cached and not any(has_excluded_content(c["text"]) for c in cached):
                return cached
        except (ValueError, KeyError, TypeError, OSError):
            pass  # a truncated or corrupted cache is never trusted
        cache.unlink(missing_ok=True)  # a damaged cache, or one that contains an excluded reference document, is discarded and rebuilt from the permitted sources
    chunks = []
    for s, p in todo:
        chunker = CHUNKERS.get(s["source_id"]) or chunk_generic
        new = chunker(s, p if (p.suffix == ".docx" and s["source_id"] in CHUNKERS) else extract_pages(p) if p.suffix == ".pdf" else p)
        if s.get("custom"):  # duplicate detection on passage level: text that is already indexed is not added twice
            have = {hashlib.sha1(c["text"].encode()).hexdigest() for c in chunks}
            new = [c for c in new if hashlib.sha1(c["text"].encode()).hexdigest() not in have]
        chunks += new
    chunks = [c for c in chunks if not is_non_english(c["text"])]
    if any(has_excluded_content(c["text"]) for c in chunks):
        raise SourceError("A source contains an external development/QA reference document; it must not be indexed.")
    for c in chunks:  # after Art. 5(1)(h), '(i)' is not a letter but the exception sub-points (i)-(iii)
        if c["source_id"] == "AIA" and c["ref"] == "Art. 5(1)(i)":
            c["ref"] = "Art. 5(1)(h) exceptions (i)–(iii)"
    seen = Counter()
    for c in chunks:
        slug = re.sub(r"[^A-Za-z0-9]+", "-", c["ref"]).strip("-")
        cid = f"{c['source_id']}:{slug}"
        seen[cid] += 1
        c["chunk_id"] = cid if seen[cid] == 1 else f"{cid}~{seen[cid]}"
    tmp = cache.with_name(cache.name + ".tmp")
    tmp.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    tmp.replace(cache)  # atomic: a crash cannot leave a half-written cache behind
    return chunks


# --------------------------------------------------------------------------- index
class Index:
    def __init__(self):
        self.chunks = build_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        self.pos = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        self._tok = [self._tokens(c["title"] + " " + c["ref"] + " " + c["text"]) for c in self.chunks]
        self.df = Counter(t for toks in self._tok for t in set(toks))
        self.avg = sum(len(t) for t in self._tok) / max(1, len(self._tok))
        self.emb = self._load_emb()
        self.validate()
        set_corpus(self.chunks)

    # ----- mandatory structure and anchors (checked after every build; a missing legal reference is an error, never silently skipped)
    def mandatory_anchors(self) -> list[tuple[str, str]]:
        """(source_id, ref) pairs the application depends on: the core Art. 5 / Art. 6 pathway plus every anchor the catalogs cite."""
        need = {("AIA", f"Art. 5(1)({x})") for x in "abcdefgh"} | {("AIA", r) for r in ("Art. 2(1)", "Art. 3", "Art. 6(1)", "Art. 6(2)", "Art. 6(3)", "Art. 6(4)", "Annex I", "Annex III")}
        for t in json.loads((config.CATALOG_DIR / "interview.json").read_text(encoding="utf-8"))["topics"]:
            for a in t.get("anchors") or []:
                sid, ref = a.split("|", 1)
                need.add((sid, ref))
        for c in json.loads((config.CATALOG_DIR / "controls.json").read_text(encoding="utf-8"))["controls"]:
            for r in c["sources"]:
                need.add((r["source_id"], r["ref"]))
        return sorted(need)

    def validate(self) -> None:
        """Raises SourceError if a present source lacks the structure or the references the application needs. Absent sources are reported by the app, not here."""
        have = {c["source_id"] for c in self.chunks}
        problems = []
        if "AIA" in have:
            refs = {re.sub(r" #\d+$", "", c["ref"]) for c in self.chunks if c["source_id"] == "AIA"}
            miss_art = [n for n in range(1, 114) if not any(r == f"Art. {n}" or r.startswith(f"Art. {n}(") or r.startswith(f"Art. {n} ") for r in refs)]
            if miss_art:
                problems.append(f"AI Act articles not found: {miss_art[:12]}")
            miss_annex = [a for a in ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII") if not any(r.startswith(f"Annex {a}") for r in refs)]
            if miss_annex:
                problems.append(f"AI Act annexes not found: {miss_annex}")
            if "Recital (180)" not in refs:
                problems.append("AI Act recital (180) not found")
        for sid, ref in self.mandatory_anchors():
            if sid in have and not self.lookup(sid, ref):
                problems.append(f"mandatory anchor not found: {sid} {ref}")
        if problems:
            raise SourceError("The source documents were not parsed into the expected structure: " + "; ".join(problems[:10]) + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))

    @staticmethod
    def _tokens(s: str) -> list[str]:
        return [w for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 2]

    # ----- embeddings (persistent cache; aligned by passage id + text digest, can be updated piece by piece)
    @staticmethod
    def _key(c: dict) -> str:
        return c["chunk_id"] + "|" + hashlib.sha1(c["text"].encode()).hexdigest()[:10]

    def _emb_path(self) -> Path:
        return config.CACHE_DIR / f"emb_{config.embed_model().replace('/', '_')}.npz"

    def _migrate_legacy(self):
        """Converts an old-format (.npy, order-based) index into an id-based format using the matching old chunk list. No API call."""
        for npy in config.CACHE_DIR.glob(f"emb_{config.embed_model().replace('/', '_')}_*.npy"):
            h = npy.stem.rsplit("_", 1)[1]
            for f in config.CACHE_DIR.glob("chunks_*.json"):
                ch = json.loads(f.read_text(encoding="utf-8"))
                if hashlib.sha256("".join(c["chunk_id"] + c["text"] for c in ch).encode()).hexdigest()[:16] == h:
                    arr = np.load(npy)
                    if len(arr) == len(ch):
                        np.savez(self._emb_path(), keys=np.array([self._key(c) for c in ch]), vecs=arr)
                        return True
        return False

    def _load_emb(self):
        p = self._emb_path()
        if not p.exists() and not self._migrate_legacy():
            self.missing = len(self.chunks)
            return None
        try:
            z = np.load(p)
            keys, vecs = z["keys"].tolist(), z["vecs"]
        except Exception:  # noqa: BLE001  (damaged or truncated file: treated as 'no embeddings yet', rebuilt by ensure_embeddings)
            p.unlink(missing_ok=True)
            self.missing = len(self.chunks)
            return None
        have = dict(zip(keys, range(len(keys))))
        dim = vecs.shape[1]
        arr = np.zeros((len(self.chunks), dim), dtype=np.float32)
        miss = 0
        for i, c in enumerate(self.chunks):
            j = have.get(self._key(c))
            if j is None:
                miss += 1
            else:
                arr[i] = vecs[j]
        self.missing = miss
        return arr if miss < len(self.chunks) else None

    def has_embeddings(self) -> bool:
        return self.emb is not None

    def mode(self) -> str:
        """hybrid = BM25 + embeddings for all passages; partial = some passages keyword-only; keyword-only = no embeddings."""
        if self.emb is None:
            return "keyword-only"
        return "hybrid" if not getattr(self, "missing", 0) else "partial"

    def refresh_embeddings(self) -> bool:
        """Loads the persistent index from disk (if present); makes no API call."""
        if self.emb is None:
            self.emb = self._load_emb()
        return self.emb is not None

    def status(self) -> dict:
        return {"chunks": len(self.chunks), "embeddings": self.emb is not None, "missing": getattr(self, "missing", len(self.chunks)),
                "embed_model": config.embed_model(), "sources": sorted({c["source_id"] for c in self.chunks})}

    def build_embeddings(self, client, batch=256, usage=None) -> int:
        """Embeds only the passages that are not in the index (called by ensure_embeddings at startup). Large batches keep the number of API requests small (the whole corpus is about six requests)."""
        todo = [i for i in range(len(self.chunks)) if self.emb is None or not np.any(self.emb[i])]
        if not todo:
            return 0
        vecs = {}
        for k in range(0, len(todo), batch):
            part = todo[k:k + batch]
            r = client.embeddings.create(model=config.embed_model(), input=[(self.chunks[i]["title"] + "\n" + self.chunks[i]["text"])[:6000] for i in part])
            for i, d in zip(part, r.data):
                vecs[i] = d.embedding
            if usage is not None and getattr(r, "usage", None):
                usage.append({"kind": "embedding_index", "prompt": r.usage.prompt_tokens, "completion": 0})
        dim = len(next(iter(vecs.values())))
        arr = self.emb if self.emb is not None else np.zeros((len(self.chunks), dim), dtype=np.float32)
        for i, v in vecs.items():
            x = np.asarray(v, dtype=np.float32)
            arr[i] = x / (np.linalg.norm(x) + 1e-9)
        final = self._emb_path()
        tmp = final.with_name(final.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array([self._key(c) for c in self.chunks]), vecs=arr)
        tmp.replace(final)  # atomic
        self.emb, self.missing = arr, 0
        return len(todo)

    # ----- search
    def bm25(self, query: str, k1=1.4, b=0.75) -> np.ndarray:
        q = self._tokens(query)
        n = len(self.chunks)
        scores = np.zeros(n)
        for i, toks in enumerate(self._tok):
            tf = Counter(toks)
            dl = len(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avg))
            scores[i] = s
        return scores

    def lookup(self, source_id: str, ref: str) -> list[dict]:
        """Direct retrieval by article/sub-point. 'Art. 5(1)' -> Art. 5(1) and its sub-points (a), (b)…"""
        out = []
        for c in self.chunks:
            if c["source_id"] != source_id:
                continue
            r = re.sub(r" #\d+$", "", c["ref"])
            if r == ref or r.startswith(ref + "(") or r.startswith(ref + " ") or r.startswith(ref + ",") or r.startswith(ref + "."):
                out.append(c)
        return out

    def retrieve(self, query: str, anchors: list[str] | None = None, k: int = 4,
                 query_vec=None, max_anchor: int = 8, source_ids: list[str] | None = None) -> list[dict]:
        """First direct citation (anchor), then hybrid (BM25 + embedding) search.
        Different sources/versions are not merged; each passage carries its own source id."""
        if not self.chunks:  # no source documents loaded: nothing to retrieve (the app says so)
            return []
        res, seen = [], set()
        for a in anchors or []:
            sid, ref = a.split("|", 1)
            for c in self.lookup(sid, ref)[:max_anchor]:
                if c["chunk_id"] not in seen:
                    seen.add(c["chunk_id"])
                    res.append(c)
        if query.strip():
            s = self.bm25(query)
            s = s / (s.max() + 1e-9)
            if self.emb is not None and query_vec is not None:
                qv = np.asarray(query_vec, dtype=np.float32)
                qv /= np.linalg.norm(qv) + 1e-9
                s = 0.5 * s + 0.5 * (self.emb @ qv)
            # authority: legislation > guidance; recitals slightly lower
            pen = np.array([0.0 if c["kind"] != "recital" else 0.08 for c in self.chunks])
            auth = np.array([(5 - c["authority"]) * 0.01 for c in self.chunks])
            s = s + auth - pen
            if source_ids:  # search only in the requested collections
                s = np.where(np.array([c["source_id"] in source_ids for c in self.chunks]), s, -1.0)
            for i in np.argsort(-s)[:k * 3]:
                c = self.chunks[i]
                if c["chunk_id"] not in seen and s[i] > 0.05:
                    seen.add(c["chunk_id"])
                    res.append(c)
                    if len(res) >= len(anchors or []) + k:
                        break
        return res


# --------------------------------------------------------------------------- support validation
# A retrieved passage is only shown as support for a question or finding if it passes this check. Retrieval never changes the classification.
SIM_ANCHOR, SIM_HYBRID = 0.15, 0.30          # minimum embedding similarity (when embeddings are available)
COVER_ANCHOR, COVER_HYBRID, COVER_NO_EMBED = 0.20, 0.34, 0.50     # minimum share of claim keywords/term groups found in the passage
_STOP = set("the and for with that this from have has are was were not but you your can does did which what when where who how any all its into than then them they their there will would could should about also only more most such been being each other these those over under between within without per via one two".split())


def _stem(w: str) -> str:
    return w[:6]


def keywords(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 3 and w not in _STOP}


WINDOW_CHARS, WINDOW_HYBRID = 450, 0.45  # distinctive terms must occur CLOSE together (inside one window of text), not scattered over a long page
_CORPUS: list[str] = []                  # lower-cased passage texts of the loaded index (document frequencies for term weights)
_DF: dict[str, int] = {}


def set_corpus(chunks: list[dict]) -> None:
    _CORPUS[:] = [((c.get("title") or "") + " " + c["text"]).lower() for c in chunks]
    _DF.clear()


def term_weight(group: list[str]) -> float:
    """Specificity of a term group: inverse document frequency of its rarest term in the loaded passages (a word such as 'system' says little; 'biometric' says a lot). 1.0 without a corpus."""
    if not _CORPUS:
        return 1.0
    best = 0.0
    for t in group:
        t = (t or "").lower()
        if t not in _DF:
            _DF[t] = sum(1 for x in _CORPUS if t in x)
        best = max(best, math.log(1 + len(_CORPUS) / (_DF[t] + 0.5)))
    return best


def _window_cov(txt: str, groups: list[list[str]], width: int = WINDOW_CHARS) -> float:
    """Largest IDF-weighted share of term groups that occur together inside one window of `width` characters."""
    w = [term_weight(g) for g in groups]
    ev = sorted((m.start(), gi) for gi, g in enumerate(groups) for t in g if t for m in re.finditer(re.escape(t.lower()), txt))
    best, j, cnt = 0.0, 0, Counter()
    for pos, gi in ev:
        cnt[gi] += 1
        while ev[j][0] < pos - width:
            cnt[ev[j][1]] -= 1
            if cnt[ev[j][1]] == 0:
                del cnt[ev[j][1]]
            j += 1
        best = max(best, sum(w[g] for g in cnt))
    return best / sum(w) if sum(w) else 0.0


def support(chunk: dict, claim: str = "", terms: list[list[str]] | None = None, sim: float | None = None, anchored: bool = False) -> dict:
    """Deterministic support check for a passage (retrieval never changes the classification; this only decides whether a passage may be SHOWN as support).
    terms: list of term groups (any-of); coverage = share of groups present in the passage. Without terms, the claim's keywords are used.
    sim: embedding similarity of the passage to the claim/query, when available.
    A passage found by search (not a curated anchor) must in addition have its distinctive terms close together (window coverage); without embeddings the bar is higher still."""
    txt = ((chunk.get("title") or "") + " " + chunk["text"]).lower()
    if terms:
        groups = terms
        cov = sum(any(t.lower() in txt for t in g) for g in terms) / len(terms)
    else:
        kw = sorted(keywords(claim))
        groups = [[k] for k in sorted(kw, key=lambda k: -term_weight([k]))[:6]]  # a long claim is judged on its six most distinctive words
        have = keywords(txt)
        cov = (len(set(kw) & have) / len(kw)) if kw else 0.0
    cmin = COVER_ANCHOR if anchored else (COVER_HYBRID if sim is not None else COVER_NO_EMBED)
    smin = SIM_ANCHOR if anchored else SIM_HYBRID
    reasons = []
    if cov < cmin:
        reasons.append(f"keyword coverage {cov:.2f} < {cmin}")
    win = None
    if not anchored:
        win = round(_window_cov(txt, groups), 2)
        if win < WINDOW_HYBRID:
            reasons.append(f"distinctive terms are not close together (proximity {win:.2f} < {WINDOW_HYBRID})")
    if not anchored and terms and len(terms) < 2 and claim:  # a single curated term (for example 'prohibited') is too weak on its own: the claim's own distinctive words must be close together too
        kw_groups = [[k] for k in sorted(keywords(claim), key=lambda k: -term_weight([k]))[:6]]
        w2 = round(_window_cov(txt, kw_groups), 2)
        if w2 < WINDOW_HYBRID:
            reasons.append(f"the claim's distinctive words are not close together (proximity {w2:.2f} < {WINDOW_HYBRID})")
    if sim is not None and sim < smin:
        reasons.append(f"embedding similarity {sim:.2f} < {smin}")
    return {"ok": not reasons, "coverage": round(cov, 2), "proximity": win, "sim": None if sim is None else round(float(sim), 2), "reason": "; ".join(reasons)}


def chunk_sim(index: "Index", chunk: dict, qvec) -> float | None:
    if qvec is None or index.emb is None:
        return None
    i = index.pos.get(chunk["chunk_id"])
    if i is None or not np.any(index.emb[i]):
        return None
    qv = np.asarray(qvec, dtype=np.float32)
    return float(index.emb[i] @ (qv / (np.linalg.norm(qv) + 1e-9)))


def topic_terms(topic: dict) -> list[list[str]]:
    """Term groups for a topic: curated 'support_terms' if given, otherwise each distinctive word of the topic hint."""
    if topic.get("support_terms"):
        return topic["support_terms"]
    return [[w] for w in sorted(keywords(topic.get("hint", "")))]


def retrieve_topic(index: "Index", topic: dict, qvec=None, max_total: int = 3, validate: bool = True) -> list[dict]:
    """Topic-specific collections. Fixed anchors (curated, checked by tools/verify_catalog.py) help find the relevant article; the passage best matching the scenario
    is added by hybrid search. Every passage must pass the support check (keywords of the topic + embedding similarity when available), otherwise it is dropped.
    Order within each source = first anchor, best scenario match, remaining anchors; sources are interleaved round-robin so each source keeps at least one passage."""
    per_source, seen = [], set()
    anchors = topic.get("anchors") or []
    terms = topic_terms(topic)
    claim = topic.get("hint", "") + " " + topic.get("default_q", "")
    for sid in dict.fromkeys((topic.get("sources") or ["AIA"]) + custom_sources_for(topic.get("concerns"))):
        mine = [a for a in anchors if a.startswith(sid + "|")]
        anc = [c for c in index.retrieve("", anchors=mine, k=0, source_ids=[sid], max_anchor=2)] if mine else []
        hyb = index.retrieve(topic["hint"], anchors=[], k=2, query_vec=qvec, source_ids=[sid]) if topic.get("hint") else []
        order = []
        for c, anchored in [(x, True) for x in anc[:1]] + [(x, False) for x in hyb[:2]] + [(x, True) for x in anc[1:]]:
            if c["chunk_id"] in seen:
                continue
            if validate:
                v = support(c, claim, terms, chunk_sim(index, c, qvec), anchored=anchored)
                if not v["ok"]:
                    continue
            seen.add(c["chunk_id"])
            order.append(c)
            if len([1 for _ in order]) >= 3:
                break
        per_source.append(order)
    out = []
    for rnd in range(max((len(o) for o in per_source), default=0)):
        for o in per_source:
            if rnd < len(o) and len(out) < max_total:
                out.append(o[rnd])
    return out


def retrieve_for_claim(index: "Index", claim: str, anchors: list[str], terms: list[list[str]] | None, source_ids: list[str], qvec=None, k: int = 2) -> list[dict]:
    """RAG retrieval for a report finding: expected anchors first, then hybrid search restricted to the relevant sources.
    Returns [{'chunk', 'support', 'anchored'}]; the caller decides how to treat unsupported passages (they are never silently shown as support)."""
    out, seen = [], set()
    for a in anchors:
        sid, ref = a.split("|", 1)
        for c in index.lookup(sid, ref)[:2]:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                out.append({"chunk": c, "anchored": True})
    for sid in source_ids:  # search every requested source separately so playbooks and guidance are not crowded out by the legal text
        for c in index.retrieve(claim, anchors=[], k=k, query_vec=qvec, source_ids=[sid]):
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"])
                out.append({"chunk": c, "anchored": False})
    for r in out:
        r["support"] = support(r["chunk"], claim, terms, chunk_sim(index, r["chunk"], qvec), anchored=r["anchored"])
    return out


def ensure_embeddings(index: "Index", client, usage: list | None = None) -> dict:
    """Part of normal startup: loads the saved index and embeds only passages that are missing. Never raises; the result says what happened."""
    index.refresh_embeddings()
    st = index.status()
    if st["embeddings"] and not st["missing"]:
        return {"ok": True, "built": 0, "mode": "hybrid", "error": None}
    if client is None:
        return {"ok": False, "built": 0, "mode": index.mode(), "error": "No API client; semantic search index could not be built (keyword search is used)."}
    try:
        n = index.build_embeddings(client, usage=usage)
        return {"ok": True, "built": n, "mode": index.mode(), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "built": 0, "mode": index.mode(), "error": "Embedding index could not be built: " + re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-***", str(e))[:200]}


def check_sources() -> None:
    """Startup check: every source that the manifest marks as required (rag + allowed for the API) must exist. The message says exactly what to do."""
    missing = [s for s in load_manifest()["sources"] if s["rag"] and s["api_transfer_allowed"] and not s.get("custom") and not (config.SOURCES_DIR / s["file"]).exists()]
    if missing:
        raise SourceError("Required source file(s) not found in " + str(config.SOURCES_DIR) + ": " + "; ".join(f"{s['file']} ({s['short']})" for s in missing)
                          + ". Copy them into the sources/ folder with exactly these names (see sources/README.md). "
                          "Restricted documents such as ISO/IEC 42001 are not required and must not be added to a public repository.")


@lru_cache(maxsize=1)
def get_index() -> Index:
    check_sources()
    return Index()
