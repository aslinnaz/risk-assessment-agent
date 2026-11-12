"""Programmatically verifies catalog citations against the actual passages (existence + keyword support).

Checks (1) the control-card source mappings in catalogs/controls.json and (2) the curated anchors of every interview topic in catalogs/interview.json:
each anchor must exist and its passage must pass the same support check that is used at runtime (rag.support)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: E402
import rag  # noqa: E402


def check_ref(idx, r):
    ch = idx.lookup(r["source_id"], r["ref"])
    if not ch:
        return False, "passage not found"
    txt = " ".join(c["text"].lower() for c in ch)
    hit = [k for k in r["keywords"] if k.lower() in txt]
    return (bool(hit), "ok" if hit else "no keyword found")


def main():
    idx = rag.get_index()
    cat = json.loads((config.CATALOG_DIR / "controls.json").read_text(encoding="utf-8"))
    bad = 0
    for c in cat["controls"]:
        for r in c["sources"]:
            ok, why = check_ref(idx, r)
            if not ok:
                bad += 1
                print(f"NOT SUPPORTED {c['id']} {r['source_id']} {r['ref']}: {why}")
    n_ctl = sum(len(c["sources"]) for c in cat["controls"])
    inter = json.loads((config.CATALOG_DIR / "interview.json").read_text(encoding="utf-8"))
    n_anc = 0
    for t in inter["topics"]:
        for a in t.get("anchors") or []:
            n_anc += 1
            sid, ref = a.split("|", 1)
            ch = idx.lookup(sid, ref)
            if not ch:
                bad += 1
                print(f"ANCHOR NOT FOUND {t['id']} {a}")
                continue
            v = rag.support(ch[0], t.get("hint", "") + " " + t.get("default_q", ""), rag.topic_terms(t), None, anchored=True)
            if not v["ok"]:
                bad += 1
                print(f"ANCHOR WEAK {t['id']} {a}: {v['reason']}")
    print(f"{n_ctl} control mappings and {n_anc} topic anchors checked, {bad} problems")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
