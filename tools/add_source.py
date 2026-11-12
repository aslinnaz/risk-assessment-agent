"""Command line for approved sources added after deployment. See sources_admin.py for the rules. Run from the project root."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sources_admin as sa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("path")
    a.add_argument("--id", required=True)
    a.add_argument("--title", required=True)
    a.add_argument("--publisher", required=True)
    a.add_argument("--version", required=True)
    a.add_argument("--date", required=True)
    a.add_argument("--kind", required=True, choices=["legislation", "framework", "guidance", "standard"])
    a.add_argument("--concerns", default="all", help="comma separated: legal, agentic, oversight, transparency, data, all")
    a.add_argument("--url")
    a.add_argument("--short")
    a.add_argument("--supersedes", help="source_id of the older version this document replaces")
    a.add_argument("--indexing-permitted", action="store_true", help="confirm that the licence allows indexing and sending passages to the embedding API")
    a.add_argument("--redistribution", action="store_true", help="the file may be redistributed with the application")
    sub.add_parser("list")
    for name in ("activate", "deactivate"):
        sub.add_parser(name).add_argument("id")
    args = ap.parse_args()
    if args.cmd == "add":
        meta = {"source_id": args.id, "title": args.title, "publisher": args.publisher, "version": args.version, "date": args.date, "kind": args.kind, "url": args.url, "short": args.short,
                "concerns": [c.strip() for c in args.concerns.split(",") if c.strip()], "supersedes": args.supersedes,
                "license": {"indexing_permitted": args.indexing_permitted, "redistribution": args.redistribution}}
        r = sa.add_source(args.path, meta)
        if not r["ok"]:
            print("REJECTED:\n  - " + "\n  - ".join(r["problems"]))
            return 1
        print(f"Registered {args.id}. Restart the application; the index is rebuilt automatically (new passages are embedded at startup).")
        return 0
    if args.cmd == "list":
        for s in sa.status():
            print(s["source_id"].ljust(16), s["state"].ljust(9), s["title"], ("— " + s["reason"]) if s.get("reason") else "")
        return 0
    ok = sa.set_active(args.id, args.cmd == "activate")
    print("done" if ok else f"unknown source id {args.id}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
