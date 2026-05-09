"""Extract a catalog from clipboard-PASS forms.

For each shadow_pipeline JSON, re-verify with the clipboard verifier
(three rules: Assembly / Mechanism / Residue) and collect the forms
that pass. Group by structural signature, build catalog entries per
CATALOG_DESIGN.md, write to JSON.

Output version is v1 (clipboard verifier with fodder integrity, settled
2026-05-05). The old v0 file (catalog_v0.json, built by the deprecated
strict_verifier) is left untouched for comparison.

Read-only — no DB writes, no fixes to the forms.
"""
import json
import sys
import sqlite3
from collections import defaultdict
from pathlib import Path

from .schema import Form, Definition, Node, LEAF_OPERATIONS
from .clipboard_verifier import verify, NO_INDICATOR_OPS
from .salvage_audit import _form_from_dict
from signature_solver.db import RefDB


# Ops that REQUIRE an indicator — the solver must find a clue word for each
INDICATOR_REQUIRED_OPS = {
    "anagram", "reversal", "container", "deletion", "hidden",
    "homophone", "acrostic",
}


def signature(node: Node) -> str:
    """Canonical structural signature string."""
    op = node.operation
    is_leaf = op in LEAF_OPERATIONS and not node.sources
    if is_leaf:
        if op == "positional":
            return f"positional[{node.positional_kind}]"
        return op
    children = [signature(c) for c in node.sources or []]
    if op == "deletion":
        return f"deletion[{node.deletion_kind}]({','.join(children)})"
    if op == "acrostic":
        return f"acrostic[{node.acrostic_kind}]({','.join(children)})"
    return f"{op}({','.join(children)})"


def structure_dict(node: Node) -> dict:
    """Pure-structure tree — no values, no source words, no indicators."""
    op = node.operation
    is_leaf = op in LEAF_OPERATIONS and not node.sources
    if is_leaf:
        d = {"op": op, "leaf": True}
        if op == "positional":
            d["positional_kind"] = node.positional_kind
        return d
    d = {"op": op,
          "children": [structure_dict(c) for c in node.sources or []]}
    if op == "deletion":
        d["deletion_kind"] = node.deletion_kind
    if op == "acrostic":
        d["acrostic_kind"] = node.acrostic_kind
    return d


def indicator_slots(node: Node, path=None) -> list:
    """List of paths to nodes that REQUIRE an indicator."""
    if path is None:
        path = []
    out = []
    op = node.operation
    is_leaf = op in LEAF_OPERATIONS and not node.sources
    if is_leaf:
        if op == "positional":
            out.append(list(path))
        return out
    if op in INDICATOR_REQUIRED_OPS:
        out.append(list(path))
    for i, c in enumerate(node.sources or []):
        out.extend(indicator_slots(c, path + ["children", str(i)]))
    return out


def leaf_kinds(node: Node) -> list:
    op = node.operation
    is_leaf = op in LEAF_OPERATIONS and not node.sources
    if is_leaf:
        if op == "positional":
            return [f"positional[{node.positional_kind}]"]
        return [op]
    out = []
    for c in node.sources or []:
        out.extend(leaf_kinds(c))
    return out


def collect_clipboard_pass(json_paths) -> list:
    """Returns list of (record, form) for every clipboard-PASS form."""
    db = RefDB()
    shadow = Path("data/shadow_blog_v0.db")
    shadow_conn = sqlite3.connect(shadow) if shadow.exists() else None
    out = []
    for p in json_paths:
        if not Path(p).exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        per_clue = data.get("per_clue")
        if not isinstance(per_clue, list):
            continue
        for r in per_clue:
            if not isinstance(r, dict) or r.get("form") is None:
                continue
            clue_text = r.get("clue_text") or r.get("clue") or ""
            if not clue_text:
                continue
            try:
                form = _form_from_dict(r["form"])
            except Exception:
                continue
            try:
                v = verify(form, clue_text, db, shadow_conn)
            except Exception:
                continue
            if v.verdict == "PASS":
                # Normalise the record so build_catalog and example
                # rendering see the same shape across sources.
                rec = dict(r)
                rec.setdefault("clue_text", clue_text)
                out.append((rec, form))
    return out


def collect_from_shadow_db(
        shadow_path: str = "data/shadow_blog_v0.db",
        master_path: str = "data/clues_master.db") -> list:
    """Read every PASS solve from shadow_db.solves, re-verify with the
    current clipboard verifier, and return the list of (record, form)
    that still pass.

    The shadow_db row carries form_json + clue_id but not clue_text;
    we look up clue_text and source / puzzle_number / clue_number from
    clues_master.db.

    Re-verification catches version drift between the verifier that
    wrote the row and the verifier that's being used to extract the
    catalog now."""
    if not Path(shadow_path).exists():
        return []
    if not Path(master_path).exists():
        return []
    db = RefDB()
    shadow_conn = sqlite3.connect(shadow_path)
    master_conn = sqlite3.connect(master_path)

    out = []
    rows = shadow_conn.execute(
        "SELECT clue_id, form_json, answer FROM solves WHERE verdict='PASS'"
    ).fetchall()
    for clue_id, form_json, answer in rows:
        clue_row = master_conn.execute(
            "SELECT clue_text, source, puzzle_number, clue_number "
            "FROM clues WHERE id=?", (clue_id,)).fetchone()
        if clue_row is None:
            continue
        clue_text, source, puzzle_number, clue_number = clue_row
        if not clue_text:
            continue
        try:
            form_dict = json.loads(form_json)
            form = _form_from_dict(form_dict)
        except Exception:
            continue
        try:
            v = verify(form, clue_text, db, shadow_conn)
        except Exception:
            continue
        if v.verdict == "PASS":
            out.append((
                {
                    "clue_id": clue_id,
                    "clue_text": clue_text,
                    "answer": answer,
                    "source": source,
                    "puzzle_number": puzzle_number,
                    "clue_number": clue_number,
                    "blog": "",
                },
                form,
            ))
    return out


def _discover_form_bearing_jsons(runs_dir: str) -> list:
    """Auto-discover JSON files in runs_dir that contain per_clue
    records with `form` fields. Excludes catalog output files."""
    out = []
    for f in sorted(Path(runs_dir).glob("*.json")):
        if f.name in ("catalog_v0.json", "catalog_v1.json"):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        per_clue = data.get("per_clue")
        if not isinstance(per_clue, list):
            continue
        if any(isinstance(r, dict) and r.get("form") is not None
                for r in per_clue):
            out.append(str(f))
    return out


def build_catalog(records) -> list:
    """Group clipboard-PASS forms by signature, build catalog entries.

    Dedup key is (clue_id, signature): a clue contributing the same
    signature from multiple sources counts once. Different clues that
    share a signature each increment the frequency."""
    by_sig: dict = defaultdict(dict)  # sig -> {clue_id: (record, form)}
    for r, form in records:
        sig = signature(form.tree)
        clue_id = r.get("clue_id")
        # Records without a clue_id (rare) are kept as unique entries.
        key = clue_id if clue_id is not None else id(r)
        by_sig[sig][key] = (r, form)

    entries = []
    for sig, items_dict in sorted(by_sig.items(),
                                   key=lambda kv: (-len(kv[1]), kv[0])):
        items = list(items_dict.values())
        first_form = items[0][1]
        entries.append({
            "id": sig,
            "structure": structure_dict(first_form.tree),
            "indicator_slots": indicator_slots(first_form.tree),
            "leaf_kinds": leaf_kinds(first_form.tree),
            "frequency": len(items),
            "examples": [
                {
                    "answer": r["answer"],
                    "clue": r["clue_text"],
                    "blog": r.get("blog") or "",
                    "source": r.get("source", ""),
                    "puzzle_number": r.get("puzzle_number", ""),
                    "clue_id": r.get("clue_id"),
                }
                for r, form in items[:5]  # cap at 5 examples per template
            ],
        })
    return entries


def main():
    runs_dir = "prototypes/universal_form_v2/runs"
    paths = sys.argv[1:] or _discover_form_bearing_jsons(runs_dir)
    print(f"Scanning {len(paths)} JSON file(s) for clipboard-PASS forms...")
    json_records = collect_clipboard_pass(paths)
    print(f"  JSON-file PASSes (re-verified):     {len(json_records)}")

    print("Reading PASS solves from shadow_db.solves...")
    shadow_records = collect_from_shadow_db()
    print(f"  shadow_db PASSes (re-verified):     {len(shadow_records)}")

    records = json_records + shadow_records
    print(f"  Combined records (pre-dedup):       {len(records)}")

    catalog = build_catalog(records)
    print(f"Distinct catalog entries:             {len(catalog)}")
    print()
    print("Top 10 by frequency:")
    for e in catalog[:10]:
        print(f"  {e['frequency']:5d}  {e['id']}")
    out_path = "prototypes/universal_form_v2/runs/catalog_v1.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v1",
            "verifier": "clipboard (three rules + fodder integrity)",
            "source_runs": paths,
            "source_shadow_db": "data/shadow_blog_v0.db",
            "total_clipboard_pass": len(records),
            "entries": catalog,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWritten {out_path}")


if __name__ == "__main__":
    main()
