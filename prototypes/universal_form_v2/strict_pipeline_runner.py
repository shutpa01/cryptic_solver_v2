"""Strict pipeline runner.

Per architecture (memory/universal_form_architecture.md):

  1. Parse blog → candidate form (existing db_anchored_mapper)
  2. Strict-verify (strict_verifier — 7 checks)
  3. Only forms that PASS contribute confirmed shadow rows
  4. Multi-round: forms that needed peer-confirmed shadow rows can pass
     in a later round
  5. Output: catalog candidates + FAIL-mode worklist

No production touched. No DB writes outside the shadow file.
"""
import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .schema import Form, Definition, Node, LEAF_OPERATIONS
from .strict_verifier import verify
from .db_anchored_mapper import map_clue_words_db
from .assembly_enumerator import assemble
from . import shadow_db
from signature_solver.db import RefDB


def fetch_clues(source: str, puzzle_range: str = None,
                limit: int = None) -> list:
    conn = sqlite3.connect("data/clues_master.db")
    sql = (
        "SELECT id, source, puzzle_number, clue_number, direction, "
        "clue_text, answer, definition, explanation "
        "FROM clues "
        "WHERE source=? AND explanation IS NOT NULL AND explanation != '' "
        "AND answer IS NOT NULL"
    )
    args = [source]
    if puzzle_range:
        lo, hi = puzzle_range.split("-")
        sql += " AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?"
        args.extend([int(lo), int(hi)])
    sql += " ORDER BY CAST(puzzle_number AS INTEGER), direction, "
    sql += "CAST(clue_number AS INTEGER)"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, args).fetchall()
    keys = ["clue_id", "source", "puzzle_number", "clue_number",
            "direction", "clue_text", "answer", "definition_db",
            "blog"]
    return [dict(zip(keys, r)) for r in rows]


def parse_to_form(clue: dict, conn) -> Form:
    """Parse one clue's blog into a candidate form, or None."""
    try:
        m = map_clue_words_db(
            clue["clue_text"],
            clue["answer"],
            clue["blog"],
            conn,
            clue.get("definition_db"),
        )
    except Exception as e:
        return None, f"map_clue_words_db error: {e}"
    if m is None or not m.tags:
        return None, "no mapping produced"
    try:
        a = assemble(m)
    except Exception as e:
        return None, f"assemble error: {e}"
    if a is None or a.form is None:
        return None, "no form produced"
    return a.form, None


def confirmed_shadow_rows_from_form(form: Form, db: RefDB) -> list:
    """Extract leaf (source, value) rows that should go in shadow.

    Only emits rows where the (source, value) is NOT already in the
    live DB — strict verifier already confirmed they're true; no need
    to duplicate live rows.
    """
    rows = []

    def walk(n: Node):
        if n.operation in LEAF_OPERATIONS and not n.sources:
            src = (n.source_word or "").strip().lower()
            val = (n.value or "").upper()
            if not src or not val:
                return
            op = n.operation
            if op == "synonym":
                if val not in db.get_synonyms(src):
                    rows.append({
                        "kind": "synonym",
                        "source_word": src,
                        "value": val,
                        "evidence": "leaf of strict-PASS form",
                    })
            elif op == "abbreviation":
                if val not in db.get_abbreviations(src):
                    rows.append({
                        "kind": "abbreviation",
                        "source_word": src,
                        "value": val,
                        "evidence": "leaf of strict-PASS form",
                    })
            elif op == "homophone":
                if val not in db.get_homophones(src):
                    rows.append({
                        "kind": "homophone",
                        "source_word": src,
                        "value": val,
                        "evidence": "leaf of strict-PASS form",
                    })
            # literal/raw/positional don't produce DB rows
        for c in n.sources or []:
            walk(c)

    if form is not None:
        walk(form.tree)
    return rows


def confirmed_indicator_rows_from_form(form: Form, db: RefDB) -> list:
    """Extract indicators that the form uses but aren't in live DB."""
    rows = []
    for n in _all_nodes(form.tree):
        if not n.indicator:
            continue
        op = n.operation
        # Map op -> DB type
        if op in ("anagram", "reversal", "container", "deletion",
                   "hidden", "homophone", "acrostic"):
            text = n.indicator.lower().strip(",.;:!?\"'()-")
            types = db.get_indicator_types(text)
            db_types_required = {
                "anagram": "anagram",
                "reversal": "reversal",
                "container": "container",
                "deletion": "deletion",
                "hidden": "hidden",
                "homophone": "homophone",
                "acrostic": "acrostic",
            }[op]
            if not any(t[0] == db_types_required for t in types):
                rows.append({
                    "kind": "indicator",
                    "source_word": text,
                    "value": text,
                    "operation": op,
                    "subtype": None,
                    "evidence": "indicator of strict-PASS form",
                })
        elif op == "positional" and not n.sources:
            text = n.indicator.lower().strip(",.;:!?\"'()-")
            types = db.get_indicator_types(text)
            if not any(t[0] in ("parts", "acrostic", "alternating",
                                 "selection") for t in types):
                rows.append({
                    "kind": "indicator",
                    "source_word": text,
                    "value": text,
                    "operation": "positional",
                    "subtype": n.positional_kind,
                    "evidence": "positional indicator of strict-PASS form",
                })
    return rows


def _all_nodes(node):
    yield node
    for c in node.sources or []:
        yield from _all_nodes(c)


def run(source, puzzle_range=None, limit=None, max_rounds=3,
        out_path=None):
    print(f"Fetching clues: source={source} range={puzzle_range} "
          f"limit={limit}")
    clues = fetch_clues(source, puzzle_range, limit)
    print(f"  {len(clues)} clues with blogs")

    # Fresh shadow DB
    shadow_conn = shadow_db.reset_shadow()
    db = RefDB()
    live_conn = sqlite3.connect("data/cryptic_new.db")

    # Parse all clues once (parsing itself is deterministic)
    print("\nParsing all clues...")
    parsed = {}
    parse_errors = Counter()
    for clue in clues:
        form, err = parse_to_form(clue, live_conn)
        parsed[clue["clue_id"]] = form
        if err:
            parse_errors[err.split(":")[0]] += 1
    print(f"  parsed: {sum(1 for f in parsed.values() if f is not None)}"
          f" / {len(parsed)} produced a candidate form")

    # Multi-round verification
    verdicts = {}
    fail_modes = Counter()
    for round_num in range(max_rounds):
        new_passes = 0
        round_writes = Counter()
        for clue in clues:
            cid = clue["clue_id"]
            if verdicts.get(cid, {}).get("verdict") == "PASS":
                continue
            form = parsed[cid]
            if form is None:
                verdicts[cid] = {"verdict": "NO_FORM", "checks": []}
                continue
            v = verify(form, clue["clue_text"], db, shadow_conn)
            verdicts[cid] = v.to_dict()
            if v.verdict == "PASS":
                new_passes += 1
                # Write confirmed shadow rows
                rows = (confirmed_shadow_rows_from_form(form, db)
                        + confirmed_indicator_rows_from_form(form, db))
                if rows:
                    counts = shadow_db.write_candidates(rows, shadow_conn)
                    for k, n in counts.items():
                        if k != "skipped":
                            round_writes[k] += n
        # Tally fail modes
        for cid, vd in verdicts.items():
            if vd.get("verdict") not in ("PASS", "NO_FORM"):
                for c in vd.get("checks", []):
                    if c.get("status") == "fail":
                        fail_modes[c["name"]] += 1
        # Reset fail_modes per round to avoid double-count: actually
        # we only want the latest; rebuild from current verdicts:
        fail_modes = Counter()
        for cid, vd in verdicts.items():
            if vd.get("verdict") == "FAIL":
                for c in vd.get("checks", []):
                    if c.get("status") == "fail":
                        fail_modes[c["name"]] += 1
        print(f"  round {round_num + 1}: +{new_passes} PASS, "
              f"shadow writes: {dict(round_writes)}")
        if new_passes == 0:
            break

    # Final tally
    summary = Counter()
    for vd in verdicts.values():
        summary[vd.get("verdict", "NO_FORM")] += 1
    print(f"\nFinal: {dict(summary)}")
    print(f"Fail modes: {dict(fail_modes)}")
    print(f"Parse errors: {dict(parse_errors)}")

    # Write per-clue results
    if out_path:
        per_clue = []
        for clue in clues:
            cid = clue["clue_id"]
            form = parsed[cid]
            per_clue.append({
                **clue,
                "form": form.to_dict() if form else None,
                "verdict": verdicts.get(cid, {}),
            })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "n": len(clues),
                "summary": dict(summary),
                "fail_modes": dict(fail_modes),
                "parse_errors": dict(parse_errors),
                "per_clue": per_clue,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nWritten {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="times")
    ap.add_argument("--puzzle-range")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.source, args.puzzle_range, args.limit,
        args.max_rounds, args.out)


if __name__ == "__main__":
    main()
