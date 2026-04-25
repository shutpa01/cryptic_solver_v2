"""Detect HIGH clues where the stored explanation claims a positional
extraction (first/last/middle/letter N of X) but the claimed letter(s)
don't actually match the source word.

E.g. `E (last letter of "conclusion")` — last of "conclusion" is N, not E.
This is the bug class that let EVOCATIVE through HIGH.

Read-only.
"""
import os
import re
import sqlite3
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")

DATE_START = "2026-04-15"
DATE_END = "2026-04-20"

# Patterns: capture (letters, positional-description, source-word)
PATTERNS = [
    # X (first letter of "word") / X (first letter of 'word')
    re.compile(r"([A-Z]+)\s*\(\s*first\s+letters?\s+of\s+[\"']?([\w\s]+?)[\"']?\s*\)", re.IGNORECASE),
    re.compile(r"([A-Z]+)\s*\(\s*last\s+letters?\s+of\s+[\"']?([\w\s]+?)[\"']?\s*\)", re.IGNORECASE),
    re.compile(r"([A-Z]+)\s*\(\s*middle\s+letters?\s+of\s+[\"']?([\w\s]+?)[\"']?\s*\)", re.IGNORECASE),
    re.compile(r"([A-Z]+)\s*\(\s*initial\s+of\s+[\"']?([\w\s]+?)[\"']?\s*\)", re.IGNORECASE),
    re.compile(r"([A-Z]+)\s*\(\s*final\s+of\s+[\"']?([\w\s]+?)[\"']?\s*\)", re.IGNORECASE),
]


def check_positional(kind, letters, source):
    """Return True if letters correctly match the positional extraction."""
    src_letters = re.sub(r"[^A-Za-z]", "", source)
    if not src_letters:
        return None  # Can't check
    L = letters.upper()
    S = src_letters.upper()
    n = len(L)
    if kind == "first":
        return S.startswith(L)
    if kind == "last":
        return S.endswith(L)
    if kind == "middle":
        if len(S) < n:
            return False
        start = (len(S) - n) // 2
        return S[start:start + n] == L
    if kind == "initial":
        return n == 1 and S.startswith(L)
    if kind == "final":
        return n == 1 and S.endswith(L)
    return None


KINDS = ["first", "last", "middle", "initial", "final"]


def main():
    conn = sqlite3.connect(CLUES_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT c.id, c.source, c.publication_date, c.puzzle_number, c.clue_number,
               c.direction, c.clue_text, c.answer, c.definition, c.wordplay_type,
               c.ai_explanation,
               se.confidence, se.model_version
        FROM clues c
        JOIN structured_explanations se ON se.clue_id = c.id
        WHERE se.confidence >= 0.70
          AND c.publication_date BETWEEN date(?) AND date(?)
          AND c.ai_explanation IS NOT NULL AND c.ai_explanation != ''
        ORDER BY c.publication_date, c.source, c.puzzle_number
        """,
        (DATE_START, DATE_END),
    ).fetchall()

    suspects = []
    skipped_manual = 0
    scanned = 0
    for r in rows:
        mv = r["model_version"] or ""
        if mv in ("manual_approve", "claude_review"):
            skipped_manual += 1
            continue
        scanned += 1
        expl = r["ai_explanation"] or ""

        flags = []
        for kind, pat in zip(KINDS, PATTERNS):
            for letters, source in pat.findall(expl):
                source = source.strip().strip(",.;:")
                ok = check_positional(kind, letters, source)
                if ok is False:
                    flags.append((kind, letters, source))

        if flags:
            suspects.append((r, flags))

    print(f"Scanned (non-manual HIGH): {scanned}")
    print(f"Skipped manual: {skipped_manual}")
    print(f"Suspects (positional lie): {len(suspects)}")
    print()
    by_src_date = Counter()
    for r, _ in suspects:
        by_src_date[(r["publication_date"], r["source"])] += 1
    for k in sorted(by_src_date):
        print(f"  {k[0]}  {k[1]:12s}  {by_src_date[k]}")
    print()

    out = os.path.join(ROOT, "data", "positional_lie_suspects.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Positional lie suspects {DATE_START}..{DATE_END}\n")
        f.write(f"# {len(suspects)} suspects\n\n")
        for r, flags in suspects:
            f.write(f"[{r['publication_date']}] {r['source']} #{r['puzzle_number']} "
                    f"{r['clue_number']}{r['direction'][0]} {r['answer']} conf={r['confidence']}\n")
            f.write(f"  clue: {r['clue_text']}\n")
            f.write(f"  def:  {r['definition']}\n")
            f.write(f"  expl: {r['ai_explanation']}\n")
            for kind, letters, source in flags:
                f.write(f"  LIE: claims '{letters}' is {kind} of '{source}' — FALSE\n")
            f.write("\n")
    print(f"Full list: {out}")

    print("\nFirst 20 suspects:")
    for r, flags in suspects[:20]:
        print(f"  [{r['publication_date']}] {r['source']:12s} #{r['puzzle_number']:6s} "
              f"{r['clue_number']}{r['direction'][0]} {r['answer']}")
        print(f"    clue: {r['clue_text']}")
        for kind, letters, source in flags:
            print(f"    LIE: '{letters}' is NOT the {kind} letter(s) of '{source}'")
        print()


if __name__ == "__main__":
    main()
