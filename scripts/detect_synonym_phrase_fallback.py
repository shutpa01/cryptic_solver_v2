"""Detect HIGH clues where the stored explanation claims a multi-word
'synonym of "phrase"' match that is ONLY being verified via the sub-word
fallback — i.e. the phrase as a whole is not in synonyms DB, only one
of its words is. These are the EVOCATIVE-class false HIGHs the current
verifier still rubber-stamps.

Read-only.
"""
import os
import re
import sqlite3
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
REF_DB = os.path.join(ROOT, "data", "cryptic_new.db")

DATE_START = "2026-04-15"
DATE_END = "2026-04-20"

PIECE_PATTERNS = [
    re.compile(r"(\w+)\s*\(\s*synonym\s+of\s+[\"']([^\"']+)[\"']\s*\)", re.IGNORECASE),
    re.compile(r"(\w+)\s*\(\s*synonym\s*=\s*\"([^\"]+)\"\s*\)", re.IGNORECASE),
]


def is_synonym(ref, word, target):
    w, t = word.lower(), target.lower()
    for w1, w2 in [(w, t), (t, w)]:
        row = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = ? AND LOWER(synonym) = ? LIMIT 1",
            (w1, w2),
        ).fetchone()
        if row:
            return True
        row = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition) = ? AND LOWER(answer) = ? LIMIT 1",
            (w1, w2),
        ).fetchone()
        if row:
            return True
    return False


def main():
    conn = sqlite3.connect(CLUES_DB)
    conn.row_factory = sqlite3.Row
    ref = sqlite3.connect(REF_DB)

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
        matches = []
        for pat in PIECE_PATTERNS:
            matches += pat.findall(expl)
        phrase_flags = []
        for result_word, source_word in matches:
            words = source_word.split()
            if len(words) < 2:
                continue  # Single-word phrase — fallback doesn't apply
            # Check: is the full phrase a synonym of result_word?
            if is_synonym(ref, source_word, result_word):
                continue  # Phrase-level match; legitimate
            # Check: is ANY sub-word (>2 chars) a synonym?
            matching_subword = None
            for w in words:
                if len(w) > 2 and is_synonym(ref, w, result_word):
                    matching_subword = w
                    break
            if matching_subword:
                # The "verification" is only coming from a sub-word — suspect
                phrase_flags.append((result_word, source_word, matching_subword))
        if phrase_flags:
            suspects.append((r, phrase_flags))

    print(f"Scanned (non-manual HIGH): {scanned}")
    print(f"Skipped manual: {skipped_manual}")
    print(f"Suspects (synonym-phrase fallback flagged): {len(suspects)}")
    print()

    by_src_date = Counter()
    for r, _ in suspects:
        by_src_date[(r["publication_date"], r["source"])] += 1
    print("By date/source:")
    for k in sorted(by_src_date):
        print(f"  {k[0]}  {k[1]:12s}  {by_src_date[k]}")
    print()

    out = os.path.join(ROOT, "data", "synonym_phrase_suspects.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Synonym-phrase fallback suspects {DATE_START}..{DATE_END}\n")
        f.write(f"# {len(suspects)} suspects\n\n")
        for r, flags in suspects:
            f.write(f"[{r['publication_date']}] {r['source']} #{r['puzzle_number']} "
                    f"{r['clue_number']}{r['direction'][0]} {r['answer']} conf={r['confidence']}\n")
            f.write(f"  clue: {r['clue_text']}\n")
            f.write(f"  def:  {r['definition']}\n")
            f.write(f"  expl: {r['ai_explanation']}\n")
            for rw, sw, mw in flags:
                f.write(f"  FLAG: '{sw}' = {rw} only matches via sub-word '{mw}'\n")
            f.write("\n")
    print(f"Full list: {out}")

    print("\nFirst 10 suspects:")
    for r, flags in suspects[:10]:
        print(f"  [{r['publication_date']}] {r['source']:12s} #{r['puzzle_number']:6s} "
              f"{r['clue_number']}{r['direction'][0]} {r['answer']}")
        print(f"    clue: {r['clue_text']}")
        print(f"    expl: {(r['ai_explanation'] or '')[:120]}")
        for rw, sw, mw in flags:
            print(f"    FLAG: '{sw}' = {rw} only verified via sub-word '{mw}'")
        print()


if __name__ == "__main__":
    main()
