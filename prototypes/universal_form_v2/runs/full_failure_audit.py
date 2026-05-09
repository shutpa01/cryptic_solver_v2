"""Full audit of every seed_failure row.

For each failure we:
  1. Re-run the CURRENT translator and verifier on the original clue
     and components JSON.
  2. Record one of three outcomes:
        now_pass   — translator produces a PASS now (rule change recovered it)
        still_fail — same or different translator failure
        verifier_fail — translator builds a form but verifier rejects (new state)
  3. For still-failing cases, classify the precise root cause by
     inspecting residue and components, not just the failure_detail
     text.

Categories (precise — no umbrella buckets):
  A. Op not yet supported  — components has a mechanism (last_letter,
     alternate_letters, core_letters, etc.) the translator can't build
  B. Acrostic indicator missing in DB — the run is structurally an
     acrostic but no clue word is typed `acrostic` in indicators DB
  C. Multi-word indicator only partially matched — residue contains a
     positional word ('start','first','last','top','head','source',
     'opening','originally','beginning','outset') that's part of an
     indicator phrase ("at the start", "from the head") that wasn't
     recognised
  D. Truncated definition — residue contains a content word that looks
     like part of the definition (heuristic: word adjacent to the
     stored definition_text, or alpha word with no positional/cryptic
     role)
  E. Pieces don't concat — data quality, components have wrong pieces
  F. Anagram missing fodder / wordplay-type problem
  G. DD missing left/right def in assembly
  H. Hidden indicator missing
  I. Tokenisation noise (bare punctuation in residue)
  J. Other / uncategorised — flag for manual review
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from prototypes.universal_form_v2.shadow_db import ensure_shadow
from prototypes.universal_form_v2.json_translator import translate_components
from prototypes.universal_form_v2.clipboard_verifier import verify, LINK_WORDS
from prototypes.universal_form_v2.surface import tokenize as _tokenize


# Single-token positional-indicator words (DB-typed) that often appear
# inside multi-word phrases like "at the start", "from the head", etc.
POSITIONAL_INDICATOR_WORDS = {
    "start", "starts", "starting",
    "first", "firstly",
    "last", "lastly",
    "head", "heads", "heading",
    "top", "tops",
    "tip", "tips",
    "end", "ending",
    "opening", "openings",
    "outset",
    "source",
    "originally", "original",
    "beginning",
    "initially", "initial",
    "primarily",
    "leader", "leaders",
    "front",
    "back",
    "middle",
    "centre", "center",
}

# Mechanism strings the translator currently can't handle
SUPPORTED_MECHANISMS = {
    "first_letter",
    "synonym",
    "abbreviation",
    "literal",
    "anagram_fodder",
    "hidden",
}


def _classify_residue_word(word: str, def_phrase: str) -> str:
    """Return a label for a non-link residue word."""
    if not word.isalpha():
        return "tokenisation_noise"
    if word in POSITIONAL_INDICATOR_WORDS:
        return "positional_indicator_word"
    # Heuristic: if the word appears as a token in def_phrase, it's
    # part of the definition we didn't fully consume.
    def_tokens = {t.lower() for t in _tokenize(def_phrase or "")}
    if word.lower() in def_tokens:
        return "definition_token_mismatch"
    return "unclassified"


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    fail_rows = s.execute(
        "SELECT clue_id, clue_text, answer, components_json, "
        "failure_kind, failure_detail FROM seed_failures"
    ).fetchall()

    print(f"Total seed_failures rows to audit: {len(fail_rows)}")
    print()

    now_pass = []
    now_verifier_fail = []
    still_fail = []  # list of (row, current_err_kind, current_err_detail)

    # First pass: re-run translator+verifier
    for fr in fail_rows:
        row = master.execute(
            "SELECT se.id AS se_id, se.clue_id, se.components, se.wordplay_types, "
            "se.definition_text, c.clue_text, c.answer "
            "FROM structured_explanations se "
            "JOIN clues c ON c.id = se.clue_id WHERE c.id = ? LIMIT 1",
            (fr["clue_id"],),
        ).fetchone()
        if not row:
            still_fail.append((fr, "no_se_row", "structured_explanations row missing"))
            continue
        form, err = translate_components(row, db)
        if err:
            still_fail.append((fr, "translation_error", err["detail"]))
            continue
        v = verify(form, row["clue_text"], db)
        if v.verdict == "PASS":
            now_pass.append((fr, form))
        else:
            fail_msgs = "; ".join(
                f"{c.name}: {c.detail}" for c in v.checks if c.status == "fail"
            )
            now_verifier_fail.append((fr, fail_msgs))

    # Coarse counts
    print("=" * 70)
    print(f"Recovered to PASS now:        {len(now_pass)}")
    print(f"Translator builds, verifier-FAIL: {len(now_verifier_fail)}")
    print(f"Still translation_error:      {len(still_fail)}")
    print()

    # Categorise the still_fail set
    print("=" * 70)
    print("Still-failing breakdown")
    print("=" * 70)

    cat_counts = Counter()
    cat_examples = defaultdict(list)

    for fr, err_kind, detail in still_fail:
        clue = fr["clue_text"]
        ans = fr["answer"]
        try:
            comp = json.loads(fr["components_json"]) if fr["components_json"] else {}
        except Exception:
            comp = {}
        pieces = comp.get("ai_pieces", [])
        mechs = {(p.get("mechanism") or "").lower() for p in pieces}

        # Classify by the actual translator error first
        if "not yet supported" in detail or any(
            m and m not in SUPPORTED_MECHANISMS for m in mechs
        ):
            unsupported = sorted(m for m in mechs
                                  if m and m not in SUPPORTED_MECHANISMS)
            label = f"A. unsupported mechanism: {','.join(unsupported)}"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "no matching indicator found" in detail:
            # Sub-classify by piece count
            fl_count = sum(1 for p in pieces
                            if (p.get("mechanism") or "").lower() == "first_letter")
            if fl_count >= 2:
                label = "B. acrostic indicator missing (2+ first_letter pieces)"
            else:
                label = "C. single first_letter, no findable indicator"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "leftover clue word(s) not on LINK_WORDS" in detail:
            # Sub-classify by what kind of non-link words are in the residue
            clue_tokens = [t.lower() for t in _tokenize(clue)]
            consumed = set()
            for p in pieces:
                for t in _tokenize(p.get("clue_word") or ""):
                    consumed.add(t.lower())
            # Approximate def consumption from stored definition_text
            row = master.execute(
                "SELECT definition_text FROM structured_explanations "
                "WHERE clue_id = ? LIMIT 1",
                (fr["clue_id"],),
            ).fetchone()
            def_text = (row["definition_text"] if row else "") or ""
            for t in _tokenize(def_text):
                consumed.add(t.lower())
            residue = [w for w in clue_tokens if w not in consumed]
            non_link = [w for w in residue if w not in LINK_WORDS]
            sub_labels = set()
            for w in non_link:
                sub_labels.add(_classify_residue_word(w, def_text))
            if "positional_indicator_word" in sub_labels:
                label = "D. residue contains positional indicator word (multi-word phrase)"
            elif "tokenisation_noise" in sub_labels and len(sub_labels) == 1:
                label = "E. residue is tokenisation noise only"
            elif "definition_token_mismatch" in sub_labels:
                label = "F. residue includes definition_text tokens"
            elif sub_labels == {"unclassified"}:
                label = "G. residue: unclassified content words"
            else:
                label = "H. residue: mixed causes"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, non_link))
            continue

        if "pieces concat to" in detail:
            label = "I. pieces concat mismatch (data quality)"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "without left_def" in detail:
            label = "J. DD missing left/right def"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "no anagram indicator" in detail:
            label = "K. anagram indicator not in DB"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "no hidden indicator" in detail:
            label = "L. hidden indicator not in DB"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "no fodder pieces" in detail:
            label = "M. anagram with no fodder"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        if "first_letter:" in detail and "must be a single alpha char" in detail:
            label = "N. first_letter data malformed (multi-letter)"
            cat_counts[label] += 1
            cat_examples[label].append((ans, clue, detail))
            continue

        label = "Z. other: " + detail[:80]
        cat_counts[label] += 1
        cat_examples[label].append((ans, clue, detail))

    # Print category counts in order
    for label, n in cat_counts.most_common():
        print(f"  {n:4d}  {label}")
    print()

    # Print examples per category (5 per)
    for label, n in cat_counts.most_common():
        print()
        print("-" * 70)
        print(f"{label} — {n} cases")
        for ex in cat_examples[label][:5]:
            ans, clue, detail = ex
            print(f"  {ans:18s}  {clue}")
            print(f"                      detail/residue: {detail}")
        if n > 5:
            print(f"  ... {n - 5} more not shown")

    # Verifier-fail examples
    if now_verifier_fail:
        print()
        print("=" * 70)
        print(f"Verifier-FAIL after current translator runs ({len(now_verifier_fail)})")
        for fr, msg in now_verifier_fail[:5]:
            print(f"  {fr['answer']:18s}  {fr['clue_text']}")
            print(f"                      {msg[:140]}")


if __name__ == "__main__":
    main()
