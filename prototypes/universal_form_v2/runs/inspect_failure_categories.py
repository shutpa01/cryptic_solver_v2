"""For each major failure category, inspect 8 actual examples and
determine concretely what's happening — not "looks like", but the
specific cause for each example based on the data.

For category C (single first_letter, no indicator):
  - Check: is the source word in the wordplay table as a known
    abbreviation? If yes -> parser mislabel. If no -> something else.

For category D (residue contains positional indicator word):
  - Show the full clue + the residue + the position of the
    positional-indicator-word in the clue. Identify the multi-word
    phrase if any.

For category G (residue: unclassified content words):
  - Show the clue + components pieces + definition_text + residue.
    Identify whether each residue word is plausibly part of definition,
    a missed wordplay piece, or something else.

For category B (acrostic indicator missing):
  - Show the pieces. Identify which clue word should have been the
    acrostic indicator. Look it up in the indicators DB and report
    its actual types.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from prototypes.universal_form_v2.shadow_db import ensure_shadow
from prototypes.universal_form_v2.json_translator import translate_components
from prototypes.universal_form_v2.surface import tokenize as _tokenize
from prototypes.universal_form_v2.clipboard_verifier import LINK_WORDS


def is_known_abbreviation(ref_db, source_word: str, value: str) -> bool:
    """Is (source_word -> value) a known abbreviation in the DB?"""
    rows = ref_db.execute(
        "SELECT 1 FROM wordplay WHERE LOWER(indicator) = LOWER(?) "
        "AND UPPER(substitution) = UPPER(?) "
        "AND category = 'abbreviation' LIMIT 1",
        (source_word, value),
    ).fetchall()
    return bool(rows)


def is_known_synonym(ref_db, source_word: str, value: str) -> bool:
    """Is (source_word -> value) a known synonym?"""
    rows = ref_db.execute(
        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = LOWER(?) "
        "AND UPPER(synonym) = UPPER(?) LIMIT 1",
        (source_word, value),
    ).fetchall()
    return bool(rows)


def get_indicator_types(db, word: str):
    return [t[0] for t in db.get_indicator_types(word.lower())]


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    ref_conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    # Pull all still-failing rows; re-classify; pull examples per category.
    fail_rows = s.execute(
        "SELECT clue_id, clue_text, answer, components_json, "
        "failure_kind, failure_detail FROM seed_failures"
    ).fetchall()

    cat_C = []  # single first_letter, no indicator
    cat_D = []  # residue contains positional indicator word
    cat_G = []  # residue: unclassified content
    cat_B = []  # acrostic indicator missing (2+ first_letter)

    POSITIONAL_INDICATOR_WORDS = {
        "start", "starts", "starting", "first", "firstly",
        "last", "lastly", "head", "heads", "heading",
        "top", "tops", "tip", "tips", "end", "ending",
        "opening", "openings", "outset", "source",
        "originally", "original", "beginning", "initially", "initial",
        "primarily", "leader", "leaders", "front", "back", "middle",
        "centre", "center",
    }

    for fr in fail_rows:
        row = master.execute(
            "SELECT se.id AS se_id, se.clue_id, se.components, "
            "se.wordplay_types, se.definition_text, se.model_version, "
            "c.clue_text, c.answer "
            "FROM structured_explanations se "
            "JOIN clues c ON c.id = se.clue_id WHERE c.id = ? LIMIT 1",
            (fr["clue_id"],),
        ).fetchone()
        if not row:
            continue
        form, err = translate_components(row, db)
        if not err:
            continue  # now-pass, skip

        try:
            comp = json.loads(fr["components_json"]) if fr["components_json"] else {}
        except Exception:
            comp = {}
        pieces = comp.get("ai_pieces", [])
        detail = err["detail"]

        if "no matching indicator found" in detail:
            fl_count = sum(1 for p in pieces
                            if (p.get("mechanism") or "").lower() == "first_letter")
            if fl_count >= 2:
                cat_B.append((fr, row, comp, detail))
            else:
                cat_C.append((fr, row, comp, detail))
        elif "leftover clue word(s) not on LINK_WORDS" in detail:
            clue_tokens = [t.lower() for t in _tokenize(fr["clue_text"])]
            consumed = set()
            for p in pieces:
                for t in _tokenize(p.get("clue_word") or ""):
                    consumed.add(t.lower())
            for t in _tokenize(row["definition_text"] or ""):
                consumed.add(t.lower())
            residue = [w for w in clue_tokens if w not in consumed]
            non_link = [w for w in residue if w not in LINK_WORDS and w.isalpha()]
            if any(w in POSITIONAL_INDICATOR_WORDS for w in non_link):
                cat_D.append((fr, row, comp, non_link))
            else:
                cat_G.append((fr, row, comp, non_link))

    # ============================================================
    # Inspect category C: single first_letter no indicator
    # ============================================================
    print("=" * 70)
    print(f"CATEGORY C — single first_letter, no findable indicator")
    print(f"  Total: {len(cat_C)}.  Checking: is the source word a")
    print(f"  known abbreviation in the DB? If yes => parser mislabel.")
    print("=" * 70)
    print()

    c_mislabel_abbr = 0
    c_mislabel_syn = 0
    c_unknown = 0
    examples = []
    for fr, row, comp, detail in cat_C:
        # Find the first_letter piece that the translator failed to
        # find an indicator for; check if its source-word is a known
        # abbreviation or synonym in the DB.
        fl_pieces = [p for p in comp.get("ai_pieces", [])
                     if (p.get("mechanism") or "").lower() == "first_letter"]
        if not fl_pieces:
            c_unknown += 1
            continue
        p = fl_pieces[0]
        src = (p.get("clue_word") or "").strip()
        val = (p.get("letters") or "").strip()
        # Strip apostrophe-s suffix for lookup
        src_clean = src.rstrip(",.;:!?")
        is_abbr = is_known_abbreviation(ref_conn, src_clean, val)
        is_syn = is_known_synonym(ref_conn, src_clean, val)
        verdict = ("known_abbreviation" if is_abbr
                    else "known_synonym" if is_syn
                    else "neither_known")
        if verdict == "known_abbreviation":
            c_mislabel_abbr += 1
        elif verdict == "known_synonym":
            c_mislabel_syn += 1
        else:
            c_unknown += 1
        examples.append((fr["answer"], fr["clue_text"], src, val, verdict))

    print(f"  source-word→value is a known abbreviation: {c_mislabel_abbr}")
    print(f"  source-word→value is a known synonym:      {c_mislabel_syn}")
    print(f"  neither known (genuine first-letter case?): {c_unknown}")
    print()
    print(f"  10 examples:")
    for ex in examples[:10]:
        print(f"    {ex[0]:18s}  cw={ex[2]!r:25s} val={ex[3]!r}  =>  {ex[4]}")
        print(f"                      clue: {ex[1]}")
    print()

    # ============================================================
    # Inspect category D: residue has positional indicator word
    # ============================================================
    print("=" * 70)
    print(f"CATEGORY D — residue contains positional indicator word")
    print(f"  Total: {len(cat_D)}.  Showing residue + clue text;")
    print(f"  the positional word's neighbours in the clue tell us")
    print(f"  the multi-word phrase.")
    print("=" * 70)
    print()
    for fr, row, comp, non_link in cat_D[:10]:
        clue_tokens = _tokenize(fr["clue_text"])
        clue_lower = [t.lower() for t in clue_tokens]
        # For each positional word in non_link, show its neighbours
        pos_words = [w for w in non_link if w in POSITIONAL_INDICATOR_WORDS]
        print(f"  {fr['answer']:18s}  {fr['clue_text']}")
        print(f"                      residue non-link: {non_link}")
        for pw in pos_words:
            try:
                idx = clue_lower.index(pw)
                # Show window
                lo = max(0, idx - 2)
                hi = min(len(clue_tokens), idx + 3)
                window = " ".join(clue_tokens[lo:hi])
                print(f"                      {pw!r} in context: '{window}'")
            except ValueError:
                pass
    print()

    # ============================================================
    # Inspect category G: residue unclassified content
    # ============================================================
    print("=" * 70)
    print(f"CATEGORY G — residue: unclassified content words")
    print(f"  Total: {len(cat_G)}. Showing clue + pieces + definition")
    print(f"  + residue, so we can see what role residue words play.")
    print("=" * 70)
    print()
    for fr, row, comp, non_link in cat_G[:10]:
        print(f"  {fr['answer']:18s}  {fr['clue_text']}")
        print(f"                      definition_text: {row['definition_text']!r}")
        for p in comp.get("ai_pieces", []):
            print(f"                      piece: "
                  f"{p.get('clue_word')!r} -> {p.get('letters')!r}  "
                  f"({p.get('mechanism')})")
        print(f"                      residue non-link: {non_link}")
    print()

    # ============================================================
    # Inspect category B: acrostic indicator missing
    # ============================================================
    print("=" * 70)
    print(f"CATEGORY B — 2+ first_letter pieces, no acrostic indicator")
    print(f"  Total: {len(cat_B)}. For each, list candidate indicator")
    print(f"  words (positional words in clue) and their DB indicator")
    print(f"  types — see if 'acrostic' is missing.")
    print("=" * 70)
    print()
    for fr, row, comp, detail in cat_B[:10]:
        print(f"  {fr['answer']:18s}  {fr['clue_text']}")
        fl_pieces = [p for p in comp.get("ai_pieces", [])
                     if (p.get("mechanism") or "").lower() == "first_letter"]
        srcs = [p.get("clue_word") for p in fl_pieces]
        print(f"                      first_letter sources: {srcs}")
        # For each token in clue that's a positional indicator word,
        # show its DB types
        clue_tokens = _tokenize(fr["clue_text"])
        for t in clue_tokens:
            tl = t.lower()
            if tl in POSITIONAL_INDICATOR_WORDS:
                types = get_indicator_types(db, tl)
                print(f"                      indicator candidate "
                      f"{t!r}: types = {types}")


if __name__ == "__main__":
    main()
