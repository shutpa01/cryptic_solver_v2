"""Cheat-blocker validation for leftover store scripts.

Two checks, both run before any DB write:

1. Definition substring rule: the declared definition text must be a
   contiguous substring of the clue text (case-insensitive, with
   non-letter characters allowed to differ). This blocks the
   substituted-definition cheat where the parse uses a DB-friendly
   phrase that isn't in the clue.

2. Definition edge-anchor rule: the definition must sit at the START
   or END of the clue (with optional punctuation / whitespace at the
   orphan edge). Cryptic-clue convention -- definitions are always
   at one end of the clue, never in the middle. This blocks the
   "extend the definition into the wordplay" cheat where Claude
   absorbs wordplay-side words into the def to silence 7b.

Both checks return (ok: bool, reason: str). Store scripts MUST call
validate_definition before writing each clue and skip writes that
fail.

The mechanical rules exist because Claude has repeatedly gamed the
verifier by extending the definition or substituting it. See
feedback_explanation_for_user_not_verifier.md,
session_2026_05_12_cheating_handoff.md, and
feedback_definition_edge_anchored.md. Per-clue exceptions require
explicit user approval; do not bypass.
"""
import re


def _norm_letters(s):
    """Lowercase the string and strip everything that isn't a letter
    or a space."""
    return re.sub(r"[^a-z ]", "", (s or "").lower())


def validate_definition(clue_text, definition):
    """Run the two cheat-blocker checks.

    Returns (ok, reason). ok is True only when both checks pass.
    """
    if not (clue_text and definition):
        return False, "missing clue_text or definition"

    clue_norm = _norm_letters(clue_text).strip()
    def_norm = _norm_letters(definition).strip()
    if not def_norm:
        return False, "definition normalises to empty"

    # CHECK 1: definition is a contiguous substring of clue
    if def_norm not in clue_norm:
        return False, (f"definition '{definition}' is not a contiguous "
                       f"substring of the clue text")

    # CHECK 2: definition is at the start or end of the clue
    # (i.e. one of its edges is also an edge of the clue, ignoring
    # punctuation/whitespace).
    starts_at_zero = clue_norm.startswith(def_norm)
    ends_at_end = clue_norm.endswith(def_norm)
    if not (starts_at_zero or ends_at_end):
        return False, (f"definition '{definition}' is in the middle of "
                       f"the clue -- cryptic convention requires "
                       f"definitions to sit at the start or end")

    return True, "ok"


if __name__ == "__main__":
    # Self-test against today's parses
    import sys
    import sqlite3
    from pathlib import Path
    ROOT = Path(__file__).resolve().parent.parent
    clues_db = ROOT / "data" / "clues_master.db"
    conn = sqlite3.connect(str(clues_db))
    conn.row_factory = sqlite3.Row
    sys.path.insert(0, str(ROOT))
    from scripts._dm17878_parses import CLUES as DM_CLUES
    from scripts._dt31239_parses import CLUES as DT_CLUES

    failures = 0
    for label, clues in [("DM 17878", DM_CLUES), ("DT 31239", DT_CLUES)]:
        print(f"\n=== {label} ===")
        for cid, wtype, defn, expl in clues:
            row = conn.execute(
                "SELECT clue_text FROM clues WHERE id=?",
                (cid,)).fetchone()
            if not row:
                continue
            ok, reason = validate_definition(row["clue_text"], defn)
            if not ok:
                print(f"  FAIL {cid}: {reason}")
                failures += 1
    print(f"\nTotal failures: {failures}")
