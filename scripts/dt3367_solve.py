"""Per-clue solver helper for DT 3367 leftover processing.

Used by Claude during the leftover review session 2026-05-03. Not part of the
production pipeline. Intended workflow per `feedback_leftover_process.md`:

  1. Live-query the leftover list
  2. For each clue: parse → coverage check → write
  3. Self-check by re-running the leftover query

This script wraps the per-clue work so each solve is a single function call
returning the verifier verdict.
"""

import json
import sqlite3
from pathlib import Path

from sonnet_pipeline.verify_explanation import ExplanationVerifier

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clues_master.db"
REF_PATH = Path(__file__).resolve().parent.parent / "data" / "cryptic_new.db"

_verifier = ExplanationVerifier()


def get_clue(clue_id: int) -> dict:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, clue_text, answer, enumeration, source, puzzle_number, "
        "       clue_number, direction "
        "FROM clues WHERE id=?", (clue_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def run_verify(clue_id: int, definition: str, wordplay_type: str, ai_explanation: str) -> dict:
    """Verify an explanation against the verifier and return the result dict."""
    c = get_clue(clue_id)
    if not c:
        return {"error": "clue not found"}
    return _verifier.verify(
        c["clue_text"], c["answer"], definition, wordplay_type, ai_explanation
    )


def show_check_summary(result: dict) -> None:
    """Print a short summary of verifier output."""
    print(f"  verdict: {result['verdict']:8} score: {result.get('score', 0)}")
    for c in result["checks"]:
        print(f"    [{c['status']:>13}] {c['check']:<18} {c['detail'][:120]}")


def write_result(
    clue_id: int,
    definition: str,
    wordplay_type: str,
    ai_explanation: str,
    confidence_pct: int,
    pieces: list,
    assembly: dict | None = None,
    definition_start: int | None = None,
    definition_end: int | None = None,
) -> None:
    """Write result to clues + structured_explanations.

    confidence_pct: 0..100 (verifier score)
    pieces: list of dicts {mechanism, clue_word, letters}
    assembly: optional dict (e.g. {"op": "anagram", "fodder": "...", "result": "..."})
    """
    components = json.dumps({
        "ai_pieces": pieces,
        "assembly": assembly or {},
        "wordplay_type": wordplay_type,
    })
    confidence = confidence_pct / 100.0
    c = get_clue(clue_id)
    if not c:
        raise ValueError(f"clue {clue_id} not found")
    conn = sqlite3.connect(str(DB_PATH))
    try:
        conn.execute(
            """INSERT OR REPLACE INTO structured_explanations
               (clue_id, source, puzzle_number, clue_number,
                definition_text, definition_start, definition_end,
                wordplay_types, components, model_version, confidence,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'claude_review', ?,
                       CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            (
                clue_id, c["source"], c["puzzle_number"],
                f"{c['clue_number']}{c['direction'][0]}",
                definition, definition_start, definition_end,
                json.dumps([wordplay_type] if isinstance(wordplay_type, str) else list(wordplay_type)),
                components,
                confidence,
            ),
        )
        conn.execute(
            """UPDATE clues SET
                 definition = ?,
                 wordplay_type = ?,
                 ai_explanation = ?,
                 has_solution = 1,
                 reviewed = 1
               WHERE id = ?""",
            (definition, wordplay_type, ai_explanation, clue_id),
        )
        conn.commit()
    finally:
        conn.close()


def queue_enrichment(
    type_: str,
    word: str,
    letters: str,
    answer: str,
    clue_text: str,
    source: str,
    puzzle_number: int,
) -> None:
    """Add a row to pending_enrichments.

    Cross-type dedup: skip if (word, letters) is already pending under any type.
    The verifier accepts a pair from synonyms_pairs OR definition_answers_augmented
    interchangeably, so queueing the same pair twice — once as 'synonym', once as
    'definition' — wastes the reviewer's time. Match: LOWER(word), UPPER(letters).
    """
    conn = sqlite3.connect(str(DB_PATH))
    try:
        existing = conn.execute(
            "SELECT 1 FROM pending_enrichments "
            "WHERE LOWER(word) = LOWER(?) AND UPPER(letters) = UPPER(?) LIMIT 1",
            (word, letters),
        ).fetchone()
        if existing:
            return
        conn.execute(
            """INSERT INTO pending_enrichments
               (type, word, letters, answer, clue_text, source, puzzle_number, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
            (type_, word, letters, answer, clue_text, source, puzzle_number),
        )
        conn.commit()
    finally:
        conn.close()


def coverage_check(pieces: list[dict]) -> list[str]:
    """Return list of (type, word, letters) tuples that aren't in the ref DB.

    Only checks synonym, abbreviation, and definition — letter-positional and
    'from clue' pieces are mechanical and don't need DB lookup.
    """
    ref = sqlite3.connect(f"file:{REF_PATH}?mode=ro", uri=True)
    ref.row_factory = sqlite3.Row
    missing = []
    for p in pieces:
        m = p.get("mechanism")
        word = (p.get("clue_word") or "").strip().lower()
        letters = (p.get("letters") or "").strip()
        if m == "synonym":
            r = ref.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(REPLACE(synonym,' ',''))=? LIMIT 1",
                (word, letters.upper().replace(" ", "")),
            ).fetchone()
            if not r:
                r2 = ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE UPPER(REPLACE(word,' ',''))=? AND LOWER(synonym)=? LIMIT 1",
                    (letters.upper().replace(" ", ""), word),
                ).fetchone()
                if not r2:
                    r3 = ref.execute(
                        "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(REPLACE(answer,' ',''))=? LIMIT 1",
                        (word, letters.upper().replace(" ", "")),
                    ).fetchone()
                    if not r3:
                        missing.append(("synonym", word, letters))
        elif m == "abbreviation":
            r = ref.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
                (word, letters.upper()),
            ).fetchone()
            if not r:
                missing.append(("abbreviation", word, letters))
    ref.close()
    return missing


def coverage_check_definition(definition: str, answer: str) -> bool:
    """Return True if (definition, answer) appears in definition_answers_augmented."""
    ref = sqlite3.connect(f"file:{REF_PATH}?mode=ro", uri=True)
    try:
        r = ref.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(REPLACE(answer,' ',''))=? LIMIT 1",
            (definition.strip().lower(), answer.upper().replace(" ", "")),
        ).fetchone()
        return bool(r)
    finally:
        ref.close()


def coverage_check_indicator(word: str, wordplay_type: str) -> bool:
    """Return True if word is indexed as an indicator of this wordplay_type."""
    ref = sqlite3.connect(f"file:{REF_PATH}?mode=ro", uri=True)
    try:
        r = ref.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=? LIMIT 1",
            (word.strip().lower(), wordplay_type),
        ).fetchone()
        return bool(r)
    finally:
        ref.close()
