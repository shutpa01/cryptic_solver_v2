"""
enrichment/api_gap_finder.py — Uses Claude API to identify missing DB entries
for partial solves.

For each partial clue (letters_still_needed != ''), sends the full solve context
to Claude and asks what entries would eliminate the remaining letters AND give
every unresolved clue word a role.

All inserts are tagged for easy removal:
  DELETE FROM synonyms_pairs WHERE source = 'api_gap_finder'
  DELETE FROM indicators    WHERE source = 'api_gap_finder'
  DELETE FROM wordplay      WHERE notes  LIKE 'api_gap_finder%'

Note: homophone inserts cannot be bulk-rolled-back (no source column in homophones table).
Track them manually from the printed output if rollback is needed.

The script does NOT re-run the main pipeline or touch the report.
Run report.py separately to verify results.

Usage:
  python -m enrichment.api_gap_finder
  python -m enrichment.api_gap_finder --dry-run
  python -m enrichment.api_gap_finder --run-id 0
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Windows terminals default to cp1252; force UTF-8 so → and other symbols print cleanly.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import anthropic
from dotenv import load_dotenv

import enrichment.common as _common
from enrichment.common import (
    get_cryptic_conn, get_pipeline_conn,
    insert_synonym_pair, insert_indicator, insert_wordplay, insert_homophone,
    InsertCounter, add_common_args, apply_common_args,
)

load_dotenv()

SOURCE_TAG = 'api_gap_finder'
MODEL = 'claude-sonnet-4-6'

# How each indicator role type transforms its source word
INDICATOR_OPERATIONS = {
    'reversal_indicator':   'reverses the letters of the source word',
    'anagram_indicator':    'scrambles the letters of the source word',
    'container_indicator':  'places one component inside another',
    'insertion_indicator':  'inserts one component into another',
    'deletion_indicator':   'removes letters from the source word',
    'parts_indicator':      'extracts specific letters from the source word',
    'homophone_indicator':  'uses the sound-alike of the source word',
}

# Subtype descriptions for parts indicators
PARTS_SUBTYPE_OPS = {
    'first_use':   'takes the FIRST letter only',
    'last_use':    'takes the LAST letter only',
    'outer_use':   'takes the FIRST and LAST letters only',
    'inner_use':   'takes the MIDDLE letters (removes first and last)',
    'head_delete': 'removes the first letter',
    'tail_delete': 'removes the last letter',
    'initial':     'takes the FIRST letter only',
    'final':       'takes the LAST letter only',
}


# ============================================================
# LOAD FAILURES
# ============================================================

def load_partial_failures(run_id: int = 0) -> List[dict]:
    """Load partial clues from stage_secondary and stage_general.

    Covers two cases:
    - letters_still_needed != '' : formula incomplete, letters unaccounted for
    - fully_resolved = 0 AND letters_still_needed = '' : formula complete but
      some clue words still have no confirmed role (Wire Principle violation)

    stage_secondary is queried first (higher priority — these clues reached
    secondary processing, meaning they have a complete formula but unresolved
    words). stage_general is queried second; the seen-set prevents duplicates.
    """
    conn = get_pipeline_conn()
    conn.row_factory = sqlite3.Row

    # stage_secondary uses improved_formula instead of formula
    sec_rows = conn.execute(
        """SELECT clue_id, clue_text, answer, letters_still_needed,
                  unresolved_words, word_roles, improved_formula AS formula
           FROM stage_secondary
           WHERE run_id = ?
             AND unresolved_words IS NOT NULL
             AND unresolved_words != '[]'
             AND (
               (letters_still_needed IS NOT NULL AND letters_still_needed != '')
               OR (fully_resolved = 0 AND (letters_still_needed IS NULL OR letters_still_needed = ''))
             )
           ORDER BY clue_id""",
        (run_id,)
    ).fetchall()

    gen_rows = conn.execute(
        """SELECT clue_id, clue_text, answer, letters_still_needed,
                  unresolved_words, word_roles, formula
           FROM stage_general
           WHERE run_id = ?
             AND unresolved_words IS NOT NULL
             AND unresolved_words != '[]'
             AND (
               (letters_still_needed IS NOT NULL AND letters_still_needed != '')
               OR (fully_resolved = 0 AND (letters_still_needed IS NULL OR letters_still_needed = ''))
             )
           ORDER BY clue_id""",
        (run_id,)
    ).fetchall()

    conn.close()

    results = []
    seen = set()
    for row in list(sec_rows) + list(gen_rows):
        answer = (row['answer'] or '').upper().replace(' ', '')
        if not answer or answer in seen:
            continue
        seen.add(answer)
        results.append({
            'clue_id':              row['clue_id'],
            'clue_text':            row['clue_text'] or '',
            'answer':               answer,
            'letters_still_needed': (row['letters_still_needed'] or '').upper(),
            'unresolved_words':     json.loads(row['unresolved_words'] or '[]'),
            'word_roles':           json.loads(row['word_roles'] or '[]'),
            'formula':              row['formula'] or '',
        })
    return results


# ============================================================
# PROMPT BUILDING
# ============================================================

def _get_role(r) -> str:
    """Return the role type from either a dict or a list/tuple role entry."""
    if isinstance(r, dict):
        return r.get('role', '')
    if isinstance(r, (list, tuple)) and len(r) > 1:
        return r[1]
    return ''


def _get_word(r) -> str:
    if isinstance(r, dict):
        return r.get('word', '')
    if isinstance(r, (list, tuple)) and len(r) > 0:
        return r[0]
    return ''


def _get_contributes(r) -> str:
    """Return what letters this role contributes."""
    if isinstance(r, dict):
        return r.get('contributes', '')
    if isinstance(r, (list, tuple)) and len(r) > 2:
        return r[2]
    return ''


def _get_source(r) -> str:
    if isinstance(r, dict):
        return r.get('source', '')
    if isinstance(r, (list, tuple)) and len(r) > 3:
        return r[3]
    return ''


def _describe_active_indicators(word_roles: list) -> str:
    """Describe each found indicator and what it does, in plain English."""
    lines = []
    for role in word_roles:
        word      = _get_word(role)
        role_type = _get_role(role)
        source    = _get_source(role)

        if not role_type or role_type not in INDICATOR_OPERATIONS:
            continue

        base_op = INDICATOR_OPERATIONS[role_type]

        # Try to extract subtype from source string e.g. "database (initial)"
        if role_type == 'parts_indicator':
            src_lower = source.lower()
            for st, st_op in PARTS_SUBTYPE_OPS.items():
                if st in src_lower:
                    base_op = st_op
                    break

        lines.append(f'  "{word}" is a {role_type} — {base_op}')

    return '\n'.join(lines) if lines else '  (none)'


def _describe_found_parts(word_roles: list) -> str:
    """Summarise what letters have already been identified and from which words."""
    lines = []
    for role in word_roles:
        word      = _get_word(role)
        role_type = _get_role(role)
        letters   = _get_contributes(role)
        source    = _get_source(role)

        if not role_type or role_type in ('unresolved',):
            continue
        if role_type in INDICATOR_OPERATIONS:
            continue  # indicators described separately

        desc = role_type
        if letters:
            desc += f' = {letters}'
        if source:
            desc += f'  ({source})'
        lines.append(f'  "{word}" : {desc}')

    return '\n'.join(lines) if lines else '  (nothing found yet)'


def _has_homophone_indicator(word_roles: list) -> bool:
    return any(_get_role(r) == 'homophone_indicator' for r in word_roles)


def _is_acrostic_gap(failure: dict) -> bool:
    """Return True if this is a code gap, not a DB gap.

    When a parts/first_use indicator is active and the first letters of the
    unresolved words spell letters_still_needed, the pipeline simply failed to
    continue the acrostic extraction. No DB entry can fix this.
    """
    word_roles = failure['word_roles']
    letters    = failure['letters_still_needed'].upper()
    unresolved = failure['unresolved_words']

    if not letters or not unresolved:
        return False

    has_first_indicator = any(
        _get_role(r) in ('parts_indicator', 'acrostic_indicator', 'letter_selector')
        for r in word_roles
    )
    if not has_first_indicator:
        return False

    first_letters = ''.join(
        w[0].upper() for w in unresolved if w and w[0].isalpha()
    )
    return first_letters == letters


def build_prompt(failure: dict) -> str:
    clue       = failure['clue_text']
    answer     = failure['answer']
    letters    = failure['letters_still_needed']
    unresolved = failure['unresolved_words']
    word_roles = failure['word_roles']
    formula    = failure['formula']

    found_parts       = _describe_found_parts(word_roles)
    active_indicators = _describe_active_indicators(word_roles)
    unresolved_text   = ', '.join(f'"{w}"' for w in unresolved)
    is_homophone      = _has_homophone_indicator(word_roles)

    # Build a list of attributed indicator words so the model can spot extensions
    attributed_indicator_words = [
        _get_word(role).lower()
        for role in word_roles
        if _get_role(role) in INDICATOR_OPERATIONS
    ]

    homophone_section = ''
    if is_homophone:
        target = letters if letters else answer
        homophone_section = f"""
HOMOPHONE CLUE
==============
A homophone indicator is active. The answer was reached by pronouncing a source
word that sounds like the target. Your tasks:
  1. Identify the homophone pair: what real word sounds like "{target}"?
     Return: {{"type": "homophone", "word": "<source_word>", "sounds_like": "{target}"}}
  2. If a single clue word is a synonym of that source word, also return a synonym entry.
     If no single clue word maps cleanly to the source, return ONLY the homophone entry.
Example: target=PLAICE → PLACE sounds like PLAICE →
  {{"type": "homophone", "word": "place", "sounds_like": "plaice"}}
"""

    indicator_extension_section = ''
    if attributed_indicator_words:
        attributed_str = ', '.join(f'"{w}"' for w in attributed_indicator_words)
        indicator_extension_section = f"""
MULTI-WORD INDICATOR EXTENSION
===============================
The following words are already attributed as indicators: {attributed_str}
An unresolved word adjacent to one of these may EXTEND it into a two-word
indicator phrase. E.g. if "sent" is already a reversal indicator and "round"
is unresolved and adjacent, the full phrase "sent round" is the real indicator.
In that case propose: {{"type": "indicator", "phrase": "sent round", "wordplay_type": "reversal", "subtype": "null"}}
"""

    if letters:
        purpose_section = (
            "The solver has partially solved a clue but is stuck. You must identify the\n"
            "missing database entries that will:\n"
            "  1. Eliminate ALL letters in \"letters_still_needed\"\n"
            "  2. Give EVERY unresolved clue word a role"
        )
        letters_line = f"Letters still needed: {letters}"
    else:
        purpose_section = (
            "The solver found the correct answer — all answer letters are accounted for.\n"
            "However, some clue words have no confirmed role (Wire Principle violation).\n"
            "Your task: identify what role each unresolved word plays. It will be one of:\n"
            "  - An indicator the pipeline missed (reversal, container, insertion, anagram, parts)\n"
            "  - Part of a multi-word indicator phrase extending an already-attributed indicator\n"
            "  - A synonym or abbreviation for a component already found another way\n"
            "Do NOT propose entries for genuine link words (in, at, the, of, a, and, to, by).\n"
            "Give EVERY unresolved clue word a role."
        )
        letters_line = "Letters still needed: (none — formula is already complete)"

    return f"""You are a database enrichment tool for a cryptic crossword solver.

YOUR PURPOSE
============
{purpose_section}

Each entry you propose is one of three types:
  synonym   — a clue word is a synonym of a real English word or proper noun
              whose letters (possibly transformed by an active indicator) match
              some or all of the letters_still_needed
  wordplay  — a clue word is a well-known abbreviation or substitution
              e.g. "I" for Roman numeral 1, "T" for temperature, "N" for knight
  indicator — a clue word/phrase signals a wordplay operation
              (reversal, container, insertion, anagram, parts/outer_use,
               parts/first_use, parts/last_use, parts/inner_use)

WORKED EXAMPLE
==============
Clue:    "Mole unfinished, temperature 1 Celsius in the sea at Rimini"
Answer:  ADRIATIC
Formula: T(temperature) + C(Celsius) + ? = ADRIATIC
Already found:
  "temperature" : substitution = T  (abbreviation)
  "Celsius"     : substitution = C  (abbreviation)
Active indicators:
  "unfinished" is a parts_indicator — removes the last letter of the source word
Letters still needed: ADRIAI
Unresolved words: "Mole", "1"

Correct entries:
  synonym  "Mole" = ADRIAN
    Reasoning: Adrian Mole is a famous fictional spy/character. Real proper noun.
    Apply "unfinished" (remove last letter): ADRIAN -> ADRIA
    ADRIA provides A,D,R,I,A — 5 of the 6 needed letters.
  wordplay "1" = I
    Reasoning: 1 is Roman numeral I. Provides the remaining letter.
  Result: ADRIA + I = ADRIAI = letters_still_needed. Both unresolved words have roles.
{homophone_section}{indicator_extension_section}
RULES
=====
1. SINGLE WORD: The "phrase" field must be a SINGLE word from the clue.
   Exception: for indicators only, two adjacent words are allowed.
2. REAL WORDS ONLY: Synonyms must be real dictionary words or proper nouns.
   Never invent a letter string to fit. If you cannot find a real word, use [].
3. VERIFY YOUR WORKING: Before proposing a synonym, confirm:
   - it is a real word
   - its letters, after any active indicator operation, are a subset of letters_still_needed
4. HOMOPHONES: If a homophone indicator is active, the synonym must SOUND LIKE
   the target — do not match letters directly.
5. MULTI-WORD INDICATORS: Adjacent unresolved words may form a two-word indicator.
   Check whether combining them with an already-attributed indicator word makes
   a recognised cryptic indicator phrase.
6. CONFIDENCE: If you are not certain, respond []. A wrong entry poisons the
   database and is far worse than no entry.

NOW SOLVE THIS CLUE
===================
Clue:    "{clue}"
Answer:  {answer}
Formula: {formula}

Already found:
{found_parts}

Active indicators:
{active_indicators}

{letters_line}
Unresolved clue words: {unresolved_text}

Respond ONLY with a JSON array. Each entry:
  {{"type": "synonym",   "phrase": "<single_clue_word>", "letters": "<REAL_WORD>"}}
  {{"type": "wordplay",  "phrase": "<single_clue_word>", "letters": "<ABBREV>", "category": "<abbreviation|roman_numeral|single_letter>"}}
  {{"type": "indicator", "phrase": "<one_or_two_words>", "wordplay_type": "<reversal|container|insertion|anagram|parts>", "subtype": "<first_use|last_use|outer_use|inner_use|head_delete|tail_delete|null>"}}
  {{"type": "homophone", "word": "<source_word>", "sounds_like": "<target_word>"}}

If nothing can be determined with confidence: []"""


# ============================================================
# API CALL
# ============================================================

def call_claude(prompt: str) -> Optional[str]:
    """Call the Claude API and return raw response text, or None on error."""
    try:
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            messages=[{'role': 'user', 'content': prompt}],
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f'    API error: {e}')
        return None


# ============================================================
# RESPONSE PARSING
# ============================================================

def parse_candidates(response_text: str) -> List[dict]:
    """Parse the API response into a list of candidate dicts."""
    if not response_text:
        return []

    text = re.sub(r'```(?:json)?\s*|\s*```', '', response_text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'(\[.*?\]|\{.*?\})', text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


# ============================================================
# INSERT
# ============================================================

def apply_candidate(conn: sqlite3.Connection,
                    candidate: dict,
                    answer: str,
                    counter: InsertCounter) -> bool:
    """Insert one candidate into the DB with SOURCE_TAG."""
    ctype  = (candidate.get('type') or '').strip().lower()
    phrase = (candidate.get('phrase') or '').strip().lower()

    if not phrase:
        return False

    if ctype == 'synonym':
        letters = (candidate.get('letters') or '').strip().upper()
        if not letters:
            return False
        inserted = insert_synonym_pair(conn, phrase, letters, source=SOURCE_TAG)
        counter.record('synonyms_pairs', inserted,
                       f'"{phrase}" = {letters}  [for {answer}]')
        return inserted

    if ctype == 'wordplay':
        letters  = (candidate.get('letters') or '').strip().upper()
        category = (candidate.get('category') or 'abbreviation').strip().lower()
        if not letters:
            return False
        inserted = insert_wordplay(conn, phrase, letters, category,
                                   confidence='medium', source_tag=SOURCE_TAG)
        counter.record('wordplay', inserted,
                       f'"{phrase}" = {letters}  ({category})  [for {answer}]')
        return inserted

    if ctype == 'indicator':
        wordplay_type = (candidate.get('wordplay_type') or '').strip().lower()
        subtype_raw   = (candidate.get('subtype') or '').strip().lower()
        subtype       = None if subtype_raw in ('', 'null', 'none') else subtype_raw
        if not wordplay_type:
            return False
        inserted = insert_indicator(conn, phrase, wordplay_type,
                                    subtype=subtype, confidence='medium',
                                    source=SOURCE_TAG)
        label = wordplay_type + (f'/{subtype}' if subtype else '')
        counter.record('indicators', inserted,
                       f'"{phrase}" = {label}  [for {answer}]')
        return inserted

    if ctype == 'homophone':
        word        = (candidate.get('word') or candidate.get('source') or '').strip().lower()
        sounds_like = (candidate.get('sounds_like') or candidate.get('homophone') or candidate.get('target') or '').strip().lower()
        if not word or not sounds_like:
            return False
        inserted = insert_homophone(conn, word, sounds_like)
        counter.record('homophones', inserted,
                       f'"{word}" sounds like "{sounds_like}"  [for {answer}]')
        return inserted

    return False


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='API gap finder — inserts missing DB entries for partial solves.')
    add_common_args(parser)
    parser.add_argument('--run-id', type=int, default=0)
    parser.add_argument('--model', default=MODEL)
    args = parser.parse_args()
    apply_common_args(args)

    failures = load_partial_failures(run_id=args.run_id)
    print(f'Found {len(failures)} partial clues\n')

    if not failures:
        print('No partial clues found. Run report.py first.')
        return

    conn    = get_cryptic_conn()
    counter = InsertCounter('api_gap_finder')

    for failure in failures:
        answer     = failure['answer']
        letters    = failure['letters_still_needed']
        unresolved = failure['unresolved_words']

        if not unresolved:
            continue

        # Skip acrostic-gap clues — the first letters of unresolved words already
        # spell letters_still_needed under an active parts/first_use indicator.
        # This is a pipeline code gap, not a DB gap; no insert can fix it.
        if _is_acrostic_gap(failure):
            print(f'[{answer}]  SKIP (acrostic code gap — not a DB issue)')
            continue

        print(f'[{answer}]  needs={letters}  unresolved={unresolved}')

        prompt     = build_prompt(failure)
        response   = call_claude(prompt)

        if not response:
            print('  no API response\n')
            continue

        candidates = parse_candidates(response)

        if not candidates:
            safe = response[:120].encode('ascii', errors='replace').decode()
            print(f'  could not parse: {safe}\n')
            continue

        for candidate in candidates:
            applied = apply_candidate(conn, candidate, answer, counter)
            ctype   = candidate.get('type', '?')
            if ctype == 'homophone':
                w   = candidate.get('word') or candidate.get('source') or '?'
                sl  = candidate.get('sounds_like') or candidate.get('homophone') or candidate.get('target') or '?'
                display = f'"{w}" sounds like "{sl}"'
            else:
                phrase  = candidate.get('phrase', '?')
                value   = candidate.get('letters') or candidate.get('wordplay_type', '?')
                display = f'"{phrase}" = {value}'
            status  = 'inserted' if applied else 'already exists'
            print(f'  {ctype:9s}  {display}  [{status}]')
        print()

    if not _common.DRY_RUN:
        conn.commit()
    conn.close()

    counter.report()

    if not _common.DRY_RUN:
        total_new = sum(counter.counts.get(t, 0) for t in counter.counts)
        if total_new:
            print('To remove all api_gap_finder entries:')
            print(f"  DELETE FROM synonyms_pairs WHERE source = '{SOURCE_TAG}'")
            print(f"  DELETE FROM indicators    WHERE source = '{SOURCE_TAG}'")
            print(f"  DELETE FROM wordplay      WHERE notes LIKE '{SOURCE_TAG}%'")
            print()
        print('Run report.py to verify results.')


if __name__ == '__main__':
    main()
