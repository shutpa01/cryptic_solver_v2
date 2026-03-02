"""Targeted backfill for 28k clues that have explanations but no extracted
definition or wordplay_type.

The main parse_explanations.py parser fails on these because:
1. "Definition: X" annotations get cut by the commentary stripper
2. Pure notation (e.g. "(PROUST)*") has no definition to extract
3. "double def" / "cryptic def" mentions don't get definitions extracted
4. Types detectable from notation but never written to clues table

This script:
- Extracts definitions from "Definition: X" patterns in explanation text
- Extracts wordplay_type from explanation notation patterns
- Falls back to definition window matching (first/last words of clue)
- Writes directly to clues.definition and clues.wordplay_type
- Only fills empty fields — never overwrites existing values

Usage:
    python -m classifier.backfill_unparsed              # dry-run
    python -m classifier.backfill_unparsed --apply      # write to DB
    python -m classifier.backfill_unparsed --verbose     # show samples
"""

import argparse
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
REF_DB = PROJECT_ROOT / "data" / "cryptic_new.db"


# --- Definition extraction patterns ---

# "Definition: 'X'" or 'Definition: "X"' with explicit quotes — try this first
DEF_QUOTED = re.compile(
    r'(?:Definition|Defn|Def)\s*[:=]\s*'
    r'[\u201c\u2018"\']'
    r'(.+?)'
    r'[\u201d\u2019"\']',
    re.IGNORECASE
)

# "Definition: X" without quotes — stop at wordplay notation or sentence boundaries
DEF_COLON = re.compile(
    r'(?:Definition|Defn|Def)\s*[:=]\s*'
    r'[\u201c\u2018"\']?\s*'         # optional open quote
    r'(.+?)(?:'                       # capture definition
    r'[\u201d\u2019"\']'             # close quote
    r'|(?:\.\s+[A-Z])'              # period + next sentence
    r'|\s+[A-Z]{2,}[=\s*+(]'        # start of wordplay notation (CAPS=, CAPS+, CAPS()
    r'|\s+\['                        # start of bracket notation
    r'|\s+anagram'                   # start of wordplay description
    r'|\s+(?:plus|around|inside|containing|in\s)' # container/charade words
    r'|;\s'                          # semicolon
    r'|$)',                           # end of string
    re.IGNORECASE
)

# --- Type detection patterns ---

TYPE_PATTERNS = [
    (re.compile(r'\bdouble\s+def(?:inition|n|\.)?', re.I), 'double_definition'),
    (re.compile(r'\bcryptic\s+def(?:inition|n|\.)?', re.I), 'cryptic_definition'),
    (re.compile(r'[A-Z]{2,}\*|\*\([A-Z\s]+\)|\([A-Z\s]+\)\*|\([A-Za-z\s]+\)\*|\banagram\b', re.I), 'anagram'),
    (re.compile(r'\bhidden\b|\blurk(?:ing)?\b|\bconcealed\b', re.I), 'hidden'),
    (re.compile(r'\brevers(?:ed?|al)\b|\bbackwards?\b', re.I), 'reversal'),
    (re.compile(r'\bhomophone\b|\bsounds?\s+like\b|\bwe hear\b|\bas heard\b', re.I), 'homophone'),
    (re.compile(r'\bfirst\s+letters?\b|\binitial(?:s|ised)?\b|\bacrostic\b', re.I), 'acrostic'),
    (re.compile(r'\bcontain(?:ing|ed|s)?\b|\binside\b|\bwrapped\b|\bengulfing\b', re.I), 'container'),
    (re.compile(r'\bwithout\b|\bdropping\b|\blosing\b|\bminus\b|\btruncated?\b|\bbeheaded\b|\bheadless\b|\bendless\b', re.I), 'deletion'),
    # "X in Y" where X and Y are CAPS — container pattern
    (re.compile(r'[A-Z]{1,}\s+in\s+[A-Z]{2,}|[A-Z]{2,}\s+in\s+[A-Z]{1,}'), 'container'),
    # "X round/around Y" where CAPS
    (re.compile(r'[A-Z]{2,}\s+(?:round|around)\s+[A-Z]'), 'container'),
    # "envelope" — container
    (re.compile(r'\benvelope\b', re.I), 'container'),
    # "alternate letters" / "even/odd letters" — alternation (hidden variant)
    (re.compile(r'\balternate\s+letters\b|\beven\s+letters\b|\bodd\s+letters\b|\balternately\b', re.I), 'hidden'),
    # "charade of" — explicit charade mention
    (re.compile(r'\bcharade\s+of\b', re.I), 'charade'),
    # "spoonerism" — spoonerism type
    (re.compile(r'\bspoonerism\b', re.I), 'spoonerism'),
    # "central letter" / "middle letter" / "heart of" — deletion/extraction
    (re.compile(r'\bcentral\s+letter\b|\bmiddle\s+letter\b|\bheart\s+of\b', re.I), 'deletion'),
    # "outside letters" / "outer letters" — container/deletion
    (re.compile(r'\boutside\s+(?:letters|of)\b|\bouter\s+letters\b', re.I), 'container'),
    # "last letter(s)" / "first letter(s)" in prose (not acrostic pattern)
    (re.compile(r'\blast\s+letters?\s+of\b', re.I), 'deletion'),
    # "cryptic description" — cryptic definition
    (re.compile(r'\bcryptic\s+description\b', re.I), 'cryptic_definition'),
]

# Charade: multiple CAPS + CAPS or explicit + signs
CHARADE_PATTERN = re.compile(r'[A-Z]{2,}\s*\+\s*[A-Z]{2,}')


def extract_definition(explanation, clue_text, answer, def_index):
    """Try to extract definition from explanation text."""
    answer_upper = answer.upper().replace(' ', '') if answer else ''

    # Method 1: Explicit "Definition: 'X'" with quotes — highest confidence
    m = DEF_QUOTED.search(explanation)
    if m:
        defn = m.group(1).strip()
        if defn and len(defn) > 1:
            # Validate: must appear in clue
            if clue_text and defn.lower() in clue_text.lower():
                return defn, 'explicit'

    # Method 2: "Definition: X" without quotes — extract and validate
    m = DEF_COLON.search(explanation)
    if m:
        defn = m.group(1).strip()
        # Clean trailing junk
        defn = re.sub(r'\s*[\u2013\u2014\-,;:]\s*$', '', defn)
        defn = re.sub(r'\s+$', '', defn)
        # Only accept if: (a) reasonable length, (b) found in clue text,
        # (c) verified against reference DB
        if defn and 2 <= len(defn) < 60 and clue_text:
            if defn.lower() in clue_text.lower():
                if (defn.lower(), answer_upper) in def_index:
                    return defn, 'explicit'

    # Method 3: "double definition" — can't reliably extract
    if re.search(r'(?i)double\s+def', explanation):
        return None, 'double_def_no_extract'

    # Method 4: "cryptic definition" — whole clue is the definition
    if re.search(r'(?i)cryptic\s+def', explanation):
        # Strip enumeration from clue text
        clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip() if clue_text else clue_text
        return clean, 'cryptic_def'

    return None, None


def extract_type(explanation):
    """Extract the primary wordplay type from explanation notation."""
    types = []
    for pattern, typ in TYPE_PATTERNS:
        if pattern.search(explanation):
            types.append(typ)

    if CHARADE_PATTERN.search(explanation):
        types.append('charade')
    elif '+' in explanation and re.search(r'[A-Z]{2,}', explanation):
        types.append('charade')

    if not types:
        return None

    # Priority: specific types first
    priority = ['double_definition', 'cryptic_definition', 'hidden', 'homophone',
                'acrostic', 'anagram', 'reversal', 'container', 'deletion', 'charade']
    for p in priority:
        if p in types:
            return p

    return types[0]


def load_definition_index(conn_ref):
    """Pre-load definition and synonym pairs into memory for fast lookup.
    Returns a set of (definition_lower, answer_upper) tuples."""
    print('Loading reference tables into memory...')
    index = set()

    # Load definition_answers_augmented
    cur = conn_ref.execute(
        'SELECT LOWER(definition), UPPER(answer) FROM definition_answers_augmented '
        'WHERE definition IS NOT NULL AND answer IS NOT NULL'
    )
    for row in cur:
        index.add((row[0], row[1].replace(' ', '')))
    def_count = len(index)

    # Load synonyms_pairs
    cur = conn_ref.execute(
        'SELECT LOWER(word), UPPER(synonym) FROM synonyms_pairs '
        'WHERE word IS NOT NULL AND synonym IS NOT NULL'
    )
    for row in cur:
        index.add((row[0], row[1].replace(' ', '')))

    print(f'  Loaded {len(index):,} pairs ({def_count:,} definitions + {len(index) - def_count:,} synonyms)')
    return index


def match_definition_from_clue(clue_text, answer, def_index):
    """Try to find definition by checking first/last N words of clue against
    pre-loaded definition/synonym index."""
    if not clue_text or not answer:
        return None

    answer_upper = answer.upper().replace(' ', '')
    words = clue_text.split()

    # Try prefixes (first 1-4 words) and suffixes (last 1-4 words)
    candidates = []
    for n in range(1, min(5, len(words))):
        prefix = ' '.join(words[:n])
        suffix = ' '.join(words[-n:])
        # Remove trailing punctuation from suffix
        suffix = re.sub(r'[?,!.;:]+$', '', suffix).strip()
        candidates.append(prefix)
        if suffix != prefix:
            candidates.append(suffix)

    for candidate in candidates:
        if len(candidate) < 2:
            continue
        if (candidate.lower(), answer_upper) in def_index:
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(description='Backfill unparsed explanation definitions and types')
    parser.add_argument('--apply', action='store_true', help='Write to DB (default: dry-run)')
    parser.add_argument('--verbose', action='store_true', help='Show sample extractions')
    parser.add_argument('--source', default='', help='Only process this source')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    conn = sqlite3.connect(str(CLUES_DB))
    conn.execute('PRAGMA journal_mode=WAL')

    conn_ref = sqlite3.connect(str(REF_DB))
    def_index = load_definition_index(conn_ref)
    conn_ref.close()

    # Fetch clues with explanation but missing definition AND/OR type
    query = '''
        SELECT id, source, clue_text, answer, explanation, definition, wordplay_type
        FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
          AND (
            (definition IS NULL OR definition = '')
            OR (wordplay_type IS NULL OR wordplay_type = '')
          )
    '''
    params = []
    if args.source:
        query += ' AND source = ?'
        params.append(args.source)
    if args.limit:
        query += f' LIMIT {args.limit}'

    rows = conn.execute(query, params).fetchall()
    print(f'Found {len(rows):,} clues with explanation but missing definition or type')

    # Stats
    def_extracted = 0
    def_explicit = 0
    def_cryptic = 0
    def_window = 0
    type_extracted = 0
    by_source = {}
    by_type = {}
    samples = []

    updates_def = []
    updates_type = []

    for row in rows:
        clue_id, source, clue_text, answer, explanation, existing_def, existing_type = row

        new_def = None
        new_type = None
        method = None

        # Extract definition if missing
        if not existing_def:
            new_def, method = extract_definition(explanation, clue_text, answer, def_index)

            # Fallback: window matching against reference DB
            if not new_def and method != 'double_def_no_extract':
                window_def = match_definition_from_clue(clue_text, answer, def_index)
                if window_def:
                    new_def = window_def
                    method = 'window'

        # Extract type if missing
        if not existing_type:
            new_type = extract_type(explanation)

        # Track updates
        if new_def:
            updates_def.append((new_def, clue_id))
            def_extracted += 1
            if method == 'explicit':
                def_explicit += 1
            elif method == 'cryptic_def':
                def_cryptic += 1
            elif method == 'window':
                def_window += 1
            by_source[source] = by_source.get(source, 0) + 1

        if new_type:
            updates_type.append((new_type, clue_id))
            type_extracted += 1
            by_type[new_type] = by_type.get(new_type, 0) + 1

        if args.verbose and (new_def or new_type) and len(samples) < 30:
            samples.append({
                'clue': clue_text[:60] if clue_text else '?',
                'answer': answer,
                'explanation': explanation[:100],
                'new_def': new_def,
                'method': method,
                'new_type': new_type,
                'source': source,
            })

    # Report
    print(f'\n{"=" * 60}')
    print(f'RESULTS ({"DRY RUN" if not args.apply else "APPLIED"})')
    print(f'{"=" * 60}')
    print(f'Definitions extracted: {def_extracted:,}')
    print(f'  Explicit (Definition: X): {def_explicit:,}')
    print(f'  Cryptic definition:       {def_cryptic:,}')
    print(f'  Window match:             {def_window:,}')
    print(f'Types extracted:       {type_extracted:,}')
    print()

    print('Definitions by source:')
    for src, cnt in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f'  {src:20} {cnt:>6,}')
    print()

    print('Types extracted:')
    for t, cnt in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f'  {t:25} {cnt:>6,}')

    if args.verbose and samples:
        print(f'\n{"=" * 60}')
        print('SAMPLES')
        print(f'{"=" * 60}')
        for s in samples:
            print(f'\n  [{s["source"]}] {s["clue"]}...')
            print(f'  Answer: {s["answer"]}')
            print(f'  Explanation: {s["explanation"]}')
            if s['new_def']:
                print(f'  -> Definition: "{s["new_def"]}" ({s["method"]})')
            if s['new_type']:
                print(f'  -> Type: {s["new_type"]}')

    if args.apply and (updates_def or updates_type):
        print(f'\nApplying {len(updates_def):,} definition updates...')
        conn.executemany(
            'UPDATE clues SET definition = ? WHERE id = ?',
            updates_def
        )
        print(f'Applying {len(updates_type):,} type updates...')
        conn.executemany(
            'UPDATE clues SET wordplay_type = ? WHERE id = ?',
            updates_type
        )
        conn.commit()
        print('Phase 1 done.')
    elif not args.apply and (updates_def or updates_type):
        print(f'\nPhase 1 dry run — {len(updates_def) + len(updates_type):,} updates pending')

    # ---------------------------------------------------------------
    # Phase 2: Cross-clue definition matching
    # If clue A has the same answer as clue B, and B has a definition,
    # check if B's definition appears as a prefix/suffix of A's clue text.
    # ---------------------------------------------------------------
    print(f'\n{"=" * 60}')
    print('PHASE 2: Cross-clue definition matching')
    print(f'{"=" * 60}')

    # Build answer -> set(definitions) from all defined clues
    print('Building answer->definitions index from clues table...')
    cur = conn.execute(
        'SELECT UPPER(answer), LOWER(definition) FROM clues '
        'WHERE definition IS NOT NULL AND definition <> "" '
        'AND answer IS NOT NULL AND answer <> ""'
    )
    answer_defs = {}
    for ans, defn in cur:
        ans_clean = ans.replace(' ', '')
        if len(defn) >= 2:
            answer_defs.setdefault(ans_clean, set()).add(defn)
    print(f'  {len(answer_defs):,} distinct answers with known definitions')

    # Fetch all undefined clues
    query2 = '''
        SELECT id, clue_text, answer FROM clues
        WHERE (definition IS NULL OR definition = '')
          AND answer IS NOT NULL AND answer <> ''
          AND clue_text IS NOT NULL AND clue_text <> ''
    '''
    params2 = []
    if args.source:
        query2 += ' AND source = ?'
        params2.append(args.source)

    undef_rows = conn.execute(query2, params2).fetchall()
    print(f'  {len(undef_rows):,} undefined clues to check')

    cross_updates = []
    cross_samples = []

    for clue_id, clue_text, answer in undef_rows:
        ans = answer.upper().replace(' ', '')
        if ans not in answer_defs:
            continue

        # Strip enumeration
        clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()
        clue_lower = clue_clean.lower()
        # Also try with trailing punctuation stripped
        clue_stripped = re.sub(r'[?,!.;:]+$', '', clue_lower).strip()

        best = None
        for defn in answer_defs[ans]:
            if len(defn) < 3:
                continue
            if clue_lower.startswith(defn) or clue_lower.endswith(defn):
                if best is None or len(defn) > len(best):
                    best = defn
            elif clue_stripped.endswith(defn):
                if best is None or len(defn) > len(best):
                    best = defn

        if best:
            # Find original case from clue
            idx = clue_lower.find(best)
            original = clue_clean[idx:idx + len(best)] if idx >= 0 else best
            cross_updates.append((original, clue_id))
            if args.verbose and len(cross_samples) < 15:
                cross_samples.append((clue_text[:55], answer, original))

    print(f'\n  Cross-clue matches: {len(cross_updates):,}')

    if args.verbose and cross_samples:
        print('  Samples:')
        for clue, ans, defn in cross_samples:
            print(f'    {clue}... = {ans} -> "{defn}"')

    if args.apply and cross_updates:
        print(f'  Applying {len(cross_updates):,} cross-clue definition updates...')
        conn.executemany(
            'UPDATE clues SET definition = ? WHERE id = ?',
            cross_updates
        )
        conn.commit()
        print('  Phase 2 done.')
    elif not args.apply and cross_updates:
        print(f'  Dry run — {len(cross_updates):,} updates pending')

    total_updates = len(updates_def) + len(updates_type) + len(cross_updates)
    if not args.apply:
        print(f'\nTotal dry run: {total_updates:,} updates — use --apply to write')

    conn.close()


if __name__ == '__main__':
    main()
