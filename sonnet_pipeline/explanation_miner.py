"""
Mine 221k human explanations from clues_master.db for synonym and abbreviation mappings.

Patterns extracted:
  CAPS(lowercase)  ->  abbreviation (if CAPS ≤ 3 chars) or synonym (if longer)
  WORD (gloss)     ->  synonym with space-separated gloss
  {low}CAPS        ->  first/last letter abbreviation
  [low]CAPS / CAPS[low]  ->  deletion source (abbreviation if 1-2 chars)

Output: JSONL of discoveries + deduplication against cryptic_new.db
"""

import re
import sys
import json
import sqlite3
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent.parent
MASTER_DB = BASE / "data" / "clues_master.db"
REF_DB = BASE / "data" / "cryptic_new.db"
OUTPUT_JSONL = BASE / "data" / "explanation_mining_discoveries.jsonl"
OUTPUT_REPORT = BASE / "data" / "explanation_mining_report.txt"

# ── regex patterns ──────────────────────────────────────────────────

# Pattern 1: CAPS(lowercase) — gloss or abbreviation
#   GENT(male), S(econd), ER(hesitation), RE(about)
#   Must have uppercase piece then (lowercase content)
PAT_CAPS_PAREN = re.compile(
    r'\b([A-Z][A-Z.\']*)\(([a-z][a-z\s\-\']*)\)'
)

# Pattern 2: WORD (gloss) with space before paren
#   RENT (cost of living), BARE (exposed), DOCKS (harbour areas)
PAT_WORD_SPACE_GLOSS = re.compile(
    r'\b([A-Z]{2,})\s+\(([a-z][a-z\s\-\',]*)\)'
)

# Pattern 3: {lowercase}CAPS — curly brace deletion (first/last letter)
#   {r}EEL, {spea}R, {dagge}R, {c}ODE
PAT_CURLY_PREFIX = re.compile(
    r'\{([a-z]+)\}([A-Z]+)'
)
PAT_CURLY_SUFFIX = re.compile(
    r'([A-Z]+)\{([a-z]+)\}'
)

# Pattern 4: [lowercase]CAPS or CAPS[lowercase] — bracket deletion
#   [pr]EVENT, BIL[ious], [ta]LENT[ed]
PAT_BRACKET_PREFIX = re.compile(
    r'\[([a-z]+)\]([A-Z]{2,})'
)
PAT_BRACKET_SUFFIX = re.compile(
    r'([A-Z]{2,})\[([a-z]+)\]'
)

# Pattern 5: "X reversed" / "reversal of X" / "X back" — less useful for synonyms
# Skip for now, focus on direct mappings

# Filter out enumeration patterns like (7), (3,5), (2-4)
PAT_ENUM = re.compile(r'^\d[\d,\-\s]*$')

# Filter out indicator labels like [in], [around], [messily]
# These are single lowercase words in brackets that are indicators, not deletions
# Common English words that are NOT word-endings — used to detect gloss vs abbreviation
# I(one) = gloss (one is a word), S(econd) = abbreviation (econd is not a word)
_COMMON_WORDS = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'love', 'nothing', 'zero', 'about', 'around', 'over', 'on', 'in', 'at', 'to',
    'it', 'he', 'she', 'we', 'me', 'us', 'man', 'woman', 'boy', 'girl',
    'old', 'new', 'good', 'bad', 'big', 'small', 'hot', 'cold',
    'gold', 'silver', 'iron', 'copper', 'lead', 'tin',
    'king', 'queen', 'knight', 'bishop', 'rook', 'pawn',
    'north', 'south', 'east', 'west',
    'right', 'left', 'up', 'down',
    'time', 'name', 'note', 'line', 'point', 'love', 'article',
    'doctor', 'saint', 'church', 'river', 'street', 'road',
    'black', 'white', 'red', 'blue', 'green', 'yellow',
    'first', 'last', 'second', 'third',
    'the', 'and', 'but', 'for', 'not', 'you', 'all',
    'can', 'had', 'her', 'was', 'are', 'his', 'how',
    'its', 'may', 'our', 'too', 'use', 'way', 'who', 'did',
}

# Noise phrases in Telegraph explanations — structural, not synonyms
NOISE_GLOSSES = {
    'from the clue', 'from surface', 'from clue', 'from the surface',
    'from the clue surface', 'in the clue', 'from clue text',
    'from the clue again', 'taken directly from surface',
}

INDICATOR_LABELS = {
    'in', 'around', 'outside', 'inside', 'within', 'about', 'containing',
    'initially', 'finally', 'first', 'last', 'messily', 'badly', 'oddly',
    'evenly', 'regularly', 'alternately', 'back', 'up', 'reversed',
    'reduced', 'briefly', 'almost', 'nearly', 'half', 'heartless',
    'headless', 'endless', 'curtailed', 'beheaded', 'emptied',
    'centre', 'center', 'middle', 'losing', 'without', 'dropping',
    'upset', 'overturned', 'scrambled', 'cooked', 'mixed', 'broken',
    'damaged', 'destroyed', 'wild', 'crazy', 'mad', 'drunk', 'smashed',
    'cut', 'clipped', 'drained', 'shelled', 'gutted', 'capped',
}


def load_word_list():
    """Load a word list from clues_master.db answers for word validation."""
    conn = sqlite3.connect(MASTER_DB)
    words = set()
    for (w,) in conn.execute("SELECT DISTINCT LOWER(answer) FROM clues WHERE LENGTH(answer) >= 3"):
        words.add(w.replace(' ', '').replace('-', '').lower())
    conn.close()
    return words


def load_existing_db():
    """Load existing synonyms and abbreviations from cryptic_new.db."""
    conn = sqlite3.connect(REF_DB)

    # Load synonyms (word -> set of synonyms)
    existing_syns = defaultdict(set)
    for word, syn in conn.execute("SELECT LOWER(word), UPPER(synonym) FROM synonyms_pairs"):
        existing_syns[word].add(syn)

    # Load definition_answers (also synonym-like)
    for word, ans in conn.execute("SELECT LOWER(definition), UPPER(answer) FROM definition_answers_augmented"):
        existing_syns[word].add(ans)

    conn.close()
    return existing_syns


def is_enumeration(text):
    """Check if text is just a number pattern like '7' or '3,5'."""
    return bool(PAT_ENUM.match(text.strip()))


def extract_mappings(explanation, answer, word_list=None):
    """Extract synonym/abbreviation mappings from a single explanation."""
    mappings = []
    answer_upper = answer.upper().strip()

    # Pattern 1: CAPS(lowercase) — abbreviation or synonym gloss
    for m in PAT_CAPS_PAREN.finditer(explanation):
        caps = m.group(1).replace('.', '').replace("'", '')  # clean dots/apostrophes
        lower = m.group(2).strip().lower()

        if is_enumeration(lower):
            continue
        if len(lower) < 2:
            continue
        if len(caps) < 1:
            continue
        if lower in NOISE_GLOSSES:
            continue

        # Decide: abbreviation (short CAPS) vs synonym (long CAPS)
        if len(caps) <= 3:
            # Abbreviation pattern: S(econd) -> full word = "second", abbrev = "S"
            # Gloss pattern: I(one) -> one = I (synonym)
            # Key distinction: in abbreviation, lower is REST of word (S+econd=second)
            # In gloss, lower is the MEANING (I means "one")
            full_word = caps.lower() + lower

            # Distinguish abbreviation from gloss using word list:
            # Abbreviation: S(econd) — full_word "second" IS a real word
            # Gloss: I(one) — full_word "ione" is NOT a real word
            is_gloss = False
            if ' ' in lower:
                # Multi-word gloss: always a synonym
                is_gloss = True
            elif word_list and full_word not in word_list:
                # full_word is not a real word, so lower is a gloss/meaning
                is_gloss = True

            if is_gloss:
                mappings.append({
                    'type': 'synonym',
                    'word': lower,
                    'value': caps.upper(),
                    'pattern': 'CAPS(lower)',
                })
            elif len(caps) <= 2:
                mappings.append({
                    'type': 'abbreviation',
                    'word': full_word,
                    'value': caps.upper(),
                    'pattern': 'CAPS(lower)',
                })
            else:
                # 3-letter: abbreviation if short completion, synonym if long gloss
                if len(lower) <= 4:
                    mappings.append({
                        'type': 'abbreviation',
                        'word': full_word,
                        'value': caps.upper(),
                        'pattern': 'CAPS(lower)',
                    })
                else:
                    mappings.append({
                        'type': 'synonym',
                        'word': lower,
                        'value': caps.upper(),
                        'pattern': 'CAPS(lower)',
                    })
        else:
            # Longer = synonym gloss: GENT(male) -> male -> GENT
            mappings.append({
                'type': 'synonym',
                'word': lower,
                'value': caps.upper(),
                'pattern': 'CAPS(lower)',
            })

    # Pattern 2: WORD (gloss) — synonym with space before paren
    for m in PAT_WORD_SPACE_GLOSS.finditer(explanation):
        caps = m.group(1)
        lower = m.group(2).strip().lower()

        if is_enumeration(lower):
            continue
        if len(lower) < 2:
            continue
        if lower in NOISE_GLOSSES:
            continue
        # Skip if this looks like it's part of a longer sentence
        if ',' in lower and len(lower.split(',')) > 2:
            continue

        mappings.append({
            'type': 'synonym',
            'word': lower,
            'value': caps.upper(),
            'pattern': 'WORD (gloss)',
        })

    # Patterns 3-4 ({low}CAPS, [low]CAPS etc.) REMOVED — they capture deletion
    # artifacts, not real abbreviations. E.g. ZIPP[i]ER -> "ier->ER" is wrong.

    return mappings


def mine_all(limit=None, source_filter=None):
    """Mine all explanations from clues_master.db."""
    print("Loading word list for validation...")
    word_list = load_word_list()
    print(f"  {len(word_list)} words loaded")

    conn = sqlite3.connect(MASTER_DB)

    query = """SELECT clue_text, answer, explanation, source FROM clues
               WHERE explanation IS NOT NULL AND explanation != ''"""
    params = []
    if source_filter:
        query += " AND source = ?"
        params.append(source_filter)
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()

    print(f"Mining {len(rows)} explanations...")

    # Aggregate: (type, word, value) -> count + examples
    agg = defaultdict(lambda: {'count': 0, 'patterns': set(), 'examples': []})
    total_mappings = 0

    for i, (clue, answer, explanation, source) in enumerate(rows):
        if i % 50000 == 0 and i > 0:
            print(f"  {i}/{len(rows)}...")

        mappings = extract_mappings(explanation, answer, word_list)
        total_mappings += len(mappings)

        for m in mappings:
            key = (m['type'], m['word'].lower(), m['value'].upper())
            agg[key]['count'] += 1
            agg[key]['patterns'].add(m['pattern'])
            if len(agg[key]['examples']) < 3:
                agg[key]['examples'].append(explanation[:100])

    print(f"Extracted {total_mappings} raw mappings -> {len(agg)} unique (type, word, value) triples")
    return agg


def deduplicate(agg, existing_syns):
    """Split into new vs already-in-DB."""
    new_entries = []
    existing_entries = []

    for (typ, word, value), info in sorted(agg.items(), key=lambda x: -x[1]['count']):
        entry = {
            'type': typ,
            'word': word,
            'value': value,
            'count': info['count'],
            'patterns': sorted(info['patterns']),
            'examples': info['examples'],
        }

        # Check if already in DB
        in_db = False
        if typ == 'synonym':
            if value in existing_syns.get(word, set()):
                in_db = True
        elif typ == 'abbreviation':
            # Abbreviations stored in synonyms_pairs too
            if value in existing_syns.get(word, set()):
                in_db = True

        entry['in_db'] = in_db
        if in_db:
            existing_entries.append(entry)
        else:
            new_entries.append(entry)

    return new_entries, existing_entries


def write_output(new_entries, existing_entries, total_explanations):
    """Write JSONL and report."""
    # Filter: require count >= 2 for new entries (noise reduction)
    confident_new = [e for e in new_entries if e['count'] >= 2]
    single_new = [e for e in new_entries if e['count'] == 1]

    # Write JSONL (confident new only)
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in confident_new:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')

    # Write report
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write(f"Explanation Mining Report\n")
        f.write(f"========================\n\n")
        f.write(f"Explanations mined: {total_explanations}\n")
        f.write(f"Total unique mappings: {len(new_entries) + len(existing_entries)}\n")
        f.write(f"Already in DB: {len(existing_entries)}\n")
        f.write(f"New (count >= 2): {len(confident_new)}\n")
        f.write(f"New (count == 1): {len(single_new)}\n\n")

        # Breakdown by type
        syn_new = [e for e in confident_new if e['type'] == 'synonym']
        abbr_new = [e for e in confident_new if e['type'] == 'abbreviation']
        f.write(f"New synonyms (count >= 2): {len(syn_new)}\n")
        f.write(f"New abbreviations (count >= 2): {len(abbr_new)}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"NEW SYNONYMS (count >= 2, sorted by frequency)\n")
        f.write(f"{'='*60}\n\n")
        for e in sorted(syn_new, key=lambda x: -x['count'])[:200]:
            f.write(f"  {e['word']} -> {e['value']}  (×{e['count']}, {e['patterns']})\n")
            for ex in e['examples'][:1]:
                f.write(f"    ex: {ex}\n")

        f.write(f"\n{'='*60}\n")
        f.write(f"NEW ABBREVIATIONS (count >= 2, sorted by frequency)\n")
        f.write(f"{'='*60}\n\n")
        for e in sorted(abbr_new, key=lambda x: -x['count'])[:200]:
            f.write(f"  {e['word']} -> {e['value']}  (×{e['count']}, {e['patterns']})\n")
            for ex in e['examples'][:1]:
                f.write(f"    ex: {ex}\n")

        # Also show high-count existing (validation)
        f.write(f"\n{'='*60}\n")
        f.write(f"CONFIRMED EXISTING (top 50 by frequency)\n")
        f.write(f"{'='*60}\n\n")
        for e in sorted(existing_entries, key=lambda x: -x['count'])[:50]:
            f.write(f"  {e['word']} -> {e['value']}  (×{e['count']})\n")

    print(f"\nReport: {OUTPUT_REPORT}")
    print(f"Discoveries JSONL: {OUTPUT_JSONL} ({len(confident_new)} entries)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mine explanations for synonym/abbreviation mappings')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of explanations to process')
    parser.add_argument('--source', type=str, default=None, help='Filter by source (times, guardian, etc.)')
    args = parser.parse_args()

    print("Loading existing DB entries for deduplication...")
    existing_syns = load_existing_db()
    print(f"  {sum(len(v) for v in existing_syns.values())} existing synonym mappings loaded")

    agg = mine_all(limit=args.limit, source_filter=args.source)
    new_entries, existing_entries = deduplicate(agg, existing_syns)

    total_exp = args.limit or 221429  # approximate
    write_output(new_entries, existing_entries, total_exp)


if __name__ == '__main__':
    main()
