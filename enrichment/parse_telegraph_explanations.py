"""
parse_telegraph_explanations.py — Extract substitutions and indicators from
Telegraph narrative explanations using regex patterns + answer constraint.

Categories handled:
  1. Structured charades: "X (word)", "single letter for X", "abbreviation for X"
  2. Anagram indicators: "anagram (INDICATOR) of FODDER"
  3. Container/insertion indicators: "X surrounds Y", "Y inside X"

Outputs potential DB insertions to documents/telegraph_enrichment_review.txt
for human review. No database writes.
"""

import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CRYPTIC_DB = PROJECT_ROOT / 'data' / 'cryptic_new.db'
OUTPUT_FILE = PROJECT_ROOT / 'documents' / 'telegraph_enrichment_review.txt'


def get_conn():
    return sqlite3.connect(CRYPTIC_DB)


def norm_letters(s):
    return re.sub(r'[^A-Za-z]', '', s or '').lower()


def load_existing_wordplay(conn):
    """Load existing wordplay entries for dedup."""
    existing = set()
    for row in conn.execute("SELECT LOWER(indicator), UPPER(substitution) FROM wordplay"):
        existing.add((row[0], row[1]))
    return existing


def load_existing_indicators(conn):
    """Load existing indicator words for dedup."""
    existing = set()
    for row in conn.execute("SELECT LOWER(word), wordplay_type FROM indicators"):
        existing.add((row[0], row[1]))
    return existing


def load_existing_synonyms(conn):
    """Load existing synonym pairs for dedup."""
    existing = set()
    for row in conn.execute("SELECT LOWER(word), LOWER(synonym) FROM synonyms_pairs"):
        existing.add((row[0], row[1]))
    return existing


def load_known_words():
    """Load a set of known English words from clue texts only.

    Uses words from clue texts (natural English) rather than answers,
    because answers include artifacts like DDAY, ITA, AONE that would
    cause false positives in the concatenation check.
    """
    import sqlite3 as _sql
    words = set()
    master = PROJECT_ROOT / 'data' / 'clues_master.db'
    conn = _sql.connect(master)
    # Every word from clue texts — natural English only
    for (clue,) in conn.execute("SELECT DISTINCT clue_text FROM clues WHERE clue_text IS NOT NULL"):
        for token in re.findall(r'[a-zA-Z]{3,}', clue.lower()):
            words.add(token)
    conn.close()
    return words


# ======================================================================
# PATTERN EXTRACTORS
# ======================================================================

# Pattern A: UPPER(lower) — e.g., "SIGH (complain)", "D (Democrat)", "W(ife)"
RE_UPPER_PAREN = re.compile(
    r'(?<![A-Za-z])([A-Z]{1,6})\s*\(([a-z][a-z\s\']*?)\)'
)

# Pattern B: "single letter for X" / "the letter X"
RE_SINGLE_LETTER = re.compile(
    r'(?:the\s+)?single\s+letter\s+for\s+(\w+)',
    re.IGNORECASE
)

# Pattern C: "abbreviation for X" / "abbreviated to X"
RE_ABBREVIATION = re.compile(
    r'(?:the\s+)?abbreviation\s+(?:for|of)\s+(?:a\s+|an\s+)?(\w[\w\s]{0,30}?)(?:\s+(?:and|followed|is|goes|gives|with|then|plus|after|before|inside|outside|around|contains|surrounds|,|\.))',
    re.IGNORECASE
)

# Pattern D: "the French/Spanish/Italian/German/Latin word for X"
RE_FOREIGN = re.compile(
    r'(?:the\s+)?(French|Spanish|Italian|German|Latin)\s+(?:word\s+)?(?:for|meaning)\s+(?:\'|\")?(\w[\w\s]{0,20}?)(?:\'|\"|,|\.|;|\s+and\s|\s+followed|\s+is|\s+goes|\s+with|\s+then)',
    re.IGNORECASE
)

# Pattern E: "chemical symbol for X"
RE_CHEMICAL = re.compile(
    r'(?:the\s+)?chemical\s+symbol\s+for\s+(\w+)',
    re.IGNORECASE
)

# Pattern F: "Roman numeral for X" / "Roman numeral X"
RE_ROMAN = re.compile(
    r'(?:the\s+)?Roman\s+(?:numeral\s+)?(?:for\s+)?(\w+)',
    re.IGNORECASE
)

# Pattern G: "anagram (INDICATOR) of FODDER" — extract the indicator
RE_ANAGRAM_INDICATOR = re.compile(
    r'(?:an?\s+)?anagram\s*\((\w[\w\s]*?)\)\s+of\s+',
    re.IGNORECASE
)
# Also: "an anagram of" preceded by indicator word
RE_ANAGRAM_INDICATOR2 = re.compile(
    r'(\w+)\s+(?:is\s+)?(?:an?\s+)?anagram\s+of\s+',
    re.IGNORECASE
)

# Pattern H: Container indicators — "X surrounds/contains/around/outside Y"
RE_CONTAINER_WORDS = re.compile(
    r'(\w+)\s+(surrounds?|contains?|wraps?|encloses?|embraces?|'
    r'outside|around|clothing|clothed\s+by|placed\s+around|'
    r'goes\s+around|goes\s+round|wrapped\s+around|wrapped\s+round)',
    re.IGNORECASE
)

# Pattern I: Insertion indicators — "Y inside/in/within X", "X inserted into Y"
RE_INSERTION_WORDS = re.compile(
    r'(inside|within|inserted\s+into|placed\s+in|goes\s+into|'
    r'goes\s+in|put\s+into|put\s+in)',
    re.IGNORECASE
)

# Pattern J: Reversal indicators — "reverse/reversal of X", "X reversed/turned"
RE_REVERSAL_WORDS = re.compile(
    r'(\w+)\s+(?:is\s+)?(?:the\s+)?(?:reverse|reversal|reversed|turned|'
    r'turning|turned\s+over|upside\s+down|back|backwards|reflected)',
    re.IGNORECASE
)

# Pattern K: "a word meaning X" / "synonym of X" / "another word for X"
RE_SYNONYM_PHRASE = re.compile(
    r'(?:a\s+)?(?:word|synonym|term)\s+(?:meaning|for|of)\s+(?:\'|\")?(\w[\w\s]{0,25}?)(?:\'|\"|,|\.|;|\s+and\s|\s+followed|\s+is|\s+goes|\s+with|\s+then|\s+gives)',
    re.IGNORECASE
)


def extract_upper_paren(explanation, answer, existing_wordplay=None, known_words=None):
    """Extract UPPER(lower) patterns and validate against answer."""
    results = []
    answer_upper = answer.upper().replace(' ', '')
    existing_wordplay = existing_wordplay or set()
    known_words = known_words or set()

    for match in RE_UPPER_PAREN.finditer(explanation):
        letters = match.group(1).upper()
        fragment = match.group(2).strip().lower()

        # Skip if letters not in answer
        if not all(c in answer_upper for c in letters):
            continue

        # Two conventions in Telegraph explanations:
        #   T(ime)         -> suffix: T from "time"       -> word = "time", sub = T
        #   GO(depart)     -> meaning: GO means "depart"  -> word = "depart", sub = GO
        #
        # Distinguish by checking if concatenation produces a real English word.
        # "time" is a word -> suffix.  "godepart" is not -> meaning.

        concat = letters.lower() + fragment
        concat_clean = re.sub(r'[^a-z]', '', concat)

        if known_words and concat_clean in known_words:
            indicator = concat_clean  # suffix pattern: full word
        else:
            indicator = fragment  # meaning pattern: fragment IS the word

        if len(indicator) < 2:
            continue

        # Skip noise and junk
        if len(indicator) < 3:
            continue
        if any(p in indicator for p in ['clue', 'from the', 'the answer',
                                         'answer is', 'this gives']):
            continue
        # Indicator must be a real English word/phrase
        indicator_clean = re.sub(r'[^a-z]', '', indicator)
        if known_words and indicator_clean not in known_words:
            # Multi-word phrases: check each word individually
            words_in_phrase = indicator.split()
            if len(words_in_phrase) <= 1:
                continue
            if not all(w in known_words for w in words_in_phrase if len(w) >= 3):
                continue

        # Determine category
        if len(letters) == 1:
            category = 'single_letter'
        elif len(letters) <= 3:
            category = 'abbreviation'
        else:
            category = 'synonym'

        results.append({
            'indicator': indicator,
            'substitution': letters,
            'category': category,
            'table': 'wordplay',
            'evidence': f'{letters}({fragment}) -> {indicator} = {letters}'
        })

    return results


def extract_single_letter(explanation, answer, existing_wordplay=None):
    """Extract 'single letter for X' — only accept if the pair is already
    known in the wordplay table, since we cannot reliably guess the letter
    (e.g. 'love' -> O not L, 'nothing' -> O not N)."""
    existing_wordplay = existing_wordplay or set()
    results = []
    for match in RE_SINGLE_LETTER.finditer(explanation):
        word = match.group(1).strip().lower()
        if len(word) < 3:
            continue

        # Check all single letters against existing wordplay
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if (word, letter) in existing_wordplay and letter in answer.upper():
                results.append({
                    'indicator': word,
                    'substitution': letter,
                    'category': 'single_letter',
                    'table': 'wordplay',
                    'evidence': f'single letter for {word} = {letter} (confirmed)'
                })
                break
    return results


def extract_abbreviation(explanation, answer):
    """Extract 'abbreviation for X' patterns."""
    results = []
    for match in RE_ABBREVIATION.finditer(explanation):
        phrase = match.group(1).strip().lower()
        # Clean trailing articles/prepositions
        phrase = re.sub(r'\s+(the|a|an|is|as|in|on|at|to)$', '', phrase)
        if len(phrase) < 2:
            continue
        results.append({
            'indicator': phrase,
            'substitution': None,  # Need answer constraint to resolve
            'category': 'abbreviation',
            'table': 'wordplay',
            'evidence': f'abbreviation for {phrase}'
        })
    return results


def extract_foreign(explanation, answer):
    """Extract foreign word patterns."""
    results = []
    for match in RE_FOREIGN.finditer(explanation):
        language = match.group(1).lower()
        word = match.group(2).strip().lower()
        if len(word) < 2:
            continue
        results.append({
            'indicator': word,
            'substitution': None,  # Need answer constraint
            'category': f'foreign_{language}',
            'table': 'wordplay',
            'evidence': f'{language} word for {word}'
        })
    return results


def extract_chemical(explanation, answer):
    """Extract chemical symbol patterns."""
    # Common chemical symbols
    SYMBOLS = {
        'copper': 'CU', 'gold': 'AU', 'silver': 'AG', 'iron': 'FE',
        'lead': 'PB', 'tin': 'SN', 'mercury': 'HG', 'sodium': 'NA',
        'potassium': 'K', 'calcium': 'CA', 'nitrogen': 'N', 'oxygen': 'O',
        'hydrogen': 'H', 'carbon': 'C', 'sulphur': 'S', 'sulfur': 'S',
        'platinum': 'PT', 'tungsten': 'W', 'zinc': 'ZN', 'nickel': 'NI',
        'aluminium': 'AL', 'aluminum': 'AL', 'phosphorus': 'P',
        'chlorine': 'CL', 'uranium': 'U', 'radium': 'RA', 'neon': 'NE',
        'argon': 'AR', 'helium': 'HE', 'lithium': 'LI', 'barium': 'BA',
        'cobalt': 'CO', 'manganese': 'MN', 'chromium': 'CR', 'iodine': 'I',
        'arsenic': 'AS', 'boron': 'B', 'silicon': 'SI', 'antimony': 'SB',
    }
    results = []
    for match in RE_CHEMICAL.finditer(explanation):
        element = match.group(1).strip().lower()
        symbol = SYMBOLS.get(element)
        if symbol and all(c in answer.upper() for c in symbol):
            results.append({
                'indicator': element,
                'substitution': symbol,
                'category': 'chemical_symbol',
                'table': 'wordplay',
                'evidence': f'chemical symbol for {element} = {symbol}'
            })
    return results


def extract_anagram_indicators(explanation):
    """Extract anagram indicator words from 'anagram (INDICATOR) of' patterns."""
    results = []

    for match in RE_ANAGRAM_INDICATOR.finditer(explanation):
        indicator = match.group(1).strip().lower()
        if len(indicator) >= 3:
            results.append({
                'word': indicator,
                'wordplay_type': 'anagram',
                'table': 'indicators',
                'evidence': f'anagram ({indicator}) of ...'
            })

    return results


def extract_container_indicators(explanation):
    """Extract container/insertion indicator words."""
    results = []
    seen = set()

    # Look for structural phrases that imply containment
    container_phrases = [
        (r'(\w+)\s+(?:is\s+)?(?:placed\s+)?(?:around|round)\b', 'container'),
        (r'(\w+)\s+(?:is\s+)?(?:wrapped|clothed|clothing)\b', 'container'),
        (r'(\w+)\s+surrounds?\b', 'container'),
        (r'(\w+)\s+contains?\b', 'container'),
        (r'(\w+)\s+embraces?\b', 'container'),
        (r'(\w+)\s+encloses?\b', 'container'),
        (r'(?:goes|put|placed|inserted)\s+(inside|into|in)\b', 'insertion'),
        (r'(inside|within)\s+\w+', 'insertion'),
    ]

    for pattern, ind_type in container_phrases:
        for match in re.finditer(pattern, explanation, re.IGNORECASE):
            word = match.group(1).strip().lower()
            if len(word) >= 2 and word not in seen:
                # Skip common non-indicator words
                skip = {'the', 'a', 'an', 'is', 'it', 'and', 'or', 'but',
                        'this', 'that', 'then', 'them', 'with', 'from', 'answer'}
                if word in skip:
                    continue
                seen.add(word)
                results.append({
                    'word': word,
                    'wordplay_type': ind_type,
                    'table': 'indicators',
                    'evidence': match.group(0).strip()[:60]
                })

    return results


def extract_synonym_phrases(explanation, answer):
    """Extract 'a word meaning X' patterns as potential synonym pairs."""
    results = []
    for match in RE_SYNONYM_PHRASE.finditer(explanation):
        phrase = match.group(1).strip().lower()
        if len(phrase) < 2 or len(phrase) > 20:
            continue
        results.append({
            'word': phrase,
            'table': 'synonym_candidate',
            'evidence': f'a word meaning {phrase}'
        })
    return results


# ======================================================================
# ANSWER-CONSTRAINED SEQUENTIAL PARSER
# ======================================================================

def parse_sequential(explanation, answer, conn, existing_wordplay=None, known_words=None):
    """
    Process explanation sequentially, using the answer as a constraint.
    For 'single letter for X', 'abbreviation for X', etc., the next
    unaccounted answer letters tell us the substitution.
    """
    answer_upper = answer.upper().replace(' ', '')
    results = []

    # Load known substitutions for validation
    known_subs = {}
    for row in conn.execute("SELECT LOWER(indicator), UPPER(substitution) FROM wordplay"):
        if row[0] not in known_subs:
            known_subs[row[0]] = []
        known_subs[row[0]].append(row[1])

    # Split explanation into segments by common delimiters
    # Many explanations use + , "followed by", "and then", "then"
    segments = re.split(
        r'\s*(?:\+|followed\s+by|and\s+then|then|,\s+then|,\s+and)\s*',
        explanation,
        flags=re.IGNORECASE
    )

    accounted = 0  # Position in answer we've accounted for

    for segment in segments:
        segment = segment.strip()
        if not segment or accounted >= len(answer_upper):
            break

        remaining = answer_upper[accounted:]

        # Try UPPER(lower) in this segment
        up_match = RE_UPPER_PAREN.search(segment)
        if up_match:
            letters = up_match.group(1).upper()
            fragment = up_match.group(2).strip().lower()
            # Same dictionary check as extract_upper_paren
            concat = letters.lower() + fragment
            concat_clean = re.sub(r'[^a-z]', '', concat)
            if concat_clean in (known_words or set()):
                word = concat_clean
            else:
                word = fragment
            if remaining.startswith(letters):
                accounted += len(letters)
                if len(letters) == 1:
                    cat = 'single_letter'
                elif len(letters) <= 3:
                    cat = 'abbreviation'
                else:
                    cat = 'synonym'
                junk = len(word) < 3 or any(p in word for p in [
                    'clue', 'from the', 'the answer', 'answer is', 'this gives'])
                if not junk:
                    results.append({
                        'indicator': word,
                        'substitution': letters,
                        'category': cat,
                        'table': 'wordplay',
                        'evidence': f'sequential: {word} = {letters}',
                        'position': accounted - len(letters)
                    })
                continue

        # Try "single letter for X"
        sl_match = RE_SINGLE_LETTER.search(segment)
        if sl_match and len(remaining) >= 1:
            word = sl_match.group(1).strip().lower()
            letter = remaining[0]
            # Only accept if pair is already known in DB — cannot reliably
            # guess the letter (e.g. 'love' -> O not L)
            if len(word) >= 3 and existing_wordplay and (word, letter) in existing_wordplay:
                results.append({
                    'indicator': word,
                    'substitution': letter,
                    'category': 'single_letter',
                    'table': 'wordplay',
                    'evidence': f'sequential: single letter for {word} = {letter}',
                    'position': accounted
                })
            accounted += 1
            continue

        # Try "abbreviation for X"
        ab_match = RE_ABBREVIATION.search(segment)
        if ab_match and len(remaining) >= 2:
            phrase = ab_match.group(1).strip().lower()
            phrase = re.sub(r'\s+(the|a|an|is|as|in|on|at|to)$', '', phrase)
            # Try known substitutions first
            if phrase in known_subs:
                for known in known_subs[phrase]:
                    if remaining.startswith(known):
                        results.append({
                            'indicator': phrase,
                            'substitution': known,
                            'category': 'abbreviation',
                            'table': 'wordplay',
                            'evidence': f'sequential: abbr for {phrase} = {known} (confirmed)',
                            'position': accounted
                        })
                        accounted += len(known)
                        break
            continue

        # Try "X from the clue" — literal word in the clue
        from_match = re.search(r'(\w+)\s+from\s+the\s+clue', segment, re.IGNORECASE)
        if from_match:
            word = from_match.group(1).upper()
            if remaining.startswith(word):
                accounted += len(word)
                continue  # Literal, no new substitution to record

    return results


# ======================================================================
# MAIN
# ======================================================================

def main():
    conn = get_conn()

    # Load existing data for dedup
    print("Loading existing DB entries for dedup...")
    existing_wordplay = load_existing_wordplay(conn)
    existing_indicators = load_existing_indicators(conn)
    existing_synonyms = load_existing_synonyms(conn)

    print(f"  Existing wordplay: {len(existing_wordplay)}")
    print(f"  Existing indicators: {len(existing_indicators)}")
    print(f"  Existing synonyms: {len(existing_synonyms)}")

    print("\nLoading dictionary of known English words...")
    known_words = load_known_words()
    print(f"  Dictionary size: {len(known_words):,} words")

    # Load telegraph explanations
    print("\nLoading telegraph explanations...")
    rows = conn.execute("""
        SELECT clue_text, answer, explanation
        FROM clues
        WHERE source IN ('telegraph', 'toughie', 'sunday_telegraph')
          AND explanation IS NOT NULL
          AND explanation != ''
          AND answer IS NOT NULL
          AND LENGTH(explanation) > 15
    """).fetchall()
    print(f"  Loaded {len(rows)} explanations")

    # Collect all discoveries
    wordplay_new = Counter()     # (indicator, substitution, category) -> count

    wordplay_evidence = defaultdict(list)
    synonyms_evidence = defaultdict(list)

    parsed = 0
    errors = 0

    for clue_text, answer, explanation in rows:
        try:
            answer_clean = (answer or '').upper().replace(' ', '')
            if not answer_clean or len(answer_clean) < 2:
                continue

            # Pattern A: UPPER(lower)
            for r in extract_upper_paren(explanation, answer_clean, existing_wordplay, known_words):
                key = (r['indicator'], r['substitution'], r['category'])
                if (r['indicator'], r['substitution']) not in existing_wordplay:
                    wordplay_new[key] += 1
                    if len(wordplay_evidence[key]) < 3:
                        wordplay_evidence[key].append(
                            f"[{answer_clean}] {clue_text[:50]}")

            # Pattern B: Single letter
            for r in extract_single_letter(explanation, answer_clean, existing_wordplay):
                key = (r['indicator'], r['substitution'], r['category'])
                if (r['indicator'], r['substitution']) not in existing_wordplay:
                    wordplay_new[key] += 1
                    if len(wordplay_evidence[key]) < 3:
                        wordplay_evidence[key].append(
                            f"[{answer_clean}] {clue_text[:50]}")

            # Pattern E: Chemical symbols
            for r in extract_chemical(explanation, answer_clean):
                key = (r['indicator'], r['substitution'], r['category'])
                if (r['indicator'], r['substitution']) not in existing_wordplay:
                    wordplay_new[key] += 1
                    if len(wordplay_evidence[key]) < 3:
                        wordplay_evidence[key].append(
                            f"[{answer_clean}] {clue_text[:50]}")

            # Sequential answer-constrained parsing
            for r in parse_sequential(explanation, answer_clean, conn, existing_wordplay, known_words):
                if r.get('substitution'):
                    key = (r['indicator'], r['substitution'], r['category'])
                    if (r['indicator'], r['substitution']) not in existing_wordplay:
                        wordplay_new[key] += 1
                        if len(wordplay_evidence[key]) < 3:
                            wordplay_evidence[key].append(
                                f"[{answer_clean}] {r['evidence']}")

            parsed += 1
            if parsed % 10000 == 0:
                print(f"  Processed {parsed}/{len(rows)}...")

        except Exception as e:
            errors += 1
            continue

    print(f"\nProcessed {parsed} explanations ({errors} errors)")

    # Filter: require frequency >= 2 for wordplay
    wordplay_filtered = {k: v for k, v in wordplay_new.items() if v >= 2}

    print(f"\nResults (after frequency filter):")
    print(f"  Wordplay entries (freq >= 2): {len(wordplay_filtered)}")

    # Write output file — one line per entry, split by target table
    print(f"\nWriting to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"Source: {parsed} explanations | Thresholds: wordplay >= 2, indicators >= 3\n\n")

        # Split into substitutions (wordplay table) vs synonyms (synonyms_pairs)
        # Substitution = many-to-one: word -> 1-2 letter abbreviation
        # Synonym = word-for-word: word -> word of similar length (3+ letters)
        substitutions = []
        synonyms = []

        for (indicator, subst, category), freq in sorted(
                wordplay_filtered.items(), key=lambda x: -x[1]):
            sub_len = len(subst.replace(' ', ''))
            if sub_len <= 2:
                substitutions.append((indicator, subst, freq))
            else:
                synonyms.append((indicator, subst, freq))

        # ---- TABLE: wordplay (substitutions) ----
        f.write(f"=== TABLE: wordplay (word -> 1-2 letter abbreviation) ===\n\n")
        for indicator, subst, freq in substitutions:
            f.write(f"{indicator:35s} -> {subst:8s}  ({freq})\n")
        f.write(f"\nCount: {len(substitutions)}\n\n")

        # ---- TABLE: synonyms_pairs (word = word) ----
        f.write(f"=== TABLE: synonyms_pairs (word = word, 3+ letters) ===\n\n")
        for indicator, subst, freq in synonyms:
            f.write(f"{indicator:35s} = {subst:8s}  ({freq})\n")
        f.write(f"\nCount: {len(synonyms)}\n\n")

        total_wordplay = len(substitutions) + len(synonyms)

        f.write(f"\nTOTALS: {len(substitutions)} substitutions, {len(synonyms)} synonyms\n")

    print(f"Done! Review file: {OUTPUT_FILE}")

    conn.close()


if __name__ == '__main__':
    main()
