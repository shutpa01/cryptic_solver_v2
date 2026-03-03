#!/usr/bin/env python3
"""
enrichment/06_phrase_corpus_lookup.py — Phrase → word mapping via corpus lookup.

Algorithm:
  1. Confirm phrase is a real crossword phrase: appears in >= MIN_HITS corpus clues
     (but not too commonly — skip if > MAX_HITS, those are noise)
  2. Use letters_still_needed from the failing clue as the sole constraint.
  3. Find the longest word W from answer vocabulary whose letters are a
     multiset subset of letters_still_needed.
  4. W is the mapping. No corpus majority voting — the answer tells us what fits.

Why this works: if "rental agreement" appears in real clues AND the only letters
still needed are LEASED, then the longest crossword word fitting LEASED is LEASE
(or EASEL — same letters, both valid for the pipeline).

Usage:
    python -m enrichment.06_phrase_corpus_lookup           # dry-run, review only
    python -m enrichment.06_phrase_corpus_lookup --insert  # write to synonyms_pairs
    python -m enrichment.06_phrase_corpus_lookup --min-hits 3 --max-hits 100
"""

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import enrichment.common as common
from enrichment.common import (
    get_clues_conn, get_cryptic_conn, get_pipeline_conn,
    insert_synonym_pair, InsertCounter,
)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FILE = PROJECT_ROOT / 'documents' / 'phrase_corpus_review.txt'

SOURCE_TAG = 'corpus_lookup'

MIN_HITS_DEFAULT = 3     # Min corpus clue hits (phrase must be a real crossword phrase)
MAX_HITS_DEFAULT = 200   # Max hits — very common words/phrases are noise
MIN_HIT_RATIO = 0.50     # Min fraction of corpus answers where candidate letters appear
MAX_PHRASE_WORDS = 3     # Max words in a candidate phrase
MIN_MAPPING_LEN = 4      # Min letters in the inferred mapping (3-letter subsets are noisy)

STOPWORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'and',
    'or', 'but', 'is', 'it', 'be', 'as', 'by', 'up', 'if', 'so', 'do',
    'not', 'no', 'its', 'are', 'was', 'has', 'had',
}


# ======================================================================
# HELPERS
# ======================================================================

def lcount(s: str) -> Counter:
    """Counter of uppercase letters only."""
    return Counter(c for c in (s or '').upper() if c.isalpha())


def is_subset(small: Counter, big: Counter) -> bool:
    """True if every letter in small appears with >= count in big."""
    return all(big[c] >= n for c, n in small.items())


def clean_answer(s: str) -> str:
    """Uppercase, letters only."""
    return re.sub(r'[^A-Za-z]', '', s or '').upper()


def clean_clue(s: str) -> str:
    """Lowercase, HTML-unescaped, strip enumeration."""
    s = html.unescape(s or '')
    s = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', s)
    return s.lower().strip()


# ======================================================================
# CORPUS LOADING
# ======================================================================

def load_corpus() -> List[Tuple[str, str]]:
    """Load all (clue_text_lower, answer_upper) from both DBs, deduplicated."""
    print("Loading corpus into memory...")
    seen: Set[Tuple[str, str]] = set()
    corpus: List[Tuple[str, str]] = []

    def _load(conn):
        for (ct, ans) in conn.execute(
            "SELECT clue_text, answer FROM clues "
            "WHERE clue_text IS NOT NULL AND answer IS NOT NULL"
        ):
            a = clean_answer(ans)
            if not (3 <= len(a) <= 15 and a.isalpha()):
                continue
            ct_lower = clean_clue(ct)
            key = (ct_lower, a)
            if key not in seen:
                seen.add(key)
                corpus.append((ct_lower, a))

    conn_master = get_clues_conn()
    _load(conn_master)
    conn_master.close()

    conn_cryptic = get_cryptic_conn()
    _load(conn_cryptic)
    conn_cryptic.close()

    print(f"  {len(corpus):,} unique (clue, answer) pairs loaded")
    return corpus


def build_inverted_index(corpus: List[Tuple[str, str]]) -> Dict[str, Set[int]]:
    """word → set of corpus indices containing that word in clue_text."""
    print("Building inverted word index...")
    index: Dict[str, Set[int]] = defaultdict(set)
    for idx, (ct, _) in enumerate(corpus):
        for word in re.findall(r"[a-z']+", ct):
            index[word].add(idx)
    print(f"  {len(index):,} unique words indexed")
    return index


def build_answer_vocab(corpus: List[Tuple[str, str]]) -> List[Tuple[str, Counter]]:
    """All unique 3-10 letter answers as (word, letter_counter), longest first."""
    print("Building answer vocabulary...")
    freq: Counter = Counter()
    for _, a in corpus:
        if 3 <= len(a) <= 10:
            freq[a.lower()] += 1

    vocab = sorted(
        [(w, lcount(w)) for w in freq],
        key=lambda x: (-len(x[0]), -freq[x[0]])
    )
    print(f"  {len(vocab):,} unique answer words")
    return vocab


# ======================================================================
# PHRASE SEARCH
# ======================================================================

def find_phrase_answers(
    phrase: str,
    inverted_index: Dict[str, Set[int]],
    corpus: List[Tuple[str, str]],
) -> List[str]:
    """Return deduplicated list of answers from corpus clues containing the phrase."""
    words = re.findall(r"[a-z']+", phrase.lower())
    if not words:
        return []

    candidates = inverted_index.get(words[0], set())
    for w in words[1:]:
        candidates = candidates & inverted_index.get(w, set())
        if not candidates:
            return []

    seen: Set[str] = set()
    answers: List[str] = []
    for idx in candidates:
        ct, ans = corpus[idx]
        if phrase.lower() in ct and ans not in seen:
            seen.add(ans)
            answers.append(ans)

    return answers


def majority_letter_counter(answers: List[str]) -> Counter:
    """
    For each letter, the max count k such that >= MIN_HIT_RATIO of answers have >= k of that letter.
    """
    n = len(answers)
    counters = [lcount(a) for a in answers]
    result = Counter()
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        per_answer = sorted([c[letter] for c in counters], reverse=True)
        for k in range(1, per_answer[0] + 1):
            if sum(1 for c in per_answer if c >= k) >= n * MIN_HIT_RATIO:
                result[letter] = k
            else:
                break
    return result


# ======================================================================
# MAPPING INFERENCE
# ======================================================================

def find_mapping(
    phrase: str,
    corpus_answers: List[str],
    answer_vocab: List[Tuple[str, Counter]],
    letters_needed: str,
    min_hits: int,
    max_hits: int,
) -> Optional[dict]:
    """
    Find the longest word W that satisfies both:
      1. Letters fit in letters_needed (constraint from failing clue)
      2. Letters appear in >= MIN_HIT_RATIO of corpus answers for this phrase
         (validates genuine semantic signal, not coincidence)

    Returns the first (longest) valid match, or None.
    """
    n = len(corpus_answers)
    if n < min_hits or n > max_hits:
        return None

    needed_c = lcount(letters_needed)
    needed_len = sum(needed_c.values())
    if needed_len < MIN_MAPPING_LEN:
        return None

    # Majority letter counter: letters that appear in >= 50% of corpus answers
    majority = majority_letter_counter(corpus_answers)

    # Effective constraint: intersection of majority and letters_needed
    effective = Counter({
        k: min(v, needed_c[k])
        for k, v in majority.items()
        if needed_c[k] > 0
    })
    if sum(effective.values()) < MIN_MAPPING_LEN:
        return None

    answer_counters = [lcount(a) for a in corpus_answers]

    for word, wc in answer_vocab:  # longest first
        if len(word) < MIN_MAPPING_LEN:
            break

        # Must fit within effective (majority ∩ letters_needed)
        if not is_subset(wc, effective):
            continue

        # Must not equal the full failing answer
        if word.upper() == letters_needed:
            continue

        # Full corpus vote
        hits = sum(1 for ac in answer_counters if is_subset(wc, ac))
        if hits >= min_hits and hits / n >= MIN_HIT_RATIO:
            return {
                'phrase': phrase,
                'mapping': word.upper(),
                'corpus_hits': n,
                'corpus_vote': f'{hits}/{n}',
                'letters_needed': letters_needed,
            }

    return None


# ======================================================================
# PHRASE GENERATION
# ======================================================================

def candidate_phrases(
    clue_text: str,
    unresolved: List[str],
    max_words: int = MAX_PHRASE_WORDS,
) -> List[str]:
    """
    All 1-max_words adjacent word groups from clue that contain >= 1 unresolved word.
    Skips single stopwords.
    """
    ct = clean_clue(clue_text)
    tokens = re.findall(r"[a-zA-Z']+", ct)
    ur_set = {re.sub(r"[^a-z]", '', w.lower()) for w in unresolved if w}

    phrases: List[str] = []
    seen: Set[str] = set()

    for i in range(len(tokens)):
        for length in range(2, min(max_words + 1, len(tokens) - i + 1)):
            chunk = tokens[i:i + length]
            phrase = ' '.join(chunk).lower()

            if phrase in seen:
                continue
            seen.add(phrase)

            chunk_clean = {re.sub(r"[^a-z]", '', w.lower()) for w in chunk}
            if not chunk_clean & ur_set:
                continue

            if len(phrase) >= 3:
                phrases.append(phrase)

    return phrases


# ======================================================================
# FAILURE LOADING
# ======================================================================

def load_failures() -> List[dict]:
    """Load stage_secondary failures with unresolved words and letters_still_needed."""
    conn = get_pipeline_conn()
    rows = conn.execute("""
        SELECT clue_text, answer, unresolved_words, letters_still_needed
        FROM stage_secondary
        WHERE fully_resolved = 0
          AND unresolved_words IS NOT NULL
          AND unresolved_words != '[]'
          AND answer IS NOT NULL
          AND letters_still_needed IS NOT NULL
          AND letters_still_needed != ''
    """).fetchall()
    conn.close()

    failures = []
    seen_answers: Set[str] = set()
    for clue_text, answer, uw_json, letters_needed in rows:
        a = clean_answer(answer)
        ln = clean_answer(letters_needed)
        if not a or not ln or a in seen_answers:
            continue
        seen_answers.add(a)
        try:
            unresolved = json.loads(uw_json or '[]')
        except Exception:
            continue
        unresolved_clean = [
            re.sub(r"[^a-z]", '', w.lower())
            for w in unresolved
            if len(re.sub(r"[^a-z]", '', w.lower())) >= 3
        ]
        if not unresolved_clean:
            continue
        failures.append({
            'clue_text': clue_text or '',
            'answer': a,
            'unresolved': unresolved_clean,
            'letters_needed': ln,
        })
    return failures


# ======================================================================
# OUTPUT
# ======================================================================

def write_review_file(discoveries: List[dict]):
    """Write candidates to human-readable review file."""
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(
            f"Phrase Corpus Lookup — {len(discoveries)} candidates\n"
            f"Min mapping length: {MIN_MAPPING_LEN} letters\n\n"
        )
        for d in discoveries:
            f.write(
                f"PHRASE:   {d['phrase']}\n"
                f"MAPPING:  {d['mapping']}  "
                f"(corpus: {d['corpus_vote']} = {int(int(d['corpus_vote'].split('/')[0])/int(d['corpus_vote'].split('/')[1])*100)}%)\n"
                f"CLUE:     {d['failing_clue'][:80]}\n"
                f"ANSWER:   {d['failing_answer']}  "
                f"(needed: {d['letters_needed']})\n\n"
            )
    print(f"Review file: {OUTPUT_FILE}")


# ======================================================================
# MAIN
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Infer phrase→word mappings from clue corpus.'
    )
    parser.add_argument('--insert', action='store_true',
                        help='Write to synonyms_pairs (default: dry-run)')
    parser.add_argument('--min-hits', type=int, default=MIN_HITS_DEFAULT)
    parser.add_argument('--max-hits', type=int, default=MAX_HITS_DEFAULT)
    args = parser.parse_args()

    if not args.insert:
        common.DRY_RUN = True
        print("[DRY RUN — no writes will be made]")

    # ── Load data ──────────────────────────────────────────────
    corpus = load_corpus()
    inverted_index = build_inverted_index(corpus)
    answer_vocab = build_answer_vocab(corpus)

    print("Loading pipeline failures...")
    failures = load_failures()
    print(f"  {len(failures)} failures to process")

    # ── Build candidate phrases ───────────────────────────────
    print("Generating candidate phrases...")
    phrase_to_failures: Dict[str, List[dict]] = defaultdict(list)
    for failure in failures:
        for phrase in candidate_phrases(failure['clue_text'], failure['unresolved']):
            phrase_to_failures[phrase].append(failure)
    print(f"  {len(phrase_to_failures):,} unique phrases to search")

    # ── Load existing synonyms for dedup ─────────────────────
    conn_cryptic = get_cryptic_conn()
    existing: Set[Tuple[str, str]] = set(
        conn_cryptic.execute("SELECT LOWER(word), LOWER(synonym) FROM synonyms_pairs")
    )
    print(f"  {len(existing):,} existing synonym pairs")

    # ── Search and infer ──────────────────────────────────────
    print("\nSearching corpus and inferring mappings...")
    discoveries: List[dict] = []
    seen_keys: Set[Tuple[str, str]] = set(existing)
    checked = 0
    had_hits = 0

    for phrase, phrase_failures in sorted(phrase_to_failures.items()):
        corpus_answers = find_phrase_answers(phrase, inverted_index, corpus)
        checked += 1

        if len(corpus_answers) < args.min_hits or len(corpus_answers) > args.max_hits:
            continue
        had_hits += 1

        for failure in phrase_failures:
            result = find_mapping(
                phrase, corpus_answers, answer_vocab,
                failure['letters_needed'],
                args.min_hits, args.max_hits,
            )
            if not result:
                continue

            key = (phrase.lower(), result['mapping'].lower())
            if key in seen_keys:
                continue
            seen_keys.add(key)

            discoveries.append({
                **result,
                'failing_clue': failure['clue_text'],
                'failing_answer': failure['answer'],
            })

        if checked % 500 == 0:
            print(f"  {checked}/{len(phrase_to_failures)} phrases checked, "
                  f"{had_hits} in range, {len(discoveries)} mappings")

    discoveries.sort(key=lambda x: (-x['corpus_hits'], -len(x['mapping'])))
    print(f"\nFound {len(discoveries)} new mappings "
          f"({had_hits}/{checked} phrases in hit range)")

    # ── Write review file ─────────────────────────────────────
    write_review_file(discoveries)

    # ── Insert ────────────────────────────────────────────────
    counter = InsertCounter('06_phrase_corpus_lookup')
    for d in discoveries:
        inserted = insert_synonym_pair(
            conn_cryptic, d['phrase'].lower(), d['mapping'].lower(), SOURCE_TAG
        )
        counter.record('synonyms_pairs', inserted,
                       f"{d['phrase']} -> {d['mapping']} ({d['corpus_hits']} hits)")

    conn_cryptic.commit()
    conn_cryptic.close()
    counter.report()


if __name__ == '__main__':
    main()
