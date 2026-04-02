"""Backfill batch solver for Double Definition and Hidden Word clues.

Adapts proven V1 archive algorithms to run against the current RefDB:
  - DD: from cryptic_solver_archive/stages/dd.py (coverage-checked)
  - Hidden: from cryptic_solver_archive/stages/lurker.py (proper suffix/prefix validation)

All output goes to JSONL — NEVER writes to the database.

Usage:
    python scripts/backfill_dd_hidden.py                    # all unsolved
    python scripts/backfill_dd_hidden.py --limit 1000       # test run
    python scripts/backfill_dd_hidden.py --source guardian   # one source
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.db import RefDB, _normalize_key

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RESULTS_PATH = os.path.join(ROOT, "data", "backfill_dd_hidden_results.jsonl")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm_letters(s):
    """Strip non-alpha, lowercase. Same as V1 resources.norm_letters."""
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


def parse_enumeration(enumeration):
    """Parse enumeration string to total letter count. '(5,3)' -> 8"""
    if isinstance(enumeration, int):
        return enumeration
    if isinstance(enumeration, str):
        digits = re.findall(r'\d+', enumeration)
        return sum(int(d) for d in digits) if digits else 0
    return 0


def strip_enumeration(clue_text):
    """Remove trailing enumeration from clue text."""
    return re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()


# ---------------------------------------------------------------------------
# Graph builder — builds V1-compatible graph from RefDB
# ---------------------------------------------------------------------------

def build_graph(ref_db):
    """Build bidirectional definition->answer graph from RefDB's synonyms data.

    V1's load_graph() built this from definition_answers_augmented + synonyms_pairs,
    both directions. RefDB.synonyms already merges both tables but only stores
    phrase->answers direction. We add the reverse direction here.

    Keys use _normalize_key() for consistency with how RefDB loaded the data.
    """
    graph = defaultdict(list)

    for key, values in ref_db.synonyms.items():
        # key is already _normalize_key'd from RefDB loading
        for v in values:
            if v not in graph[key]:
                graph[key].append(v)
            # Reverse direction: answer -> phrase
            rev_key = _normalize_key(v)
            if rev_key:
                key_val = key.upper()
                if key_val not in graph[rev_key]:
                    graph[rev_key].append(key_val)

    return dict(graph)


# ---------------------------------------------------------------------------
# DD solver — adapted from cryptic_solver_archive/stages/dd.py
# ---------------------------------------------------------------------------

def generate_definition_windows(clue_text):
    """Generate candidate definition substrings from both ends of the clue.

    Adapted from cryptic_solver_archive/stages/definition.py lines 54-81.
    """
    clue_text = (
        clue_text
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )

    words = clue_text.split()
    windows = set()

    for i in range(len(words)):
        window1 = " ".join(words[:i + 1])
        window2 = " ".join(words[-(i + 1):])
        if window1:
            windows.add(window1.strip())
        if window2:
            windows.add(window2.strip())

    # Apostrophe bifurcation — try both with and without possessive S
    expanded = set(windows)
    for w in windows:
        if "\u2019s" in w or "'s" in w:
            # "lout's" -> "louts" (plural form)
            expanded.add(w.replace("\u2019s", "s").replace("'s", "s"))
            # "lout's" -> "lout" (possessive stripped)
            expanded.add(w.replace("\u2019s", "").replace("'s", ""))

    return list(expanded)


def _extract_words(clue_text):
    """Extract words from clue, removing enumeration."""
    text = re.sub(r'\(\d+(?:,\d+)*\)\s*$', '', clue_text).strip()
    return text.split()


def _get_candidates_for_phrase(phrase, graph, total_len=None):
    """Get all candidates for windows generated from a phrase.

    Returns dict: {normalized_candidate: [(window, raw_candidate), ...]}

    Adapted from cryptic_solver_archive/stages/dd.py lines 41-68.
    Uses _normalize_key instead of V1 clean_key for graph lookups.
    """
    candidates = defaultdict(list)

    windows = generate_definition_windows(phrase)

    for window in windows:
        key = _normalize_key(window)
        if not key:
            continue

        graph_candidates = graph.get(key)
        if not graph_candidates:
            continue

        for cand in graph_candidates:
            cand_norm = norm_letters(cand)

            if total_len is not None and len(cand_norm) != total_len:
                continue

            candidates[cand_norm].append((window, cand))

    return candidates


def generate_dd_hypotheses(clue_text, graph, total_len=None, answer=None):
    """True Double Definition detection.

    Adapted from cryptic_solver_archive/stages/dd.py lines 71-149.
    Algorithm unchanged — only the key normalization function differs.

    A DD fires only if:
    1. Clue splits into LEFT and RIGHT parts
    2. LEFT independently produces candidate X
    3. RIGHT independently produces the SAME candidate X
    4. The two definition windows cover nearly all clue words (max 1 uncovered)
    5. If answer provided, candidate must match it

    Returns at most ONE hit.
    """
    if not clue_text:
        return None

    words = _extract_words(clue_text)

    if len(words) < 2:
        return None

    for split_point in range(1, len(words)):
        left_phrase = ' '.join(words[:split_point])
        right_phrase = ' '.join(words[split_point:])

        left_candidates = _get_candidates_for_phrase(left_phrase, graph, total_len)
        right_candidates = _get_candidates_for_phrase(right_phrase, graph, total_len)

        overlap = set(left_candidates.keys()) & set(right_candidates.keys())

        if answer and overlap:
            answer_norm = norm_letters(answer)
            overlap = {c for c in overlap if c == answer_norm}

        if overlap:
            left_word_count = split_point
            right_word_count = len(words) - split_point

            for cand_norm in overlap:
                best_left = max(left_candidates[cand_norm],
                                key=lambda x: len(x[0].split()))
                best_right = max(right_candidates[cand_norm],
                                 key=lambda x: len(x[0].split()))

                left_window, left_answer = best_left
                right_window, right_answer = best_right

                left_covered = len(left_window.split())
                right_covered = len(right_window.split())
                uncovered = (left_word_count - left_covered) + (right_word_count - right_covered)

                if uncovered > 1:
                    continue

                return {
                    "answer": left_answer,
                    "left_def": left_window,
                    "right_def": right_window,
                    "split_point": split_point,
                }

    return None


# ---------------------------------------------------------------------------
# Hidden word solver — adapted from cryptic_solver_archive/stages/lurker.py
# ---------------------------------------------------------------------------

def _letters_only_stream(clue_text):
    """Convert clue to letters-only lowercase stream."""
    return "".join(ch.lower() for ch in clue_text if ch.isalpha())


def _word_spans_letters_only(clue_text):
    """Word spans in letters-only coordinates. Each span is [start, end)."""
    spans = []
    idx = 0
    in_word = False
    start = 0

    for ch in clue_text:
        if ch.isalpha():
            if not in_word:
                in_word = True
                start = idx
            idx += 1
        else:
            if in_word:
                spans.append((start, idx))
                in_word = False

    if in_word:
        spans.append((start, idx))

    return spans


def _is_valid_lurker_span(span, word_spans):
    """A valid lurker span must:
      - cross at least one word boundary (span 2+ words)
      - take a proper suffix of the first word
      - take a proper prefix of the last word
      - include complete middle words (if any)
    """
    s, e = span

    touched = []
    for ws, we in word_spans:
        if s < we and e > ws:
            touched.append((ws, we))

    if len(touched) < 2 or len(touched) > 3:
        return False

    # Check touched words are adjacent
    for i in range(len(touched) - 1):
        if touched[i][1] != touched[i + 1][0]:
            return False

    # First word: must start inside (proper suffix)
    w1s, w1e = touched[0]
    if not (w1s < s < w1e):
        return False

    # Last word: must end inside (proper prefix)
    wLs, wLe = touched[-1]
    if not (wLs < e < wLe):
        return False

    return True


def definition_candidates(clue_text, enumeration, graph):
    """Find candidate answers via definition window lookups.

    Adapted from cryptic_solver_archive/stages/definition.py lines 10-51.
    Uses _normalize_key instead of V1 clean_key for graph lookups.

    Returns dict with candidates (set), support (dict: candidate -> set of windows).
    """
    ARTICLES = ("a ", "an ", "the ")

    windows = generate_definition_windows(clue_text)

    candidates = set()
    support = {}

    for dp in windows:
        dp_norm = dp.replace("\u2018", "'").replace("\u2019", "'")
        key = _normalize_key(dp_norm)
        if key in graph:
            for cand in graph[key]:
                cand_norm = norm_letters(cand)
                if len(cand_norm) == enumeration:
                    candidates.add(cand)
                    support.setdefault(cand, set()).add(dp)

    # Article-variant matches
    for dp in windows:
        dp_norm = dp.replace("\u2018", "'").replace("\u2019", "'")
        dp_key = _normalize_key(dp_norm)
        for art in ARTICLES:
            key = _normalize_key(art + dp_key)
            if key in graph:
                for cand in graph[key]:
                    cand_norm = norm_letters(cand)
                    if len(cand_norm) == enumeration:
                        candidates.add(cand)
                        support.setdefault(cand, set()).add(dp)

    return {"candidates": list(candidates), "support": support}


def try_hidden(clue_text, answer, graph, enumeration):
    """Check if the answer is hidden in the clue, with definition confirmation.

    Follows the V1 pipeline discipline:
    1. Run definition engine to get candidates from definition windows
    2. Check that the KNOWN ANSWER is among those candidates (definition confirmed)
    3. Run lurker to find the answer hidden in the clue text
    4. Return result with the verified definition

    Returns dict with direction, spanning words, and definition — or None.
    """
    # Step 1: Get definition-confirmed candidates
    def_result = definition_candidates(clue_text, enumeration, graph)
    candidates = def_result["candidates"]
    support = def_result["support"]

    # Step 2: Check that the known answer is among definition candidates
    answer_norm = norm_letters(answer)
    matching_candidate = None
    for cand in candidates:
        if norm_letters(cand) == answer_norm:
            matching_candidate = cand
            break

    if matching_candidate is None:
        return None  # Answer not confirmed by definition engine

    # Step 3: Run lurker for ONLY the known answer
    text = strip_enumeration(clue_text)
    hypotheses = _lurker_search(text, enumeration, [matching_candidate])

    if not hypotheses:
        return None

    # Step 4: Get the definition window that confirmed this answer
    hit = hypotheses[0]
    def_windows = support.get(matching_candidate, set())
    best_def = max(def_windows, key=lambda w: len(w.split())) if def_windows else None
    if best_def:
        best_def = best_def.strip().strip(".,;:!?\"'()-")

    return {
        "direction": hit["direction"],
        "words": _find_spanning_words_from_hit(hit, text),
        "definition": best_def,
    }


def _lurker_search(clue_text, enumeration, candidates):
    """Core lurker search — adapted from V1 _candidate_bounded_hypotheses.

    Only searches for candidates provided by the definition engine.
    """
    if not candidates:
        return []

    norm_candidates = {}
    for c in candidates:
        nc = norm_letters(c)
        if len(nc) == enumeration:
            norm_candidates[nc] = c

    if not norm_candidates:
        return []

    stream = _letters_only_stream(clue_text)
    n = len(stream)
    if n < enumeration:
        return []

    word_spans = _word_spans_letters_only(clue_text)
    hypotheses = []

    for i in range(0, n - enumeration + 1):
        span = (i, i + enumeration)

        if not _is_valid_lurker_span(span, word_spans):
            continue

        window = stream[i:i + enumeration]

        if window in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[window],
                "direction": "forward",
                "span": span,
            })

        rev = window[::-1]
        if rev in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[rev],
                "direction": "reverse",
                "span": span,
            })

    return hypotheses


def _find_spanning_words_from_hit(hit, clue_text):
    """Find the clue words that the hidden span covers."""
    span = hit["span"]
    word_spans = _word_spans_letters_only(clue_text)
    words = clue_text.split()
    return _find_spanning_words(span, word_spans, words)


def _find_spanning_words(span, word_spans, words):
    """Find the clue words that the hidden span covers."""
    s, e = span
    spanning = []
    for idx, (ws, we) in enumerate(word_spans):
        if s < we and e > ws:
            if idx < len(words):
                spanning.append(words[idx])
    return " ".join(spanning)


# ---------------------------------------------------------------------------
# Highlight hidden letters in the clue (for explanation text)
# ---------------------------------------------------------------------------

def _highlight_hidden(clue_text, answer, direction):
    """Capitalise the hidden letters within the clue words for the explanation."""
    answer_clean = norm_letters(answer)
    text = strip_enumeration(clue_text)

    stream = _letters_only_stream(text)
    enum_len = len(answer_clean)

    target = answer_clean if direction == "forward" else answer_clean[::-1]

    idx = stream.find(target)
    if idx < 0:
        return text

    # Map stream positions back to original text positions
    char_positions = []
    for pos, ch in enumerate(text):
        if ch.isalpha():
            char_positions.append(pos)

    if idx + enum_len > len(char_positions):
        return text

    result = list(text)
    for j in range(idx, idx + enum_len):
        orig_pos = char_positions[j]
        result[orig_pos] = result[orig_pos].upper()
    # Lowercase the rest
    for j in range(len(char_positions)):
        if j < idx or j >= idx + enum_len:
            orig_pos = char_positions[j]
            result[orig_pos] = result[orig_pos].lower()

    return "".join(result)


# ---------------------------------------------------------------------------
# JSONL output helpers
# ---------------------------------------------------------------------------

def make_dd_result(clue_id, dd_result, answer, source, puzzle_number, clue_number):
    """Build JSONL record for a DD solve."""
    left_def = dd_result["left_def"]
    right_def = dd_result["right_def"]

    return {
        "clue_id": clue_id,
        "action": "dd_solve",
        "payload": {
            "definition": "Double definition",
            "wordplay_type": "double_definition",
            "ai_explanation": "Double definition",
            "has_solution": 1,
            "reviewed": 1,
            "confidence": 1.0,
            "components": {
                "ai_pieces": [],
                "assembly": {
                    "op": "double_definition",
                    "left_def": left_def,
                    "right_def": right_def,
                },
                "wordplay_type": "double_definition",
            },
            "wordplay_types": ["double_definition"],
            "definition_start": None,
            "definition_end": None,
            "model_version": "v1_dd_backfill",
            "source": source,
            "puzzle_number": puzzle_number,
            "clue_number": clue_number,
        }
    }


def make_hidden_result(clue_id, hidden_result, clue_text, answer, source, puzzle_number, clue_number):
    """Build JSONL record for a hidden word solve."""
    direction = hidden_result["direction"]
    spanning_words = hidden_result["words"]
    definition = hidden_result.get("definition")
    wtype = "hidden_reversed" if direction == "reverse" else "hidden"

    highlighted = _highlight_hidden(clue_text, answer, direction)
    if direction == "reverse":
        expl = 'Hidden reversed in "%s"' % highlighted
    else:
        expl = 'Hidden in "%s"' % highlighted

    return {
        "clue_id": clue_id,
        "action": "hidden_solve",
        "payload": {
            "definition": definition,
            "wordplay_type": wtype,
            "ai_explanation": expl,
            "has_solution": 1,
            "reviewed": 1,
            "confidence": 1.0,
            "components": {
                "ai_pieces": [{
                    "clue_word": spanning_words,
                    "letters": answer.upper(),
                    "mechanism": wtype,
                }],
                "assembly": {"op": wtype, "words": spanning_words},
                "wordplay_type": wtype,
            },
            "wordplay_types": [wtype],
            "definition_start": None,
            "definition_end": None,
            "model_version": "v1_hidden_backfill",
            "source": source,
            "puzzle_number": puzzle_number,
            "clue_number": clue_number,
        }
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backfill DD + Hidden solves")
    parser.add_argument("--limit", type=int, default=0, help="Max clues to process (0=all)")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    parser.add_argument("--output", type=str, default=RESULTS_PATH, help="Output JSONL path")
    args = parser.parse_args()

    print("Loading RefDB...")
    t0 = time.time()
    ref_db = RefDB()
    print(f"  RefDB loaded in {time.time() - t0:.1f}s")

    print("Building graph...")
    t0 = time.time()
    graph = build_graph(ref_db)
    print(f"  Graph built in {time.time() - t0:.1f}s — {len(graph):,} keys")

    print("Loading unsolved clues...")
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    query = """
        SELECT id, source, puzzle_number, clue_number, clue_text, answer, enumeration
        FROM clues
        WHERE answer IS NOT NULL AND answer != ''
          AND (has_solution IS NULL OR has_solution = 0)
    """
    params = []
    if args.source:
        query += " AND source = ?"
        params.append(args.source)
    query += " ORDER BY publication_date DESC"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    print(f"  {len(rows):,} unsolved clues loaded")

    # Process
    dd_count = 0
    hidden_count = 0
    processed = 0
    results = []

    t_start = time.time()

    for cid, source, puzzle_number, clue_number, clue_text, answer, enumeration in rows:
        processed += 1

        answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
        if not answer_clean or len(answer_clean) < 2:
            continue

        total_len = len(norm_letters(answer))

        # Try hidden first — requires definition confirmation via graph
        hidden_result = try_hidden(clue_text, answer_clean, graph, total_len)
        if hidden_result:
            hidden_count += 1
            results.append(make_hidden_result(
                cid, hidden_result, clue_text, answer, source, puzzle_number, clue_number))
            continue

        # Try DD
        dd_result = generate_dd_hypotheses(
            clue_text, graph, total_len=total_len, answer=answer_clean)
        if dd_result:
            dd_count += 1
            results.append(make_dd_result(
                cid, dd_result, answer, source, puzzle_number, clue_number))

        if processed % 10000 == 0:
            elapsed = time.time() - t_start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  {processed:,} processed — {hidden_count:,} hidden, {dd_count:,} DD "
                  f"({rate:.0f} clues/sec)")

    elapsed = time.time() - t_start

    # Write results
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Processed: {processed:,}")
    print(f"  Hidden:    {hidden_count:,}")
    print(f"  DD:        {dd_count:,}")
    print(f"  Total:     {hidden_count + dd_count:,}")
    print(f"  Output:    {args.output}")


if __name__ == "__main__":
    main()
