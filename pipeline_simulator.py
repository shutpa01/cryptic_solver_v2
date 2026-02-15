# solver/solver_engine/pipeline_simulator.py
#
# Pipeline simulator:
# - runs clues through ALL stages
# - preserves evidence
# - optional wordplay_type filter
# - ENHANCED: Now includes forwarded anagram cohort analysis
# - ENHANCED: Now persists stage data to SQLite for debugging

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Dict, Any

from resources import (
    connect_db, connect_clues_db,
    load_graph,
    parse_enum,
    norm_letters,
    clean_key,
    matches_enumeration,
)

from stages.definition_edges import definition_candidates
from stages.dd import generate_dd_hypotheses
from stages.anagram import generate_anagram_hypotheses
from stages.lurker import generate_lurker_hypotheses

# Evidence System Integration for Analysis
try:
    from stages.evidence import \
        ComprehensiveWordplayDetector

    EVIDENCE_SYSTEM_AVAILABLE = True
except ImportError:
    EVIDENCE_SYSTEM_AVAILABLE = False

# Pipeline Persistence Integration
try:
    from persistence import start_run, save_stage

    PERSISTENCE_AVAILABLE = True
except ImportError:
    PERSISTENCE_AVAILABLE = False
    print("WARNING: Pipeline persistence not available - stage data will not be saved")

# ==============================
# SIMULATOR CONFIGURATION
# ==============================

MAX_CLUES = 40
WORDPLAY_TYPE = "all"  # e.g. "all", "anagram", "lurker", "dd"
ONLY_MISSING_DEFINITION = False  # show only clues where answer NOT in def candidates
MAX_DISPLAY = 50  # max number of clues to print
SINGLE_CLUE_MATCH = ""  # normalised substring match on clue_text (highest priority)
SOURCE = "telegraph"  # e.g. "telegraph", "times", "guardian", "ft", "independent" (
# empty = all)
PUZZLE_NUMBER = "30000"  # e.g. "29345" (empty = all)

# NEW: Use known answer as single candidate (skip definition candidate generation)
USE_KNOWN_ANSWER = True  # When True, answer becomes the only candidate

# NEW: Forwarded cohort analysis settings
ANALYZE_FORWARDED_ANAGRAMS = False  # Disable for explanation system development - focus on successes
MAX_FORWARDED_SAMPLES = 50  # Max forwarded samples to show

# NEW: Successful anagram analysis for explanation system development
ANALYZE_SUCCESSFUL_ANAGRAMS = False  # Enable successful anagram analysis
MAX_SUCCESSFUL_SAMPLES = 25  # Max successful samples to show

# NEW: Persistence settings
ENABLE_PERSISTENCE = True  # Set to False to disable stage persistence
EXCLUDE_SOLVED = False  # Exclude clues that already have a solution

_ENUM_RE = re.compile(r"\(\d+(?:,\d+)*\)")


def _norm_continuous(s: str) -> str:
    """Lowercase, letters only, no spaces."""
    return re.sub(r"[^a-z]", "", s.lower())


def _match_single_clue(query: str, clue_text: str) -> bool:
    """
    Match query against clue_text using continuous normalised letters,
    but tolerate extra words in the clue.
    Example:
      query: "top removed vehicle in struggle"
      clue : "Top removed vehicle in a struggle"
    should match.
    """
    q_words = [w for w in re.findall(r"[A-Za-z]+", query.lower()) if w]
    if not q_words:
        return False
    # Build a regex that matches the words in order with any letters between them.
    pattern = ".*".join(re.escape(_norm_continuous(w)) for w in q_words)
    return re.search(pattern, _norm_continuous(clue_text)) is not None


def _clean_window(w: str) -> str:
    w = _ENUM_RE.sub("", w)
    w = w.strip(" ,.;:-")
    return " ".join(w.split())


def _length_filter(cands: List[str], enumeration: str) -> List[str]:
    """
    Filter candidates by enumeration.
    For single-word enumerations (7): check total letter count
    For multi-word enumerations (6,3,5): require word pattern to match
    """
    total_len = parse_enum(enumeration)
    parts = re.findall(r'\d+', enumeration or "")
    is_multi_word = len(parts) > 1
    pattern = [int(p) for p in parts]

    result = []
    for c in cands:
        # First check: total letter count must match
        if len(norm_letters(c)) != total_len:
            continue

        # For multi-word enumerations, check word pattern
        if is_multi_word:
            # Split candidate into words
            words = c.split()

            if len(words) == len(pattern):
                # Check each word length matches pattern
                word_lengths = [len(norm_letters(w)) for w in words]
                if word_lengths == pattern:
                    result.append(c)
            # Single-word candidates rejected for multi-word enumerations
            # (AUTHORITY rejected for (2,4,3) even though it's 9 letters)
        else:
            # Single-word enumeration: just total length check
            result.append(c)

    return result


def _analyze_forwarded_anagram(clue_text: str, answer: str, candidates: List[str],
                               enumeration: str) -> Dict[str, Any]:
    """Analyze why an anagram case was forwarded (not solved)."""

    analysis = {
        "clue": clue_text,
        "answer": answer,
        "candidates_sample": candidates[:8],  # Show first 8 for readability
        "answer_in_candidates": answer in [norm_letters(c) for c in candidates],
        "indicators_detected": [],
        "evidence_system_available": EVIDENCE_SYSTEM_AVAILABLE,
        "evidence_system_result": "not_tested",
        "failure_reason": "unknown"
    }

    if not EVIDENCE_SYSTEM_AVAILABLE:
        analysis["failure_reason"] = "evidence_system_not_available"
        return analysis

    try:
        # Test evidence system to see what it found
        detector = ComprehensiveWordplayDetector()

        # Check indicator detection first
        indicators = detector.detect_wordplay_indicators(clue_text)
        analysis["indicators_detected"] = indicators.get('anagram', [])

        if not indicators.get('anagram'):
            analysis["failure_reason"] = "no_anagram_indicators"
            analysis["evidence_system_result"] = "skipped_no_indicators"
            return analysis

        # If indicators found, test evidence system
        evidence_list = detector.analyze_clue_for_anagram_evidence(
            clue_text=clue_text,
            candidates=candidates,
            enumeration=enumeration,
            debug=False
        )

        if evidence_list:
            analysis["evidence_system_result"] = f"found_{len(evidence_list)}_evidence"
            analysis[
                "failure_reason"] = "evidence_found_but_original_missed"  # Shouldn't happen in additive mode
        else:
            analysis["evidence_system_result"] = "no_evidence_found"
            analysis["failure_reason"] = "evidence_system_failed_to_find"

    except Exception as e:
        analysis["evidence_system_result"] = f"error: {str(e)[:50]}"
        analysis["failure_reason"] = "evidence_system_error"

    return analysis


def _analyze_successful_anagram(clue_text: str, answer: str, candidates: List[str],
                                anagram_hits: List[Dict],
                                window_support: Dict[str, List[str]],
                                enumeration: str) -> Dict[str, Any]:
    """Analyze successful anagram case for explanation system development."""

    # Take the first (best) anagram hit
    best_hit = anagram_hits[0] if anagram_hits else {}

    # Find which definition window provided the answer
    definition_window = None
    normalized_answer = norm_letters(answer)
    for window, window_candidates in window_support.items():
        normalized_candidates = [norm_letters(c) for c in window_candidates]
        if normalized_answer in normalized_candidates:
            definition_window = window
            break

    analysis = {
        "clue": clue_text,
        "answer": answer,
        "definition_window": definition_window,
        "anagram_evidence": {
            "candidate": best_hit.get("answer", ""),
            "fodder_words": best_hit.get("fodder_words", []),
            "fodder_letters": best_hit.get("fodder_letters", ""),
            "evidence_type": best_hit.get("evidence_type", "unknown"),
            "solve_type": best_hit.get("solve_type", "unknown"),
            "confidence": best_hit.get("confidence", 0.0),
            "needed_letters": best_hit.get("needed_letters", ""),
            "excess_letters": best_hit.get("excess_letters", "")
        },
        "enumeration": enumeration,
        "candidates_sample": candidates[:8]
    }

    # Calculate remaining words (for explanation system development)
    clue_words = clue_text.split()
    accounted_words = set()

    # Add definition window words (if found)
    if definition_window:
        accounted_words.update(definition_window.split())

    # Add anagram fodder words
    if best_hit.get("fodder_words"):
        accounted_words.update(best_hit["fodder_words"])

    # Add common anagram indicators (basic set - could be expanded)
    anagram_indicators = {"confused", "mixed", "jumbled", "corrupted", "converts",
                          "exceptional", "comic", "arranged", "changed", "reformed",
                          "twisted", "mangled", "disturbed", "wrong", "badly", "odd"}

    for word in clue_words:
        clean_word = word.strip('.,!?:;()').lower()
        if clean_word in anagram_indicators:
            accounted_words.add(clean_word)

    # Calculate remaining words
    remaining_words = []
    for word in clue_words:
        clean_word = word.strip('.,!?:;()')
        if clean_word.lower() not in [w.lower() for w in accounted_words]:
            remaining_words.append(clean_word)

    analysis["remaining_words"] = remaining_words
    analysis["accounted_words"] = list(accounted_words)

    return analysis


def run_pipeline_probe(
        max_clues: int = None,
        wordplay_type: str = None,
) -> List[Dict[str, Any]]:
    if max_clues is None:
        max_clues = MAX_CLUES
    if wordplay_type is None:
        wordplay_type = WORDPLAY_TYPE
    wp_filter = wordplay_type.lower()

    # ---- START PERSISTENCE RUN ----
    run_id = None
    if PERSISTENCE_AVAILABLE and ENABLE_PERSISTENCE:
        run_id = start_run()

    conn = connect_db()  # reference tables (synonyms, wordplay, indicators)
    graph = load_graph(conn)
    clues_conn = connect_clues_db()  # clue selection
    cur = clues_conn.cursor()

    # -----------------------------
    # SOURCE SELECTION
    # -----------------------------
    if SINGLE_CLUE_MATCH:
        cur.execute(
            """
            SELECT id, clue_text, enumeration, answer, wordplay_type, source, puzzle_number
            FROM clues
            """
        )
        rows = [
            r for r in cur.fetchall()
            if _match_single_clue(SINGLE_CLUE_MATCH, r[1])
        ]
    else:
        # Build dynamic WHERE clause from filters
        conditions = []
        params = []

        if wp_filter != "all":
            conditions.append("LOWER(wordplay_type) = ?")
            params.append(wp_filter)
        if SOURCE:
            conditions.append("LOWER(source) = ?")
            params.append(SOURCE.lower())
        if PUZZLE_NUMBER:
            conditions.append("puzzle_number = ?")
            params.append(PUZZLE_NUMBER)
        if EXCLUDE_SOLVED:
            conditions.append("(has_solution IS NULL OR has_solution = 0)")

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(max_clues)

        cur.execute(
            f"""
            SELECT id, clue_text, enumeration, answer, wordplay_type, source, puzzle_number
            FROM clues
            {where_clause}
            ORDER BY RANDOM()
            LIMIT ?
            """,
            params,
        )
        rows = cur.fetchall()

    results: List[Dict[str, Any]] = []

    # NEW: Forwarded anagram cases collection
    forwarded_anagram_cases = []

    # NEW: Successful anagram cases collection
    successful_anagram_cases = []

    overall = {
        "clues": 0,
        "clues_with_def_match": 0,
        "clues_with_anagram": 0,
        "clues_with_lurker": 0,
        "clues_with_dd": 0,
        "gate_failed": 0,  # NEW: Clues filtered by definition gate
        # NEW: Forwarded analysis stats
        "forwarded_anagrams": 0,
        "forwarded_no_indicators": 0,
        "forwarded_evidence_failed": 0,
        "forwarded_system_error": 0,
        # NEW: Successful analysis stats
        "successful_anagrams": 0,
        "successful_exact": 0,
        "successful_partial": 0,
        "successful_deletion": 0,
    }

    # ---- SAVE INPUT STAGE ----
    input_records = []
    for row in rows:
        clue_id, clue, enum, answer_raw, wp_type, source, puzzle_number = row
        input_records.append({
            'id': clue_id,
            'clue_text': clue,
            'answer': answer_raw,
            'enumeration': enum,
            'wordplay_type': wp_type,
            'source': source,
            'puzzle_number': puzzle_number
        })

    if run_id is not None:
        save_stage('input', run_id, input_records)

    # ---- STAGE RECORDS ----
    dd_stage_records = []
    definition_stage_records = []
    definition_failed_records = []
    lurker_stage_records = []

    for row in rows:
        clue_id, clue, enum, answer_raw, wp_type, source, puzzle_number = row
        answer = norm_letters(answer_raw)
        total_len = parse_enum(enum)

        record: Dict[str, Any] = {
            "id": clue_id,
            "clue": clue,
            "enumeration": enum,
            "answer": answer,
            "answer_raw": answer_raw,
            "wordplay_type": wp_type,
        }

        # ---- STAGE 1: Double Definition (checked FIRST) ----
        dd_hits = generate_dd_hypotheses(
            clue_text=clue,
            graph=graph,
            answer=answer_raw,
        )

        # Replicate master_solver behaviour: enforce enumeration on DD hits
        dd_hits = [
            h for h in dd_hits
            if len(norm_letters(h["answer"])) == total_len
        ]

        # Check if DD found the correct answer
        dd_answer_present = answer in {norm_letters(h["answer"]) for h in dd_hits}

        # Save DD stage record
        dd_stage_records.append({
            'id': clue_id,
            'clue_text': clue,
            'answer': answer_raw,
            'double_definition': dd_hits
        })

        # If DD found correct answer, count it and skip to next clue
        if dd_answer_present:
            overall["clues"] += 1
            overall["clues_with_dd"] += 1
            continue

        # ---- STAGE 2: Definition ----
        def_result = definition_candidates(
            clue_text=clue,
            enumeration=enum,
            graph=graph,
        )

        raw_windows = [
            _clean_window(w)
            for w in def_result.get("definition_windows", [])
            if w and _clean_window(w)
        ]

        raw_candidates = def_result.get("candidates", []) or []
        flat_candidates = _length_filter(raw_candidates, enum)

        # NEW: Use known answer as single candidate if enabled
        if USE_KNOWN_ANSWER:
            flat_candidates = [answer_raw]
            definition_answer_present = True
        # ---- WINDOW → CANDIDATES (INVERTED SUPPORT) ----
        window_support: Dict[str, List[str]] = defaultdict(list)
        # All windows, including those with zero candidates
        window_candidates_by_window: Dict[str, List[str]] = {w: [] for w in raw_windows}

        for window in raw_windows:
            window_key = clean_key(window)

            keys = [window_key]
            for art in ("a ", "an ", "the "):
                keys.append(clean_key(art + window_key))

            for key in keys:
                if key not in graph:
                    continue
                for cand in graph[key]:
                    if USE_KNOWN_ANSWER:
                        match = norm_letters(cand) == answer
                    else:
                        match = cand in flat_candidates
                    if match:
                        window_support[window].append(cand)
                        window_candidates_by_window[window].append(cand)

        for w in window_support:
            window_support[w] = sorted(set(window_support[w]))

        for w in window_candidates_by_window:
            window_candidates_by_window[w] = sorted(set(window_candidates_by_window[w]))

        windows_with_hits = {w: c for w, c in window_support.items() if c}

        # ---- Save Definition Stage Record ----
        if not USE_KNOWN_ANSWER:
            definition_answer_present = (
                    answer in {norm_letters(c) for c in flat_candidates}
            )
        # else: already set to True above

        # Build candidate → windows mapping (inverse of window_support)
        candidate_to_windows = {}
        for window, cands in windows_with_hits.items():
            for cand in cands:
                if cand not in candidate_to_windows:
                    candidate_to_windows[cand] = []
                if window not in candidate_to_windows[cand]:
                    candidate_to_windows[cand].append(window)

        definition_stage_records.append({
            'id': clue_id,
            'clue_text': clue,
            'answer': answer_raw,
            'definition_candidates': flat_candidates,
            'answer_in_candidates': definition_answer_present,
            'support': candidate_to_windows
        })

        # ---- DEFINITION GATE: Skip if answer not in candidates ----
        # Cannot solve if correct answer isn't even a candidate
        # Skip gate when using known answer (always present)
        if not definition_answer_present and not USE_KNOWN_ANSWER:
            # Save to definition_failed for debugging
            definition_failed_records.append({
                'id': clue_id,
                'clue_text': clue,
                'answer': answer_raw,
                'definition_candidates': flat_candidates
            })
            overall["clues"] += 1
            overall["gate_failed"] += 1
            continue

        # ---- STAGE 3: Anagrams ----
        anag_hits = generate_anagram_hypotheses(
            clue_text=clue,
            enumeration=enum,  # Pass raw enumeration - anagram_stage now normalizes it
            candidates=flat_candidates,
        )

        # NEW: Forwarded anagram analysis
        if not anag_hits and ANALYZE_FORWARDED_ANAGRAMS and len(
                forwarded_anagram_cases) < MAX_FORWARDED_SAMPLES:
            # This is a forwarded anagram case - analyze why
            analysis = _analyze_forwarded_anagram(clue, answer_raw, flat_candidates, enum)
            forwarded_anagram_cases.append(analysis)

            # Update forwarded stats
            overall["forwarded_anagrams"] += 1
            if analysis["failure_reason"] == "no_anagram_indicators":
                overall["forwarded_no_indicators"] += 1
            elif analysis["failure_reason"] == "evidence_system_failed_to_find":
                overall["forwarded_evidence_failed"] += 1
            elif analysis["failure_reason"] in ["evidence_system_error",
                                                "evidence_system_not_available"]:
                overall["forwarded_system_error"] += 1

        # NEW: Successful anagram analysis for explanation system development
        if anag_hits and ANALYZE_SUCCESSFUL_ANAGRAMS and len(
                successful_anagram_cases) < MAX_SUCCESSFUL_SAMPLES:
            # This is a successful anagram case - analyze for explanation system
            analysis = _analyze_successful_anagram(
                clue_text=clue,
                answer=answer_raw,
                candidates=flat_candidates,
                anagram_hits=anag_hits,
                window_support=windows_with_hits,
                enumeration=enum
            )
            successful_anagram_cases.append(analysis)

            # Update successful stats
            overall["successful_anagrams"] += 1
            if analysis["anagram_evidence"]["evidence_type"] == "exact":
                overall["successful_exact"] += 1
            elif analysis["anagram_evidence"]["evidence_type"] == "partial":
                overall["successful_partial"] += 1
            elif analysis["anagram_evidence"]["evidence_type"] == "deletion":
                overall["successful_deletion"] += 1

        # ---- STAGE 4: Lurkers ----
        lurk_hits = generate_lurker_hypotheses(
            clue_text=clue,
            enumeration=total_len,
            candidates=flat_candidates,
        )

        # Save lurker stage record
        lurker_stage_records.append({
            'id': clue_id,
            'clue_text': clue,
            'answer': answer_raw,
            'lurkers': lurk_hits
        })

        # ---- HIT CLASSIFICATION (REPORTING ONLY) ----
        # Note: DD clues already exited the pipeline earlier
        hit_types = []
        if windows_with_hits:
            hit_types.append("definition")
        if anag_hits:
            hit_types.append("anagram")
        if lurk_hits:
            hit_types.append("lurker")

        hit_any = bool(hit_types)

        overall["clues"] += 1
        if definition_answer_present:
            overall["clues_with_def_match"] += 1
        if anag_hits:
            overall["clues_with_anagram"] += 1
        if lurk_hits:
            overall["clues_with_lurker"] += 1

        # Reporting filter: show only clues where answer is NOT in definition candidates
        # (unless SINGLE_CLUE_MATCH is explicitly set)
        if ONLY_MISSING_DEFINITION and not SINGLE_CLUE_MATCH:
            if definition_answer_present:
                continue

        record["summary"] = {
            "definition_windows_with_hits": len(windows_with_hits),
            "definition_candidates": len(set(flat_candidates)),
            "anagram_hits": len(anag_hits),
            "lurker_hits": len(lurk_hits),
            "hit_any": hit_any,
            "hit_types": hit_types,
            "answer_in_definition_candidates": definition_answer_present,
        }

        record["window_support"] = windows_with_hits
        record["window_candidates_by_window"] = window_candidates_by_window
        record["definition_candidates"] = flat_candidates
        record["definition_answer_present"] = definition_answer_present
        record["anagrams"] = anag_hits
        record["lurkers"] = lurk_hits

        results.append(record)

    conn.close()

    # ---- SAVE STAGES ----
    if run_id is not None:
        save_stage('dd', run_id, dd_stage_records)
        save_stage('definition', run_id, definition_stage_records)
        save_stage('definition_failed', run_id, definition_failed_records)
        save_stage('anagram', run_id, results)
        save_stage('lurker', run_id, lurker_stage_records)

    # Add forwarded analysis to overall stats
    overall["forwarded_anagram_cases"] = forwarded_anagram_cases

    # Add successful analysis to overall stats
    overall["successful_anagram_cases"] = successful_anagram_cases

    return results, overall


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    data, overall = run_pipeline_probe()

    print("\n=== PIPELINE SIMULATOR ===")
    print("POPULATION SUMMARY:")
    total_processed = overall['clues'] + overall['clues_with_dd']
    print(f"  clues processed           : {total_processed}")
    print(f"  clues w/ DD hit           : {overall['clues_with_dd']} (exit pipeline)")
    print(f"  gate failed (no def match): {overall['gate_failed']}")
    print(f"  clues w/ def answer match : {overall['clues_with_def_match']}")
    print(f"  clues w/ anagram hit      : {overall['clues_with_anagram']}")
    print(f"  clues w/ lurker hit       : {overall['clues_with_lurker']}")

    # NEW: Forwarded anagram analysis section
    if ANALYZE_FORWARDED_ANAGRAMS and overall.get("forwarded_anagram_cases"):
        print(f"\n=== FORWARDED ANAGRAM COHORT ANALYSIS ===")
        print(f"  total forwarded           : {overall['forwarded_anagrams']}")
        print(f"  no indicators detected    : {overall['forwarded_no_indicators']}")
        print(f"  evidence system failed    : {overall['forwarded_evidence_failed']}")
        print(f"  system errors             : {overall['forwarded_system_error']}")
        print(
            f"\nSample forwarded cases (showing first {len(overall['forwarded_anagram_cases'])}):")
        print("-" * 80)

        for i, case in enumerate(overall["forwarded_anagram_cases"], 1):
            print(f"[{i}] CLUE: {case['clue']}")
            print(f"    ANSWER: {case['answer']}")
            print(
                f"    CANDIDATES: {', '.join(case['candidates_sample'])}{'...' if len(case['candidates_sample']) == 8 else ''}")
            print(f"    ANSWER IN CANDIDATES: {case['answer_in_candidates']}")
            print(
                f"    INDICATORS DETECTED: {', '.join(case['indicators_detected']) if case['indicators_detected'] else 'none'}")
            print(f"    EVIDENCE SYSTEM: {case['evidence_system_result']}")
            print(f"    FAILURE REASON: {case['failure_reason']}")
            print("-" * 80)

    # NEW: Successful anagram analysis section
    if ANALYZE_SUCCESSFUL_ANAGRAMS and overall.get("successful_anagram_cases"):
        print(f"\n=== SUCCESSFUL ANAGRAM COHORT ANALYSIS ===")
        print(f"  total successful          : {overall['successful_anagrams']}")
        print(f"  exact matches             : {overall['successful_exact']}")
        print(f"  partial matches           : {overall['successful_partial']}")
        print(f"  deletion matches          : {overall['successful_deletion']}")
        print(
            f"\nSample successful cases (showing first {len(overall['successful_anagram_cases'])}):")
        print("-" * 80)

        for i, case in enumerate(overall["successful_anagram_cases"], 1):
            print(f"[{i}] CLUE: {case['clue']}")
            print(f"    ANSWER: {case['answer']}")
            print(f"    DEFINITION: {case['definition_window']}")
            print(f"    ANAGRAM EVIDENCE:")
            print(f"      Type: {case['anagram_evidence']['evidence_type']}")
            print(f"      Solve Type: {case['anagram_evidence']['solve_type']}")
            print(f"      Fodder: {' + '.join(case['anagram_evidence']['fodder_words'])}")
            print(
                f"      Confidence: {case['anagram_evidence']['confidence'] if isinstance(case['anagram_evidence']['confidence'], (int, float)) else 'N/A'}")
            if case['anagram_evidence']['needed_letters']:
                print(
                    f"      Needed letters: {case['anagram_evidence']['needed_letters']}")
            if case['anagram_evidence']['excess_letters']:
                print(
                    f"      Excess letters: {case['anagram_evidence']['excess_letters']}")
            print(
                f"    REMAINING WORDS: {', '.join([w for w in case['remaining_words'] if not w.isdigit()]) if case['remaining_words'] else 'none'}")
            print(f"    ACCOUNTED FOR: {', '.join(case['accounted_words'])}")
            print("-" * 80)

    print()
    for i, r in enumerate(data[:MAX_DISPLAY], 1):
        print(f"[{i}] CLUE: {r['clue']}")
        print(f"    TYPE: {r['wordplay_type']}")
        print(f"    ENUM: {r['enumeration']}")
        print(f"    ANSWER: {r['answer_raw']}")
        print(f"    SUMMARY: {r['summary']}")
        print(
            f"    HIT: {r['summary']['hit_any']} "
            f"({', '.join(r['summary']['hit_types']) or 'none'}) | "
            f"ANSWER IN DEF CANDIDATES: {r['summary']['answer_in_definition_candidates']}"
        )

        # ---- WORDPLAY HIT DETAILS (REPORTING ONLY) ----
        if r["anagrams"]:
            print("    ANAGRAM HITS:")
            for h in r["anagrams"]:
                print(f"      {h}")

        if r["lurkers"]:
            print("    LURKER HITS:")
            for h in r["lurkers"]:
                print(f"      {h}")

        print(f"    WINDOW → CANDIDATES: {r['window_support']}")

        print("    ALL WINDOWS → CANDIDATES:")
        for w, cands in sorted(
                r["window_candidates_by_window"].items(),
                key=lambda x: (len(x[0].split()), x[0].lower())
        ):
            if cands:
                print(f"      {w} → {', '.join(cands)}")
        print("-" * 60)