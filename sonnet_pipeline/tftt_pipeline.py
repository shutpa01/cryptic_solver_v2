"""TFTT Pipeline — Times puzzle solver using human explanations + Haiku parsing.

Workflow:
  1. Scrape TFTT blog for human explanation of each clue
  2. Parse explanation with Haiku into structured pieces ($0.001/clue)
  3. Score each parse with our confidence system (start at 100, deduct for problems)
  4. Only call Sonnet for clues scoring below threshold (expensive)
  5. Store all results in clues_master.db

Usage:
    python -m sonnet_pipeline.tftt_pipeline 29494           # run specific puzzle
    python -m sonnet_pipeline.tftt_pipeline 29494 --dry-run # parse + score, no DB writes
    python -m sonnet_pipeline.tftt_pipeline 29494 --no-fallback  # skip Sonnet fallback
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

from dotenv import load_dotenv
load_dotenv()

import anthropic

# -- Paths --
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLUES_DB = os.path.join(BASE_DIR, "data", "clues_master.db")

# -- Models --
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

# -- Scoring threshold: below this, fall back to Sonnet --
FALLBACK_THRESHOLD = 70

# -- Haiku parsing prompt --
PARSE_PROMPT = """You are a cryptic crossword explanation parser. Given a clue, its answer, and a human-written explanation of the wordplay, extract the structured pieces.

Output ONLY valid JSON with:
- "definition": the exact substring of the clue that defines the answer
- "wordplay_type": one of: charade, container, anagram, deletion, hidden, reversal, homophone, double_definition, cryptic_definition, acrostic, substitution, spoonerism
- "pieces": array of objects, each with:
  - "clue_word": the word(s) from the clue this piece comes from
  - "letters": the uppercase letters this piece contributes to the answer
  - "mechanism": one of: synonym, abbreviation, literal, anagram_fodder, first_letter, last_letter, reversal, hidden, deletion, alternate_letters, core_letters, sound_of
- "_reasoning": one-line summary

Rules:
- Each piece must map to the SMALLEST unit: individual words, not lumped phrases.
- Indicator words (anagram indicators, reversal indicators, etc.) are NOT pieces — they signal operations but contribute no letters.
- Link words (in, for, with, etc.) are NOT pieces.
- The definition words are NOT pieces.
- pieces letters must concatenate (or combine via the wordplay_type operation) to spell the answer.
- For anagrams: pieces are the raw fodder letters BEFORE rearrangement.
- For containers: show outer and inner pieces separately.
- For hidden words: one piece with the spanning clue words.
- For reversals: use mechanism "reversal".
- For deletions: the "letters" field must contain ONLY the letters that REMAIN after deletion (the letters that appear in the answer), NOT the original word. Also include "source" (the word before deletion) and "deleted" (the removed letters). E.g. if FARCE loses R, the piece is {"clue_word": "travesty", "letters": "FACE", "mechanism": "deletion", "source": "FARCE", "deleted": "R"}.
- Blog notation: parenthesised lowercase letters like (t)RASH or CRE(a)TION mean those letters are REMOVED. E.g. (t)RASH = {"clue_word": "rubbish", "letters": "RASH", "mechanism": "deletion", "source": "TRASH", "deleted": "T"}. CRE(a)TION = {"clue_word": "work", "letters": "CRETION", "mechanism": "deletion", "source": "CREATION", "deleted": "A"}.
- For double definitions (DD): use wordplay_type "double_definition" with no pieces needed.
- For homophones: use mechanism "sound_of" and the letters field should contain the letters that the word SOUNDS LIKE, which spell part or all of the answer.
- The concatenation of all pieces' "letters" fields MUST exactly spell the answer. Verify this before responding.

Return ONLY valid JSON, no other text."""


def clean(s):
    """Strip non-alpha characters and uppercase."""
    return re.sub(r'[^A-Za-z]', '', s).upper()


# ============================================================
# Step 1: Fetch TFTT explanations
# ============================================================

def fetch_tftt(puzzle_number):
    """Fetch and parse TFTT page for a puzzle. Returns list of clue dicts.

    Tries multiple URL patterns since TFTT changed their slug format:
    - Old: times-cryptic-XXXXX
    - New: XXXXX-title-slug (discovered via WordPress search API)
    """
    import requests

    sys.path.insert(0, os.path.join(BASE_DIR, "scraper", "timesforthetimes"))
    from timesforthetimes_scraper import parse_page

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36'
    }

    # Try old URL pattern first (fast)
    old_url = "https://timesforthetimes.co.uk/times-cryptic-%d" % puzzle_number
    try:
        resp = requests.get(old_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            clues = parse_page(resp.text, puzzle_number)
            if clues:
                return clues
    except Exception:
        pass

    # Try WordPress search API to find the correct slug
    search_url = "https://timesforthetimes.co.uk/wp-json/wp/v2/posts"
    try:
        resp = requests.get(search_url, headers=headers, timeout=15, params={
            "search": str(puzzle_number),
            "per_page": 5,
            "categories": "11,21",  # Daily Cryptic, Weekend Cryptic
        })
        if resp.status_code == 200:
            posts = resp.json()
            for post in posts:
                slug = post.get("slug", "")
                if str(puzzle_number) in slug:
                    post_url = post.get("link") or "https://timesforthetimes.co.uk/%s" % slug
                    resp2 = requests.get(post_url, headers=headers, timeout=15)
                    if resp2.status_code == 200:
                        clues = parse_page(resp2.text, puzzle_number)
                        if clues:
                            return clues
    except Exception as e:
        print("    TFTT search error: %s" % e)

    return None


# ============================================================
# Step 2: Parse explanation with Haiku
# ============================================================

def parse_with_haiku(client, clue_text, answer, explanation):
    """Send one clue to Haiku for parsing. Returns (parsed_dict, usage) or (None, None)."""
    user_msg = "Clue: %s\nAnswer: %s\nHuman explanation: %s" % (clue_text, answer, explanation)

    try:
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2000,
            temperature=0,
            system=PARSE_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3].strip()

        parsed = json.loads(text)
        return parsed, response.usage
    except json.JSONDecodeError:
        return None, None
    except Exception as e:
        print("    Haiku error: %s" % e)
        return None, None


# ============================================================
# Step 3: Score the parse
# ============================================================

def score_parse(parsed, answer, ref_db):
    """Score a Haiku-parsed explanation using our UX-based confidence system.

    Start at 100, deduct for problems a user would spot.
    Returns (score, list of (reason, delta) tuples).
    """
    if not parsed:
        return 0, [("no parse", -100)]

    # Double definitions and cryptic definitions have no pieces — that's correct
    wtype = parsed.get("wordplay_type", "")
    if wtype in ("double_definition", "cryptic_definition"):
        if parsed.get("definition"):
            return 90, []  # DD/CD with definition = good
        return 70, [("DD/CD without definition", -30)]

    if not parsed.get("pieces"):
        return 0, [("no pieces", -100)]

    pieces = parsed["pieces"]
    answer_clean = clean(answer)
    reasons = []
    score = 100

    # Check pieces concatenate to answer
    piece_letters = "".join(clean(p.get("letters", "")) for p in pieces)
    if piece_letters != answer_clean:
        if sorted(piece_letters) == sorted(answer_clean):
            pass  # anagram — letters present but rearranged, OK
        else:
            score -= 30
            reasons.append(("pieces don't spell answer: %s != %s" % (piece_letters, answer_clean), -30))

    # Per-piece checks
    for p in pieces:
        mech = p.get("mechanism", "")
        letters = clean(p.get("letters", ""))
        clue_word = (p.get("clue_word") or "").strip()

        if not letters:
            continue

        # Synonym: check if the value is a real word
        if mech == "synonym":
            if letters and not ref_db.is_real_word(letters):
                score -= 60
                reasons.append(("nonsense synonym: %s=%s" % (clue_word, letters), -60))
            else:
                # Check if confirmed in DB
                syns = ref_db.get_synonyms(clue_word.lower())
                if letters not in syns:
                    score -= 20
                    reasons.append(("unconfirmed synonym: %s=%s" % (clue_word, letters), -20))

        # Abbreviation: check DB
        elif mech == "abbreviation":
            abbrs = ref_db.get_abbreviations(clue_word.lower())
            if letters not in [a.upper() for a in abbrs]:
                # Short abbreviations (1-2 chars) are common and hard to verify
                if len(letters) <= 2:
                    score -= 5
                    reasons.append(("unconfirmed abbreviation: %s=%s" % (clue_word, letters), -5))
                else:
                    score -= 10
                    reasons.append(("unconfirmed long abbreviation: %s=%s" % (clue_word, letters), -10))

        # Deletion: fragment is mechanically derived, no nonsense check needed.
        # The letters are what remain after removing letters from a source word —
        # CROS (from CROSS-S) or CRETION (from CREATION-A) are valid fragments.
        elif mech == "deletion":
            pass  # mechanically verified by yields check

        # Homophone: check DB
        elif mech == "sound_of":
            homos = ref_db.get_homophones(clue_word.lower())
            if letters not in homos:
                score -= 15
                reasons.append(("unconfirmed homophone: %s=%s" % (clue_word, letters), -15))

        # Literal, anagram_fodder, first_letter, last_letter, hidden, reversal,
        # alternate_letters, core_letters — mechanically verifiable, no penalty

    # Definition check
    definition = parsed.get("definition", "")
    if definition:
        clue_lower = clue_text_for_def_check if 'clue_text_for_def_check' in dir() else ""
        # Can't check position without clue text — skip for now

    return max(0, min(100, score)), reasons


# ============================================================
# Step 4: Sonnet fallback
# ============================================================

def solve_with_sonnet(client, clue_text, answer, enrichment, enricher, homo_engine, ref_db):
    """Fall back to full Sonnet pipeline for a single clue."""
    from .solver import solve_clue
    example_messages = []  # not used with thinking mode
    result = solve_clue(
        clue_text, answer, enrichment, enricher, homo_engine,
        example_messages, ref_db=ref_db
    )
    return result


# ============================================================
# Step 5: Store results
# ============================================================

def store_tftt_result(conn, clue_id, parsed, score, definition_from_tftt, raw_explanation=""):
    """Store a TFTT-parsed result into clues_master.db.

    Updates: definition, wordplay_type, ai_explanation, explanation, has_solution, reviewed.
    Also writes to structured_explanations.
    """
    if not parsed or not parsed.get("pieces"):
        return

    pieces = parsed["pieces"]
    wordplay_type = parsed.get("wordplay_type", "unknown")
    definition = parsed.get("definition") or definition_from_tftt or ""

    # Build human-readable explanation from pieces
    parts = []
    for p in pieces:
        clue_word = p.get("clue_word", "?")
        letters = p.get("letters", "?")
        mechanism = p.get("mechanism", "?")
        if mechanism in ("literal", "anagram_fodder"):
            parts.append("%s(%s)" % (letters, clue_word))
        elif mechanism == "synonym":
            parts.append("%s(synonym of \"%s\")" % (letters, clue_word))
        elif mechanism == "abbreviation":
            parts.append("%s(abbr. \"%s\")" % (letters, clue_word))
        elif mechanism == "first_letter":
            parts.append("%s(first letter of \"%s\")" % (letters, clue_word))
        elif mechanism == "last_letter":
            parts.append("%s(last letter of \"%s\")" % (letters, clue_word))
        elif mechanism == "reversal":
            parts.append("%s(\"%s\" reversed)" % (letters, clue_word))
        elif mechanism == "hidden":
            parts.append("%s(hidden in \"%s\")" % (letters, clue_word))
        elif mechanism == "deletion":
            source_word = p.get("source", "")
            deleted = p.get("deleted", "")
            if source_word and deleted:
                parts.append("%s(%s minus %s, \"%s\")" % (letters, source_word, deleted, clue_word))
            elif source_word:
                parts.append("%s(from %s, \"%s\")" % (letters, source_word, clue_word))
            else:
                parts.append("%s(deletion from \"%s\")" % (letters, clue_word))
        elif mechanism == "alternate_letters":
            parts.append("%s(alternate letters of \"%s\")" % (letters, clue_word))
        elif mechanism == "sound_of":
            parts.append("%s(sounds like \"%s\")" % (letters, clue_word))
        else:
            parts.append("%s(%s: \"%s\")" % (letters, mechanism, clue_word))

    explanation = " + ".join(parts)
    if wordplay_type not in ("charade",):
        explanation += " [%s]" % wordplay_type
    explanation += "; definition: \"%s\"" % definition

    # Update clues table
    conn.execute("""
        UPDATE clues SET
            definition = COALESCE(NULLIF(definition, ''), ?),
            wordplay_type = ?,
            ai_explanation = ?,
            explanation = COALESCE(NULLIF(explanation, ''), ?),
            has_solution = 1,
            reviewed = ?
        WHERE id = ?
    """, (
        definition,
        wordplay_type,
        explanation,
        raw_explanation,
        1 if score >= 80 else 0,
        clue_id,
    ))

    # Write structured_explanations
    confidence = score / 100.0
    components = json.dumps({
        "ai_pieces": pieces,
        "wordplay_type": wordplay_type,
        "source": "tftt+haiku",
    })

    # Check if row exists
    existing = conn.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()

    puzzle_row = conn.execute(
        "SELECT puzzle_number, clue_number FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    pnum = puzzle_row[0] if puzzle_row else ""
    cnum = puzzle_row[1] if puzzle_row else ""

    if existing:
        conn.execute("""
            UPDATE structured_explanations SET
                definition_text = ?, wordplay_types = ?, components = ?,
                model_version = ?, confidence = ?, source = ?,
                puzzle_number = ?, clue_number = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE clue_id = ?
        """, (
            definition, json.dumps([wordplay_type]), components,
            "tftt+haiku", confidence, "times",
            pnum, cnum, clue_id,
        ))
    else:
        conn.execute("""
            INSERT INTO structured_explanations
            (clue_id, definition_text, wordplay_types, components,
             model_version, confidence, source, puzzle_number, clue_number,
             created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?,
                    CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (
            clue_id, definition, json.dumps([wordplay_type]), components,
            "tftt+haiku", confidence, "times",
            pnum, cnum,
        ))

    conn.commit()


# ============================================================
# Main pipeline
# ============================================================

def run_tftt_pipeline(puzzle_number, write_db=False, no_fallback=False):
    """Run the full TFTT pipeline for a Times puzzle."""

    print("=" * 70)
    print("TFTT Pipeline — Times puzzle %d" % puzzle_number)
    print("=" * 70)

    # Step 1: Fetch TFTT
    print("\n--- Step 1: Fetching TFTT blog ---")
    tftt_clues = fetch_tftt(puzzle_number)

    if not tftt_clues:
        print("  No TFTT page found for puzzle %d" % puzzle_number)
        return

    print("  Fetched %d clues from TFTT" % len(tftt_clues))

    # Load RefDB for scoring
    print("\n--- Loading RefDB ---")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    # Connect to clues_master to match TFTT clues to DB rows
    conn = sqlite3.connect(CLUES_DB, timeout=30)

    # Build lookup: answer -> (clue_id, clue_text) from clues_master
    db_rows = conn.execute("""
        SELECT id, clue_number, clue_text, answer
        FROM clues WHERE source = 'times' AND puzzle_number = ?
    """, (str(puzzle_number),)).fetchall()

    if not db_rows:
        print("  Puzzle %d not in clues_master.db — run the daily scraper first" % puzzle_number)
        conn.close()
        return

    # Match by answer
    db_by_answer = {}
    for cid, cnum, ctext, answer in db_rows:
        key = clean(answer)
        db_by_answer[key] = (cid, cnum, ctext, answer)

    # Step 2+3: Parse with Haiku and score
    print("\n--- Step 2+3: Haiku parsing + scoring ---")
    client = anthropic.Anthropic()
    haiku_cost = 0
    sonnet_cost = 0

    high_count = 0
    medium_count = 0
    low_count = 0
    fallback_count = 0
    failed_count = 0

    for tc in tftt_clues:
        answer_key = clean(tc["answer"])
        db_match = db_by_answer.get(answer_key)

        if not db_match:
            print("  %s %s — no DB match" % (tc["clue_number"], tc["answer"]))
            failed_count += 1
            continue

        clue_id, clue_num, clue_text, db_answer = db_match
        explanation = tc.get("explanation", "")
        tftt_definition = tc.get("definition", "")

        if not explanation:
            print("  %s %s — no TFTT explanation" % (clue_num, db_answer))
            failed_count += 1
            continue

        # Parse with Haiku
        parsed, usage = parse_with_haiku(client, clue_text, db_answer, explanation)

        if usage:
            haiku_cost += (usage.input_tokens * 0.80 + usage.output_tokens * 4.0) / 1_000_000

        if not parsed:
            print("  %s %s — Haiku parse failed" % (clue_num, db_answer))
            low_count += 1
            continue

        # Validate pieces produce the answer
        piece_letters = "".join(clean(p.get("letters", "")) for p in parsed.get("pieces", []))
        answer_clean = clean(db_answer)

        yields_ok = (piece_letters == answer_clean or
                     sorted(piece_letters) == sorted(answer_clean))

        # Score
        score, reasons = score_parse(parsed, db_answer, ref_db)

        # Determine band
        if score >= 80:
            band = "HIGH"
            high_count += 1
        elif score >= FALLBACK_THRESHOLD:
            band = "MEDIUM"
            medium_count += 1
        else:
            band = "LOW"
            low_count += 1

        reason_str = ", ".join("%s(%d)" % (r, d) for r, d in reasons) if reasons else "clean"

        print("  %s %-20s %3d %s  yields:%s  %s" % (
            clue_num, db_answer, score, band,
            "OK" if yields_ok else "FAIL",
            reason_str
        ))

        # Store if scoring well
        if write_db and score >= FALLBACK_THRESHOLD:
            store_tftt_result(conn, clue_id, parsed, score, tftt_definition)

        # Step 4: Sonnet fallback for low scores
        if score < FALLBACK_THRESHOLD and not no_fallback:
            print("    -> Sonnet fallback...", end=" ", flush=True)
            try:
                from .solver import solve_clue
                from clue_enricher import ClueEnricher
                from .homophone_engine import HomophoneEngine

                enricher = ClueEnricher()
                homo_engine = HomophoneEngine()
                enrichment = enricher.enrich(clue_text, db_answer)

                result = solve_clue(
                    clue_text, db_answer, enrichment, enricher, homo_engine,
                    [], ref_db=ref_db
                )

                if result and result.get("validation"):
                    s_score = result["validation"].get("score", 0)
                    s_conf = result["validation"].get("confidence", "?")
                    print("score=%d (%s)" % (s_score, s_conf))
                    fallback_count += 1

                    if write_db:
                        from .solver import store_result
                        store_result(conn, clue_id,
                                     result.get("ai_output"),
                                     result.get("assembly"),
                                     result["validation"],
                                     result.get("tier", "SONNET"))
                else:
                    print("no result")
            except Exception as e:
                print("error: %s" % e)

    conn.close()

    # Summary
    print("\n" + "=" * 70)
    print("Results: %d HIGH, %d MEDIUM, %d LOW, %d failed" % (
        high_count, medium_count, low_count, failed_count))
    print("Fallback Sonnet calls: %d" % fallback_count)
    print("Cost: Haiku $%.4f" % haiku_cost)
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="TFTT Pipeline for Times puzzles")
    parser.add_argument("puzzle", type=int, help="Puzzle number")
    parser.add_argument("--write-db", action="store_true", help="Write results to clues_master.db")
    parser.add_argument("--dry-run", action="store_true", help="Parse and score only, no DB writes or Sonnet calls")
    parser.add_argument("--no-fallback", action="store_true", help="Skip Sonnet fallback for low scores")
    args = parser.parse_args()

    run_tftt_pipeline(
        args.puzzle,
        write_db=args.write_db,
        no_fallback=args.dry_run or args.no_fallback,
    )


if __name__ == "__main__":
    main()
