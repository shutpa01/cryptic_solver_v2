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
SONNET_MODEL = "claude-sonnet-4-6"

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
    if not parsed:
        return

    wtype = parsed.get("wordplay_type", "")
    pieces = parsed.get("pieces") or []

    # Cryptic definitions and double definitions legitimately have no pieces
    if not pieces and wtype not in ("cryptic_definition", "double_definition"):
        return
    wordplay_type = parsed.get("wordplay_type", "unknown")
    definition = parsed.get("definition") or definition_from_tftt or ""

    # Run the assembler to determine how pieces fit together
    from .solver import assemble, clean
    piece_letters = [clean(p.get("letters", "")) for p in pieces if p.get("letters")]
    answer_row = conn.execute("SELECT answer FROM clues WHERE id = ?", (clue_id,)).fetchone()
    answer = answer_row[0] if answer_row else ""
    assembly = assemble(
        conn.execute("SELECT clue_text FROM clues WHERE id = ?", (clue_id,)).fetchone()[0],
        answer, piece_letters, ai_wtype=wordplay_type,
    )

    # Build explanation from assembly result (proper container/anagram/reversal descriptions)
    if assembly:
        from .report import _describe_assembly
        explanation = _describe_assembly(assembly, pieces, answer=answer)
        if not explanation:
            explanation = ""
        # Add definition
        if explanation and definition:
            explanation += "; definition: \"%s\"" % definition
        elif definition:
            explanation = "Definition: \"%s\"" % definition
    else:
        # Fallback: simple concatenation if assembler can't solve it
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
            elif mechanism == "hidden":
                parts.append("%s(hidden in \"%s\")" % (letters, clue_word))
            else:
                parts.append("%s(%s: \"%s\")" % (letters, mechanism, clue_word))
        explanation = " + ".join(parts)
        if wordplay_type not in ("charade",):
            explanation += " [%s]" % wordplay_type
        explanation += "; definition: \"%s\"" % definition

    # Run mechanical verifier BEFORE updating clues — gate has_solution on verifier score
    from .verify_explanation import ExplanationVerifier
    clue_row = conn.execute(
        "SELECT clue_text, answer FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    _verifier = getattr(store_tftt_result, '_verifier', None)
    if _verifier is None:
        _verifier = ExplanationVerifier()
        store_tftt_result._verifier = _verifier
    v_result = _verifier.verify(
        clue_row[0] if clue_row else "", clue_row[1] if clue_row else "",
        definition, wordplay_type, explanation,
    )
    confidence = v_result["score"] / 100.0 if v_result else score / 100.0
    v_score = v_result["score"] if v_result else 0

    # Update clues table — use verifier score to gate has_solution and reviewed
    conn.execute("""
        UPDATE clues SET
            definition = COALESCE(NULLIF(definition, ''), ?),
            wordplay_type = ?,
            ai_explanation = ?,
            explanation = COALESCE(NULLIF(explanation, ''), ?),
            has_solution = ?,
            reviewed = ?
        WHERE id = ?
    """, (
        definition,
        wordplay_type,
        explanation,
        raw_explanation,
        1 if v_score > 0 else 0,
        1 if v_score >= 70 else 0,
        clue_id,
    ))
    components = json.dumps({
        "ai_pieces": pieces,
        "assembly": assembly,
        "wordplay_type": wordplay_type,
        "source": "tftt+haiku",
    })

    # Check if row exists
    existing = conn.execute(
        "SELECT id FROM structured_explanations WHERE clue_id = ?", (clue_id,)
    ).fetchone()

    puzzle_row = conn.execute(
        "SELECT source, puzzle_number, clue_number FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    src = puzzle_row[0] if puzzle_row else ""
    pnum = puzzle_row[1] if puzzle_row else ""
    cnum = puzzle_row[2] if puzzle_row else ""

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

    # Queue enrichment gaps (unverifiable synonyms/abbreviations/definitions/
    # homophones, plus missing indicators extracted from the explanation's
    # quoted annotations).
    if v_result:
        try:
            import sqlite3 as _sq
            _ref = _sq.connect(
                os.path.join(BASE_DIR, "data", "cryptic_new.db"), timeout=10
            )
            clue_text = clue_row[0] if clue_row else ""
            answer_val = clue_row[1] if clue_row else ""

            # Collect candidate gaps: (type, word, letters) tuples
            _gaps_to_queue = []
            _indicator_types = set()

            for _gc in v_result.get("checks", []):
                _status = _gc.get("status", "")
                _detail = _gc.get("detail", "")
                _ctype = _gc.get("check", "")

                if _status == "unverifiable" and _ctype in (
                    "synonym", "abbreviation", "definition"
                ):
                    _gm = re.match(r"'(.+?)'\s*(?:=|->)\s*(\w+)", _detail)
                    if _gm:
                        _gaps_to_queue.append(
                            (_ctype, _gm.group(1).strip().lower(),
                             _gm.group(2).strip().upper())
                        )

                elif _ctype == "homophone" and "not in DB" in _detail:
                    _gm = re.match(r"'(.+?)' sounds like '(.+?)': not in DB", _detail)
                    if _gm:
                        _gaps_to_queue.append(
                            ("homophone", _gm.group(1).strip().lower(),
                             _gm.group(2).strip().upper())
                        )

                elif _ctype == "indicator" and _status == "wrong":
                    _im = re.search(r"no '([^']+)' indicator", _detail)
                    if _im:
                        _indicator_types.add(_im.group(1))

            # Extract candidate indicator words from the explanation's quoted
            # annotations. Skip hidden-spans (internal uppercase) and long
            # phrases. Queue each candidate for each needed indicator type.
            if _indicator_types and explanation:
                _quoted = re.findall(r'"([^"]+)"', explanation)
                for _qt in _indicator_types:
                    for _q in _quoted:
                        _qs = _q.strip()
                        if (_qs and len(_qs) <= 30
                                and not any(c.isupper() for c in _qs[1:])):
                            _gaps_to_queue.append(
                                ("indicator", _qs.lower(), _qt.upper())
                            )

            for _gt, _gw, _gl in _gaps_to_queue:
                _already = False
                if _gt == "synonym":
                    _already = _ref.execute(
                        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
                        (_gw, _gl),
                    ).fetchone() is not None
                elif _gt == "abbreviation":
                    _already = _ref.execute(
                        "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=?",
                        (_gw, _gl),
                    ).fetchone() is not None
                elif _gt == "definition":
                    _already = _ref.execute(
                        "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=?",
                        (_gw, _gl),
                    ).fetchone() is not None
                elif _gt == "homophone":
                    _already = _ref.execute(
                        "SELECT 1 FROM homophones WHERE "
                        "(LOWER(word)=? AND LOWER(homophone)=?) OR "
                        "(LOWER(word)=? AND LOWER(homophone)=?)",
                        (_gw, _gl.lower(), _gl.lower(), _gw),
                    ).fetchone() is not None
                elif _gt == "indicator":
                    _already = _ref.execute(
                        "SELECT 1 FROM indicators WHERE LOWER(word)=? AND LOWER(wordplay_type)=?",
                        (_gw, _gl.lower()),
                    ).fetchone() is not None

                if _already:
                    continue

                _rejected = conn.execute(
                    "SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
                    (_gt, _gw, _gl),
                ).fetchone()
                if _rejected:
                    continue

                _existing = conn.execute(
                    "SELECT 1 FROM pending_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
                    (_gt, _gw, _gl),
                ).fetchone()
                if _existing:
                    continue

                conn.execute(
                    "INSERT OR IGNORE INTO pending_enrichments (type, word, letters, answer, clue_text, source, puzzle_number) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (_gt, _gw, _gl, answer_val, clue_text, src, pnum),
                )

            _ref.close()
        except Exception:
            pass  # Don't let gap collection failure block the pipeline

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
        print("  No TFTT page found — falling through to Sonnet pipeline")
        import subprocess
        cmd = [sys.executable, "-m", "sonnet_pipeline.run",
               "--mode", "1", "--no-review",
               "--source", "times", str(puzzle_number)]
        if write_db:
            cmd.append("--write-db")
        result = subprocess.run(cmd, cwd=BASE_DIR)
        sys.exit(result.returncode)

    print("  Fetched %d clues from TFTT" % len(tftt_clues))

    # Always save raw blog explanations to the clues table
    conn_save = sqlite3.connect(CLUES_DB, timeout=30)
    blog_saved = 0
    for tc in tftt_clues:
        if not tc.get("explanation"):
            continue
        answer_clean = clean(tc["answer"])
        result = conn_save.execute(
            "UPDATE clues SET explanation = ? "
            "WHERE source = 'times' AND puzzle_number = ? "
            "AND UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ? "
            "AND (explanation IS NULL OR explanation = '')",
            (tc["explanation"], str(puzzle_number), answer_clean),
        )
        if result.rowcount > 0:
            blog_saved += 1
    conn_save.commit()
    conn_save.close()
    if blog_saved:
        print("  Saved %d blog explanations to DB" % blog_saved)

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
            store_tftt_result(conn, clue_id, parsed, score, tftt_definition, explanation)

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
