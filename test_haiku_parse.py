"""Test: Can Haiku parse human-written cryptic clue explanations into structured format?

Standalone test script — reads TFTT explanations and asks Haiku to produce
structured pieces in our format. No DB writes, no pipeline dependencies.

Usage: python -m sonnet_pipeline.test_haiku_parse
"""

import json
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

MODEL = "claude-haiku-4-5-20251001"

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
- Each piece must map to the SMALLEST unit: individual words, not lumped phrases
- Indicator words (anagram indicators, reversal indicators, etc.) are NOT pieces — they signal operations but contribute no letters
- Link words (in, for, with, etc.) are NOT pieces
- The definition words are NOT pieces
- pieces letters must concatenate (or combine via the wordplay_type operation) to spell the answer
- For anagrams: pieces are the raw fodder letters BEFORE rearrangement
- For containers: show outer and inner pieces separately
- For hidden words: one piece with the spanning clue words
- For reversals: use mechanism "reversal"
- For deletions: show the piece AFTER deletion with mechanism "deletion"

Return ONLY valid JSON, no other text."""

# TFTT puzzle 29494 — human explanations from timesforthetimes.co.uk
CLUES = [
    {
        "number": "1a",
        "clue": "Belittle bridge opponents with 'mug to lead' (2,4)",
        "answer": "DO DOWN",
        "explanation": "Bridge opponents are W and N. DODO (the mug) leads them. DO + DO + W + N.",
    },
    {
        "number": "4a",
        "clue": "Called again about faulty gadget (8)",
        "answer": "RETAGGED",
        "explanation": "RE (about) + anagram of GADGET = RETAGGED",
    },
    {
        "number": "10a",
        "clue": "Criminal lawyer and con man spoke at last in good nick (6,3)",
        "answer": "NEARLY NEW",
        "explanation": "Anagram of LAWYER + last letters of coN maN spokE = NEARLY NEW. Definition: in good nick",
    },
    {
        "number": "11a",
        "clue": "Outgoing type stripped twice in store (5)",
        "answer": "TROVE",
        "explanation": "EXTROVERT with outside letters removed twice: eXTROVERt -> XTROVER -> TROVE. Definition: store",
    },
    {
        "number": "12a",
        "clue": "Bring back one beer to accompany a big cigar (7)",
        "answer": "REGALIA",
        "explanation": "Reverse of I (one) LAGER + A = REGALIA. Definition: big cigar",
    },
    {
        "number": "13a",
        "clue": "Seek further use for components within grey vehicle (7)",
        "answer": "RECYCLE",
        "explanation": "Hidden in gREy + CYCLE (vehicle) = RE + CYCLE. Definition: Seek further use for",
    },
    {
        "number": "14a",
        "clue": "Spot sheep collectively shifting over to middle of field (5)",
        "answer": "FLECK",
        "explanation": "FLOCK (sheep collectively) changes O to E (middle of fiEld) = FLECK. Definition: Spot",
    },
    {
        "number": "15a",
        "clue": "Community has charging point finally switched on (8)",
        "answer": "TOWNSHIP",
        "explanation": "T (point finally) + OWNS (has) + HIP (switched on) = TOWNSHIP. Definition: Community",
    },
    {
        "number": "18a",
        "clue": "City hospital to decline work (3,5)",
        "answer": "SAN DIEGO",
        "explanation": "SAN (hospital/sanatorium) + DIE (decline) + GO (work) = SAN DIEGO. Definition: City",
    },
    {
        "number": "20a",
        "clue": "Writer's miserable doctor tried, having neglected case (5)",
        "answer": "DREAR",
        "explanation": "DR (doctor) + hEARd (tried, with case/outside letters removed) = DREAR. Definition: Writer's miserable",
    },
    {
        "number": "23a",
        "clue": "Drained theatre nurse maybe collars food supplier (7)",
        "answer": "CATERER",
        "explanation": "TE (theatre drained) inside CARER (nurse) = CA(TE)RER. Definition: food supplier",
    },
    {
        "number": "25a",
        "clue": "Revolutionary Green leader? That could be incendiary (7)",
        "answer": "WARHEAD",
        "explanation": "WAR (RAW reversed = revolutionary) + HEAD (leader) = WARHEAD. Definition: That could be incendiary",
    },
    {
        "number": "26a",
        "clue": "Catches slowcoach delaying son's appearance (5)",
        "answer": "NAILS",
        "explanation": "SNAIL (slowcoach) with S moved to end = NAILS. Definition: Catches",
    },
    {
        "number": "27a",
        "clue": "Fools lease land west of London (9)",
        "answer": "BERKSHIRE",
        "explanation": "BERKS (fools) + HIRE (lease) = BERKSHIRE. Definition: land west of London",
    },
    {
        "number": "28a",
        "clue": "Solid shelf delivered with fashionable red backing (8)",
        "answer": "CYLINDER",
        "explanation": "Homophone of SILL (shelf) = CYL + IN (fashionable) + DER (RED reversed) = CYLINDER. Definition: Solid",
    },
    {
        "number": "29a",
        "clue": "Act like court has sealed the end for MS Outlook (6)",
        "answer": "ASPECT",
        "explanation": "APE (act like) + CT (court sealed/shortened) + S (end of MS) = ASPECT. Definition: Outlook. Wait — recheck: A(S)PE + CT. S inside APE + CT.",
    },
    {
        "number": "1d",
        "clue": "Fine curtailed Trump's flakiness? (8)",
        "answer": "DANDRUFF",
        "explanation": "DANDY (fine) curtailed = DAND + RUFF (trump in cards) = DANDRUFF. Definition: flakiness",
    },
    {
        "number": "2d",
        "clue": "Trail broken leg, bearing pain (7)",
        "answer": "DRAGGLE",
        "explanation": "Anagram of LEG inside DRAG (pain) = DR(AGG)LE. Definition: Trail",
    },
    {
        "number": "3d",
        "clue": "Popular Liberal upset butcher in garden (4-5)",
        "answer": "WELL-LIKED",
        "explanation": "L (Liberal) reversed + KILL (butcher) inside WEED (garden) = WE(LLKI)ED. Definition: Popular",
    },
    {
        "number": "5d",
        "clue": "Actor and journalist on identical charges having pinched fuel (6,8)",
        "answer": "EDWARD WOODWARD",
        "explanation": "ED (journalist) + WARD (charge) + WOOD (fuel) + WARD (charge) = EDWARD WOODWARD. Definition: Actor",
    },
    {
        "number": "6d",
        "clue": "Casual job rejected by great old buffoon (5)",
        "answer": "ANTIC",
        "explanation": "GIGANTIC minus GIG (casual job) = ANTIC. Definition: old buffoon",
    },
    {
        "number": "7d",
        "clue": "Coaching failed to yield first-class dumplings (7)",
        "answer": "GNOCCHI",
        "explanation": "Anagram of COACHING minus A (first-class) = GNOCCHI. Definition: dumplings",
    },
    {
        "number": "8d",
        "clue": "Increase strength of hideaway, covering leak up (6)",
        "answer": "DEEPEN",
        "explanation": "DEN (hideaway) around PEE reversed (leak up) = DE(EPE)N. Wait — DEE(PEN). DEN around EP (PEE reversed minus E?). Actually: D(EEPE)N — DEN containing reversed PEEP? No. DEN around EEP (PEE reversed) = DEEPEN.",
    },
    {
        "number": "9d",
        "clue": "Enter bar in felt buckles? That's an embarrassment (6,8)",
        "answer": "ENFANT TERRIBLE",
        "explanation": "Anagram of ENTER BAR IN FELT = ENFANT TERRIBLE. Definition: an embarrassment",
    },
    {
        "number": "16d",
        "clue": "Complaint papers raised ulcer acquired by relative (9)",
        "answer": "SIDEROSIS",
        "explanation": "ID (papers) reversed = DI, then SORE reversed = EROS inside SIS (relative)? Actually: SI(DEROS)IS — SIS containing ID reversed (DI) + SORE reversed (EROS). S(IDEROS)IS.",
    },
    {
        "number": "17d",
        "clue": "One's inclined to bend the limits of radioisotope dating (8)",
        "answer": "GRADIENT",
        "explanation": "Anagram of RE (limits of RadioisotopE) + DATING = GRADIENT. Definition: One's inclined",
    },
    {
        "number": "19d",
        "clue": "Pile of workers want to go topless — Benny, for example (7)",
        "answer": "ANTHILL",
        "explanation": "WANT minus W (topless) = ANT + HILL (Benny Hill) = ANTHILL. Definition: Pile of workers",
    },
    {
        "number": "21d",
        "clue": "Loner's suitable moment to return, before crowds (7)",
        "answer": "EREMITE",
        "explanation": "EMIT (suitable moment? No — TIME reversed = EMIT) + ERE (before) = ERE + MITE. Wait: EREMITE = ERE (before) + MIT (TIME reversed minus E?) Actually: ERE + MITE (TIME reversed? No EMIT). Hmm. ERE (before) + MITE (small amount/crowds?). Definition: Loner",
    },
    {
        "number": "22d",
        "clue": "Beautiful actions excuse turning up now and then (6)",
        "answer": "SCENIC",
        "explanation": "Alternate letters of aCtIoNs ExCuSe reversed = SCENIC. Definition: Beautiful",
    },
    {
        "number": "24d",
        "clue": "Partly lifting ottoman is ordinarily a help when varnishing (5)",
        "answer": "ROSIN",
        "explanation": "Hidden reversed in ottomaN IS ORdinarily = NISOR reversed = ROSIN. Definition: a help when varnishing",
    },
]


def parse_one(client, clue_data):
    """Send one clue to Haiku for parsing. Returns parsed JSON or error string."""
    user_msg = (
        "Clue: %s\nAnswer: %s\nHuman explanation: %s"
        % (clue_data["clue"], clue_data["answer"], clue_data["explanation"])
    )

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            temperature=0,
            system=PARSE_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]  # remove first line (```json)
            if text.endswith("```"):
                text = text[:-3].strip()
        parsed = json.loads(text)
        return parsed, response.usage
    except json.JSONDecodeError:
        return {"_error": "Invalid JSON", "_raw": text}, None
    except Exception as e:
        return {"_error": str(e)}, None


def validate_pieces(parsed, answer):
    """Check if pieces concatenate to the answer."""
    pieces = parsed.get("pieces", [])
    if not pieces:
        return "no pieces"
    letters = "".join(p.get("letters", "") for p in pieces)
    answer_clean = answer.replace(" ", "").replace("-", "").upper()
    letters_clean = letters.replace(" ", "").replace("-", "").upper()

    if letters_clean == answer_clean:
        return "PASS"
    if sorted(letters_clean) == sorted(answer_clean):
        return "PASS (anagram)"
    return "FAIL: %s != %s" % (letters_clean, answer_clean)


def main():
    client = anthropic.Anthropic()

    total_cost = 0
    results = []

    print("=" * 70)
    print("Haiku Explanation Parser Test — Puzzle 29494")
    print("Model: %s" % MODEL)
    print("=" * 70)

    for clue_data in CLUES:
        parsed, usage = parse_one(client, clue_data)

        if usage:
            # Haiku pricing: $0.80/M input, $4/M output
            cost = (usage.input_tokens * 0.80 + usage.output_tokens * 4.0) / 1_000_000
            total_cost += cost
        else:
            cost = 0

        validation = validate_pieces(parsed, clue_data["answer"])
        pieces = parsed.get("pieces", [])

        status = "OK" if "PASS" in validation else "FAIL"
        results.append(status)

        print("\n%s. %s = %s" % (clue_data["number"], clue_data["clue"][:50], clue_data["answer"]))
        print("  Definition: %s" % parsed.get("definition", "?"))
        print("  Type: %s" % parsed.get("wordplay_type", "?"))
        for p in pieces:
            print("    %s -> %s (%s)" % (
                p.get("clue_word", "?"),
                p.get("letters", "?"),
                p.get("mechanism", "?"),
            ))
        print("  Yields: %s" % validation)
        if parsed.get("_reasoning"):
            print("  Reasoning: %s" % parsed["_reasoning"])
        if parsed.get("_error"):
            print("  ERROR: %s" % parsed["_error"])
        print("  [%s] $%.4f" % (status, cost))

    print("\n" + "=" * 70)
    n_pass = results.count("OK")
    print("Results: %d/%d passed (%.0f%%)" % (n_pass, len(results), 100 * n_pass / len(results)))
    print("Total cost: $%.4f" % total_cost)
    print("=" * 70)


if __name__ == "__main__":
    main()
