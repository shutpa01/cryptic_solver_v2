"""API-assisted solver for clues the mechanical solver can't handle.

Instead of asking the API to reason from scratch, we provide rich
structured evidence from the word analyser, turning the task into
constrained selection rather than open-ended reasoning.

Single-pass design: one API call returns structured JSON directly.
"""

import json
import re

from anthropic import Anthropic
from dotenv import load_dotenv

from .evidence import format_evidence, format_failed_solve_context

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are an expert cryptic crossword analyst. You will be given:
1. A cryptic clue with its answer already known
2. The definition portion (already identified)
3. Detailed word-by-word analysis from a reference database showing each word's possible roles

Your job: using ONLY the provided evidence, determine how the wordplay produces the answer.

RULES:
- Every letter in the answer must be accounted for
- Indicator words (anagram, reversal, container, deletion, hidden, homophone indicators) signal the mechanism but contribute NO letters
- Link words (for, with, in, to, of, etc.) are connective — they contribute NO letters
- Use the DB lookups preferentially — if a word has a confirmed abbreviation or synonym that fits, use it
- The definition is at the start or end of the clue and has already been identified for you

OUTPUT FORMAT — respond with ONLY a JSON object:
{
  "wordplay_type": "charade|container|anagram|deletion|hidden|reversal|homophone|double_definition|cryptic_definition|acrostic|substitution",
  "pieces": [
    {"word": "clue word(s)", "letters": "UPPERCASE LETTERS", "role": "synonym|abbreviation|literal|anagram_fodder|first_letter|last_letter|outer_letters|middle_letter|alternate_letters|reversal|hidden|homophone|indicator|link"},
    ...
  ],
  "assembly": "brief description of how pieces combine",
  "confidence": "high|medium|low"
}

CRITICAL VALIDATION RULE — your answer will be mechanically verified:
- For charade: the "letters" fields concatenated left-to-right MUST exactly spell the answer
- For reversal: "letters" must show the REVERSED form (post-reversal), so concatenation spells the answer
- For container: "letters" of outer piece must be SPLIT showing where inner goes, OR concatenation must spell the answer
- For deletion: "letters" must show what REMAINS after deletion, so concatenation spells the answer
- For anagram: "letters" is the raw fodder BEFORE rearrangement (sorted letters must match answer)
- For homophone: "letters" must be the SPELLED-OUT form that matches the answer
- Indicators and link words MUST have "letters": "" (empty string)
- Do NOT include deleted/removed letters in any piece's "letters" field

SELF-CHECK: Before outputting, mentally concatenate all non-empty "letters" values. For non-anagram types, this concatenation must exactly equal the answer. If it doesn't, fix your pieces.

If you cannot determine the wordplay, return {"wordplay_type": "unknown", "pieces": [], "assembly": "", "confidence": "low"}"""


def api_solve(clue_text, answer, definition, solve_result, db):
    """Call the API with structured evidence to solve a clue.

    Args:
        clue_text: full clue text
        answer: known answer
        definition: definition portion of the clue
        solve_result: SolveResult from mechanical solver (has analyses)
        db: RefDB instance

    Returns:
        (parsed_json, tokens_in, tokens_out) or (None, tokens_in, tokens_out)
    """
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    # Build evidence block
    evidence = format_evidence(
        solve_result.analyses,
        solve_result.phrases,
        answer_clean,
        clue_text=clue_text,
        definition=definition,
    )

    # Add failed solve context if available
    failed_context = format_failed_solve_context(solve_result, answer_clean)

    user_msg = evidence
    if failed_context:
        user_msg += "\n" + failed_context

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "{"},  # force JSON output
        ],
    )

    raw = "{" + response.content[0].text.strip()
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens

    parsed = _parse_json_response(raw)

    return parsed, tokens_in, tokens_out


def _parse_json_response(raw):
    """Extract JSON from API response, handling markdown fences."""
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def validate_api_result(parsed, answer, db=None):
    """Mechanically verify the API result produces the answer.

    Two layers of validation:
    1. Letter verification — concatenation/anagram must produce the answer
    2. DB lookup verification — claimed synonyms/abbreviations must exist in DB

    Returns (is_valid, reason) tuple.
    """
    from . import executor

    if not parsed or not isinstance(parsed, dict):
        return False, "no valid response"

    wtype = parsed.get("wordplay_type", "")
    pieces = parsed.get("pieces", [])

    if wtype == "unknown":
        return False, "API could not determine wordplay"

    if not pieces:
        return False, "no pieces"

    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    # Extract contributing pieces (skip indicators and links)
    contribs = []
    for p in pieces:
        role = p.get("role", "")
        letters = p.get("letters", "").upper().replace(" ", "").replace("-", "")
        if role in ("indicator", "link") or not letters:
            continue
        contribs.append((letters, role, p.get("word", "?")))

    if not contribs:
        return False, "no contributing letters"

    all_letters = [c[0] for c in contribs]

    # --- Layer 1: Letter verification ---
    letter_valid, letter_reason = _verify_letters(
        wtype, all_letters, contribs, answer_clean, executor
    )

    if not letter_valid:
        return False, letter_reason

    # --- Layer 2: DB lookup verification ---
    if db is not None:
        db_valid, db_reason = _verify_db_lookups(pieces, db)
        if not db_valid:
            return False, f"{letter_reason} BUT {db_reason}"

    return True, letter_reason


def _verify_letters(wtype, all_letters, contribs, answer_clean, executor):
    """Layer 1: Verify that pieces produce the answer via string operations."""

    # --- Anagram ---
    if wtype == "anagram":
        fodder = "".join(all_letters)
        if executor.check_anagram(fodder, answer_clean):
            return True, "anagram verified"
        return False, f"anagram mismatch: sorted({fodder}) != sorted({answer_clean})"

    # --- Hidden ---
    if wtype == "hidden":
        fodder = "".join(all_letters)
        if answer_clean in fodder:
            return True, "hidden verified"
        return False, "answer not found in hidden span"

    # --- Charade ---
    if wtype == "charade":
        concat = "".join(all_letters)
        if concat == answer_clean:
            return True, "charade verified"
        return False, f"charade mismatch: {concat} != {answer_clean}"

    # --- Reversal ---
    if wtype == "reversal":
        concat = "".join(all_letters)
        if concat[::-1] == answer_clean:
            return True, "reversal verified (full reverse)"
        if _try_reversal_combos(contribs, answer_clean):
            return True, "reversal verified (partial reverse)"
        return False, f"reversal mismatch: {concat} reversed = {concat[::-1]} != {answer_clean}"

    # --- Container ---
    if wtype == "container":
        if _try_container_combos(contribs, answer_clean):
            return True, "container verified"
        return False, "container: no valid insertion found"

    # --- Deletion ---
    if wtype == "deletion":
        if _try_deletion_combos(contribs, answer_clean):
            return True, "deletion verified"
        concat = "".join(all_letters)
        if concat == answer_clean:
            return True, "deletion verified (pieces already show result)"
        return False, "deletion: no valid removal found"

    # --- Homophone ---
    if wtype == "homophone":
        concat = "".join(all_letters)
        if concat == answer_clean:
            return True, "homophone verified (spelling matches)"
        if executor.check_anagram(concat, answer_clean):
            return True, "homophone verified (letters match)"
        return False, f"homophone mismatch: {concat} != {answer_clean}"

    # --- Double/cryptic definition ---
    if wtype in ("double_definition", "cryptic_definition"):
        return True, "definition type (unverifiable)"

    # --- Substitution ---
    if wtype == "substitution":
        concat = "".join(all_letters)
        if concat == answer_clean:
            return True, "substitution verified"
        return False, f"substitution mismatch: {concat} != {answer_clean}"

    # --- Acrostic ---
    if wtype == "acrostic":
        concat = "".join(all_letters)
        if concat == answer_clean:
            return True, "acrostic verified"
        return False, f"acrostic mismatch: {concat} != {answer_clean}"

    return False, f"unknown wordplay type: {wtype}"


def _verify_db_lookups(pieces, db):
    """Check that claimed synonyms/abbreviations actually exist in the DB.

    Returns (is_valid, reason) — only fails if a bogus lookup is found.
    """
    bogus = []

    for p in pieces:
        role = p.get("role", "")
        word = p.get("word", "").strip()
        letters = p.get("letters", "").upper().replace(" ", "").replace("-", "")

        if not word or not letters:
            continue

        if role == "synonym":
            # Check if any word in the clue phrase maps to these letters
            words_to_check = word.lower().split()
            found = False
            for w in words_to_check:
                syns = db.get_synonyms(w)
                if letters in syns:
                    found = True
                    break
            if not found:
                # Also check if it's a valid literal (short word = itself)
                w_alpha = "".join(c for c in word.upper() if c.isalpha())
                if w_alpha == letters:
                    found = True
            if not found:
                bogus.append(f"{word}->{letters} (synonym not in DB)")

        elif role == "abbreviation":
            words_to_check = word.lower().split()
            found = False
            for w in words_to_check:
                abbrs = db.get_abbreviations(w)
                if letters in abbrs:
                    found = True
                    break
            if not found:
                bogus.append(f"{word}->{letters} (abbreviation not in DB)")

        elif role == "homophone":
            words_to_check = word.lower().split()
            found = False
            for w in words_to_check:
                homos = db.get_homophones(w)
                if letters in homos:
                    found = True
                    break
                # Also check synonyms that sound like the answer
                syns = db.get_synonyms(w)
                if letters in syns:
                    found = True
                    break
            if not found:
                bogus.append(f"{word}->{letters} (homophone not in DB)")

    if bogus:
        return False, "bogus lookups: " + "; ".join(bogus)
    return True, "all lookups verified"


def _try_reversal_combos(contribs, answer):
    """Try reversing one or more pieces and concatenating."""
    letters_list = [c[0] for c in contribs]
    n = len(letters_list)

    # Try reversing each single piece
    for i in range(n):
        parts = list(letters_list)
        parts[i] = parts[i][::-1]
        if "".join(parts) == answer:
            return True

    # Try reversing all pieces
    parts = [l[::-1] for l in letters_list]
    if "".join(parts) == answer:
        return True

    # Try reversing concatenation of all
    if "".join(letters_list)[::-1] == answer:
        return True

    # Try reversed order of pieces (each piece intact)
    if "".join(reversed(letters_list)) == answer:
        return True

    # Try reversed order with each piece also reversed
    if "".join(l[::-1] for l in reversed(letters_list)) == answer:
        return True

    return False


def _try_container_combos(contribs, answer):
    """Try all combinations of inner/outer pieces to verify container."""
    from . import executor

    letters_list = [c[0] for c in contribs]
    n = len(letters_list)

    if n < 2:
        return False

    # Two pieces: try each as outer/inner
    if n == 2:
        a, b = letters_list
        if executor.try_container(a, b, answer):
            return True
        if executor.try_container(b, a, answer):
            return True
        return False

    # Three or more pieces: try pairs as outer/inner, concat rest
    from itertools import permutations

    for perm in permutations(range(n)):
        ordered = [letters_list[i] for i in perm]

        # Try: first piece is outer, second is inner, rest concat after
        for outer_idx in range(n):
            for inner_idx in range(n):
                if outer_idx == inner_idx:
                    continue
                outer = ordered[outer_idx]
                inner = ordered[inner_idx]
                rest = [ordered[k] for k in range(n)
                        if k != outer_idx and k != inner_idx]

                # inner inside outer, then concat rest
                for ins_pos in range(1, len(outer)):
                    result = outer[:ins_pos] + inner + outer[ins_pos:]
                    # Try rest before, after, or split
                    rest_str = "".join(rest)
                    if result + rest_str == answer:
                        return True
                    if rest_str + result == answer:
                        return True

                    # Try rest pieces interleaved
                    if len(rest) == 1:
                        r = rest[0]
                        if r + result == answer:
                            return True
                        if result + r == answer:
                            return True

        # Only check first permutation for 3+ pieces to limit explosion
        if n >= 3:
            break

    return False


def _try_deletion_combos(contribs, answer):
    """Try deletion: one piece is the base, another is removed."""
    from . import executor

    letters_list = [c[0] for c in contribs]
    roles = [c[1] for c in contribs]
    n = len(letters_list)

    if n < 2:
        return False

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            base = letters_list[i]
            remove = letters_list[j]
            # Try removing 'remove' from 'base'
            if executor.try_deletion(base, remove, answer):
                return True

            # Also try: base with first/last letter removed
            if base[1:] == answer:
                return True
            if base[:-1] == answer:
                return True

        # Try: concat remaining pieces after deletion
        remaining = [letters_list[k] for k in range(n) if k != i]
        remaining_str = "".join(remaining)
        if remaining_str == answer:
            return True

    return False
