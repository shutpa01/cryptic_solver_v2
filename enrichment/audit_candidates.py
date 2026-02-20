"""
audit_candidates.py — Anthropic API audit for self-learning candidates.

Reads documents/self_learning_candidates.txt (one JSON per line, written by
05_self_learning_enrichment.py --dry-run) and asks Claude Haiku whether each
candidate is a valid DB entry.

Verdict for each candidate:
  YES    — approve for insertion
  NO     — reject
  UNSURE — flag for manual review

Output written to documents/self_learning_audit.txt (one JSON per line,
same fields as input plus: verdict, reason).

Usage:
  python -m enrichment.audit_candidates
  python -m enrichment.audit_candidates --candidates documents/self_learning_candidates.txt
  python -m enrichment.audit_candidates --candidates documents/self_learning_candidates.txt --verbose
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent

# Use cheapest model — these are simple YES/NO/UNSURE judgements.
AUDIT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 150
SLEEP_BETWEEN_CALLS = 0.5  # seconds — polite rate limiting


# ============================================================
# PROMPT BUILDERS
# ============================================================

def build_prompt(candidate: dict) -> str:
    """Build the audit prompt for a single candidate."""
    ctype = candidate.get('type', '')
    clue = candidate.get('clue', '(unknown clue)')

    if ctype == 'definition_pair':
        phrase = candidate['phrase']
        answer = candidate['answer']
        return (
            f"Cryptic crossword question.\n\n"
            f"Clue: \"{clue}\"\n"
            f"Answer: {answer}\n\n"
            f"Could the phrase \"{phrase}\" reasonably serve as the definition "
            f"component for {answer} in this cryptic crossword clue?\n\n"
            f"Reply with exactly one of YES, NO, or UNSURE on the first line, "
            f"followed by one sentence explaining why."
        )

    elif ctype == 'indicator':
        phrase = candidate['phrase']
        subtype = candidate.get('subtype', candidate.get('wordplay_type', 'unknown'))
        container = candidate.get('container', '')
        letters = candidate.get('letters', '')
        return (
            f"Cryptic crossword question.\n\n"
            f"Clue: \"{clue}\"\n"
            f"Answer: {candidate['answer']}\n\n"
            f"Is \"{phrase}\" a valid {subtype} indicator in cryptic crosswords? "
            f"(Context: it is being proposed as the indicator that the outer "
            f"letters of \"{container}\" give \"{letters}\".)\n\n"
            f"Reply with exactly one of YES, NO, or UNSURE on the first line, "
            f"followed by one sentence explaining why."
        )

    elif ctype in ('wordplay', 'synonym'):
        word = candidate.get('phrase', candidate.get('word', ''))
        letters = candidate.get('letters', candidate.get('answer', ''))
        return (
            f"Cryptic crossword question.\n\n"
            f"Clue: \"{clue}\"\n\n"
            f"Is \"{word}\" → \"{letters}\" a valid abbreviation or substitution "
            f"in cryptic crosswords?\n\n"
            f"Reply with exactly one of YES, NO, or UNSURE on the first line, "
            f"followed by one sentence explaining why."
        )

    else:
        return (
            f"Cryptic crossword question.\n\n"
            f"Clue: \"{clue}\"\n"
            f"Candidate entry: {json.dumps(candidate)}\n\n"
            f"Is this a valid entry for a cryptic crossword solving database?\n\n"
            f"Reply with exactly one of YES, NO, or UNSURE on the first line, "
            f"followed by one sentence explaining why."
        )


# ============================================================
# VERDICT PARSING
# ============================================================

def parse_verdict(response_text: str) -> tuple[str, str]:
    """Extract (verdict, reason) from model response.

    Expects the first token to be YES, NO, or UNSURE.
    Falls back to scanning first two lines if the format is slightly off.
    """
    text = response_text.strip()
    first_line = text.split('\n')[0].strip()

    # Try first word of first line
    first_word = re.split(r'[\s,.\-:]', first_line)[0].upper()
    if first_word in ('YES', 'NO', 'UNSURE'):
        # Reason is everything after the verdict word/line
        remaining = text[len(first_line):].strip()
        if not remaining:
            remaining = first_line[len(first_word):].strip().lstrip('.,: ')
        reason = remaining[:300]  # cap length
        return first_word, reason

    # Fallback: scan for YES/NO/UNSURE anywhere in first two lines
    for line in text.splitlines()[:2]:
        for verdict in ('YES', 'NO', 'UNSURE'):
            if verdict in line.upper():
                reason = text.replace(line, '').strip()[:300]
                return verdict, reason or line

    return 'UNSURE', text[:300]


# ============================================================
# AUDIT LOOP
# ============================================================

def audit_candidates(candidates_path: Path, output_path: Path,
                     verbose: bool = False) -> list:
    """Audit all candidates in candidates_path, write results to output_path."""
    if not candidates_path.exists():
        print(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    lines = [l.strip() for l in candidates_path.read_text(encoding='utf-8').splitlines()
             if l.strip()]
    if not lines:
        print("Candidates file is empty.")
        return []

    candidates = []
    for i, line in enumerate(lines, 1):
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping line {i} (JSON error: {e})")

    print(f"Candidates to audit: {len(candidates)}")
    print(f"Model: {AUDIT_MODEL}\n")

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in environment / .env")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for i, candidate in enumerate(candidates, 1):
        pattern = candidate.get('pattern', '?')
        ctype = candidate.get('type', '?')
        phrase = candidate.get('phrase', '')
        answer = candidate.get('answer', '')

        print(f"[{i}/{len(candidates)}] Pattern {pattern} | {ctype} | "
              f"'{phrase}' -> {answer}")

        prompt = build_prompt(candidate)

        if verbose:
            print(f"  Prompt:\n    {prompt.replace(chr(10), chr(10) + '    ')}")

        try:
            message = client.messages.create(
                model=AUDIT_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = message.content[0].text
        except Exception as e:
            print(f"  ERROR calling API: {e}")
            verdict, reason = 'UNSURE', f'API error: {e}'
            response_text = ''

        verdict, reason = parse_verdict(response_text)
        print(f"  Verdict: {verdict}  |  {reason[:100]}")

        result = {**candidate, 'verdict': verdict, 'reason': reason}
        results.append(result)

        if i < len(candidates):
            time.sleep(SLEEP_BETWEEN_CALLS)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\nAudit written to: {output_path}")

    # Summary
    counts = {'YES': 0, 'NO': 0, 'UNSURE': 0}
    for r in results:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    print(f"Summary: YES={counts['YES']}  NO={counts['NO']}  UNSURE={counts['UNSURE']}")

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Anthropic API audit for self-learning candidates')
    parser.add_argument(
        '--candidates',
        default=str(PROJECT_ROOT / 'documents' / 'self_learning_candidates.txt'),
        help='Input candidates file (default: documents/self_learning_candidates.txt)'
    )
    parser.add_argument(
        '--output',
        default=str(PROJECT_ROOT / 'documents' / 'self_learning_audit.txt'),
        help='Output audit file (default: documents/self_learning_audit.txt)'
    )
    parser.add_argument('--verbose', action='store_true',
                        help='Print full prompt for each candidate')
    args = parser.parse_args()

    audit_candidates(
        candidates_path=Path(args.candidates),
        output_path=Path(args.output),
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
