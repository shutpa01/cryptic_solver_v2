"""Stage 2: Classify parsed explanations against the 46 known catalog signatures.

Maps ParseResult (from notation_parser) → catalog signature label.
Measures coverage: what % of verified parses match a known signature?
"""

import sqlite3
import sys
import re
from collections import Counter
from typing import Optional, Tuple

sys.path.insert(0, '.')
from cryptic_taxonomy.analysis.notation_parser import parse_explanation, ParseResult


def classify(result: ParseResult) -> Tuple[str, str]:
    """Classify a ParseResult into a signature label.

    Returns:
        (signature_label, token_sequence)
        e.g. ("ABR+SYN charade", "ABR_F · SYN_F")
    """
    if not result.verified:
        return ("unverified", "")

    op = result.operation
    pieces = result.pieces

    # Whole-clue types (no wordplay decomposition)
    if op == 'double_definition':
        return ("double_definition", "DOUBLE_DEFINITION")
    if op == 'cryptic_definition':
        return ("cryptic_definition", "CRYPTIC_DEFINITION")
    if op == 'hidden':
        return ("hidden", "HID_F")
    if op == 'homophone':
        return ("homophone", "HOM_F")

    # Pure anagram
    if op == 'anagram':
        return ("anagram", "ANA_F")

    # Pure reversal
    if op == 'reversal':
        return ("SYN reversal", "SYN_F [REV]")

    # Map piece source_types to token vocabulary
    token_seq = []
    for p in pieces:
        t = p.source_type
        if t == 'ABR':
            token_seq.append('ABR_F')
        elif t == 'SYN':
            token_seq.append('SYN_F')
        elif t == 'DEL':
            # Deletion: the piece remains after deletion
            # Size heuristic: short = abbreviation-like, long = synonym-like
            if len(p.letters) <= 2:
                token_seq.append('ABR_F')
            else:
                token_seq.append('SYN_F')
        elif t == 'CON':
            # Container: already assembled. Map to SYN_F+SYN_F (the two parts)
            # since the catalog represents containers as two fodder tokens + CON_I
            token_seq.append('SYN_F')
            token_seq.append('SYN_F')
        elif t == 'REV':
            token_seq.append('SYN_F')  # reversed synonym
        elif t == 'ANA':
            token_seq.append('ANA_F')
        elif t == 'RAW':
            token_seq.append('RAW')
        elif t == 'HOM':
            token_seq.append('HOM_F')
        elif t == 'HID':
            token_seq.append('HID_F')
        else:
            token_seq.append(t)

    seq_str = ' · '.join(token_seq)

    # Determine signature label based on operation + token sequence
    if op in ('container', 'container_charade'):
        # Container operations map to SYN_F,SYN_F with CON_I indicator
        label = '+'.join(token_seq) + ' container'
    elif op == 'reversal_container':
        label = '+'.join(token_seq) + ' reversal_container'
    elif op == 'charade':
        label = '+'.join(token_seq) + ' charade'
    elif op == 'anagram_charade':
        label = '+'.join(token_seq) + ' anagram_charade'
    elif op == 'reversal_charade':
        label = '+'.join(token_seq) + ' reversal_charade'
    elif op == 'del':
        label = '+'.join(token_seq) + ' deletion'
    elif op == 'synonym':
        label = 'synonym'
    elif op == 'abr':
        label = 'abbreviation'
    else:
        label = f'{op}:{seq_str}'

    return (label, seq_str)


def run_classification():
    """Run classification on all explanations and report results."""
    conn = sqlite3.connect('data/times_explanations.db')
    rows = conn.execute(
        'SELECT answer, explanation FROM clues '
        'WHERE explanation IS NOT NULL AND explanation != ""'
    ).fetchall()
    conn.close()

    total = len(rows)
    verified = 0
    sig_counts = Counter()
    op_counts = Counter()

    for answer, expl in rows:
        result = parse_explanation(expl, answer)
        if result.verified:
            verified += 1
        label, seq = classify(result)
        sig_counts[label] += 1
        op_counts[result.operation] += 1

    print(f"Total: {total}, Verified: {verified} ({100*verified/total:.1f}%)")
    print()

    # Show signature distribution (top 50)
    print(f"{'Signature':50s} {'Count':>7s} {'%':>6s} {'Cum%':>6s}")
    print('-' * 72)
    cum = 0
    for sig, cnt in sig_counts.most_common(50):
        pct = 100 * cnt / total
        cum += pct
        print(f"{sig:50s} {cnt:7d} {pct:5.1f}% {cum:5.1f}%")

    # Build set of all catalog labels for matching
    from signature_solver.catalog import CATALOG as cat_entries
    catalog_labels = {e.label for e in cat_entries}

    # Also build set of token sequences in catalog
    catalog_token_seqs = set()
    for e in cat_entries:
        seq = '+'.join(e.tokens)
        catalog_token_seqs.add((seq, e.operation))

    # Operation categories that are whole-clue types (always in catalog)
    whole_clue = {'anagram', 'hidden', 'double_definition', 'cryptic_definition',
                  'homophone', 'reversal', 'reversal_container'}

    matched = 0
    unmatched = Counter()

    for sig, cnt in sig_counts.items():
        if sig == 'unverified':
            continue

        is_match = False

        # Whole-clue types
        if sig in whole_clue:
            is_match = True
        # Check by label match
        elif sig in catalog_labels:
            is_match = True
        # Check by token sequence + operation
        else:
            # Extract tokens from sig label
            parts = sig.rsplit(' ', 1)
            if len(parts) == 2:
                tok_str = parts[0].replace('+', '+')
                op_str = parts[1]
                # Check if any catalog entry has matching tokens
                for e in cat_entries:
                    e_seq = '+'.join(e.tokens)
                    if e_seq == tok_str and op_str in e.operation:
                        is_match = True
                        break

        if is_match:
            matched += cnt
        else:
            unmatched[sig] = cnt

    print()
    print("=" * 72)
    print("CATALOG COVERAGE (79 entries)")
    print("=" * 72)
    print(f"\nVerified parses: {verified}")
    print(f"Matched to catalog: {matched} ({100*matched/verified:.1f}%)")
    print(f"Not in catalog: {verified - matched} ({100*(verified-matched)/verified:.1f}%)")

    if unmatched:
        print(f"\nTop unmatched (potential new entries):")
        for sig, cnt in unmatched.most_common(20):
            print(f"  {sig:50s} {cnt:6d}")


if __name__ == '__main__':
    run_classification()
