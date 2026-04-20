"""The hidden code: does surface grammar predict cryptic structure?

For each clue in the word_roles dataset:
1. POS-tag the wordplay window as a sentence fragment (surface reading)
2. Compare the POS sequence to the known cryptic role sequence
3. Measure: how strongly does POS predict cryptic role?
4. Look for POS-sequence patterns that map to specific role sequences

The hypothesis: setters are constrained by English grammar. A word's
part of speech in the surface reading strongly predicts its cryptic
function. If true, this is the hidden code — the structural correspondence
between language and wordplay that the setter is forced to employ.
"""

import sqlite3
import sys
import re
import spacy
from collections import Counter, defaultdict

sys.path.insert(0, '.')


def run():
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')

    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    # Get all clues with their words and roles
    clue_ids = [r[0] for r in conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles'
    ).fetchall()]

    print(f"POS-tagging {len(clue_ids)} clues...")

    # Collect: (pos_tag, cryptic_role) pairs across all words
    pos_role_pairs = Counter()
    # Collect: POS sequence -> role sequence
    pos_seq_to_role_seq = defaultdict(lambda: Counter())
    # Per-POS role distribution
    pos_to_role = defaultdict(lambda: Counter())
    # Per-role POS distribution
    role_to_pos = defaultdict(lambda: Counter())
    # Raw counts
    total_words = 0

    for i, clue_id in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")

        rows = conn.execute('''
            SELECT word_text, assigned_role, clue_text
            FROM word_roles WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        roles = [r[1] for r in rows]
        clue_text = rows[0][2] if rows else ''

        # POS-tag the wordplay window as a phrase
        # Join words into a string for spaCy
        wp_text = ' '.join(words)
        doc = nlp(wp_text)

        # Align spaCy tokens back to our words
        # spaCy may tokenize differently (e.g. "parliament's" -> "parliament" + "'s")
        # Use simple alignment: walk through spaCy tokens matching to our words
        spacy_tokens = list(doc)
        pos_tags = []
        si = 0
        for word in words:
            if si < len(spacy_tokens):
                # Find the spaCy token(s) that cover this word
                pos_tags.append(spacy_tokens[si].pos_)
                # Advance past any sub-tokens of this word
                consumed = len(spacy_tokens[si].text)
                si += 1
                while consumed < len(word) and si < len(spacy_tokens):
                    consumed += len(spacy_tokens[si].text) + 1
                    si += 1
            else:
                pos_tags.append('X')

        if len(pos_tags) != len(roles):
            continue  # alignment failed

        # Record pairs
        for pos, role in zip(pos_tags, roles):
            pos_role_pairs[(pos, role)] += 1
            pos_to_role[pos][role] += 1
            role_to_pos[role][pos] += 1
            total_words += 1

        # Record sequences (for clues with 3-6 words, most common lengths)
        if 3 <= len(words) <= 6:
            pos_seq = tuple(pos_tags)
            role_seq = tuple(roles)
            pos_seq_to_role_seq[pos_seq][role_seq] += 1

    conn.close()

    # === ANALYSIS ===
    print(f"\nTotal words tagged: {total_words}")

    # 1. For each POS tag, what cryptic role does it most commonly play?
    print(f"\n{'='*70}")
    print("POS TAG -> CRYPTIC ROLE DISTRIBUTION")
    print(f"{'='*70}")
    print(f"{'POS':8s} {'Total':>7s} | {'Top Role':12s} {'%':>6s} | {'2nd Role':12s} {'%':>6s} | {'3rd Role':12s} {'%':>6s}")
    print("-" * 85)

    for pos in sorted(pos_to_role.keys(), key=lambda x: -sum(pos_to_role[x].values())):
        total = sum(pos_to_role[pos].values())
        if total < 50:
            continue
        top3 = pos_to_role[pos].most_common(3)
        line = f"  {pos:6s} {total:7d} |"
        for role, cnt in top3:
            pct = 100 * cnt / total
            line += f" {role:12s} {pct:5.1f}% |"
        print(line)

    # 2. For each cryptic role, what POS tag is most common?
    print(f"\n{'='*70}")
    print("CRYPTIC ROLE -> POS TAG DISTRIBUTION")
    print(f"{'='*70}")
    print(f"{'Role':15s} {'Total':>7s} | {'Top POS':8s} {'%':>6s} | {'2nd POS':8s} {'%':>6s} | {'3rd POS':8s} {'%':>6s}")
    print("-" * 85)

    for role in sorted(role_to_pos.keys(), key=lambda x: -sum(role_to_pos[x].values())):
        total = sum(role_to_pos[role].values())
        if total < 20:
            continue
        top3 = role_to_pos[role].most_common(3)
        line = f"  {role:13s} {total:7d} |"
        for pos, cnt in top3:
            pct = 100 * cnt / total
            line += f" {pos:8s} {pct:5.1f}% |"
        print(line)

    # 3. Strongest POS->role correlations (where POS is highly predictive)
    print(f"\n{'='*70}")
    print("STRONGEST POS -> ROLE PREDICTIONS (>60% of one role)")
    print(f"{'='*70}")
    for pos in sorted(pos_to_role.keys(), key=lambda x: -sum(pos_to_role[x].values())):
        total = sum(pos_to_role[pos].values())
        if total < 50:
            continue
        top_role, top_cnt = pos_to_role[pos].most_common(1)[0]
        pct = 100 * top_cnt / total
        if pct >= 60:
            print(f"  {pos:8s} -> {top_role:12s} ({pct:.1f}%, n={total})")

    # 4. POS sequence -> role sequence patterns
    print(f"\n{'='*70}")
    print("POS SEQUENCE -> ROLE SEQUENCE (most common patterns)")
    print(f"{'='*70}")

    # Find POS sequences where the top role sequence dominates
    strong_patterns = []
    for pos_seq, role_counts in pos_seq_to_role_seq.items():
        total = sum(role_counts.values())
        if total < 5:
            continue
        top_role_seq, top_cnt = role_counts.most_common(1)[0]
        pct = 100 * top_cnt / total
        strong_patterns.append((pos_seq, top_role_seq, top_cnt, total, pct))

    strong_patterns.sort(key=lambda x: -x[2])

    print(f"\nTop 30 most common POS sequences with dominant role pattern:")
    print(f"{'POS Sequence':40s} {'Role Sequence':50s} {'N':>5s} {'Tot':>5s} {'%':>6s}")
    print("-" * 110)
    for pos_seq, role_seq, cnt, total, pct in strong_patterns[:30]:
        pos_str = ' '.join(pos_seq)
        role_str = ' '.join(role_seq)
        print(f"  {pos_str:38s} {role_str:50s} {cnt:5d} {total:5d} {pct:5.1f}%")

    # 5. High-confidence patterns (>70% dominant, n>=10)
    confident = [(p, r, c, t, pct) for p, r, c, t, pct in strong_patterns
                 if pct >= 70 and t >= 10]
    confident.sort(key=lambda x: -x[3])

    print(f"\n{'='*70}")
    print(f"HIGH-CONFIDENCE PATTERNS (>70% dominant, n>=10): {len(confident)}")
    print(f"{'='*70}")
    total_covered = sum(t for _, _, _, t, _ in confident)
    total_correct = sum(c for _, _, c, _, _ in confident)
    print(f"Total clues covered: {total_covered}")
    print(f"Correctly predicted: {total_correct} ({100*total_correct/total_covered:.1f}%)")
    print()
    for pos_seq, role_seq, cnt, total, pct in confident[:20]:
        pos_str = ' '.join(pos_seq)
        role_str = ' '.join(role_seq)
        print(f"  {pos_str:38s} -> {role_str:50s} ({cnt}/{total} = {pct:.0f}%)")


if __name__ == '__main__':
    run()
