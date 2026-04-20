"""Do operation types have characteristic POS sequences?

For each known operation (charade, container, reversal, anagram, etc.),
extract the fine-grained POS sequence of the wordplay window and look
for patterns. If reversals have a distinctive grammatical signature
that's different from containers, that's the hidden code.
"""

import sqlite3
import sys
import spacy
from collections import Counter, defaultdict

sys.path.insert(0, '.')


def run():
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')

    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    clue_ids = conn.execute('''
        SELECT DISTINCT clue_id, operation
        FROM word_roles WHERE word_position = 0
    ''').fetchall()

    print(f"Analysing POS sequences for {len(clue_ids)} clues...")

    # operation -> list of POS sequences
    op_pos_seqs = defaultdict(list)
    # operation -> Counter of POS sequences
    op_pos_counts = defaultdict(lambda: Counter())
    # Also store the role sequence alongside for comparison
    op_paired = defaultdict(lambda: Counter())  # op -> Counter of (pos_seq, role_seq)

    for i, (clue_id, operation) in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")

        rows = conn.execute('''
            SELECT word_text, assigned_role
            FROM word_roles WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        roles = [r[1] for r in rows]

        wp_text = ' '.join(words)
        doc = nlp(wp_text)

        spacy_tokens = list(doc)
        pos_tags = []
        si = 0
        for word in words:
            if si < len(spacy_tokens):
                pos_tags.append(spacy_tokens[si].tag_)
                consumed = len(spacy_tokens[si].text)
                si += 1
                while consumed < len(word) and si < len(spacy_tokens):
                    consumed += len(spacy_tokens[si].text) + 1
                    si += 1
            else:
                pos_tags.append('XX')

        if len(pos_tags) != len(roles):
            continue

        pos_seq = tuple(pos_tags)
        role_seq = tuple(roles)
        op_pos_seqs[operation].append(pos_seq)
        op_pos_counts[operation][pos_seq] += 1
        op_paired[operation][(pos_seq, role_seq)] += 1

    conn.close()

    # === ANALYSIS ===

    # 1. For each operation, show the most common POS sequences
    print(f"\n{'='*90}")
    print("MOST COMMON POS SEQUENCES BY OPERATION")
    print(f"{'='*90}")

    for op in sorted(op_pos_counts.keys(), key=lambda x: -len(op_pos_seqs[x])):
        total = len(op_pos_seqs[op])
        if total < 30:
            continue
        print(f"\n--- {op} ({total} clues) ---")
        print(f"  {'POS Sequence':50s} {'N':>5s} {'%':>6s}")
        print(f"  {'-'*65}")
        cum = 0
        for seq, cnt in op_pos_counts[op].most_common(15):
            pct = 100 * cnt / total
            cum += pct
            seq_str = ' '.join(seq)
            print(f"  {seq_str:50s} {cnt:5d} {pct:5.1f}%")
            if cum > 50:
                break

    # 2. Distinctive sequences: POS sequences that are much more common
    # in one operation than others
    print(f"\n{'='*90}")
    print("DISTINCTIVE POS SEQUENCES (>3x more likely in one op than average)")
    print(f"{'='*90}")

    # Build global frequency of each POS sequence
    global_counts = Counter()
    global_total = 0
    for op, seqs in op_pos_seqs.items():
        for seq in seqs:
            global_counts[seq] += 1
            global_total += 1

    distinctive = []
    for op in sorted(op_pos_counts.keys(), key=lambda x: -len(op_pos_seqs[x])):
        op_total = len(op_pos_seqs[op])
        if op_total < 30:
            continue
        for seq, cnt in op_pos_counts[op].most_common(50):
            if cnt < 5:
                continue
            op_rate = cnt / op_total
            global_rate = global_counts[seq] / global_total
            if global_rate > 0 and op_rate / global_rate > 3.0:
                distinctive.append((op, seq, cnt, op_total, op_rate, global_rate))

    distinctive.sort(key=lambda x: -(x[4] / x[5]))
    print(f"\n  {'Operation':20s} {'POS Sequence':45s} {'N':>4s} {'Op%':>6s} {'Glob%':>6s} {'Ratio':>6s}")
    print(f"  {'-'*95}")
    for op, seq, cnt, op_total, op_rate, global_rate in distinctive[:30]:
        seq_str = ' '.join(seq)
        ratio = op_rate / global_rate
        print(f"  {op:20s} {seq_str:45s} {cnt:4d} {100*op_rate:5.1f}% {100*global_rate:5.1f}% {ratio:5.1f}x")

    # 3. For the top distinctive sequences, show what role sequence they map to
    print(f"\n{'='*90}")
    print("DISTINCTIVE SEQUENCES: POS -> ROLE MAPPING")
    print(f"{'='*90}")

    for op, seq, cnt, op_total, op_rate, global_rate in distinctive[:15]:
        seq_str = ' '.join(seq)
        ratio = op_rate / global_rate
        print(f"\n  {op}: {seq_str} (n={cnt}, {ratio:.1f}x overrepresented)")

        # What role sequences does this POS sequence produce in this operation?
        role_counts = Counter()
        for (ps, rs), count in op_paired[op].items():
            if ps == seq:
                role_counts[rs] += count

        for role_seq, rc in role_counts.most_common(3):
            role_str = ' '.join(role_seq)
            print(f"    -> {role_str:50s} ({rc}/{cnt} = {100*rc/cnt:.0f}%)")

    # 4. Bigram analysis: consecutive POS tag pairs by operation
    print(f"\n{'='*90}")
    print("POS BIGRAMS DISTINCTIVE TO EACH OPERATION")
    print(f"{'='*90}")

    op_bigrams = defaultdict(lambda: Counter())
    global_bigrams = Counter()
    global_bi_total = 0

    for op, seqs in op_pos_seqs.items():
        for seq in seqs:
            for j in range(len(seq) - 1):
                bigram = (seq[j], seq[j+1])
                op_bigrams[op][bigram] += 1
                global_bigrams[bigram] += 1
                global_bi_total += 1

    for op in ['container', 'reversal_charade', 'anagram', 'charade', 'del']:
        op_bi_total = sum(op_bigrams[op].values())
        if op_bi_total < 50:
            continue
        print(f"\n  --- {op} ---")
        dist_bigrams = []
        for bigram, cnt in op_bigrams[op].most_common(100):
            if cnt < 5:
                continue
            op_rate = cnt / op_bi_total
            global_rate = global_bigrams[bigram] / global_bi_total
            if global_rate > 0 and op_rate / global_rate > 2.0:
                dist_bigrams.append((bigram, cnt, op_rate, global_rate))

        dist_bigrams.sort(key=lambda x: -(x[2] / x[3]))
        for bigram, cnt, op_rate, global_rate in dist_bigrams[:8]:
            ratio = op_rate / global_rate
            print(f"    {bigram[0]:6s} {bigram[1]:6s}  n={cnt:4d}  {100*op_rate:5.1f}% vs {100*global_rate:5.1f}% ({ratio:.1f}x)")


if __name__ == '__main__':
    run()
