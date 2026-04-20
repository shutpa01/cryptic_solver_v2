"""Step 2+3: Enrich word_roles with DB features and analyse patterns.

Adds per-word features:
  - has_indicator: which indicator types this word has in the DB
  - has_synonym: does it have synonyms in RefDB
  - has_abbreviation: does it have abbreviations in RefDB
  - is_link_word: is it in the LINK_WORDS set
  - letter_count: number of alpha chars in this word
  - window_letter_total: total alpha chars across all wordplay words
  - letter_budget_ratio: window_letter_total / answer_length

Then analyses:
  - Role distribution by feature
  - Letter budget ratio by operation
  - Indicator presence vs operation type
  - Triage accuracy: how often do features predict the correct operation?
"""

import sqlite3
import sys
import re
from collections import Counter, defaultdict

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS, INDICATOR_TYPE_TO_TOKEN


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


def enrich():
    """Add feature columns to word_roles table."""
    print("Loading indicator DB...")
    ind_db = IndicatorDB()

    print("Loading RefDB for synonym/abbreviation lookups...")
    ref_conn = sqlite3.connect('data/cryptic_new.db', timeout=30)

    # Build quick lookup sets
    syn_words = set()
    for (word,) in ref_conn.execute("SELECT DISTINCT word FROM synonyms_pairs"):
        syn_words.add(word.lower().strip())
    for (defn,) in ref_conn.execute(
        "SELECT DISTINCT definition FROM definition_answers_augmented WHERE definition IS NOT NULL"
    ):
        syn_words.add(defn.lower().strip())

    abbr_words = set()
    for (word,) in ref_conn.execute("SELECT DISTINCT indicator FROM wordplay"):
        abbr_words.add(word.lower().strip())
    ref_conn.close()

    print(f"  {len(syn_words):,} synonym source words, {len(abbr_words):,} abbreviation source words")

    # Open word_roles DB
    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    # Add columns if they don't exist
    existing_cols = {row[1] for row in conn.execute('PRAGMA table_info(word_roles)')}
    new_cols = [
        ('has_indicator_types', 'TEXT'),
        ('has_synonym', 'INTEGER'),
        ('has_abbreviation', 'INTEGER'),
        ('is_link_word', 'INTEGER'),
        ('letter_count', 'INTEGER'),
        ('window_letter_total', 'INTEGER'),
        ('letter_budget_ratio', 'REAL'),
    ]
    for col, typ in new_cols:
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE word_roles ADD COLUMN {col} {typ}')
    conn.commit()

    # Process in batches by clue_id
    clue_ids = [r[0] for r in conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles'
    ).fetchall()]
    print(f"Enriching {len(clue_ids)} clues...")

    batch_updates = []
    for i, clue_id in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")
            conn.executemany('''
                UPDATE word_roles SET
                    has_indicator_types=?, has_synonym=?, has_abbreviation=?,
                    is_link_word=?, letter_count=?, window_letter_total=?,
                    letter_budget_ratio=?
                WHERE clue_id=? AND word_position=?
            ''', batch_updates)
            conn.commit()
            batch_updates = []

        rows = conn.execute('''
            SELECT word_position, word_text, answer_length
            FROM word_roles WHERE clue_id=?
            ORDER BY word_position
        ''', (clue_id,)).fetchall()

        # Compute window-level features
        window_letter_total = sum(
            len(re.sub(r'[^a-zA-Z]', '', r[1])) for r in rows
        )
        answer_length = rows[0][2] if rows else 0
        ratio = window_letter_total / answer_length if answer_length > 0 else 0

        for pos, word_text, _ in rows:
            w = _clean(word_text)
            letter_count = len(re.sub(r'[^a-zA-Z]', '', word_text))

            # Indicator types
            ind_types = ind_db.get_indicator_types(word_text)
            ind_str = ','.join(sorted(ind_types)) if ind_types else None

            # Synonym/abbreviation presence
            has_syn = 1 if w in syn_words else 0
            has_abbr = 1 if w in abbr_words else 0

            # Also check without trailing s
            if not has_syn and len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
                has_syn = 1 if w[:-1] in syn_words else 0
            if not has_abbr and len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
                has_abbr = 1 if w[:-1] in abbr_words else 0

            # Link word
            is_lnk = 1 if w in LINK_WORDS else 0

            batch_updates.append((
                ind_str, has_syn, has_abbr, is_lnk,
                letter_count, window_letter_total, ratio,
                clue_id, pos
            ))

    # Final batch
    if batch_updates:
        conn.executemany('''
            UPDATE word_roles SET
                has_indicator_types=?, has_synonym=?, has_abbreviation=?,
                is_link_word=?, letter_count=?, window_letter_total=?,
                letter_budget_ratio=?
            WHERE clue_id=? AND word_position=?
        ''', batch_updates)
        conn.commit()

    print("Enrichment done.\n")
    return conn


def analyse(conn):
    """Run the analysis and print results."""

    print("=" * 70)
    print("STEP 3: ANALYSIS")
    print("=" * 70)

    # --- 1. Letter budget ratio by operation ---
    print("\n--- Letter Budget Ratio by Operation ---")
    print(f"{'Operation':25s} {'Clues':>6s} {'Mean':>6s} {'Median':>7s} {'Min':>5s} {'Max':>5s}")
    print("-" * 60)
    for op, cnt, avg, mn, mx in conn.execute('''
        SELECT operation, COUNT(DISTINCT clue_id) as n_clues,
               ROUND(AVG(letter_budget_ratio), 2),
               ROUND(MIN(letter_budget_ratio), 2),
               ROUND(MAX(letter_budget_ratio), 2)
        FROM word_roles
        WHERE word_position = 0
        GROUP BY operation
        ORDER BY n_clues DESC
    ''').fetchall():
        # Get median via subquery
        med = conn.execute('''
            SELECT ROUND(letter_budget_ratio, 2) FROM (
                SELECT letter_budget_ratio, ROW_NUMBER() OVER (ORDER BY letter_budget_ratio) as rn,
                       COUNT(*) OVER () as total
                FROM word_roles WHERE operation=? AND word_position=0
            ) WHERE rn = total/2 + 1
        ''', (op,)).fetchone()
        med_val = med[0] if med else '?'
        print(f"  {op:23s} {cnt:6d} {avg:6.2f} {str(med_val):>7s} {mn:5.2f} {mx:5.2f}")

    # --- 2. Role distribution by feature ---
    print("\n--- What role do words with indicator DB hits actually play? ---")
    print(f"{'Actual Role':15s} {'Has Ind DB':>10s} {'No Ind DB':>10s} {'% Ind DB':>9s}")
    print("-" * 50)
    for role, with_ind, without_ind in conn.execute('''
        SELECT assigned_role,
               SUM(CASE WHEN has_indicator_types IS NOT NULL THEN 1 ELSE 0 END),
               SUM(CASE WHEN has_indicator_types IS NULL THEN 1 ELSE 0 END)
        FROM word_roles
        GROUP BY assigned_role
        ORDER BY SUM(CASE WHEN has_indicator_types IS NOT NULL THEN 1 ELSE 0 END) DESC
    ''').fetchall():
        total = with_ind + without_ind
        pct = 100 * with_ind / total if total else 0
        print(f"  {role:13s} {with_ind:10d} {without_ind:10d} {pct:8.1f}%")

    # --- 3. How predictive is indicator DB hit for BEING an indicator? ---
    print("\n--- Words with indicator DB hits: what role do they actually play? ---")
    total_with_ind = conn.execute(
        'SELECT COUNT(*) FROM word_roles WHERE has_indicator_types IS NOT NULL'
    ).fetchone()[0]
    print(f"Total words with indicator DB hits: {total_with_ind}")
    for role, cnt in conn.execute('''
        SELECT assigned_role, COUNT(*) as cnt
        FROM word_roles
        WHERE has_indicator_types IS NOT NULL
        GROUP BY assigned_role
        ORDER BY cnt DESC
    ''').fetchall():
        pct = 100 * cnt / total_with_ind
        print(f"  {role:15s} {cnt:7d} ({pct:5.1f}%)")

    # --- 4. Synonym feature vs actual role ---
    print("\n--- Words with synonym DB hits: what role do they play? ---")
    total_with_syn = conn.execute(
        'SELECT COUNT(*) FROM word_roles WHERE has_synonym = 1'
    ).fetchone()[0]
    print(f"Total words with synonym hits: {total_with_syn}")
    for role, cnt in conn.execute('''
        SELECT assigned_role, COUNT(*) as cnt
        FROM word_roles WHERE has_synonym = 1
        GROUP BY assigned_role ORDER BY cnt DESC
    ''').fetchall():
        pct = 100 * cnt / total_with_syn
        print(f"  {role:15s} {cnt:7d} ({pct:5.1f}%)")

    # --- 5. Abbreviation feature vs actual role ---
    print("\n--- Words with abbreviation DB hits: what role do they play? ---")
    total_with_abbr = conn.execute(
        'SELECT COUNT(*) FROM word_roles WHERE has_abbreviation = 1'
    ).fetchone()[0]
    print(f"Total words with abbreviation hits: {total_with_abbr}")
    for role, cnt in conn.execute('''
        SELECT assigned_role, COUNT(*) as cnt
        FROM word_roles WHERE has_abbreviation = 1
        GROUP BY assigned_role ORDER BY cnt DESC
    ''').fetchall():
        pct = 100 * cnt / total_with_abbr
        print(f"  {role:15s} {cnt:7d} ({pct:5.1f}%)")

    # --- 6. Operation triage: indicator types present vs actual operation ---
    print("\n--- Operation Triage: indicator types present in clue vs actual operation ---")
    print("(For each clue, what indicator types exist among its words?)")

    clue_data = conn.execute('''
        SELECT clue_id, operation,
               GROUP_CONCAT(DISTINCT has_indicator_types) as all_ind_types
        FROM word_roles
        WHERE word_position = 0
        GROUP BY clue_id
    ''').fetchall()

    # For each clue, get all indicator types across all its words
    triage = defaultdict(lambda: Counter())
    for clue_id, operation, _ in clue_data:
        ind_types = set()
        for (types_str,) in conn.execute(
            'SELECT DISTINCT has_indicator_types FROM word_roles WHERE clue_id=? AND has_indicator_types IS NOT NULL',
            (clue_id,)
        ).fetchall():
            if types_str:
                for t in types_str.split(','):
                    ind_types.add(t.strip())

        key = frozenset(ind_types) if ind_types else frozenset(['NONE'])
        triage[key][operation] += 1

    # Show top indicator-type combinations and what operations they predict
    print(f"\n{'Indicator Types Present':45s} {'Top Operation':25s} {'Count':>6s} {'Accuracy':>8s}")
    print("-" * 90)
    combos = sorted(triage.items(), key=lambda x: -sum(x[1].values()))
    for ind_types, op_counts in combos[:25]:
        total = sum(op_counts.values())
        top_op, top_cnt = op_counts.most_common(1)[0]
        accuracy = 100 * top_cnt / total
        types_str = ', '.join(sorted(ind_types)) if ind_types else 'NONE'
        print(f"  {types_str:43s} {top_op:25s} {total:6d} {accuracy:7.1f}%")
        # Show runner-up if accuracy < 80%
        if accuracy < 80 and len(op_counts) > 1:
            for op, cnt in op_counts.most_common(3)[1:]:
                pct = 100 * cnt / total
                print(f"  {'':43s} {op:25s} {cnt:6d} {pct:7.1f}%")

    # --- 7. Combined triage: letter budget + indicators ---
    print("\n--- Combined Triage: letter budget band + indicator presence ---")

    clue_features = conn.execute('''
        SELECT DISTINCT clue_id, operation, letter_budget_ratio
        FROM word_roles WHERE word_position = 0
    ''').fetchall()

    def budget_band(ratio):
        if ratio < 0.8:
            return '<0.8'
        elif ratio <= 1.2:
            return '0.8-1.2'
        elif ratio <= 1.8:
            return '1.2-1.8'
        else:
            return '>1.8'

    combined = defaultdict(lambda: Counter())
    for clue_id, operation, ratio in clue_features:
        band = budget_band(ratio)

        # Get indicator types for this clue
        ind_types = set()
        for (types_str,) in conn.execute(
            'SELECT DISTINCT has_indicator_types FROM word_roles WHERE clue_id=? AND has_indicator_types IS NOT NULL',
            (clue_id,)
        ).fetchall():
            if types_str:
                for t in types_str.split(','):
                    ind_types.add(t.strip())

        has_ana = 'anagram' in ind_types
        has_rev = 'reversal' in ind_types
        has_con = 'container' in ind_types or 'insertion' in ind_types
        has_del = 'deletion' in ind_types
        has_hid = 'hidden' in ind_types

        # Build triage key
        signals = []
        signals.append(f'budget:{band}')
        if has_ana: signals.append('ANA_I')
        if has_rev: signals.append('REV_I')
        if has_con: signals.append('CON_I')
        if has_del: signals.append('DEL_I')
        if has_hid: signals.append('HID_I')
        if not (has_ana or has_rev or has_con or has_del or has_hid):
            signals.append('no_indicator')

        key = ' + '.join(signals)
        combined[key][operation] += 1

    print(f"\n{'Triage Signals':55s} {'Top Op':20s} {'N':>5s} {'Acc':>6s} {'2nd Op':20s}")
    print("-" * 115)
    for key, op_counts in sorted(combined.items(), key=lambda x: -sum(x[1].values()))[:30]:
        total = sum(op_counts.values())
        top_op, top_cnt = op_counts.most_common(1)[0]
        accuracy = 100 * top_cnt / total
        second = op_counts.most_common(2)
        second_str = f"{second[1][0]} ({100*second[1][1]/total:.0f}%)" if len(second) > 1 else ""
        print(f"  {key:53s} {top_op:20s} {total:5d} {accuracy:5.1f}% {second_str}")

    # --- 8. Summary stats ---
    print("\n--- Summary ---")
    total_clues = len(clue_features)
    # Count clues where top triage prediction matches actual operation
    correct = 0
    for key, op_counts in combined.items():
        top_op, top_cnt = op_counts.most_common(1)[0]
        correct += top_cnt
    print(f"Total clues: {total_clues}")
    print(f"Correctly predicted by triage: {correct} ({100*correct/total_clues:.1f}%)")


if __name__ == '__main__':
    conn = enrich()
    analyse(conn)
    conn.close()
