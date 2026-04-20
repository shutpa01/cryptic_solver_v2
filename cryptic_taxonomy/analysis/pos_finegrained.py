"""Fine-grained POS analysis: do specific grammatical forms predict cryptic roles?

Uses spaCy's tag_ (Penn Treebank tags) instead of coarse pos_:
  VBG = gerund ("breaking", "holding") — likely indicators?
  VBN = past participle ("broken", "mixed") — likely indicators?
  VBD = past tense ("backed", "lost") — likely indicators?
  NNP = proper noun ("George", "London") — likely abbreviations?
  JJ = adjective ("short", "old") — indicators or synonyms?
  RB = adverb ("back", "up", "around") — indicators?
  IN = preposition/conjunction ("in", "about", "with") — links or indicators?
  , = comma — piece boundary?
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

    clue_ids = [r[0] for r in conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles'
    ).fetchall()]

    print(f"POS-tagging {len(clue_ids)} clues with fine-grained tags...")

    pos_to_role = defaultdict(lambda: Counter())
    role_to_pos = defaultdict(lambda: Counter())
    total_words = 0

    # Also track: for indicator roles specifically, what fine POS are they?
    indicator_roles = {'ANA_I', 'CON_I', 'REV_I', 'DEL_I', 'HID_I', 'HOM_I'}

    for i, clue_id in enumerate(clue_ids):
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

        for pos, role in zip(pos_tags, roles):
            pos_to_role[pos][role] += 1
            role_to_pos[role][pos] += 1
            total_words += 1

    conn.close()

    print(f"\nTotal words tagged: {total_words}")
    print(f"Unique fine-grained tags: {len(pos_to_role)}")

    # 1. Full POS -> role table
    print(f"\n{'='*80}")
    print("FINE-GRAINED POS -> CRYPTIC ROLE")
    print(f"{'='*80}")
    print(f"{'Tag':6s} {'Description':25s} {'N':>6s} | {'#1 Role':10s} {'%':>5s} | {'#2 Role':10s} {'%':>5s} | {'#3':10s} {'%':>5s}")
    print("-" * 100)

    tag_descriptions = {
        'NN': 'noun singular',
        'NNS': 'noun plural',
        'NNP': 'proper noun',
        'NNPS': 'proper noun plural',
        'VB': 'verb base',
        'VBD': 'verb past',
        'VBG': 'verb gerund',
        'VBN': 'verb past part',
        'VBP': 'verb non-3rd pres',
        'VBZ': 'verb 3rd pres',
        'JJ': 'adjective',
        'JJR': 'adjective compar',
        'JJS': 'adjective superl',
        'RB': 'adverb',
        'RBR': 'adverb compar',
        'RBS': 'adverb superl',
        'IN': 'preposition/subconj',
        'DT': 'determiner',
        'CC': 'coordinating conj',
        'PRP': 'personal pronoun',
        'PRP$': 'possessive pronoun',
        'MD': 'modal',
        'TO': 'to',
        'RP': 'particle',
        'WDT': 'wh-determiner',
        'WP': 'wh-pronoun',
        'WRB': 'wh-adverb',
        'EX': 'existential there',
        'CD': 'cardinal number',
        'FW': 'foreign word',
        'UH': 'interjection',
        ',': 'comma',
        '.': 'period',
        ':': 'colon',
        '``': 'open quote',
        "''": 'close quote',
        '-LRB-': 'left bracket',
        '-RRB-': 'right bracket',
        'HYPH': 'hyphen',
        'NFP': 'superfluous punct',
        'POS': 'possessive ending',
    }

    for pos in sorted(pos_to_role.keys(), key=lambda x: -sum(pos_to_role[x].values())):
        total = sum(pos_to_role[pos].values())
        if total < 20:
            continue
        desc = tag_descriptions.get(pos, '')
        top3 = pos_to_role[pos].most_common(3)
        line = f"  {pos:5s} {desc:25s} {total:6d} |"
        for role, cnt in top3:
            pct = 100 * cnt / total
            line += f" {role:10s} {pct:4.1f}% |"
        print(line)

    # 2. Focus on indicator roles — what POS tags are they?
    print(f"\n{'='*80}")
    print("INDICATOR ROLES: WHAT POS TAGS ARE THEY?")
    print(f"{'='*80}")
    for role in ['ANA_I', 'CON_I', 'REV_I', 'DEL_I']:
        total = sum(role_to_pos[role].values())
        if total < 10:
            continue
        print(f"\n  {role} (n={total}):")
        for pos, cnt in role_to_pos[role].most_common(10):
            pct = 100 * cnt / total
            desc = tag_descriptions.get(pos, '')
            print(f"    {pos:6s} {desc:25s} {cnt:5d} ({pct:5.1f}%)")

    # 3. Strongest predictions — where one role dominates
    print(f"\n{'='*80}")
    print("STRONG PREDICTIONS (one role >50%, n>=50)")
    print(f"{'='*80}")
    for pos in sorted(pos_to_role.keys(), key=lambda x: -sum(pos_to_role[x].values())):
        total = sum(pos_to_role[pos].values())
        if total < 50:
            continue
        top_role, top_cnt = pos_to_role[pos].most_common(1)[0]
        pct = 100 * top_cnt / total
        if pct >= 50:
            desc = tag_descriptions.get(pos, '')
            print(f"  {pos:6s} {desc:25s} -> {top_role:10s} ({pct:.1f}%, n={total})")

    # 4. The key question: do VBG/VBN strongly predict indicator?
    print(f"\n{'='*80}")
    print("KEY QUESTION: DO VERB FORMS PREDICT INDICATORS?")
    print(f"{'='*80}")
    for tag in ['VBG', 'VBN', 'VBD', 'VB', 'VBP', 'VBZ']:
        total = sum(pos_to_role[tag].values())
        if total < 20:
            continue
        ind_count = sum(pos_to_role[tag].get(r, 0) for r in indicator_roles)
        syn_count = pos_to_role[tag].get('SYN_F', 0)
        lnk_count = pos_to_role[tag].get('LNK', 0)
        abr_count = pos_to_role[tag].get('ABR_F', 0)
        print(f"  {tag:5s} ({tag_descriptions.get(tag, ''):20s}) n={total:5d}: "
              f"INDICATOR={100*ind_count/total:5.1f}%  SYN_F={100*syn_count/total:5.1f}%  "
              f"LNK={100*lnk_count/total:5.1f}%  ABR_F={100*abr_count/total:5.1f}%")

    # 5. Adverbs and prepositions — the user specifically asked
    print(f"\n{'='*80}")
    print("ADVERBS AND PREPOSITIONS IN DETAIL")
    print(f"{'='*80}")
    for tag in ['RB', 'RBR', 'RBS', 'IN', 'RP', 'TO']:
        total = sum(pos_to_role[tag].values())
        if total < 20:
            continue
        print(f"\n  {tag} ({tag_descriptions.get(tag, '')}, n={total}):")
        for role, cnt in pos_to_role[tag].most_common(8):
            pct = 100 * cnt / total
            if pct >= 2:
                print(f"    {role:12s} {cnt:5d} ({pct:5.1f}%)")

    # 6. Punctuation
    print(f"\n{'='*80}")
    print("PUNCTUATION ROLES")
    print(f"{'='*80}")
    for tag in [',', '.', ':', 'HYPH', '``', "''", '-LRB-', '-RRB-', 'NFP']:
        total = sum(pos_to_role[tag].values())
        if total < 5:
            continue
        print(f"\n  {tag} ({tag_descriptions.get(tag, '')}, n={total}):")
        for role, cnt in pos_to_role[tag].most_common(5):
            pct = 100 * cnt / total
            print(f"    {role:12s} {cnt:5d} ({pct:5.1f}%)")


if __name__ == '__main__':
    run()
