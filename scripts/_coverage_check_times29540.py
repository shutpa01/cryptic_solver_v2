"""Coverage check for Times 29540 leftover pass.

For each (synonym/abbreviation/indicator/definition/homophone) claim in the
parses, query the reference DB. Print OK / MISS. Genuine misses are what
gets queued to pending_enrichments — duplicates of in-DB rows do not.
"""
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')

ref = sqlite3.connect('data/cryptic_new.db')
ref.row_factory = sqlite3.Row

# Claim list: (kind, ...)
# kinds: syn(word, target), def(definition, answer), abbr(word, letters),
#        ind(word, wordplay_type), hom(w1, w2)
checks = [
    # ===== 1a ASITCOMES =====
    ('syn',  'funny show', 'SITCOM'),
    ('ind',  'knocked', 'deletion'),
    ('def',  'Straight', 'ASITCOMES'),

    # ===== 6a GRASS (DD) =====
    ('def',  'blades', 'GRASS'),
    ('def',  'Split', 'GRASS'),

    # ===== 9a GAMELAN =====
    ('syn',  'something to play', 'GAME'),
    ('syn',  'track', 'LANE'),
    ('ind',  'short', 'deletion'),
    ('def',  'Percussion ensemble', 'GAMELAN'),

    # ===== 10a TROPHIC =====
    ('syn',  'parallel', 'TROPIC'),
    ('ind',  'eating', 'container'),
    ('def',  'related to nutrition', 'TROPHIC'),

    # ===== 11a THROB =====
    ('def',  'Beat', 'THROB'),

    # ===== 12a INGESTION =====
    ('ind',  'flying', 'anagram'),
    ('def',  'for feeding', 'INGESTION'),
    ('def',  'feeding', 'INGESTION'),

    # ===== 14a PIE (DD) =====
    ('def',  'Sweet or savoury dish', 'PIE'),
    ('def',  'mixed type', 'PIE'),

    # ===== 15a EVENINGSTAR =====
    ('ind',  'breaks', 'anagram'),
    ('def',  'Heavenly body', 'EVENINGSTAR'),

    # ===== 17a ADIMEADOZEN =====
    ('syn',  'port', 'ADEN'),
    ('syn',  'one', 'I'),
    ('syn',  'drink', 'MEAD'),
    ('syn',  'Australian', 'OZ'),
    ('ind',  'opening', 'container'),
    ('def',  'nothing special', 'ADIMEADOZEN'),

    # ===== 19a FEW =====
    ('hom',  'few', 'phew'),
    ('syn',  "I'm relieved", 'PHEW'),
    ('ind',  'to hear', 'homophone'),
    ('ind',  'hear', 'homophone'),
    ('def',  'Only a handful', 'FEW'),
    ('def',  'a handful', 'FEW'),

    # ===== 22a ELAND =====
    ('syn',  'river', 'CAM'),
    ('syn',  'with', 'AND'),
    ('ind',  'leaving', 'deletion'),
    ('def',  'antelope', 'ELAND'),

    # ===== 24a REISSUE =====
    ('syn',  'Queen', 'ER'),
    ('syn',  'girl, possibly', 'ISSUE'),
    ('syn',  'girl', 'ISSUE'),
    ('ind',  'rejected', 'reversal'),
    ('def',  'put out again', 'REISSUE'),

    # ===== 26a EXACTOR =====
    ('syn',  'no longer', 'EX'),
    ('syn',  'one taking a part', 'ACTOR'),
    ('def',  'Demanding individual', 'EXACTOR'),

    # ===== 27a DWELT =====
    ('syn',  'Raised mark', 'WELT'),
    ('def',  'resided', 'DWELT'),

    # ===== 28a HORSEPLAY =====
    ('def',  'High jinks', 'HORSEPLAY'),

    # ===== 1d ARGOT =====
    ('syn',  'understood', 'GOT'),
    ('def',  'secret language', 'ARGOT'),

    # ===== 2d IMMERSE =====
    ('syn',  '1', 'I'),
    ('abbr', 'millimetre', 'MM'),
    ('syn',  'millimetre', 'MM'),
    ('syn',  'tongue', 'ERSE'),
    ('def',  'Dip', 'IMMERSE'),

    # ===== 3d CALABRESE =====
    ('syn',  'Suit', 'CASE'),
    ('syn',  'party', 'LAB'),
    ('syn',  'concerned with', 'RE'),
    ('ind',  'hosting', 'container'),
    ('def',  'vegetarian food', 'CALABRESE'),

    # ===== 4d MANBITESDOG =====
    ('def',  'Newsworthy story', 'MANBITESDOG'),

    # ===== 5d SOT =====
    ('def',  'Sponge', 'SOT'),

    # ===== 6d GROSS (DD) =====
    ('def',  "I'm disgusted", 'GROSS'),
    ('def',  '24 six-packs', 'GROSS'),

    # ===== 7d ATHEIST =====
    ('syn',  'witnessing', 'AT'),
    ('syn',  'robbery', 'HEIST'),
    ('def',  'Infidel', 'ATHEIST'),

    # ===== 8d SECONDROW =====
    ('ind',  'collapse', 'anagram'),
    ('def',  'Part of scrum', 'SECONDROW'),

    # ===== 13d GRIZZLYBEAR =====
    ('hom',  'grizzly', 'grisly'),
    ('hom',  'bear', 'bare'),
    ('syn',  'gruesome', 'GRISLY'),
    ('syn',  'display', 'BARE'),
    ('ind',  'for the audience', 'homophone'),
    ('ind',  'audience', 'homophone'),
    ('def',  'Beastly thing', 'GRIZZLYBEAR'),

    # ===== 14d PLASTERED (DD) =====
    ('def',  'Lit up', 'PLASTERED'),
    ('def',  'as ceilings, perhaps', 'PLASTERED'),

    # ===== 16d GINGERALE =====
    ('syn',  'drink', 'GIN'),
    ('ind',  'rum', 'anagram'),
    ('abbr', 'energy', 'E'),
    ('def',  'Drink', 'GINGERALE'),

    # ===== 18d IMAGINE =====
    ('syn',  'this writer', 'I'),
    ('syn',  'soldier', 'GI'),
    ('syn',  'shock', 'MANE'),
    ('ind',  'in', 'container'),
    ('def',  'Picture', 'IMAGINE'),

    # ===== 19d FRACTAL =====
    ('ind',  'adrift', 'anagram'),
    ('abbr', 'American', 'A'),
    ('abbr', 'lake', 'L'),
    ('def',  'Complex pattern', 'FRACTAL'),

    # ===== 21d INSET =====
    ('syn',  'popular', 'IN'),
    ('syn',  'class', 'SET'),
    ('def',  'Introduce', 'INSET'),

    # ===== 23d DERBY =====
    ('syn',  'raised', 'BRED'),
    ('ind',  'lifted', 'reversal'),
    ('def',  'Hat', 'DERBY'),

    # ===== 25d ETH =====
    ('ind',  'delivered in', 'hidden'),
    ('ind',  'delivered', 'hidden'),
    ('def',  'Old letter', 'ETH'),
]

results = {'OK': [], 'MISS': []}
for chk in checks:
    kind = chk[0]
    if kind == 'syn':
        _, w, target = chk
        r = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1",
            (w, target)).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1",
            (target, w)).fetchone()
        rd = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=LOWER(?) AND UPPER(answer)=UPPER(?) LIMIT 1",
            (w, target)).fetchone()
        ok = r is not None or r2 is not None or rd is not None
        label = f"SYN  {w!r} -> {target!r}"
    elif kind == 'def':
        _, d, a = chk
        r = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=LOWER(?) AND UPPER(answer)=UPPER(?) LIMIT 1",
            (d, a)).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1",
            (d, a)).fetchone()
        ok = r is not None or r2 is not None
        label = f"DEF  {d!r} -> {a!r}"
    elif kind == 'abbr':
        _, w, letters = chk
        r = ref.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=LOWER(?) AND UPPER(substitution)=UPPER(?) LIMIT 1",
            (w, letters)).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1",
            (w, letters)).fetchone()
        ok = r is not None or r2 is not None
        label = f"ABBR {w!r} -> {letters!r}"
    elif kind == 'ind':
        _, w, wt = chk
        r = ref.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=LOWER(?) AND LOWER(wordplay_type)=LOWER(?) LIMIT 1",
            (w, wt)).fetchone()
        ok = r is not None
        label = f"IND  {w!r} [{wt}]"
    elif kind == 'hom':
        _, w1, w2 = chk
        r = ref.execute(
            "SELECT 1 FROM homophones WHERE (LOWER(word)=LOWER(?) AND LOWER(homophone)=LOWER(?)) OR (LOWER(word)=LOWER(?) AND LOWER(homophone)=LOWER(?)) LIMIT 1",
            (w1, w2, w2, w1)).fetchone()
        ok = r is not None
        label = f"HOM  {w1!r} ~ {w2!r}"
    status = 'OK' if ok else 'MISS'
    results[status].append(label)
    print(f"  {status:4} {label}")

print()
print(f"Summary: {len(results['OK'])} OK, {len(results['MISS'])} MISS")
print()
print("Genuine MISSES (to queue as pending_enrichments):")
for m in results['MISS']:
    print(f"  {m}")
