"""One-shot coverage check for Guardian 30003 leftover pass."""
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
ref = sqlite3.connect('data/cryptic_new.db')
ref.row_factory = sqlite3.Row

checks = [
    # 6a MUGGLE
    ('syn', 'simpleton', 'MUG'),
    ('def', "he can't magically fix", 'MUGGLE'),
    ('ind', 'broken', 'anagram'),
    # 9a SPOTON
    ('syn', 'something to eat with', 'SPOON'),
    ('hom', 'tea', 'T'),
    ('ind', 'say', 'homophone'),
    ('ind', 'coming in', 'container'),
    ('def', 'Exactly', 'SPOTON'),
    ('def', 'exactly', 'SPOTON'),
    # 10a MARINADE
    ('ind', 'Pelting', 'anagram'),
    ('ind', 'pelting', 'anagram'),
    ('def', 'a liquid mixture over some food', 'MARINADE'),
    ('def', 'liquid mixture', 'MARINADE'),
    # 11a RELAYRACE - cryptic def
    ('def', "One may start this contest, but one won't finish", 'RELAYRACE'),
    ('def', 'contest', 'RELAYRACE'),
    # 13a FREED
    ('abbr', 'Fine', 'F'),
    ('abbr', 'fine', 'F'),
    ('syn', 'supplier of venison', 'DEER'),
    ('ind', 'returns', 'reversal'),
    ('def', 'out of jail', 'FREED'),
    # 17a FLYTIP
    ('syn', 'insect', 'FLY'),
    ('syn', 'end', 'TIP'),
    ('def', 'illegally dispose of it', 'FLYTIP'),
    ('def', 'illegally dispose', 'FLYTIP'),
    # 18a RENDER - DD
    ('def', 'Hand over', 'RENDER'),
    ('def', 'hand over', 'RENDER'),
    ('def', 'thin coat', 'RENDER'),
    ('def', 'first thin coat', 'RENDER'),
    ('syn', 'Hand over', 'RENDER'),
    ('syn', 'thin coat', 'RENDER'),
    # 19a BARREL - DD
    ('def', 'Large quantity of beer', 'BARREL'),
    ('def', 'large quantity of beer', 'BARREL'),
    ('def', 'over which one is powerless', 'BARREL'),
    ('def', 'beer container', 'BARREL'),
    ('syn', 'large quantity of beer', 'BARREL'),
    # 21a MARSH
    ('syn', 'planet', 'MARS'),
    ('def', 'planet', 'MARS'),
    ('abbr', 'hot', 'H'),
    ('def', 'waterlogged area', 'MARSH'),
    # 22a WOLVERINE
    ('ind', 'Drunken', 'anagram'),
    ('ind', 'drunken', 'anagram'),
    ('def', 'a glutton', 'WOLVERINE'),
    ('def', 'glutton', 'WOLVERINE'),
    # 25a INCIDENT
    ('syn', 'part of', 'IN'),
    ('syn', 'In', 'IN'),
    ('abbr', 'CID', 'CID'),
    ('syn', 'detective force', 'CID'),
    ('abbr', 'Criminal Investigation Department', 'CID'),
    ('ind', 'first off', 'deletion'),
    ('def', 'disturbance', 'INCIDENT'),
    # 26a DAMAGE
    ('syn', 'Barrier', 'DAM'),
    ('syn', 'barrier', 'DAM'),
    ('syn', 'to become weaker', 'AGE'),
    ('syn', 'become weaker', 'AGE'),
    ('def', 'a likely result of crash', 'DAMAGE'),
    ('def', 'result of crash', 'DAMAGE'),
    # 28a CRISES
    ('abbr', 'Charlie', 'C'),
    ('syn', 'gets up', 'RISES'),
    ('def', 'Crucial moments', 'CRISES'),
    # 29a BERGAMOT
    ('syn', 'mass of ice', 'BERG'),
    ('abbr', 'car test', 'MOT'),
    ('abbr', 'MOT', 'MOT'),
    ('def', 'Essential oil', 'BERGAMOT'),
    # 2d LAP - CD
    ('def', 'Part of body disappearing as one stands', 'LAP'),
    # 4d LONERANGER
    ('abbr', 'left hand', 'L'),
    ('abbr', 'right hand', 'R'),
    ('abbr', 'left', 'L'),
    ('abbr', 'right', 'R'),
    ('abbr', 'hand', 'L'),
    ('abbr', 'hand', 'R'),
    ('syn', 'fury', 'ANGER'),
    ('def', 'Wild West law enforcer', 'LONERANGER'),
    # 6d MARE - DD
    ('def', 'Horse', 'MARE'),
    ('def', 'horse', 'MARE'),
    ('def', 'sea on the moon', 'MARE'),
    ('def', 'that appears on the moon', 'MARE'),
    # 7d GENERATOR
    ('ind', 'Shivering', 'anagram'),
    ('ind', 'shivering', 'anagram'),
    ('def', 'source of energy', 'GENERATOR'),
    # 12d ENTERTAINER
    ('ind', 'Somehow', 'anagram'),
    ('ind', 'somehow', 'anagram'),
    ('def', 'job in showbiz', 'ENTERTAINER'),
    # 14d ELSALVADOR
    ('ind', 'moved', 'anagram'),
    ('def', 'the country', 'ELSALVADOR'),
    ('def', 'country', 'ELSALVADOR'),
    # 16d LANDSLIPS - CD
    ('def', 'Naturally, they may fall off a cliff', 'LANDSLIPS'),
    ('def', 'they may fall off a cliff', 'LANDSLIPS'),
    # 20d HOTTUB
    ('syn', 'Centre', 'HUB'),
    ('syn', 'centre', 'HUB'),
    ('abbr', 'unwarranted', 'OTT'),
    ('abbr', 'over-the-top', 'OTT'),
    ('abbr', 'over the top', 'OTT'),
    ('ind', 'installs', 'container'),
    ('ind', 'installs', 'insertion'),
    ('def', 'bath', 'HOTTUB'),
    # 23d RUMBA
    ('syn', 'strong drink', 'RUM'),
    ('abbr', 'airline', 'BA'),
    ('abbr', 'British Airways', 'BA'),
    ('def', 'A Cuban export', 'RUMBA'),
    ('def', 'Cuban export', 'RUMBA'),
    ('def', 'Cuban dance', 'RUMBA'),
    # 24d MEWS
    ('hom', 'muse', 'mews'),
    ('hom', 'mews', 'muse'),
    ('syn', 'ponder', 'MUSE'),
    ('syn', 'Ponder', 'MUSE'),
    ('ind', 'aloud', 'homophone'),
    ('def', 'old stables', 'MEWS'),
    # 27d GOO
    ('syn', 'satisfactory', 'GOOD'),
    ('ind', 'not entirely', 'deletion'),
    ('def', 'Sticky stuff', 'GOO'),
    ('def', 'sticky stuff', 'GOO'),
]

for chk in checks:
    kind = chk[0]
    if kind == 'syn':
        _, w, target = chk
        r = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (w, target)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (target, w)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  SYN  {w!r} <-> {target!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'def':
        _, d, a = chk
        r = ref.execute("SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=LOWER(?) AND UPPER(answer)=UPPER(?) LIMIT 1", (d, a)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (d, a)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  DEF  {d!r} -> {a!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'abbr':
        _, w, letters = chk
        r = ref.execute("SELECT 1 FROM wordplay WHERE LOWER(indicator)=LOWER(?) AND UPPER(substitution)=UPPER(?) LIMIT 1", (w, letters)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (w, letters)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  ABBR {w!r} -> {letters!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'ind':
        _, w, wt = chk
        r = ref.execute("SELECT 1 FROM indicators WHERE LOWER(word)=LOWER(?) AND LOWER(wordplay_type)=LOWER(?) LIMIT 1", (w, wt)).fetchone()
        ok = r is not None
        print(f"  IND  {w!r} [{wt}]: {'OK' if ok else 'MISS'}")
    elif kind == 'hom':
        _, w1, w2 = chk
        r = ref.execute("SELECT 1 FROM homophones WHERE (LOWER(word)=LOWER(?) AND LOWER(homophone)=LOWER(?)) OR (LOWER(word)=LOWER(?) AND LOWER(homophone)=LOWER(?)) LIMIT 1", (w1, w2, w2, w1)).fetchone()
        ok = r is not None
        print(f"  HOM  {w1!r} ~ {w2!r}: {'OK' if ok else 'MISS'}")
