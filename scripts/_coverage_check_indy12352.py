"""Coverage check for planned Indy 12352 parses."""
import sqlite3, sys
sys.stdout.reconfigure(encoding='utf-8')
ref = sqlite3.connect('data/cryptic_new.db')

CHECKS = [
    # 1a CAMP
    ('syn', 'hat', 'CAP'),
    ('abbr', 'maiden', 'M'),
    ('ind', 'wearing', 'container'),
    ('def', 'Theatrical', 'CAMP'),
    # 3a APPRENTICE
    ('syn', 'Program', 'APP'),
    ('abbr', 'run', 'R'),
    ('syn', 'attract', 'ENTICE'),
    ('def', 'trainee', 'APPRENTICE'),
    # 9a WINDSOR — substitution, mostly FAIL
    ('def', 'castle', 'WINDSOR'),
    ('syn', 'Victor', 'WINNER'),
    ('abbr', 'award', 'DSO'),
    # 11a PERUSED
    ('syn', 'a', 'PER'),
    ('syn', 'Yankee', 'US'),
    ('abbr', 'editor', 'ED'),
    ('def', 'Went over', 'PERUSED'),
    # 12a LOGCABIN
    ('def', "backwoods' location", 'LOGCABIN'),
    ('def', 'backwoods location', 'LOGCABIN'),
    # 13a HEADS — DD
    ('def', 'On board facilities', 'HEADS'),
    ('def', 'on board facilities', 'HEADS'),
    ('def', 'leaders', 'HEADS'),
    # 15a PERFORMINGARTS
    ('def', 'ballet and opera?', 'PERFORMINGARTS'),
    ('def', 'ballet and opera', 'PERFORMINGARTS'),
    # 17a FORTUNECOOKIES — CD
    ('def', 'culinary treats', 'FORTUNECOOKIES'),
    ('def', 'Lots may be wrapped up in these culinary treats', 'FORTUNECOOKIES'),
    # 21a ISAAC
    ('abbr', 'Type of investment', 'ISA'),
    ('abbr', 'investment account', 'ISA'),
    ('abbr', 'investment', 'ISA'),
    ('abbr', 'account', 'AC'),
    ('def', "Ishmael's half-brother", 'ISAAC'),
    # 22a SOCIETAL
    ('syn', 'former', 'LATE'),
    ('abbr', 'independent', 'I'),
    ('syn', 'island', 'COS'),
    ('ind', 'returns', 'reversal'),
    ('def', 'Community', 'SOCIETAL'),
    # 24a REGRETS
    ('syn', 'fliers', 'EGRETS'),
    ('def', 'feels remorse', 'REGRETS'),
    # 25a LEAKOUT — hidden
    ('def', 'emerge', 'LEAKOUT'),
    # 26a BRASSBANDS
    ('syn', 'money', 'BRASS'),
    ('syn', 'belts', 'BANDS'),
    ('def', 'Players', 'BRASSBANDS'),
    # 27a STUD
    ('syn', 'room', 'STUDY'),
    ('def', 'Boss', 'STUD'),
    ('ind', 'backing', 'reversal'),  # often reversal indicator
    ('ind', 'backing out', 'deletion'),
    # 1d COWSLIPS
    ('abbr', 'Commanding Officer', 'CO'),
    ('syn', 'underskirts', 'SLIPS'),
    ('def', 'bloomers', 'COWSLIPS'),
    # 2d MANAGER
    ('ind', 'Naked', 'deletion'),
    ('ind', 'dancing', 'anagram'),
    ('def', "Michael O'Neill?", 'MANAGER'),
    ('def', 'Michael ONeill', 'MANAGER'),
    # 4d PARLIAMENT
    ('syn', 'Train', 'PARENT'),
    ('syn', 'post', 'MAIL'),
    ('ind', 'carrying', 'container'),
    ('ind', 'around', 'reversal'),
    ('def', 'assembly', 'PARLIAMENT'),
    # 5d RAPT
    ('def', 'Fascinated', 'RAPT'),
    # 6d NORTHCAROLINA
    ('ind', 'tipsy', 'anagram'),
    ('def', 'State', 'NORTHCAROLINA'),
    # 7d INSTANT
    ('abbr', 'insurance', 'INS'),
    ('syn', 'worker', 'ANT'),
    ('abbr', 'Thailand', 'T'),
    ('ind', 'touring', 'container'),
    ('def', 'Second', 'INSTANT'),
    # 8d ELDEST
    ('syn', 'The Spanish', 'EL'),
    ('abbr', 'duke', 'D'),
    ('abbr', 'established', 'EST'),
    ('def', 'most senior', 'ELDEST'),
    # 10d SEASONTICKETS
    ('ind', 'involved', 'anagram'),
    ('def', 'travel passes', 'SEASONTICKETS'),
    # 14d UNSCHOOLED
    ('syn', 'A', 'UN'),
    ('syn', 'a', 'UN'),
    ('syn', 'local group', 'SCHOOL'),
    ('abbr', 'education', 'ED'),
    ('def', 'illiterate', 'UNSCHOOLED'),
    # 16d ISOLATED
    ('abbr', 'India', 'I'),
    ('syn', 'very', 'SO'),
    ('syn', 'happy', 'ELATED'),
    ('abbr', 'England', 'E'),
    ('ind', 'to leave', 'deletion'),
    ('def', 'cut off', 'ISOLATED'),
    # 18d OTALGIA
    ('syn', 'Longing', 'NOSTALGIA'),
    ('abbr', 'partners', 'NS'),
    ('def', 'source of pain', 'OTALGIA'),
    # 19d INTROIT
    ('syn', 'Fashionable', 'IN'),
    ('syn', 'communist', 'TROT'),
    ('abbr', 'one', 'I'),
    ('ind', 'collects', 'container'),
    ('def', 'piece of music', 'INTROIT'),
    # 20d MIDRIB
    ('syn', 'Note', 'MI'),
    ('abbr', 'medic', 'DR'),
    ('def', 'vein', 'MIDRIB'),
    # 23d ASIA
    ('ind', 'Regularly', 'alternate'),
    ('def', "approximately 60% of the world's population", 'ASIA'),
    ('def', "60% of the world's population", 'ASIA'),
]

for chk in CHECKS:
    kind = chk[0]
    if kind == 'syn':
        _, w, target = chk
        r = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (w, target)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (target, w)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  SYN  {w!r:35} <-> {target!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'def':
        _, d, a = chk
        r = ref.execute("SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=LOWER(?) AND UPPER(answer)=UPPER(?) LIMIT 1", (d, a)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (d, a)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  DEF  {d!r:50} -> {a!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'abbr':
        _, w, letters = chk
        r = ref.execute("SELECT 1 FROM wordplay WHERE LOWER(indicator)=LOWER(?) AND UPPER(substitution)=UPPER(?) LIMIT 1", (w, letters)).fetchone()
        r2 = ref.execute("SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=LOWER(?) AND UPPER(synonym)=UPPER(?) LIMIT 1", (w, letters)).fetchone()
        ok = r is not None or r2 is not None
        print(f"  ABBR {w!r:35} -> {letters!r}: {'OK' if ok else 'MISS'}")
    elif kind == 'ind':
        _, w, wt = chk
        r = ref.execute("SELECT 1 FROM indicators WHERE LOWER(word)=LOWER(?) AND LOWER(wordplay_type)=LOWER(?) LIMIT 1", (w, wt)).fetchone()
        ok = r is not None
        print(f"  IND  {w!r:35} [{wt}]: {'OK' if ok else 'MISS'}")
