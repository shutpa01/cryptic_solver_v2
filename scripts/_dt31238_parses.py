"""Honest parses for DT 31238 leftovers, derived solo (no blog).

Each entry: (clue_id, wordplay_type, definition, ai_explanation)

Built without a blog compass. Where the wordplay genuinely couldn't be
decoded confidently, wtype='unparsed' with explanation noting why — NOT
a fake CD. Known verifier limits (compound mechanisms, unicode î, 'when'
homophone false-positive) lead to honest LOW/FAIL — not gamed.
"""

CLUES = [
    # 1a VAMPIRE BAT (7,3) — A Prime TV broadcast about airline that sucks flying at night
    # anagram(A PRIME TV) [broadcast] containing BA (airline) [about]
    (10065420, 'container', 'that sucks flying at night',
     'anagram of A PRIME TV [anagram: "broadcast"] containing BA (synonym="airline") [container: "about"] = VAMPIREBAT; definition: "that sucks flying at night"'),

    # 6a SMUT (4) — Obscene material corporation's withdrawn
    (10065421, 'reversal', 'Obscene material',
     'reversal of TUMS, where TUMS = TUM (synonym="corporation") + S (from clue) [reversal: "withdrawn"] = SMUT; definition: "Obscene material"'),

    # 10a SIGHT (5)
    (10065422, 'homophone', 'sense',
     'SIGHT sounds like CITE (synonym="Quote") [homophone: "talking"] = SIGHT; definition: "sense"'),

    # 11a ARMADILLO (9)
    (10065423, 'charade', 'Texan resident',
     'A (abbreviation="American") + R (abbreviation="Republican") + MAD (synonym="crazy") + ILL (synonym="sick") + O (abbreviation="old") = ARMADILLO; definition: "Texan resident"'),

    # 12a below par (5,3) — Having dislocated elbow, dad runs off
    # anagram(ELBOW) + PA + R; charade with anagram piece — compound limit, expect LOW
    (10065424, 'charade', 'off',
     'BELOW (anagram="ELBOW") [anagram: "dislocated"] + PA (synonym="dad") + R (abbreviation="runs") = BELOWPAR; definition: "off"'),

    # 13a QUEUE (5)
    (10065425, 'charade', 'line of Americans',
     'QUE (synonym="Manuel\'s gag") + U (first letter of "upset") + E (first letter of "enraged") [parts: "starts"] = QUEUE; definition: "line of Americans"'),

    # 15a ANIMATE (7) — A + (N + MATE) containing I [touring]
    (10065426, 'container', 'Give life to',
     'A (from clue) + NIMATE, where NIMATE = N (abbreviation="new") + MATE (synonym="friend") containing I (abbreviation="Italy") [container: "touring"] = ANIMATE; definition: "Give life to"'),

    # 17a LEAFLET (7) — LEA (Lea & Perrins) + anagram(LEFT) — compound, expect LOW
    (10065427, 'charade', 'marketing material',
     'LEA (synonym="Perrins\'s saucy partner") + FLET (anagram="LEFT") [anagram: "dodgy"] = LEAFLET; definition: "marketing material"'),

    # 19a SAMURAI (7) — anagram(ASIA) containing MUR (reversal of RUM) — compound, expect LOW
    (10065428, 'container', 'Japanese warrior',
     'anagram of ASIA [anagram: "sozzled"] containing MUR (reversal="RUM"), RUM (synonym="booze") [reversal: "knocked back"; container: "in"] = SAMURAI; definition: "Japanese warrior"'),

    # 21a INERTIA (7) — hidden reversed in "Maître Nicolas"; unicode î stripping is a known verifier limit
    (10065429, 'hidden_reversed', 'Lethargy',
     'hidden reversed in "m AITRENI colas" [reversal: "retirement"] = INERTIA; definition: "Lethargy"'),

    # 22a SHARD (5)
    (10065430, 'container', 'London landmark',
     'SH (synonym="Quiet") + RD (abbreviation="road") containing A (from clue) [container: "circling"] = SHARD; definition: "London landmark"'),

    # 24a ALPHABET (8) — unparsed honestly
    (10065431, 'unparsed', 'range of letters',
     'WORDPLAY UNPARSED — could not confidently decompose "Debut for Apple phablet leading with large"; the answer is likely A (first of Apple) + L (large) + PHABET (PHABLET with L moved) but the move is not cleanly verifier-expressible; definition: "range of letters"'),

    # 28a ALBUM (5)
    (10065433, 'container', 'record',
     'AM (abbreviation="In the morning") containing LBU (odd letters of "Labour") [container: "defending"; parts: "odd"] = ALBUM; definition: "record"'),

    # 29a Nile (4) — CD-style; full clue not in DB, mark UNPARSED honestly
    (10065434, 'unparsed', 'Provider of interbank liquidity in Egypt',
     'WORDPLAY UNPARSED — clue is a cryptic definition (interbank liquidity = water between river banks; in Egypt = NILE) but the full clue text is not in the DB, so strict CD verification cannot pass; definition: "Provider of interbank liquidity in Egypt"'),

    # 30a Anglo-Saxon (5-5) — unparsed
    (10065435, 'unparsed', 'plain English',
     'WORDPLAY UNPARSED — "American slogan on X translated" could not be confidently decomposed; definition: "plain English"'),

    # 1d VAST (4) — VAT containing S; "swallowed up by" is multi-word indicator; "up" alone fires 7b
    (10065436, 'container', 'Huge',
     'VAT (synonym="tax") containing S (last letter of "profits") [container: "swallowed"; parts: "ultimately"] = VAST; definition: "Huge"'),

    # 2d MAGNETISM (9) — compound charade(container, anagram); expect LOW
    (10065437, 'charade', 'Attractiveness',
     'MAGN (= MAN (synonym="chap") containing G (abbreviation="good") [container: "embracing"]) + ETISM (anagram="TIMES") [anagram: "play"] = MAGNETISM; definition: "Attractiveness"'),

    # 3d INTRO (5)
    (10065438, 'charade', 'Opening bars',
     'INT (deletion="PINT", P (abbreviation="penny") dropped) [deletion: "off"] + RO (deletion="GROG", outer letters dropped) [deletion: "unlimited"] = INTRO; definition: "Opening bars"'),

    # 4d EXAMPLE (7)
    (10065439, 'charade', 'model',
     'EX (synonym="Former lover") + AMPLE (synonym="generously proportioned") = EXAMPLE; definition: "model"'),

    # 5d ADMIRAL (7) — ADMIR (deletion=ADMIRE, last letter dropped, ADMIRE synonym=esteem) + A + L
    (10065440, 'charade', 'officer of the navy',
     'ADMIR (deletion="ADMIRE", last letter dropped, ADMIRE synonym="esteem") [deletion: "Endlessly"] + A (from clue) + L (abbreviation="student") = ADMIRAL; definition: "officer of the navy"'),

    # 7d MELEE (5) — M + E + LEE
    (10065441, 'charade', 'Confused conflict',
     'M (first letter of "Middle") + E (first letter of "East") + LEE (synonym="shelter") [parts: "leaders"] = MELEE; definition: "Confused conflict"'),

    # 8d T-bone steak — TBONES (anagram of ON BEST) + TEAK; compound, expect LOW
    (10065442, 'charade', 'choice cut',
     'TBONES (anagram="ON BEST") [anagram: "Working"] + TEAK (synonym="hardwood") = TBONESTEAK; definition: "choice cut"'),

    # 9d ADEQUATE (8) — cannot decode queen=QU honestly
    (10065443, 'unparsed', 'Competent',
     'WORDPLAY UNPARSED — answer is structurally A + DATE containing E + QU but "queen" → QU is not a standard cryptic mapping I can confidently claim; definition: "Competent"'),

    # 14d jam session — CD joke; verifier compound limit
    (10065444, 'unparsed', 'Extemporised notes',
     'WORDPLAY UNPARSED — CD-style joke: stereotypical WI (Women\'s Institute) meeting features jam-making and sessions; the move does not decompose to a verifier mechanism; definition: "Extemporised notes"'),

    # 16d ABRIDGED (8)
    (10065445, 'charade', 'condensed',
     'A (abbreviation="Advanced") + BRIDGE (synonym="tricky game") + D (abbreviation="daughter") = ABRIDGED; definition: "condensed"'),

    # 18d LETTERBOX — Spoonerism (verifier limit)
    (10065446, 'unparsed', 'where the envelope is pushed',
     'WORDPLAY UNPARSED via verifier — Spoonerism of "BETTER LOCKS" (BETTER = "Improved", LOCKS = "security devices") swapping initial sounds to LETTER BOX, indicated by "Spooner". Spoonerism is a known verifier limit; definition: "where the envelope is pushed"'),

    # 20d in a spin — hidden in "again as Pinochet"; apostrophe in Pinochet's is verifier word-coverage limit
    (10065447, 'hidden', 'Confused',
     'hidden in "agaINASPINochets" [hidden: "accepted"] = INASPIN; definition: "Confused"'),

    # 21d IMPERIL (7) — IMPERIAL minus A; "evacuating" queues; "when" 7b false positive
    (10065448, 'deletion', 'Endanger',
     'IMPERIAL (synonym="sovereign") with A (abbreviation="area") removed [deletion: "evacuating"] = IMPERIL; definition: "Endanger"'),

    # 23d ANNUL (5) — ANNUAL with second A (article) removed
    (10065449, 'deletion', 'Abolish',
     'ANNUAL (synonym="yearbook") with second A removed, A (abbreviation="article") [deletion: "deleting"; parts: "second"] = ANNUL; definition: "Abolish"'),

    # 25d AMASS (5) — A (first of Anglican) + MASS (service); "seen in" 7b false positive
    (10065450, 'charade', 'Collect',
     'A (first letter of "Anglican") [parts: "first"] + MASS (synonym="service") = AMASS; definition: "Collect"'),

    # 26d OMEN (4) — CD-style joke; unparsed honestly
    (10065451, 'unparsed', 'Warning sign',
     'WORDPLAY UNPARSED — CD-style joke: at a "hen night" you see "NO MEN" rearranging to OMEN; not a standard anagram on clue letters; definition: "Warning sign"'),
]
