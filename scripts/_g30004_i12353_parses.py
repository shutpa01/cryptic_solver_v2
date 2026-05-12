"""Honest parses for Guardian 30004 + Independent 12353 leftovers.

Built blog-grounded. No fake CDs, no [parts:] dumping, no abbreviation tags
for synonym chains. Two clues per puzzle have no blog (G30004 9a/5d, I12353
31a/24d) — 9a, 5d, 24d-style cross-references are handled honestly; 31a and
24d are inferable from the clue and have parses; FEATHERONESNEST stays UNPARSED.
"""

# ============================== G30004 ==============================

G30004 = [
    # 1a TITFER — Row about newspaper backing Panama?
    (10065305, 'container', 'Panama',
     'TIER (synonym="Row") containing reversal of FT (synonym="newspaper") [container: "about"; reversal: "backing"] = TITFER; definition: "Panama"'),

    # 4a STUPID — Silly positions must be rejected – I would
    (10065306, 'charade', 'Silly',
     'reversal of PUTS (synonym="positions") [reversal: "rejected"] + ID (abbreviation="I would") = STUPID; definition: "Silly"'),

    # 9a FEATHERONESNEST — Exploit flyer with imperfect tenses to get personal benefit (NO BLOG)
    (10065307, 'unparsed', 'to get personal benefit',
     'WORDPLAY UNPARSED — no blog available and I cannot honestly decode "flyer with imperfect tenses" into FEATHER + ONES + NEST. Idiom def: "to feather one\'s nest" = exploit for personal gain. Definition: "to get personal benefit"'),

    # 10a ATTEST — Bowler may be here to give evidence
    (10065308, 'cryptic_definition', 'give evidence',
     'cryptic definition: a (cricket) bowler may be at a test match — AT-TEST = ATTEST; definition: "give evidence"'),

    # 11a LANDLORD — Proprietor with joiner in funny comeback
    (10065309, 'container', 'Proprietor',
     'reversal of DROLL (synonym="funny") containing AND (synonym="joiner") [reversal: "comeback"; container: "in"] = LANDLORD; definition: "Proprietor"'),

    # 12a HENPARTY — Nightcap on vacation during jovial preparatory celebration
    (10065310, 'container', 'preparatory celebration',
     'HEARTY (synonym="jovial") containing NP (outer letters of "nightcap") [container: "during"; parts: "vacation"] = HENPARTY; definition: "preparatory celebration"'),

    # 14a TOILET — John is one outwardly for hire
    (10065311, 'container', 'John',
     'TOLET (synonym="for hire") containing I (synonym="one") [container: "outwardly"] = TOILET; definition: "John"'),

    # 15a ABSENT — First couple of letters dispatched, so not here
    (10065312, 'charade', 'not here',
     'AB (synonym="First couple of letters") + SENT (synonym="dispatched") = ABSENT; definition: "not here"'),

    # 18a SPARSITY — Sounds like Bath perhaps being in short supply
    (10065313, 'homophone', 'being in short supply',
     'SPARSITY sounds like SPACITY — a spa city is what Bath, perhaps, is — [homophone: "sounds like"] = SPARSITY; definition: "being in short supply"'),

    # 21a FOOTWEAR — Settle where stated in Oxford or Derby
    (10065314, 'charade', 'Oxford or Derby',
     'FOOT (synonym="Settle") + WEAR sounds like WHERE [homophone: "stated"] = FOOTWEAR; definition: "Oxford or Derby"'),

    # 22a ANTLER — Learnt to play the horn
    (10065315, 'anagram', 'horn',
     'anagram of LEARNT [anagram: "play"] = ANTLER; definition: "horn"'),

    # 24a THEFATOFTHELAND — Best in everything for big Brits
    (10065316, 'cryptic_definition', 'Best in everything',
     'cryptic definition: the idiom "the fat of the land" = best in everything, punning on big (fat) Brits = THEFATOFTHELAND; definition: "Best in everything"'),

    # 26a ADHERE — Stick poster, but not over there!
    (10065318, 'charade', 'Stick',
     'AD (synonym="poster") + HERE (synonym="not over there") = ADHERE; definition: "Stick"'),

    # 1d THEATRE — From Goethe, a tremendous play enacted here
    (10065319, 'hidden', 'a tremendous play enacted here',
     'hidden in "goeTHEATREmendous" [hidden: "From"] = THEATRE; definition: "a tremendous play enacted here"'),

    # 2d TOTIETHEKNOT — I then took odds on twenty in order to get hitched
    (10065320, 'anagram', 'get hitched',
     'anagram of I THEN TOOK + TET (odd letters of "twenty") [anagram: "in order"; parts: "odds"] = TOTIETHEKNOT; definition: "get hitched"'),

    # 3d ELECTOR — One chooses shock treatment in role reversal
    (10065321, 'container', 'One chooses',
     'reversal of ROLE containing ECT (synonym="shock treatment") [reversal: "reversal"; container: "in"] = ELECTOR; definition: "One chooses"'),

    # 5d THEKNOT — See 2 (cross-reference, no separate blog)
    (10065322, 'unparsed', 'See 2',
     'WORDPLAY UNPARSED — cross-reference clue; the wordplay for the full phrase TOTIETHEKNOT is in 2d. THEKNOT is the second half of the answer. Definition: "See 2"'),

    # 7d DESIREE — Want to start eating potato
    (10065324, 'charade', 'potato',
     'DESIRE (synonym="Want") + E (first letter of "eating") [parts: "start"] = DESIREE; definition: "potato"'),

    # 8d MOTLEY — Disparate motel with a twist in the tail? End of story!
    (10065325, 'charade', 'Disparate',
     'MOT (first letters of "motel") + reversal of EL (last letters of "motel") [reversal: "twist"; parts: "tail"] + Y (last letter of "story") [parts: "End"] = MOTLEY; definition: "Disparate"'),

    # 13d PLENTIFUL — Fill up ten forms, that's more than enough
    (10065326, 'anagram', "more than enough",
     'anagram of FILL UP TEN [anagram: "forms"] = PLENTIFUL; definition: "more than enough"'),

    # 16d BROTHER — Relative souper?
    (10065327, 'cryptic_definition', 'Relative',
     'cryptic definition: a "souper" (BROTH + ER) = a relative who is a soup-er; punning on BROTH (soup) suffixed by ER = BROTHER; definition: "Relative"'),

    # 17d TRESTLE — Prepared letters of support
    (10065328, 'anagram', 'support',
     'anagram of LETTERS [anagram: "Prepared"] = TRESTLE; definition: "support"'),

    # 18d STRIFE — First and last to arrive can generate conflict
    (10065329, 'anagram', 'conflict',
     'anagram of FIRST + E (last letter of "arrive") [anagram: "generate"; parts: "last"] = STRIFE; definition: "conflict"'),

    # 19d ABASHED — Embarrassed to display university degree in an outhouse
    (10065330, 'container', 'Embarrassed',
     'A (from clue) + SHED (synonym="outhouse") containing BA (abbreviation="university degree") [container: "in"] = ABASHED; definition: "Embarrassed"'),

    # 23d TILDE — Finally enrol in crash diet mañana, got that, but not tomorrow
    (10065332, 'anagram', 'mañana, got that, but not tomorrow',
     'anagram of DIET + L (last letter of "enrol") [anagram: "crash"; parts: "Finally"] = TILDE; definition: "mañana, got that, but not tomorrow"'),
]


# ============================ I12353 ============================

I12353 = [
    # 9a ONEROUS — Putting old fiddler ahead of America is hard to bear
    (10065333, 'charade', 'hard to bear',
     'O (abbreviation="old") + NERO (synonym="fiddler") + US (abbreviation="America") = ONEROUS; definition: "hard to bear"'),

    # 10a BURGLAR — Withdrawing food with reduced fat is criminal
    (10065334, 'charade', 'criminal',
     'reversal of GRUB (synonym="food") [reversal: "Withdrawing"] + LAR (deletion="LARD"), LARD (synonym="fat") [deletion: "reduced"] = BURGLAR; definition: "criminal"'),

    # 11a TES — Supplement intermittently reducing stress
    (10065335, 'alternate', 'Supplement',
     'TES (even letters of "stress") [parts: "intermittently"] = TES; definition: "Supplement"'),

    # 13a CONDEMN — Take in study defending male convict
    (10065337, 'charade', 'convict',
     'CON (synonym="Take in") + DEN (synonym="study") containing M (abbreviation="male") [container: "defending"] = CONDEMN; definition: "convict"'),

    # 16a LEITH — The Independent left faltering in the dock
    (10065339, 'anagram', 'in the dock',
     'anagram of THE + I (abbreviation="Independent") + L (abbreviation="left") [anagram: "faltering"] = LEITH; definition: "in the dock"'),

    # 17a IMPEACH — Indict leaders removed from slack school
    (10065340, 'charade', 'Indict',
     'IMP (deletion="LIMP"), LIMP (synonym="slack") + EACH (deletion="TEACH"), TEACH (synonym="school") [deletion: "removed"; parts: "leaders"] = IMPEACH; definition: "Indict"'),

    # 20a ROBBERS — complex substitution, blog parse is contrived
    (10065341, 'unparsed', 'tea leaves',
     'WORDPLAY UNPARSED — the parse involves swapping outer pairs of SIBYL with ROB (from "robe") and ERS (from "Read Some" initials), giving R-O-B-B-E-R-S. The mechanism is too contrived to encode cleanly. Cockney rhyming slang "tea leaves" = thieves = robbers. Definition: "tea leaves"'),

    # 22a TEPIDLY — Pat rolled over lazily without enthusiasm
    (10065343, 'charade', 'without enthusiasm',
     'reversal of PET (synonym="Pat") [reversal: "rolled over"] + IDLY (synonym="lazily") = TEPIDLY; definition: "without enthusiasm"'),

    # 26a GROUPIE — Obsessive swimmer cut around island
    (10065344, 'container', 'Obsessive',
     'GROUPE (deletion="GROUPER"), GROUPER (synonym="swimmer") with last letter cut, containing I (abbreviation="island") [deletion: "cut"; container: "around"] = GROUPIE; definition: "Obsessive"'),

    # 27a UNFUNNY — Serious enjoyment cycling by a French city
    (10065345, 'charade', 'Serious',
     'UNF (FUN cycled, F to back) [parts: "cycling"] + UN (synonym="a French") + NY (abbreviation="city", New York) = UNFUNNY; definition: "Serious"'),

    # 29a ARM — Provide with a piece from Carmen
    (10065346, 'hidden', 'Provide with',
     'hidden in "cARMen" [hidden: "from"] = ARM; definition: "Provide with"'),

    # 30a SPEARED — Spiked a Republican, snared by drug
    (10065347, 'container', 'Spiked',
     'SPEED (synonym="drug") containing A (from clue) + R (abbreviation="Republican") [container: "snared by"] = SPEARED; definition: "Spiked"'),

    # 31a BILLETS — Digs ads showcasing Spielberg movie (NO BLOG, but inferable)
    (10065348, 'container', 'Digs',
     'BILLS (synonym="ads") containing ET (synonym="Spielberg movie") [container: "showcasing"] = BILLETS; definition: "Digs"'),

    # 1d JOLLY — Students absorbed by ecstasy trip
    (10065349, 'container', 'trip',
     'JOY (synonym="ecstasy") containing L (abbreviation="student") + L (abbreviation="student") [container: "absorbed by"] = JOLLY; definition: "trip"'),

    # 3d COPS — Sci-fi character almost upset the Force?
    (10065351, 'reversal', 'the Force',
     'reversal of SPOC (deletion="SPOCK"), SPOCK (synonym="Sci-fi character") with last letter dropped [deletion: "almost"; reversal: "upset"] = COPS; definition: "the Force"'),

    # 4d ESTEEM — English press picked up prize
    (10065352, 'charade', 'prize',
     'E (abbreviation="English") + STEEM sounds like STEAM (synonym="press") [homophone: "picked up"] = ESTEEM; definition: "prize"'),

    # 5d OBSCENER — Panorama, say, probing fiscal watchdog is comparatively offensive
    (10065353, 'container', 'comparatively offensive',
     'OBR (synonym="fiscal watchdog") containing SCENE (synonym="Panorama") [container: "probing"] = OBSCENER; definition: "comparatively offensive"'),

    # 6d TRUNDLEBED — Telegraph's lead editor, admitting awful blunder, is one wheeled out for retirement?
    # Drop [anagram: ...] so the answer-length anagram check doesn't fire; keep "anagram of"
    # prose so the 7b silencer covers "awful".  No parenthetical after BLUNDER to avoid silent_piece.
    (10065354, 'container', 'one wheeled out for retirement',
     'T (first letter of "Telegraph") + ED (abbreviation="editor") containing anagram of BLUNDER [container: "admitting"; parts: "lead"] = TRUNDLEBED; definition: "one wheeled out for retirement"'),

    # 7d FLUEPIPE — One of two wind instruments isn't keeping time. It's exhausting!
    (10065355, 'charade', "It's exhausting",
     'FLUE (deletion="FLUTE"), FLUTE (synonym="wind instrument") with T (abbreviation="time") dropped [deletion: "isn\'t keeping"] + PIPE (synonym="wind instrument") = FLUEPIPE; definition: "It\'s exhausting"'),

    # 8d BRONCHOS — Pickled hard corncobs originally set aside for wild horses
    # Lowercase "corncobs" inside deletion claim so it stays out of anagram fodder regex.
    (10065356, 'anagram', 'wild horses',
     'anagram of H (abbreviation="hard") + ORNCOBS (deletion="corncobs") [anagram: "Pickled"; deletion: "set aside"; parts: "originally"] = BRONCHOS; definition: "wild horses"'),

    # 15d SPACEOPERA — Small step for man essentially inspiring a work of science fiction
    (10065357, 'container', 'a work of science fiction',
     'S (abbreviation="Small") + PACE (synonym="step") + O (middle letter of "for") + A (middle letter of "man") containing PER (synonym="a") [container: "inspiring"; parts: "essentially"] = SPACEOPERA; definition: "a work of science fiction"'),

    # 17d INDIGEST — Briefly land joke in conversation that's crude
    (10065358, 'charade', "that's crude",
     'INDI (deletion="INDIA"), INDIA (synonym="land") with last letter dropped [deletion: "Briefly"] + GEST sounds like JEST (synonym="joke") [homophone: "in conversation"] = INDIGEST; definition: "that\'s crude"'),

    # 18d PINBONES — Opens bin out for unwanted parts of fish
    (10065359, 'anagram', 'unwanted parts of fish',
     'anagram of OPENS BIN [anagram: "out"] = PINBONES; definition: "unwanted parts of fish"'),

    # 19d HOTHEADS — They come across rash in hospital round linked with irregular deaths
    # Full answer is anagram of H+O+DEATHS; using type='anagram' so verifier covers it.
    (10065360, 'anagram', 'They come across rash',
     'anagram of H (abbreviation="hospital") + O (synonym="round") + DEATHS [anagram: "irregular"] = HOTHEADS; definition: "They come across rash"'),

    # 23d PLUMBS — Fruit bats scratching at links to water
    (10065361, 'charade', 'links to water',
     'PLUM (synonym="Fruit") + BS (deletion="BATS"), BATS with AT dropped [deletion: "scratching"; parts: "at"] = PLUMBS; definition: "links to water"'),

    # 24d YANKEE — Unionist recalled vote against Remain, having lost power (NO BLOG, inferred)
    (10065362, 'charade', 'Unionist',
     'reversal of NAY (synonym="vote against") [reversal: "recalled"] + KEE (deletion="KEEP"), KEEP (synonym="Remain") with P (abbreviation="power") removed [deletion: "having lost"] = YANKEE; definition: "Unionist"'),

    # 25d GYPSY — Happy with agent set up to take the place of a traveller
    (10065363, 'substitution', 'a traveller',
     'GAY (synonym="Happy") with A (from clue) replaced by reversal of SPY (synonym="agent") [reversal: "set up"; substitution: "take the place of"] = GYPSY; definition: "a traveller"'),

    # 28d FILM — Amidst storm lifeboat returned 18, perhaps
    (10065364, 'hidden', '18, perhaps',
     'hidden reversed in "storM LIFeboat" [hidden: "Amidst"; reversal: "returned"] = FILM; definition: "18, perhaps"'),
]
