"""Honest parses for DT 31239 leftovers, derived solo (no blog).

Each entry: (clue_id, wordplay_type, definition, ai_explanation).

Built without a blog compass. Where the wordplay genuinely couldn't be
decoded confidently, wtype='unparsed' with explanation noting why -- NOT
a fake CD. Known verifier limits (charade-positional indicators,
compound deletion/container shapes) lead to honest LOW/FAIL.
"""

CLUES = [
    # 1a BALLPOINT PENS (10,4) -- "Stalls behind Globe stage for writers"
    # BALL (Globe) + POINT (stage) + PENS (Stalls), "behind" as positional
    (10065756, 'charade', 'writers',
     'BALL (synonym="Globe") + POINT (synonym="stage") + PENS (synonym="Stalls") [parts: "behind"] = BALLPOINTPENS; definition: "writers"'),

    # 9a ETRUSCANS (9) -- "English PM Liz inspiring May and old Italians"
    # E + TRUS(CAN)S; clean inner pieces so word-by-word shows letters
    (10065757, 'container', 'old Italians',
     'E (abbreviation="English") + TRUSS (synonym="PM Liz") containing CAN (synonym="May") [container: "inspiring"] = ETRUSCANS; definition: "old Italians"'),

    # 10a CARGO (5) -- "Goods carried back by yobbo gracelessly"
    # hidden reversed in "yobbO GRACelessly"; anchored
    (10065758, 'hidden_reversed', 'Goods',
     'CARGO (= hidden reversed in "yobbO GRACelessly") [hidden: "carried"; reversal: "back"] = CARGO; definition: "Goods"'),

    # 12a IRIS (4) -- "Husband leaving from Cork perhaps - that's a bloomer"
    # IRISH (from Cork perhaps) minus H (Husband) = IRIS
    (10065760, 'deletion', "that's a bloomer",
     'IRIS (deletion="IRISH", H (abbreviation="Husband") dropped, IRISH synonym="from Cork perhaps") [deletion: "leaving"] = IRIS; definition: "that\'s a bloomer"'),

    # 13a DELI (4) -- "I was in front looking back in food shop"
    # ILED reversed = DELI; ILED = I (from clue) + LED (was in front)
    (10065761, 'reversal', 'food shop',
     'ILED reversed = DELI, where ILED = I (from clue) + LED (synonym="was in front") [reversal: "looking back"]; definition: "food shop"'),

    # 15a RIBCAGE (7) -- "Make fun of Oscar winner's bony frame"
    # RIB (Make fun of) + CAGE (Oscar winner = Nicolas Cage) + 's possessive
    (10065762, 'charade', 'bony frame',
     'RIB (synonym="Make fun of") + CAGE (synonym="Oscar winner") + S (from clue) = RIBCAGE; definition: "bony frame"'),

    # 17a GROUNDS (7) -- "Areas for e.g. football and golf, games of golf"
    # G (golf NATO) + ROUNDS (games of golf)
    (10065763, 'charade', 'Areas for e.g. football',
     'G (abbreviation="golf") + ROUNDS (synonym="games of golf") = GROUNDS; definition: "Areas for e.g. football"'),

    # 18a GUM TREE (3,4) -- "Greet criminal, admitting hesitation? One's in a sticky situation up here"
    # anagram(GREET) containing UM (hesitation) -- compound
    (10065764, 'container', "One's in a sticky situation up here",
     'anagram of GREET containing UM (synonym="hesitation") [anagram: "criminal"; container: "admitting"] = GUMTREE; definition: "One\'s in a sticky situation up here"'),

    # 20a VENISON (7) -- "Meat in oven's ruined"
    # piece-source form so assembly verifies
    (10065765, 'anagram', 'Meat',
     'VENISON (anagram="in oven\'s") [anagram: "ruined"] = VENISON; definition: "Meat"'),

    # 21a NOSE (4) -- "Perhaps trunk and niece's case will contain old seconds"
    # NE (outer of "niece") containing O+S
    (10065766, 'container', 'Perhaps trunk',
     'NE (outer letters of "niece") [parts: "case"] containing O (abbreviation="old") + S (abbreviation="seconds") [container: "contain"] = NOSE; definition: "Perhaps trunk"'),

    # 22a ACER (4) -- "Shrub expert's seen by river"
    # ACE (expert) + S (from clue, possessive) + R (river)? No: ACE+R only fits ACER, no S
    # Actually ACER is 4 letters: A-C-E-R = ACE + R. "expert's" possessive 's is part
    # of clue surface. Mark "seen" as charade joiner indicator.
    (10065767, 'charade', 'Shrub',
     'ACE (synonym="expert") + R (abbreviation="river") [parts: "seen by"] = ACER; definition: "Shrub"'),

    # 23a ALTER (5) -- "Change key on returning"
    # ALT (key) + RE reversed (on returning) = ALT + ER
    (10065768, 'charade', 'Change',
     'ALT (synonym="key") + ER (reversal of "RE"), where RE (synonym="on") [reversal: "returning"] = ALTER; definition: "Change"'),

    # 26a FRAIL (5) -- "Female water bird is vulnerable"
    # F (Female) + RAIL (water bird)
    (10065769, 'charade', 'vulnerable',
     'F (abbreviation="Female") + RAIL (synonym="water bird") = FRAIL; definition: "vulnerable"'),

    # 27a PRISONERS (9) -- "These people rioting occasionally, uproar and sirens?"
    # anagram of POR (alternate letters of "uproar") + SIRENS = PRISONERS
    (10065770, 'anagram', 'These people',
     'anagram of POR SIRENS [anagram: "rioting"], POR (alternate letters of "uproar") [parts: "occasionally"] = PRISONERS; definition: "These people"'),

    # 28a TRANSPORT CAFE (9,4) -- "Strong emotion about fine European eatery"
    # TRANSPORT + CA (about) + F (fine) + E (European)
    (10065771, 'charade', 'eatery',
     'TRANSPORT (synonym="Strong emotion") + CA (abbreviation="about") + F (abbreviation="fine") + E (abbreviation="European") = TRANSPORTCAFE; definition: "eatery"'),

    # 1d BEEF STROGANOFF (4,9) -- "Forget beans, sadly unavailable, and get another dish"
    # BEEFSTROGAN (anagram fodder = "Forget beans") + OFF (unavailable)
    (10065772, 'charade', 'another dish',
     'BEEFSTROGAN (anagram="Forget beans") [anagram: "sadly"] + OFF (synonym="unavailable") = BEEFSTROGANOFF; definition: "another dish"'),

    # 2d LOREN (5) -- "Actress learning lines, essentially"
    # LORE (learning) + N (middle letter of "lines")
    (10065773, 'charade', 'Actress',
     'LORE (synonym="learning") + N (middle letter of "lines") [parts: "essentially"] = LOREN; definition: "Actress"'),

    # 3d PUSHCHAIRS (10) -- "Promote president's vehicles for small charges"
    # PUSH (Promote) + CHAIR (president) + S (possessive)
    (10065774, 'charade', 'vehicles for small charges',
     'PUSH (synonym="Promote") + CHAIR (synonym="president") + S (from clue) = PUSHCHAIRS; definition: "vehicles for small charges"'),

    # 4d IMAGINE (7) -- "Proclaim a gin excellent, somewhat fancy"
    # hidden in "proclaIM A GIN Excellent"; anchored
    (10065775, 'hidden', 'fancy',
     'IMAGINE (= hidden in "proclaIM A GIN Excellent") [hidden: "somewhat"] = IMAGINE; definition: "fancy"'),

    # 5d TASTING (7) -- "Sampling T'Pau now and then, and Police singer"
    # TA (alt of T'Pau) + STING (Sting from Police)
    (10065776, 'charade', 'Sampling',
     'TA (alternate letters of "T\'Pau") [parts: "now and then"] + STING (synonym="Police singer") = TASTING; definition: "Sampling"'),

    # 6d ETCH (4) -- "Cut, grumpy having lost skin"
    # TETCHY (grumpy) minus outer letters
    (10065777, 'deletion', 'Cut',
     'ETCH (deletion="TETCHY", outer letters dropped, TETCHY synonym="grumpy") [deletion: "having lost skin"] = ETCH; definition: "Cut"'),

    # 7d SERGEANTS (9) -- "Officers get near waves on board ship"
    # SS (ship) containing anagram(get near) [waves; on board]
    (10065778, 'container', 'Officers',
     'SS (abbreviation="ship") containing anagram of GET NEAR [anagram: "waves"; container: "on board"] = SERGEANTS; definition: "Officers"'),

    # 8d ROBINSON CRUSOE (8,6) -- "One meeting Friday in work?"
    # Cryptic definition: Robinson Crusoe meets Man Friday in Defoe's novel
    (10065779, 'cryptic_definition', 'One meeting Friday in work',
     'cryptic definition: One meeting Friday in work -- Robinson Crusoe meets Man Friday in Defoe\'s novel; definition: "One meeting Friday in work"'),

    # 14d ROUNDABOUT (10) -- "Fairground Attraction, band beginning to argue and fight"
    # ROUND (band) + A (first of argue) + BOUT (fight)
    (10065780, 'charade', 'Fairground Attraction',
     'ROUND (synonym="band") + A (first letter of "argue") [parts: "beginning"] + BOUT (synonym="fight") = ROUNDABOUT; definition: "Fairground Attraction"'),

    # 16d BUMP-START (4-5) -- "First of breakdowns and Trump sat fidgeting, car needing this?"
    # B (first of breakdowns) + anagram(Trump sat) [fidgeting]
    (10065781, 'charade', 'car needing this',
     'B (first letter of "breakdowns") [parts: "First of"] + UMPSTART (anagram="Trump sat") [anagram: "fidgeting"] = BUMPSTART; definition: "car needing this"'),

    # 19d ESCAPES (7) -- "Bolts engineer sees with cap"
    # piece-source (fodder words separated by "with" -- anagram_source check
    # fails contiguity but assembly verifies)
    (10065782, 'anagram', 'Bolts',
     'ESCAPES (anagram="sees cap") [anagram: "engineer"] = ESCAPES; definition: "Bolts"'),

    # 20d VERTIGO (7) -- "Picture of yogi Trevor's holding up"
    # hidden reversed in "yOGI TREVor's"; anchored
    (10065783, 'hidden_reversed', 'Picture',
     'VERTIGO (= hidden reversed in "yOGI TREVor\'s") [hidden: "holding"; reversal: "up"] = VERTIGO; definition: "Picture"'),

    # 24d THETA (5) -- "Two articles absorbing tense character from Athens"
    # THE + A absorbing T = THE-T-A
    (10065784, 'container', 'character from Athens',
     'THE (synonym="Two articles") + A (synonym="Two articles") containing T (abbreviation="tense") [container: "absorbing"] = THETA; definition: "character from Athens"'),

    # 25d FLEA (4) -- "Small jumper, feminine, bound to be too short"
    # F (feminine) + LEA (LEAP minus last letter)
    (10065785, 'charade', 'Small jumper',
     'F (abbreviation="feminine") + LEA (deletion="LEAP", last letter dropped, LEAP synonym="bound") [deletion: "to be too short"] = FLEA; definition: "Small jumper"'),
]
