"""Honest parses for DM 17878 leftovers, derived solo (no blog).

Each entry: (clue_id, wordplay_type, definition, ai_explanation).
"""

CLUES = [
    # 1a BURKINAFASO (7,4) -- "Four banks I suspect limiting a West African country"
    # anagram of FOUR BANKS I [suspect] containing A [limiting]
    (10065967, 'container', 'West African country',
     'anagram of FOUR BANKS I [anagram: "suspect"] containing A (from clue) [container: "limiting"] = BURKINAFASO; definition: "West African country"'),

    # 9a ELAPSES (7) -- "Goes by gap in centre for test"
    # ES (middle of test) wrapping LAPSE (gap)
    (10065968, 'container', 'Goes by',
     'ES (middle letters of "test") [parts: "centre"] containing LAPSE (synonym="gap") [container: "in"] = ELAPSES; definition: "Goes by"'),

    # 10a OUTSELL (7) -- "Do better than a commercial rival in publicised period ignoring pressure"
    # OUT (publicised) + SELL, where SELL = SPELL (period) minus P (pressure)
    (10065969, 'charade', 'Do better than a commercial rival',
     'OUT (synonym="publicised") + SELL (deletion="SPELL", P (abbreviation="pressure") dropped, SPELL synonym="period") [deletion: "ignoring"] = OUTSELL; definition: "Do better than a commercial rival"'),

    # 11a URN (3) -- "Thing to hold flowers in your neighbourhood"
    # hidden in "yoUR Neighbourhood"
    (10065970, 'hidden', 'Thing to hold flowers',
     'URN (= hidden in "yoUR Neighbourhood") [hidden: "in"] = URN; definition: "Thing to hold flowers"'),

    # 12a FLOUNCE (7) -- "Frill in full oddly with little weight"
    # FL (odd letters of "full") + OUNCE (little weight)
    (10065971, 'charade', 'Frill',
     'FL (odd letters of "full") [parts: "oddly"] + OUNCE (synonym="little weight") = FLOUNCE; definition: "Frill"'),

    # 13a EVENING (7) -- "Figure missing sun in good part of the day"
    # EVEN (SEVEN - S) + IN + G = EVEN + IN + G
    (10065972, 'charade', 'part of the day',
     'EVEN (deletion="SEVEN", S (abbreviation="sun") dropped, SEVEN synonym="Figure") [deletion: "missing"] + IN (from clue) + G (abbreviation="good") = EVENING; definition: "part of the day"'),

    # 14a YES (3) -- "Sign of positivity in envoy, essentially"
    # hidden in "envoY ESsentially"
    (10065973, 'hidden', 'Sign of positivity',
     'YES (= hidden in "envoY ESsentially") [hidden: "in"] = YES; definition: "Sign of positivity"'),

    # 15a AWFUL (5) -- "Terrible fine facing university breaking law somehow"
    # anagram of F + U + LAW; fodder claim non-contiguous so anagram_source
    # check fails -- expected verifier limit
    (10065974, 'anagram', 'Terrible',
     'AWFUL (anagram="fine university law") [anagram: "somehow"]; F (abbreviation="fine") + U (abbreviation="university") + LAW (from clue) = AWFUL; definition: "Terrible"'),

    # 17a SIGHT (5) -- "Mention location for spectacle"
    # SIGHT sounds like SITE (location)
    (10065975, 'homophone', 'spectacle',
     'SIGHT sounds like SITE (synonym="location") [homophone: "Mention"] = SIGHT; definition: "spectacle"'),

    # 18a EARLS (5) -- "Expensive diamonds stolen, loss outwardly for titled figures"
    # DEAR (expensive) - D (diamonds) = EAR, then LS (outer letters of loss, outwardly)
    (10065976, 'charade', 'titled figures',
     'EAR (deletion="DEAR", D (abbreviation="diamonds") dropped, DEAR synonym="expensive") [deletion: "stolen"] + LS (outer letters of "loss") [parts: "outwardly"] = EARLS; definition: "titled figures"'),

    # 20a RAWER (5) -- "More intense bishop full of wonderment"
    # RR (Bishop = Right Reverend) containing AWE (wonderment) = R-AWE-R
    (10065977, 'container', 'More intense',
     'RR (abbreviation="bishop") containing AWE (synonym="wonderment") [container: "full of"] = RAWER; definition: "More intense"'),

    # 22a EAR (3) -- "Musical aptitude among learners"
    # hidden in "lEARners"
    (10065978, 'hidden', 'Musical aptitude',
     'EAR (= hidden in "lEARners") [hidden: "among"] = EAR; definition: "Musical aptitude"'),

    # 24a CECILIA (7) -- "In France, this trouble returns for woman"
    # CECI (this in French) + LIA (reversal of AIL = trouble)
    (10065979, 'charade', 'woman',
     'CECI (synonym="In France, this") + LIA (reversal of "AIL"), AIL synonym="trouble" [reversal: "returns"] = CECILIA; definition: "woman"'),

    # 25a LIMPEST (7) -- "Slim pet given treatment is most weak"
    # anagram of "Slim pet"
    (10065980, 'anagram', 'most weak',
     'LIMPEST (anagram="Slim pet") [anagram: "treatment"] = LIMPEST; definition: "most weak"'),

    # 26a MOO (3) -- "Sound from Hereford?"
    # Cryptic definition: Hereford is a breed of cattle; cattle moo
    (10065981, 'cryptic_definition', 'Sound from Hereford',
     'cryptic definition: Hereford is a breed of cattle; the sound a cow makes is MOO; definition: "Sound from Hereford"'),

    # 27a PROVIDE (7) -- "Supply liquid over dip"
    # anagram of "over dip" = PROVIDE
    (10065982, 'anagram', 'Supply',
     'PROVIDE (anagram="over dip") [anagram: "liquid"] = PROVIDE; definition: "Supply"'),

    # 28a GIVEOFF (4,3) -- "Discharge chap around four"
    # GEOFF (chap) containing IV (four) = G+IV+EOFF = GIVEOFF
    (10065983, 'container', 'Discharge',
     'GEOFF (synonym="chap") containing IV (roman_numeral="four") [container: "around"] = GIVEOFF; definition: "Discharge"'),

    # 29a SCEPTICALLY (11) -- "Accept silly novel in a doubtful way"
    # anagram of "Accept silly"
    (10065984, 'anagram', 'in a doubtful way',
     'SCEPTICALLY (anagram="Accept silly") [anagram: "novel"] = SCEPTICALLY; definition: "in a doubtful way"'),

    # 1d BOACONSTRICTORS (3,12) -- "Rude types holding tacit scorn curiously for heavy-bodied snakes"
    # BOORS containing anagram(TACIT SCORN) = BO-ACONSTRICT-ORS
    (10065985, 'container', 'heavy-bodied snakes',
     'BOORS (synonym="Rude types") containing anagram of TACIT SCORN [container: "holding"; anagram: "curiously"] = BOACONSTRICTORS; definition: "heavy-bodied snakes"'),

    # 2d ROSANNA (7) -- "Rooms regularly linked to a pair of names with a female"
    # ROS (alt letters of rooms) + ANNA (a pair of names, AN+NA palindrome)
    (10065986, 'charade', 'a female',
     'ROS (alternate letters of "rooms") [parts: "regularly"] + ANNA (synonym="a pair of names") = ROSANNA; definition: "a female"'),

    # 3d ISSUE (5) -- "First person to take legal action about society in contentious matter"
    # I + (SUE containing S) = I + S + SUE
    (10065987, 'charade', 'contentious matter',
     'I (synonym="First person") + SSUE, where SUE (synonym="take legal action") containing S (abbreviation="society") [container: "about"] = ISSUE; definition: "contentious matter"'),

    # 4d ATONEBLOW (2,3,4) -- "How a boxer might be felled in single operation?"
    # Cryptic definition: at one blow = in a single operation; a boxer felled by one punch
    (10065988, 'cryptic_definition', 'in single operation',
     'cryptic definition: at one blow means in a single operation; how a boxer is felled with one punch; definition: "in single operation"'),

    # 5d ANTLERS (7) -- "Learnt about origin of special feature of deer"
    # anagram of LEARNT + S (origin of special) - charade of anagram+letter
    (10065989, 'charade', 'feature of deer',
     'ANTLER (anagram="LEARNT") [anagram: "about"] + S (first letter of "special") [first letter: "origin"] = ANTLERS; definition: "feature of deer"'),

    # 6d OPENINGCEREMONY (7,8) -- "Work ordered in emergency on start of major event"
    # OP (work) + IN + EMERGENCY + ON anagrammed (ordered) = OPENINGCEREMONY
    (10065990, 'anagram', 'start of major event',
     'OPENINGCEREMONY (anagram="work in emergency on") [anagram: "ordered"]; OP (abbreviation="work") + IN (from clue) + EMERGENCY (from clue) + ON (from clue) = OPENINGCEREMONY; definition: "start of major event"'),

    # 7d BELFRY (6) -- "Source of rings in service?"
    # Cryptic definition: a belfry is where church bells hang and ring during a service
    (10065991, 'cryptic_definition', 'Source of rings',
     'cryptic definition: a belfry is the part of a church tower where bells hang and ring; "in service" alludes to a church service where bells are rung; definition: "Source of rings"'),

    # 8d SLIGHT (6) -- "Minimal insult"
    # Double definition
    (10065992, 'double_definition', 'Minimal',
     'double definition: Minimal = SLIGHT, insult = SLIGHT; definition: "Minimal"'),

    # 16d FIRMAMENT (9) -- "Company with last word over time in field of activity"
    # FIRM + AMEN + T
    (10065993, 'charade', 'field of activity',
     'FIRM (synonym="Company") + AMEN (synonym="last word") + T (abbreviation="time") = FIRMAMENT; definition: "field of activity"'),

    # 18d EXCEPT (6) -- "Omit former point about outsiders in charge"
    # (EX + PT) containing CE (outer of charge)
    (10065994, 'container', 'Omit',
     'EXPT, where EX (synonym="former") + PT (abbreviation="point"), containing CE (outer letters of "charge") [container: "about"; parts: "outsiders"] = EXCEPT; definition: "Omit"'),

    # 19d SALTIRE (7) -- "Seasoned sailor with fury is cross"
    # SALT + IRE
    (10065995, 'charade', 'cross',
     'SALT (synonym="Seasoned sailor") + IRE (synonym="fury") = SALTIRE; definition: "cross"'),

    # 21d REMOVAL (7) -- "Withdrawing stupidly over meal with no end of spite"
    # anagram of "over meal" minus E (end of spite)
    (10065996, 'anagram', 'Withdrawing',
     'REMOVAL (anagram="OVER MEAL", with E (last letter of "spite") removed) [anagram: "stupidly"; deletion: "with no end of"] = REMOVAL; definition: "Withdrawing"'),

    # 23d RATIFY (6) -- "Approve a supply on the rise on board railway"
    # RY containing (A + TIF reversal of FIT)
    (10065997, 'container', 'Approve',
     'RY (abbreviation="railway") containing ATIF, where ATIF = A (from clue) + TIF (reversal of "FIT"), FIT (synonym="supply") [reversal: "on the rise"; container: "on board"] = RATIFY; definition: "Approve"'),

    # 25d LOGIC (5) -- "Sound reasoning from soldier in short part of canal"
    # LOC (LOCK minus K) containing GI = LO-GI-C
    (10065998, 'container', 'Sound reasoning',
     'LOC (deletion="LOCK", last letter dropped, LOCK synonym="part of canal") [deletion: "short"] containing GI (abbreviation="soldier") [container: "in"] = LOGIC; definition: "Sound reasoning"'),
]
