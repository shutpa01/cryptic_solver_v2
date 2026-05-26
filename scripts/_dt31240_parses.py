"""DT 31240 leftover parses."""

# (clue_id, wordplay_type, definition, explanation)
CLUES = [
    # 1a REPUDIATE (9) - Disown and reject charity inspired by celebrity
    # anagram of AID (charity) + REPUTE (celebrity) = REPUDIATE
    (10066029, 'anagram', 'Disown and reject',
     'AID (synonym="charity") + REPUTE (synonym="celebrity") [anagram: "inspired"] = REPUDIATE; definition: "Disown and reject"'),

    # 6a GUILD (5) - Fellowship in Surrey town missing for daughter
    # GUILDFORD minus FORD = GUILD; FORD = FOR (literal from clue) + D (daughter)
    (10066030, 'deletion', 'Fellowship',
     'GUILD (deletion="GUILDFORD", FORD dropped, FORD = FOR from clue + D abbreviation="daughter", GUILDFORD synonym="Surrey town") [deletion: "missing"] = GUILD; definition: "Fellowship"'),

    # 9a CHORALE (7) - Reportedly pen hymn tune
    # pen = CORAL (sea pen/coral), reportedly = homophone indicator; CORAL sounds like CHORALE
    (10066031, 'homophone', 'hymn tune',
     'CORAL (synonym="pen") [homophone: "reportedly"] = CHORALE; definition: "hymn tune"'),

    # 10a ILLNESS (7) - Ailment caused by inactivity, sister at heart conceded
    (10066032, 'unparsed', 'Ailment',
     'WORDPLAY UNPARSED; definition: "Ailment"'),

    # 11a STING (5) - Swindle Geordie vocalist
    # DD: STING = swindle / STING = Geordie (Newcastle) vocalist
    (10066033, 'double_definition', 'Swindle Geordie vocalist',
     'STING (synonym="Swindle") / STING (synonym="Geordie vocalist"); definition: "Swindle Geordie vocalist"'),

    # 12a TEMPERATE (9) - Calm Greek character salesperson encountered travelling west
    # MET + REP + ETA all reversed (travelling west) = TEM + PER + ATE = TEMPERATE
    (10066034, 'reversal', 'Calm',
     'MET (synonym="encountered") + REP (synonym="salesperson") + ETA (synonym="Greek character") [reversal: "travelling west"] = TEMPERATE; definition: "Calm"'),

    # 13a CONTRACT BRIDGE (8,6) - Card game cut short by dental work
    # CONTRACT (cut short) + BRIDGE (dental work)
    (10066035, 'charade', 'Card game',
     'CONTRACT (synonym="cut short") + BRIDGE (synonym="dental work") = CONTRACT BRIDGE; definition: "Card game"'),

    # 20a LAST LAUGH (4,5) - Battling, stall a huge, almost final victory
    # anagram of STALL A HUG (huge almost = HUGE minus E) = LAST LAUGH
    (10066037, 'anagram', 'final victory',
     'LAST LAUGH (anagram="STALL A HUG") [deletion: "almost"] [anagram: "Battling"] = LAST LAUGH; definition: "final victory"'),

    # 22a INANE (5) - Repeatedly stripped Mini lad knew to be empty
    # middle letters of mINi + lAd + kNEw = IN + A + NE = INANE
    (10066038, 'charade', 'to be empty',
     'IN (middle letters of "Mini") + A (middle letter of "lad") + NE (middle letters of "knew") [deletion: "stripped"] = INANE; definition: "to be empty"'),

    # 23a STAND-IN (5-2) - Substitute Stockport's wingers plus one close to exhaustion
    # S+T (wingers of Stockport) + AND (plus) + I (one) + N (close to exhaustion)
    (10066039, 'charade', 'Substitute',
     'ST (outer letters of "Stockport") [parts: "wingers"] + AND (synonym="plus") + I (synonym="one") + N (last letter of "exhaustion") [last letter: "close to"] = STAND-IN; definition: "Substitute"'),

    # 24a INVOICE (7) - Bill, able to sing well
    # DD: INVOICE = bill / IN VOICE = able to sing well
    (10066040, 'double_definition', 'Bill, able to sing well',
     'INVOICE (synonym="Bill") / INVOICE (synonym="able to sing well"); definition: "Bill, able to sing well"'),

    # 25a DRYER (5) - Right to stop one staining towel perhaps
    # R (right) inside DYER (staining) = DRYER
    (10066041, 'container', 'towel perhaps',
     'DYER (synonym="one staining") containing R (abbreviation="right") [container: "to stop"] = DRYER; definition: "towel perhaps"'),

    # 26a ENTERTAIN (9) - Consider record that isn't regularly chosen
    # ENTER (record) + TAIN (alternate letters of "that isn't": T,A,I,N) = ENTERTAIN
    (10066042, 'charade', 'Consider',
     'ENTER (synonym="record") + TAIN (alternate letters of "that isn\'t") [alternating: "regularly"] = ENTERTAIN; definition: "Consider"'),

    # 1d RECESS (6) - Trump's break on ship circling Crete, needing vacation
    # def = "Trump's break"; RE (on/about) + CE (outer letters of Crete) + SS (ship) = RECESS
    (10066043, 'charade', "Trump's break",
     'RE (synonym="on") + CE (outer letters of "Crete") [deletion: "needing vacation"] + SS (abbreviation="ship") = RECESS; definition: "Trump\'s break"'),

    # 2d PROVISO (7) - Reservation in Barbados Ivor ponders over
    # PROVISO hidden reversed in "Barbados Ivor ponders" — verifier cannot handle hidden+reversal
    (10066044, 'unparsed', 'Reservation',
     'WORDPLAY UNPARSED; definition: "Reservation"'),

    # 3d DRAUGHT EXCLUDER (7,8) - Maybe one refusing beer that reduces wind internally?
    # cryptic definition
    (10066045, 'cryptic_definition', 'Maybe one refusing beer',
     'WORDPLAY UNPARSED; definition: "Maybe one refusing beer"'),

    # 5d ENIGMATIC (9) - Puzzling item in a broadcast about golf clubs
    (10066047, 'unparsed', 'Puzzling',
     'WORDPLAY UNPARSED; definition: "Puzzling"'),

    # 6d GOLDEN RETRIEVER (6,9) - Go back upset across island after information on adopting elderly dog
    # GOLDEN: GEN (information) containing OLD (elderly) = GOLDEN
    # RETRIEVER: unparsed
    (10066048, 'unparsed', 'dog',
     'GOLDEN: GEN (synonym="information") containing OLD (synonym="elderly") [container: "adopting"] = GOLDEN; RETRIEVER: WORDPLAY UNPARSED; definition: "dog"'),

    # 7d ICELAND (7) - Country retail store
    # DD: ICELAND = country / ICELAND = retail store
    (10066049, 'double_definition', 'Country retail store',
     'ICELAND (synonym="Country") / ICELAND (synonym="retail store"); definition: "Country retail store"'),

    # 8d DISHEVEL (8) - Mess up food course First Lady left
    # DISH (food course) + EVE (First Lady) + L (left) = DISHEVEL
    (10066050, 'charade', 'Mess up',
     'DISH (synonym="food course") + EVE (synonym="First Lady") + L (abbreviation="left") = DISHEVEL; definition: "Mess up"'),

    # 14d AFFLUENCE (9) - Wasting hour, naff clue he composed for "wealth"
    # anagram of NAFF + CLUE + HE minus H (wasting hour = H) = AFFLUENCE
    (10066051, 'anagram', 'wealth',
     'AFFLUENCE (anagram="NAFF CLUE HE") [deletion: "Wasting hour"] [anagram: "composed"] = AFFLUENCE; definition: "wealth"'),

    # 15d IDOLISED (8) - Suspect Lois died greatly admired
    # anagram of LOIS + DIED = IDOLISED
    (10066052, 'anagram', 'greatly admired',
     'LOIS + DIED [anagram: "Suspect"] = IDOLISED; definition: "greatly admired"'),

    # 17d TUSCANY (7) - American is able to visit extremely touristy Italian region
    # T (first of touristy) + US (American) + CAN (is able to) + Y (last of touristy) = TUSCANY
    (10066053, 'charade', 'Italian region',
     'T (first letter of "touristy") + US (synonym="American") + CAN (synonym="is able to") + Y (last letter of "touristy") = TUSCANY; definition: "Italian region"'),

    # 18d OCARINA (7) - Regularly coach adroit Noah making musical instrument
    # alternate letters of cOaCh AdRoIt NoAh = O,C,A,R,I,N,A = OCARINA
    (10066054, 'alternating', 'musical instrument',
     'OCARINA (alternate letters of "coach adroit Noah") [alternating: "Regularly"] = OCARINA; definition: "musical instrument"'),

    # 19d SEVERN (6) - Banker always found amongst Poles
    # S (south pole) + EVER (always) + N (north pole) = SEVERN; Poles = S+N
    (10066055, 'charade', 'Banker',
     'S (abbreviation="Poles") + EVER (synonym="always") + N (abbreviation="Poles") = SEVERN; definition: "Banker"'),

    # 21d HOIST (5) - Raise army to defend garrisons essentially
    # HOST (army) containing I (middle letter of garrisons) = HOIST
    (10066056, 'container', 'Raise',
     'HOST (synonym="army") containing I (middle letter of "garrisons") [container: "to defend"] = HOIST; definition: "Raise"'),
]
