"""DM 17879 leftover parses."""

CLUES = [
    # 1a LISBURN (7) - Catalogue reduced brand somewhere in Northern Ireland
    # LIST reduced (T dropped) = LIS + BURN (brand) = LISBURN
    (10066242, 'charade', 'somewhere in Northern Ireland',
     'LIS (deletion="LIST", T dropped, LIST synonym="catalogue") [deletion: "reduced"] + BURN (synonym="brand") = LISBURN; definition: "somewhere in Northern Ireland"'),

    # 5a DUCATS (6) - Old coins in area surrounded by pipes
    # A (area) inside DUCTS (pipes) = DUCATS
    (10066243, 'container', 'Old coins',
     'DUCTS (synonym="pipes") containing A (abbreviation="area") [container: "surrounded by"] = DUCATS; definition: "Old coins"'),

    # 9a TAILEND (4,3) - Net laid out in final part of event?
    # anagram of NET LAID = TAILEND; "out" = anagram indicator
    (10066244, 'anagram', 'in final part of event?',
     'TAILEND (anagram="NET LAID") [anagram: "out"] = TAILEND; definition: "in final part of event?"'),

    # 10a SPARKLE (7) - Show liveliness in flash of light
    # DD: SPARKLE = show liveliness / flash of light
    (10066245, 'double_definition', 'Show liveliness in flash of light',
     'SPARKLE (synonym="Show liveliness") / SPARKLE (synonym="flash of light"); definition: "Show liveliness in flash of light"'),

    # 11a LAW (3) - Speaker's learning regulation
    # LORE (learning) sounds like LAW; "speaker" = homophone indicator
    (10066246, 'homophone', 'regulation',
     'LAW sounds like LORE (synonym="learning") [homophone: "speaker"] = LAW; definition: "regulation"'),

    # 12a STAGINGPOST (7,4) - Got past sign needing repair in place to stop en route
    # anagram of GOT PAST SIGN = STAGINGPOST; "needing repair" = anagram indicator
    (10066247, 'anagram', 'in place to stop en route',
     'STAGINGPOST (anagram="GOT PAST SIGN") [anagram: "needing repair"] = STAGINGPOST; definition: "in place to stop en route"'),

    # 13a PATIO (5) - Pressure facing a group right away in area adjoining a house
    # P (pressure) + ATIO (RATIO with R removed, "away" = deletion indicator) = PATIO
    (10066248, 'charade', 'area adjoining a house',
     'P (abbreviation="pressure") + A (from clue) + TIO (deletion="TRIO", R dropped, R abbreviation="right", TRIO synonym="group") [deletion: "away"] = PATIO; definition: "area adjoining a house"'),

    # 14a STAIRWELL (9) - Reportedly, look that's fixed properly in part of a residential block
    # STARE (look) sounds like STAIR + WELL (properly) = STAIRWELL
    (10066249, 'charade', 'part of a residential block',
     'STAIR sounds like STARE (synonym="look that\'s fixed") [homophone: "Reportedly"] + WELL (synonym="properly") = STAIRWELL; definition: "part of a residential block"'),

    # 16a ALTERABLE (9) - Tear label off - typifying some errors?
    # anagram of TEAR LABEL = ALTERABLE; "off" = anagram indicator
    (10066250, 'anagram', 'typifying some errors?',
     'ALTERABLE (anagram="TEAR LABEL") [anagram: "off"] = ALTERABLE; definition: "typifying some errors?"'),

    # 17a CECII (5) - Chap bringing back some local ice cream
    # CECII hidden reversed in "local ice cream"; LICEC hidden in span
    (10066251, 'hidden', 'Chap',
     'CECII (hidden reversed in "local ice cream") [hidden: "some"] [reversal: "back"] = CECII; definition: "Chap"'),

    # 19a SEEEYETOEYE (3,3,2,3) - Agree to visit detectives about first signs of this offence
    (10066252, 'unparsed', 'Agree',
     'WORDPLAY UNPARSED; definition: "Agree"'),

    # 22a ELM (3) - Intermittently, see lime or another tree
    # alternate (even) letters of "see lime" = E,L,M = ELM
    (10066253, 'alternating', 'another tree',
     'ELM (alternate letters of "see lime") [alternating: "Intermittently"] = ELM; definition: "another tree"'),

    # 23a CAITLIN (7) - Talc I replaced at home for Irish female
    # anagram of TALC + I + IN (at home) = CAITLIN
    (10066254, 'anagram', 'Irish female',
     'CAITLIN (anagram="TALC I IN") [anagram: "replaced"] = CAITLIN; definition: "Irish female"'),

    # 24a GUMTREE (7) - Met urge possibly to find source of eucalyptus
    # anagram of MET URGE = GUMTREE; "possibly" = anagram indicator
    (10066255, 'anagram', 'source of eucalyptus',
     'GUMTREE (anagram="MET URGE") [anagram: "possibly"] = GUMTREE; definition: "source of eucalyptus"'),

    # 26a SENATE (6) - Name in constituency beginning to endorse governing body
    # N (name) inside SEAT (constituency) = SENAT + E (first of endorse) = SENATE
    (10066256, 'charade', 'governing body',
     'N (abbreviation="name") inside SEAT (synonym="constituency") [container: "in"] + E (first letter of "endorse") [first letter: "beginning to"] = SENATE; definition: "governing body"'),

    # 27a SCRUMPS (7) - Steals apples and what remains in sacks oddly
    (10066257, 'unparsed', 'Steals apples',
     'WORDPLAY UNPARSED; definition: "Steals apples"'),

    # 1d LETSLIP (3,4) - Inadvertently disclose one in a field supporting type of service
    # LET (type of service) + SLIP (one in a field = fielding position) = LETSLIP
    (10066258, 'charade', 'Inadvertently disclose',
     'LET (synonym="type of service") + SLIP (synonym="one in a field") = LETSLIP; definition: "Inadvertently disclose"'),

    # 2d SWIMWITHTHETIDE (4,4,3,4) - Conform to prevailing opinion, as sensible types in a dip do?
    (10066259, 'cryptic_definition', 'Conform to prevailing opinion',
     'WORDPLAY UNPARSED; definition: "Conform to prevailing opinion"'),

    # 3d USE (3) - Employ funster on and off
    # alternate (even) letters of FUNSTER = U,S,E = USE
    (10066260, 'alternating', 'Employ',
     'USE (alternate letters of "funster") [alternating: "on and off"] = USE; definition: "Employ"'),

    # 5d DISSIPATE (9) - Paid sites in trouble get to vanish
    # anagram of PAID SITES = DISSIPATE; "in trouble" = anagram indicator
    (10066262, 'anagram', 'get to vanish',
     'DISSIPATE (anagram="PAID SITES") [anagram: "in trouble"] = DISSIPATE; definition: "get to vanish"'),

    # 6d CRAIG (5) - Scottish male crashing car in Glasgow initially
    # anagram of CAR + I (first of "in") + G (first of "Glasgow") = CRAIG
    (10066263, 'anagram', 'Scottish male',
     'CAR (from clue) + I (first letter of "in") + G (first letter of "Glasgow") [first letter: "initially"] [anagram: "crashing"] = CRAIG; definition: "Scottish male"'),

    # 7d TAKEONESCUEFROM (4,4,3,4) - Follow the example of individual in EU taskforce deployed over month
    (10066264, 'unparsed', 'Follow the example of',
     'WORDPLAY UNPARSED; definition: "Follow the example of"'),

    # 8d DENTAL (6) - Relating to canines?
    (10066265, 'cryptic_definition', 'Relating to canines?',
     'WORDPLAY UNPARSED; definition: "Relating to canines?"'),

    # 12d SCOUR (5) - Comb this courtyard in part
    # SCOUR hidden in "this courtyard"; "in part" = hidden indicator
    (10066266, 'hidden', 'Comb',
     'SCOUR (hidden in "this courtyard") [hidden: "in"] [hidden: "part"] = SCOUR; definition: "Comb"'),

    # 14d SUBSTANCE (9) - Newspaper employee has opinion in matter?
    # SUB (newspaper employee) + STANCE (opinion) = SUBSTANCE
    (10066267, 'charade', 'matter?',
     'SUB (synonym="newspaper employee") + STANCE (synonym="opinion") = SUBSTANCE; definition: "matter?"'),

    # 16d ALSACE (6) - French region additionally snubbed by expert
    # ALSO snubbed (O dropped) = ALS + ACE (expert) = ALSACE
    (10066269, 'charade', 'French region',
     'ALS (deletion="ALSO", O dropped, ALSO synonym="additionally") [deletion: "snubbed"] + ACE (synonym="expert") = ALSACE; definition: "French region"'),

    # 18d LUMBERS (7) - Plods in wood in the U.S. with son
    # LUMBER (wood in the U.S.) + S (son) = LUMBERS
    (10066270, 'charade', 'Plods',
     'LUMBER (synonym="wood in the U.S.") + S (abbreviation="son") = LUMBERS; definition: "Plods"'),

    # 20d YALTA (5) - Revolutionary at ordinary port in the Black Sea
    # anagram of AT + LAY (ordinary) = YALTA; "Revolutionary" = anagram indicator
    (10066271, 'anagram', 'port in the Black Sea',
     'YALTA (anagram="AT LAY") [anagram: "Revolutionary"] = YALTA; definition: "port in the Black Sea"'),

    # 25d MAR (3) - Damage area for selling goods (not half)
    # MARKET (area for selling goods) not half = MAR; "not half" = deletion indicator
    (10066273, 'deletion', 'Damage',
     'MAR (deletion="MARKET", KET dropped, MARKET synonym="area for selling goods") [deletion: "not half"] = MAR; definition: "Damage"'),
]
