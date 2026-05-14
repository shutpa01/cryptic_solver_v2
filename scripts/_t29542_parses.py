"""Honest parses for Times 29542 leftovers, derived from TFTT blog.

Each entry: (clue_id, wordplay_type, definition, ai_explanation).
"""

CLUES = [
    # 1a RIBALDRY (8) -- "Blair cracked boring, licentious jokes"
    # Blog: anagram of BLAIR and DRY (boring), cracked = indicator
    (10065999, 'anagram', 'licentious jokes',
     'RIBALDRY (anagram="BLAIR DRY") [anagram: "cracked"]; BLAIR (from clue) + DRY (synonym="boring") = RIBALDRY; definition: "licentious jokes"'),

    # 5a ICECAP (3,3) -- "Always a cool head in charge, step back"
    # IC (abbreviation for in charge) + PACE (step) reversed (back)
    (10066000, 'charade', 'Always a cool head',
     'IC (abbreviation="in charge") + ECAP (reversal of "PACE"), PACE synonym="step" [reversal: "back"] = ICECAP; definition: "Always a cool head"'),

    # 10a GONEWITHTHEWIND (4,4,3,4) -- "High tide: newt now swimming in famous picture"
    # Blog: anagram (swimming) of HIGH TIDE NEWT NOW
    (10066001, 'anagram', 'famous picture',
     'GONEWITHTHEWIND (anagram="HIGH TIDE NEWT NOW") [anagram: "swimming"]; HIGH (from clue) + TIDE (from clue) + NEWT (from clue) + NOW (from clue) = GONEWITHTHEWIND; definition: "famous picture"'),

    # 11a UTOPIAN (7) -- "Ideal university best in surrounding area"
    # Blog: U + TOP + A inside IN = U + TOP + IAN = UTOPIAN
    (10066002, 'container', 'Ideal',
     'U (abbreviation="university") + TOP (synonym="best") + IAN, where IN (from clue) containing A (abbreviation="area") [container: "surrounding"] = IAN; U + TOP + IAN = UTOPIAN; definition: "Ideal"'),

    # 12a NEMESIS (7) -- "Goddess, saint and Magi from the East, not West"
    # Blog: S + (WISEMEN - W) reversed = SISEMEN reversed = NEMESIS
    (10066003, 'reversal', 'Goddess',
     'S (abbreviation="saint") + ISEMEN (deletion="WISEMEN", W dropped, WISEMEN synonym="Magi") [deletion: "not West"; reversal: "from the East"] = NEMESIS; definition: "Goddess"'),

    # 13a ASPERGES (8) -- "Religious rite snake vocally encourages?"
    # Blog: ASP (snake) + ERGES sounds like URGES (encourages), vocally = homophone indicator
    (10066004, 'homophone', 'Religious rite',
     'ASP (synonym="snake") + ERGES sounds like URGES (synonym="encourages") [homophone: "vocally"] = ASPERGES; definition: "Religious rite"'),

    # 15a RAITA (5) -- "Inferior cricketer leaves B&B with a side"
    # Blog: RABBIT (inferior cricketer) - BB (the two Bs of B&B) = RAIT, + A (from clue) = RAITA
    (10066005, 'charade', 'side',
     'RAIT (deletion="RABBIT", BB dropped, RABBIT synonym="inferior cricketer") [deletion: "leaves"] + A (from clue) = RAITA; definition: "side"'),

    # 18a ERNST (5) -- "Year one sits periodically for artist"
    # Blog: alternate letters of "year one sits" = E,R,N,S,T = ERNST
    (10066006, 'alternating', 'artist',
     'ERNST (alternate letters of "year one sits") [parts: "periodically"] = ERNST; definition: "artist"'),

    # 20a RESONATE (8) -- "Thunder from returning beak judge welcomes"
    # Blog: NOSE (beak) reversed (returning) inside RATE (judge, welcomes = container)
    (10066007, 'container', 'Thunder',
     'RATE (synonym="judge") containing ESON (reversal of "NOSE"), NOSE synonym="beak" [reversal: "returning"; container: "welcomes"] = RESONATE; definition: "Thunder"'),

    # 23a LASAGNE (7) -- "Way to get to grips with spinach dish"
    # Blog: LANE (way) containing SAG (spinach) = LASAGNE
    (10066008, 'container', 'dish',
     'LANE (synonym="way") containing SAG (synonym="spinach") [container: "to get to grips with"] = LASAGNE; definition: "dish"'),

    # 25a MINORCA (7) -- "Maidens popular with killer in holiday island"
    # Blog: M (maidens) + IN (popular) + ORCA (killer whale) = MINORCA
    (10066009, 'charade', 'holiday island',
     'M (abbreviation="maidens") + IN (synonym="popular") + ORCA (synonym="killer") = MINORCA; definition: "holiday island"'),

    # 26a WEARTHETROUSERS (4,3,8) -- "Pathetic people causing stir outside Globe are in control"
    # Blog: WETROUSERS (WET + ROUSERS) containing EARTH (Globe) = WEARTHETROUSERS
    (10066010, 'container', 'are in control',
     'WETROUSERS containing EARTH (synonym="Globe") [container: "outside"]; WET (synonym="pathetic") + ROUSERS (synonym="causing stir") = WETROUSERS; WETROUSERS containing EARTH = WEARTHETROUSERS; definition: "are in control"'),

    # 27a RETORT (6) -- "Chemist obtains liquids from this counter"
    # Blog: double definition
    (10066011, 'double_definition', 'Chemist obtains liquids from this',
     'double definition: Chemist obtains liquids from this = RETORT, counter = RETORT; definition: "Chemist obtains liquids from this"'),

    # 1d RAGOUT (6) -- "Kid not allowed spicy food"
    # Blog: RAG (to kid) + OUT (not allowed) = RAGOUT
    (10066013, 'charade', 'spicy food',
     'RAG (synonym="kid") + OUT (synonym="not allowed") = RAGOUT; definition: "spicy food"'),

    # 2d BENJONSON (3,6) -- "One missed among orders taken by good French dramatist"
    # Blog: BON (good French) containing ENJONS (ENJOINS - I/one) = BENJONSON
    (10066014, 'container', 'dramatist',
     'BON (synonym="good French") containing ENJONS [container: "taken by"]; ENJONS from ENJOINS (synonym="orders") with I (synonym="one") dropped [deletion: "missed"] = BENJONSON; definition: "dramatist"'),

    # 3d LOWLIFE (7) -- "Prison sentence involving bird for criminal?"
    # Blog: LIFE (prison sentence) containing OWL (bird) = LOWLIFE
    (10066015, 'container', 'criminal?',
     'LIFE (synonym="prison sentence") containing OWL (synonym="bird") [container: "involving"] = LOWLIFE; definition: "criminal?"'),

    # 4d RATON (3,2) -- "Hurried round to shop"
    # Blog: RAN (hurried) containing TO (from clue) = RATON
    (10066016, 'container', 'shop',
     'RAN (synonym="hurried") containing TO (from clue) [container: "round"] = RATON; definition: "shop"'),

    # 6d CREAMER (7) -- "One shrieking denied son milk substitute"
    # Blog: SCREAMER (one shrieking) - S (son) = CREAMER
    (10066017, 'deletion', 'milk substitute',
     'CREAMER (deletion="SCREAMER", S (abbreviation="son") dropped, SCREAMER synonym="one shrieking") [deletion: "denied"] = CREAMER; definition: "milk substitute"'),

    # 7d CHIPS (5) -- "Carpenter put power in lives, overseen by church"
    # Blog: CH (church) + IPS where IS (lives) containing P (power)
    (10066018, 'container', 'Carpenter',
     'CH (abbreviation="church") + IPS, where IS (synonym="lives") containing P (abbreviation="power") [container: "in"] = IPS; CH + IPS = CHIPS; definition: "Carpenter"'),

    # 8d PEDESTAL (8) -- "Operate bike going round established base"
    # Blog: PEDAL (operate bike) containing EST (established) = PEDESTAL
    (10066019, 'container', 'base',
     'PEDAL (synonym="operate bike") containing EST (abbreviation="established") [container: "going round"] = PEDESTAL; definition: "base"'),

    # 9d STENOSIS (8) -- "Constriction sets in so badly"
    # Blog: anagram (badly) of SETS IN SO = STENOSIS
    (10066020, 'anagram', 'Constriction',
     'STENOSIS (anagram="sets in so") [anagram: "badly"] = STENOSIS; definition: "Constriction"'),

    # 14d GARDENER (8) -- "Worker at Versailles possibly planting forest in Germany"
    # Blog: GER (Germany) containing ARDEN (forest) = GARDENER
    (10066021, 'container', 'Worker at Versailles possibly',
     'GER (abbreviation="Germany") containing ARDEN (synonym="forest") [container: "planting"] = GARDENER; definition: "Worker at Versailles possibly"'),

    # 16d INTERFERE (9) -- "Barge in Bury rent free"
    # Blog: INTER (bury) + anagram of FREE (rent = torn = anagram indicator)
    (10066022, 'charade', 'Barge in',
     'INTER (synonym="bury") + FERE (anagram="free") [anagram: "rent"] = INTERFERE; definition: "Barge in"'),

    # 17d BELLOWER (8) -- "Bull almost hit cow?"
    # Blog: BEL (BELT minus last letter, almost) + LOWER (cow)
    (10066023, 'charade', 'Bull',
     'BEL (deletion="BELT", last letter dropped, BELT synonym="hit") [deletion: "almost"] + LOWER (synonym="cow") = BELLOWER; definition: "Bull"'),

    # 19d TIGHTER (7) -- "Faster animal embracing hard time"
    # Blog: TIGER (animal) containing H (hard) + T (time) = TIGHTER
    (10066024, 'container', 'Faster',
     'TIGER (synonym="animal") containing HT, where H (abbreviation="hard") + T (abbreviation="time") [container: "embracing"] = TIGHTER; definition: "Faster"'),

    # 21d NONSUCH (7) -- "Previously unrivalled thing in Rome not so great?"
    # Blog: NON (not in Latin) + SUCH (so great) = NONSUCH
    (10066025, 'charade', 'Previously unrivalled thing',
     'NON (synonym="not in Rome") + SUCH (synonym="so great") = NONSUCH; definition: "Previously unrivalled thing"'),

    # 24d SHAFT (5) -- "Upright female in small fedora?"
    # Blog: S (small) + HAFT where HAT (fedora) containing F (female) = HAFT; S + HAFT = SHAFT
    (10066027, 'container', 'Upright',
     'S (abbreviation="small") + HAFT, where HAT (synonym="fedora") containing F (abbreviation="female") [container: "in"] = HAFT; S + HAFT = SHAFT; definition: "Upright"'),

    # 25d MARIA (5) -- "Yacht sanctuary keeps out November seas"
    # Blog: MARINA (yacht sanctuary) - N (November) = MARIA
    (10066028, 'deletion', 'seas',
     'MARIA (deletion="MARINA", N (abbreviation="November") dropped, MARINA synonym="yacht sanctuary") [deletion: "keeps out"] = MARIA; definition: "seas"'),
]
