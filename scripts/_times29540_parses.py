"""Honest parses for Times 29540 leftovers, derived from fifteensquared blog.

Each entry: (clue_id, wordplay_type, definition, ai_explanation)

Designed to be importable for both the coverage-check pass and the
store-and-verify pass. Built blog-grounded — no fake CDs, no [parts:] dumping,
no abbreviation tags for synonym chains.
"""

CLUES = [
    # 1a ASITCOMES (2,2,5) — Straight entertainers with stuffing knocked out by a funny show
    # Blog: A , SITCOM (funny show), E {ntertainer} S [with stuffing knocked out]
    # "knocked" and "out" both deletion indicators; "stuffing" parts indicator
    (10065273, 'charade', 'Straight',
     'A (from clue) + SITCOM (synonym="funny show") + ES (outer letters of "entertainers") [deletion: "knocked"; deletion: "out"; parts: "stuffing"] = ASITCOMES; definition: "Straight"'),

    # 6a GRASS (5) — Split blades. DD; "Split" = inform on = GRASS, "blades" = blades of grass
    (10065274, 'double_definition', 'blades',
     'double definition: Split = GRASS, blades = GRASS [parts: "Split"]; definition: "blades"'),

    # 9a GAMELAN (7) — Percussion ensemble with something to play, short track
    # Blog: GAME (something to play), LAN {e} (track) [short]
    (10065275, 'charade', 'Percussion ensemble',
     'GAME (synonym="something to play") + LAN (deletion="LANE"), LANE (synonym="track") [deletion: "short"] = GAMELAN; definition: "Percussion ensemble"'),

    # 10a TROPHIC (7) — Parallel eating habits originally related to nutrition
    # Blog: TROP^I C (parallel) containing [eating] H {abits} [originally]
    (10065276, 'container', 'related to nutrition',
     'TROPIC (synonym="parallel") containing H (first letter of "habits") [container: "eating"; parts: "originally"] = TROPHIC; definition: "related to nutrition"'),

    # 11a THROB (5) — Beat time, having raised orchestral baton first of all
    # Blog: T {ime} + H {aving} + R {aised} + O {rchestral} + B {aton} [first of all]
    (10065277, 'charade', 'Beat',
     'T (first letter of "time") + H (first letter of "having") + R (first letter of "raised") + O (first letter of "orchestral") + B (first letter of "baton") [parts: "first"] = THROB; definition: "Beat"'),

    # 12a INGESTION (9) — I go in nest, after flying about, for feeding
    # Blog: Anagram [after flying about] of I GO IN NEST
    (10065278, 'anagram', 'for feeding',
     'anagram of I GO IN NEST [anagram: "flying"; anagram: "about"] = INGESTION; definition: "for feeding"'),

    # 14a PIE (3) — Sweet or savoury dish, mixed type. DD; second def = printing term "pie = mixed type"
    (10065279, 'double_definition', 'Sweet or savoury dish',
     'double definition: Sweet or savoury dish = PIE, mixed type = PIE [parts: "mixed type"]; definition: "Sweet or savoury dish"'),

    # 15a EVENINGSTAR (7,4) — Heavenly body gave interns breaks
    # Blog: Anagram [breaks] of GAVE INTERNS
    (10065280, 'anagram', 'Heavenly body',
     'anagram of GAVE INTERNS [anagram: "breaks"] = EVENINGSTAR; definition: "Heavenly body"'),

    # 17a ADIMEADOZEN (1,4,1,5) — One drink, Australian opening port, nothing special
    # Blog: I (one) + MEAD (drink) + OZ (Australian) contained by [opening] AD^EN (port)
    (10065281, 'container', 'nothing special',
     'ADEN (synonym="port") containing I (synonym="One") + MEAD (synonym="drink") + OZ (synonym="Australian") [container: "opening"] = ADIMEADOZEN; definition: "nothing special"'),

    # 19a FEW (3) — Only a handful I'm relieved to hear?
    # Blog: Aural wordplay [to hear]: "phew" (I'm relieved)
    (10065282, 'homophone', 'Only a handful',
     'FEW sounds like PHEW — an exclamation of relief — [homophone: "hear"] = FEW; definition: "Only a handful"'),

    # 22a ELAND (5) — Camel leaving river with antelope
    # Blog: {Cam}EL (leaving river - R.Cam), AND (with)
    (10065284, 'charade', 'antelope',
     'EL (deletion="CAMEL"), CAMEL (from clue) with CAM (synonym="river") removed + AND (synonym="with") [deletion: "leaving"] = ELAND; definition: "antelope"'),

    # 24a REISSUE (7) — Queen rejected by girl, possibly put out again?
    # Blog: ER (Queen) reversed [rejected], ISSUE (girl, possibly - offspring)
    (10065285, 'charade', 'put out again',
     'reversal of ER (synonym="Queen") [reversal: "rejected"] + ISSUE (synonym="girl, possibly") = REISSUE; definition: "put out again"'),

    # 26a EXACTOR (7) — Demanding individual, one no longer taking a part?
    # Blog: EX-ACTOR (one no longer taking a part)
    (10065286, 'charade', 'Demanding individual',
     'EX (synonym="no longer") + ACTOR (synonym="one taking a part") = EXACTOR; definition: "Demanding individual"'),

    # 27a DWELT (5) — Raised mark on back of hand resided
    # Blog: {han}D [back of...], WELT (raised mark)
    (10065287, 'charade', 'resided',
     'D (last letter of "hand") + WELT (synonym="Raised mark") [parts: "back of"] = DWELT; definition: "resided"'),

    # 28a HORSEPLAY (9) — High jinks in Equus, perhaps
    # Blog: HORSE PLAY (Equus, perhaps - Equus is a play by Peter Shaffer about a horse)
    # CD: the whole "in Equus, perhaps" cryptically clues HORSE+PLAY (Equus = play about horses).
    (10065288, 'cryptic_definition', 'High jinks',
     'cryptic definition: Equus, perhaps, is a horse play — Equus by Peter Shaffer is famously about a horse — = HORSEPLAY; definition: "High jinks"'),

    # 1d ARGOT (5) — Answer that's not filled in, understood in secret language
    # Blog: A{nswe}R [that's not filled in], GOT (understood)
    (10065289, 'charade', 'secret language',
     'AR (outer letters of "answer") + GOT (synonym="understood") = ARGOT; definition: "secret language"'),

    # 2d IMMERSE (7) — Dip very short tongue?
    # Blog: 1 MM (very short - 1 millimetre), ERSE (tongue)
    # "very short" = "1mm" cryptically, giving I + MM. No clean indicator
    # for that move — accept the word-coverage penalty rather than [parts:] dump.
    (10065290, 'charade', 'Dip',
     'I (synonym="1") + MM (abbreviation="millimetre") + ERSE (synonym="tongue") = IMMERSE; definition: "Dip"'),

    # 3d CALABRESE (9) — Suit hosting party concerned with vegetarian food
    # Blog: CA^SE (law suit) containing [hosting] LAB (political party) + RE (concerned with)
    (10065291, 'container', 'vegetarian food',
     'CASE (synonym="Suit") containing LAB (synonym="party") + RE (synonym="concerned with") [container: "hosting"] = CALABRESE; definition: "vegetarian food"'),

    # 4d MANBITESDOG (3,5,3) — Newsworthy story best friend chewed over here?
    # CD - journalistic adage "dog bites man isn't news but man bites dog is"; best friend = dog
    (10065292, 'cryptic_definition', 'Newsworthy story',
     'cryptic definition: a man biting his best friend, a dog, is the journalistic example of a newsworthy story = MANBITESDOG; definition: "Newsworthy story"'),

    # 5d SOT (3) — Sponge cakes go flat in the end
    # Blog: {cake}S + {g}O + {fla}T [in the end]. Sponge = heavy drinker = sot.
    (10065293, 'charade', 'Sponge',
     'S (last letter of "cakes") + O (last letter of "go") + T (last letter of "flat") [parts: "in the end"] = SOT; definition: "Sponge"'),

    # 6d GROSS (5) — I'm disgusted by 24 six-packs? DD. 24x6=144 (gross).
    (10065294, 'double_definition', "I'm disgusted",
     'double definition: I\'m disgusted = GROSS, 24 six-packs = GROSS (24 x 6 = 144 = a gross) [parts: "24 six-packs"]; definition: "I\'m disgusted"'),

    # 7d ATHEIST (7) — Infidel witnessing robbery?
    # Blog: AT HEIST (witnessing robbery)
    (10065295, 'charade', 'Infidel',
     'AT (synonym="witnessing") + HEIST (synonym="robbery") = ATHEIST; definition: "Infidel"'),

    # 8d SECONDROW (6,3) — Part of scrum down, score after collapse
    # Blog: Anagram [after collapse] of DOWN SCORE
    (10065296, 'anagram', 'Part of scrum',
     'anagram of DOWN SCORE [anagram: "collapse"] = SECONDROW; definition: "Part of scrum"'),

    # 13d GRIZZLYBEAR (7,4) — Beastly thing and gruesome display for the audience
    # Blog: GRIZZLY/"grisly" (gruesome) + BEAR/"bare" (display) [for the audience]
    (10065297, 'homophone', 'Beastly thing',
     'GRIZZLY sounds like GRISLY (synonym="gruesome") + BEAR sounds like BARE (synonym="display") [homophone: "for the audience"] = GRIZZLYBEAR; definition: "Beastly thing"'),

    # 14d PLASTERED (9) — Lit up — as ceilings, perhaps? DD.
    (10065298, 'double_definition', 'Lit up',
     'double definition: Lit up = PLASTERED, as ceilings, perhaps = PLASTERED [parts: "as ceilings, perhaps"]; definition: "Lit up"'),

    # 16d GINGERALE (6,3) — Drink large rum and energy drink
    # Blog: GIN (drink), anagram [rum] of LARGE, then E (energy)
    (10065299, 'charade', 'Drink',
     'GIN (synonym="drink") + anagram of LARGE [anagram: "rum"] + E (abbreviation="energy") = GINGERALE; definition: "Drink"'),

    # 18d IMAGINE (7) — Picture this writer and soldier in shock
    # Blog: I (this writer), then GI (soldier) contained by [in] MA^NE (shock - hair)
    (10065300, 'container', 'Picture',
     'I (synonym="this writer") + MANE (synonym="shock") containing GI (synonym="soldier") [container: "in"] = IMAGINE; definition: "Picture"'),

    # 19d FRACTAL (7) — Complex pattern generated by craft adrift on American lake
    # Blog: Anagram [adrift] of CRAFT, then A (American), L (lake)
    (10065301, 'charade', 'Complex pattern',
     'anagram of CRAFT [anagram: "adrift"] + A (abbreviation="American") + L (abbreviation="lake") = FRACTAL; definition: "Complex pattern"'),

    # 21d INSET (5) — Introduce class who are popular?
    # Blog: IN (popular), SET (class)
    (10065302, 'charade', 'Introduce',
     'IN (synonym="popular") + SET (synonym="class") = INSET; definition: "Introduce"'),

    # 23d DERBY (5) — Hat raised, lifted with courtesy, ultimately
    # Blog: BRED (raised) reversed [lifted], {courtes}Y [ultimately]
    (10065303, 'charade', 'Hat',
     'reversal of BRED (synonym="raised") [reversal: "lifted"] + Y (last letter of "courtesy") [parts: "ultimately"] = DERBY; definition: "Hat"'),

    # 25d ETH (3) — Old letter delivered in time, thankfully
    # Blog: Hidden [delivered in] {tim}E TH{ankfully}
    (10065304, 'hidden', 'Old letter',
     'hidden in "timE THankfully" [hidden: "in"] = ETH; definition: "Old letter"'),
]
