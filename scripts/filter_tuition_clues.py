"""Filter tuition_clues.json to keep only 10 clues per type with common everyday answers."""
import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tuition_clues.json"

# Hand-picked indices (0-based) for each type — common words, clear mechanisms
SELECTIONS = {
    "anagram": [
        0,   # ESTIMATE - Guess teatimes varied
        3,   # PHEASANT - Heats pan to cook game bird
        6,   # GENERATE - Create concoction of green tea
        11,  # BALANCE - Ban lace pants to create stability
        12,  # AUTHORISE - Hires out a buggy producing permit
        16,  # ELBOW - Joint below or bowel given treatment
        20,  # USEFUL - Reconstruct flue with us? That'll be handy
        22,  # PROGNOSIS - Terribly poor signs seeing medical forecast
        4,   # OTHERWISE - Their woes compounded if not
        10,  # SOMETIMES - Crackers seem moist on occasion
    ],
    "charade": [
        8,   # PATTERN - Glib talk close to fashion model
        13,  # CHASTE - Pure Charlie? It's superior to speed
        17,  # EXAMPLE - Former partner, full-figured specimen
        18,  # OPERA - Aida perhaps got naked for each adult
        21,  # TATTOO - Rubbish as well, Scottish entertainer?
        25,  # CEASE - Refrain from church when finally contrite
        26,  # PALLID - Friend having top that's rather washed out
        27,  # CRUEL - Heartless Conservative left to admit regret
        1,   # NOTE - Minute mark
        3,   # MEANT - Signified average temperature
    ],
    "container": [
        0,   # RACY - Blue fish catching cold
        5,   # MAPLE - Quietly cutting gentleman's tree
        6,   # CLOT - Lake bed holds solidified matter
        7,   # PARKA - Old man covers chest in warm coat
        8,   # CHAP - Fellow from hotel wearing headgear
        17,  # PHOTO - Italian banker fences recently stolen picture
        22,  # IMAGE - Concern of PR department that is saving publication
        29,  # CLIENT - Person engaging lawyer Neil is back in court
        14,  # ROTUND - Start to tuck into sandwich and roly-poly
        27,  # BEANO - Exclude nothing about sweetheart for party
    ],
    "deletion": [
        0,   # RASH - Spots litter leader's left
        2,   # FACE - Be confronted by heartless travesty
        10,  # FAME - Celebrity, female, arrived topless
        14,  # GRASP - Clutch pot son knocked off piano
        16,  # PRESENT - Tense leader ignoring papers
        17,  # OATH - Unwilling to forgo large bond
        18,  # ATLAS - World map eventually detailed
        20,  # DARE - Cut short shady European venture
        24,  # OVERT - Public secret initially suppressed
        25,  # NOEL - A Christmas Carol - heartless story
    ],
    "double_definition": [
        0,   # WELL - Quite possibly fortunate
        3,   # DARTS - Shoots game
        4,   # GRUB - Little wriggler eats
        6,   # LATE - Departed behind schedule
        8,   # DISCO - Club legend is counting cups
        9,   # PASS - Achieve exam success while away
        10,  # BALL - Clog dance
        12,  # SEAT - Spanish car: you may sit in it
        13,  # ORDER - System command
        17,  # CUTE - Clever and sweetly pleasing
    ],
    "hidden": [
        2,   # LYING - Whitefly in greenhouse to some extent remaining
        4,   # RATHER - Somewhat in clover at hers
        12,  # ETHOS - Greet host, holding spirit
        20,  # YOURS - The solver's joy? Our setter welcomes it
        21,  # ARDENT - Passionate, somewhat wayward entrepreneur
        23,  # REFER - Pass on preference to have case of money removed
        11,  # RINGLET - Answering letter enclosing lock of hair
        16,  # OVERSPILL - Discovers pillagers pinching surplus
        25,  # TESS - Hardy character demonstrates stamina, to an extent
        6,   # PHANTOM - Triumphant ombudsman's hiding illusion
    ],
    "reversal": [
        1,   # OGRE - Monster therefore returns
        5,   # SLAM - Criticise some informal styles making a comeback
        13,  # PLANS - Designs aircraft base rejected
        16,  # KNOT - Hit back in tie
        0,   # TRAMS - Vehicles Sting held up
        19,  # SEMINAR - Half rushed back for teaching session
        22,  # SANE - Reasonable except when noun plays part of verb
        17,  # TOSH - Drunkard back with hard stuff
        21,  # BURROW - Polish brought west to fight make dugouts?
        11,  # MANIOC - Mother raised money as source of meal
    ],
    "homophone": [
        2,   # MEDDLE - Interfere in award, one hears
        3,   # ASCENT - Mounting agreement by the sound of it
        5,   # KNEADS - Manipulates desires in audition
        6,   # LESSENS - Reduces lectures for the audience
        10,  # EATEN - Consumed tuck, maybe in school, reportedly
        0,   # IDEALS - Confession of drug pusher, originally sacrificing
        7,   # PURSUIT - Interest announced for every case
        9,   # TWOSOME - Overheard extremely remarkable couple
        1,   # SUPERSEDE - Replace exceptionally good, highly ranked player
        4,   # CORSAIR - Reportedly bawdy song for pirate
    ],
    "acrostic": None,  # Only 3, keep all
    "cryptic_definition": [
        1,   # ICELAND - Country store?
        3,   # EPIC - Impressive digital snap?
        5,   # BOMB - Incendiary device in packet?
        6,   # NAAN - This is the same round bread
        9,   # SPACEBAR - One often depressed during a long sentence?
        14,  # NUDISM - One barely believes in it?
        15,  # CELLIST - Performer who will take a big bow?
        19,  # SERVICE - Tree maintenance check
        22,  # LETTERS - Landlords perhaps providing B&Bs?
        23,  # YELLOW - Cowardly manner in which to express pain?
    ],
}

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

for wtype, indices in SELECTIONS.items():
    if indices is None:
        continue  # keep all
    original = data["types"][wtype]["clues"]
    filtered = [original[i] for i in indices if i < len(original)]
    data["types"][wtype]["clues"] = filtered
    print(f"{wtype}: {len(original)} -> {len(filtered)}")

# Write back
with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"\nUpdated {DATA_PATH}")
