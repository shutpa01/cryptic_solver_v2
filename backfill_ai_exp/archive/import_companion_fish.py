"""Import Fish, Fireplaces, Fireworks, Firths, Aquarium Fish, Fishing from scan."""
import sqlite3

DB_PATH = "data/cryptic_new.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT LOWER(definition), UPPER(answer) FROM definition_answers_augmented")
existing = set((r[0], r[1]) for r in c.fetchall())
print(f"Existing pairs: {len(existing)}")

inserted = 0
skipped = 0

def add(definitions, words):
    global inserted, skipped, existing
    for word in words:
        word_upper = word.upper().strip()
        if not word_upper:
            continue
        for defn in definitions:
            key = (defn.lower(), word_upper)
            if key in existing:
                skipped += 1
                continue
            c.execute("INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
                      (defn, word_upper, "crossword_companion"))
            existing.add(key)
            inserted += 1

# ============ FIREPLACES ============
fireplaces = [
    "KILN", "OVEN", "FORGE", "GRATE", "INGLE", "STOVE",
    "BOILER", "HEARTH", "BONFIRE", "BRAZIER",
    "FIREBOX", "FURNACE", "GAS FIRE", "CAMPFIRE", "OPEN FIRE",
    "BACKBOILER", "INCINERATOR", "WOOD BURNING",
    "ELECTRIC FIRE", "PARAFFIN STOVE",
]
add(["fireplace", "fire", "hearth"], fireplaces)

# ============ FIREWORKS ============
fireworks = [
    "CAKE", "MINE", "PIOY", "DEVIL", "FLARE", "GERBE", "PEEOY",
    "PLOYE", "SHELL", "SQUIB", "WHEEL", "BANGER",
    "FIZGIG", "FIZGIG", "MAROON", "PETARD", "ROCKET", "CRACKER",
    "SERPENT", "VOLCANO", "FLIP-FLOP", "FOUNTAIN", "PINWHEEL", "SLAP-BANG",
    "SPARKLER", "WHIZ-BANG", "FIREDRAKE", "GIRANDOLA", "GIRANDOLE",
    "SKY-ROCKET", "THROW-DOWN", "WATERFALL", "WHITE-BANG",
    "GOLDEN RAIN", "INDIAN FIRE", "TOURBILLON",
    "FIRECRACKER", "FIREWRITING", "JUMPING-JACK", "ROMAN CANDLE", "TOURBILLON",
    "CATHERINE WHEEL", "CHINESE CRACKER", "INDOOR FIREWORK",
    "PHARAOH'S SERPENT", "WATERLOO CRACKER",
]
add(["firework", "pyrotechnic"], fireworks)

# ============ FIRTHS ============
firths = [
    "TAY", "LORN", "WIDE", "CLYDE", "FORTH", "LORNE", "MORAY",
    "BEAULY", "SOLWAY", "THAMES", "DORNOCH", "WESTRAY",
    "CROMARTY", "PENTLAND", "STORNAY", "SZCZECIN",
    "INVERNESS", "NORTH RONALDSAY",
]
add(["firth", "inlet", "estuary"], firths)

# ============ FISH (2-letter) ============
fish_2 = ["AI", "ID"]

# ============ FISH (3-letter) ============
fish_3 = [
    "AHI", "AYU", "BAR", "BIB", "COD", "DAB", "DAR", "EEL", "GAR",
    "IDE", "RAY", "SAI", "BARB", "BASS", "BLAY", "BLEY", "BRIT",
    "CARP", "CERO", "CHAR", "CHUB", "DACE", "DARE", "DART", "DORY",
    "FUGU", "GADE", "GABY", "HAKE", "HOKI", "LING",
    "LUCE", "MOKI", "OPAH", "ORFE", "PIKE", "POPE", "POUR",
    "RUDD", "RUFF", "SCAD", "SCAT", "SCUP", "SEER", "SEIR",
    "SHAD", "SILD", "SOLE", "TANG", "TUNA", "WELS",
]

# ============ FISH (5-letter) ============
fish_5 = [
    "ABLET", "BASSE", "BLAIN", "BLEAK", "BREAM", "BRILL", "CHARR",
    "CISCO", "COBIA", "COLEY", "DARIO", "DORAS", "DOREE",
    "ELOPS", "GRUNT", "GUPPY", "LANCE", "LOACH", "LYTHE",
    "MANTA", "MOLLY", "PERCA", "PERCH", "PIRAI", "PLATY",
    "POGGE", "ROACH", "RUFFE", "SARGO", "SAURY", "SHARK",
    "SKATE", "SMELT", "SPRAT", "SQUID", "TENCH", "TETRA",
    "TORSK", "TROUT", "TUNNY", "WAHOO", "WHELK", "WHIFF", "ZEBRA",
]

# ============ FISH (6-letter) ============
fish_6 = [
    "ANABAS", "BARBEL", "BICHIR", "BIGEYE", "BLENNY", "BONITO",
    "BOWFIN", "BURBOT", "BUMALO", "CALLOP", "CAPLIN", "CARIBE",
    "COMBER", "CONGER", "COTTID", "COTTUS", "CUNNER", "DARTER",
    "DENTEX", "DOCTOR", "DORADO", "GADOID", "GULPER", "GUNNEL",
    "GURAMI", "GURNETT",
    "INANGA", "JERKER", "KIPPER", "LAUNCE", "LOUVAR", "MAHAR",
    "MAGRE", "MARLIN", "MEAGRE", "MEDAKA", "MINNOW", "MULLET",
    "MURENA", "PLAICE", "PIRANHA", "PUFFER", "REMORA",
    "ROBALO", "ROUGHY", "RUNNER", "SAITHE", "SALMON", "SANDER",
    "SARDEL", "SARGUS", "SHANNY", "SHINER", "SKELLY",
    "SPARID", "SUCKER", "TAILOR", "TARPON", "TAUTOG",
    "TURBOT", "VENDIS", "WRASSE", "ZANDER", "ZINGEL",
]

# ============ FISH (7-letter) ============
fish_7 = [
    "ALEWIFE", "ANCHOVY", "AZURINE", "CATFISH", "BENIFISH",
    "BLOATER", "BOX-FISH", "BUFFALO", "CABEZON", "CANDIRU",
    "CAPELIN", "CATFISH", "CAVALLA", "CAVALLY", "CHIMERA",
    "CICHLID", "CLUPEID", "CODFISH", "COWFISH", "CRUCIAN",
    "CRUSIAN", "DOGFISH", "DOLPHIN", "DRUMMER", "ESCOLAR",
    "GARFISH", "GARPIKE", "GARNISH", "GOLDENEYE", "GOURAMIS",
    "GROUPER", "GROWLER", "GRUNION", "GUDGEON", "GURNARD",
    "HADDOCK", "HALIBUT", "HERRING", "HOGFISH", "HOUTING",
    "ICE FISH", "INK-FISH", "KAHAWAI", "LAMPREY", "LAMPUKI",
    "MAHSEER", "MEDACCA", "MOONEVE", "MORWONG", "MUDFISH",
    "MUREANA", "OAR FISH", "OCTOPUS", "OLD WIFE", "PANCHAR",
    "PIG-FISH", "PIRANHA", "POLLACK", "POLLOCK", "POMFRET",
    "POMPANO", "RAT-TAIL", "REDFISH", "RED MOKI", "SANDEEL",
    "SARDINE", "SCULPIN", "SEA BASS", "SEACOCK", "SEA PIKE",
    "SILURID", "SLEEPER", "SNAPPER", "STERLET",
    "SUNFISH", "TIDDLER", "TORPEDO", "VENDACE", "VENDISS",
    "WHITING",
]

# ============ FISH (8+ letter) ============
fish_8plus = [
    "ALBACORE", "ANABLEPS", "ARAPAUZA", "ATHERINE", "BANDFISH",
    "BILLFISH", "BLOWFISH", "BLUEFISH", "BONEFISH", "BRISTLING",
    "BULLHEAD", "CARANGID", "CHARACID", "CHARACIN", "CHIMAERA",
    "CLUPETID", "COALFISH", "COELACAN", "COW-PILOT",
    "DEVILRAY", "DRAGONET", "DRUMFISH", "EAGLE RAY",
    "FALLFISH", "FILE FISH", "FLATHEAD", "FLOUNDER",
    "FORKTAIL", "FROGFISH",
    "GAMBUSIA", "GILT-HEAD", "GOAT FISH", "GOLDFISH",
    "GRAYLING", "GREENEYE", "GREY-FISH", "HAIR-TAIL",
    "HALFBEAK", "HARDHEAD", "JOHN DORY", "KABELJOU",
    "KINGFISH", "LUDERICK",
    "LUNGFISH", "MACKEREL", "MANTA RAY", "MESOHAEM",
    "MONKFISH", "MOONFISH", "MORAY EEL", "MULLOWAY",
    "NANNYGAI", "PILCHARD", "PIPEFISH",
    "RASCASSE", "RED BELLY", "ROCK BASS", "ROCKFISH",
    "ROCKLING", "ROSEFISH", "SAILFISH", "SANDFISH",
    "SARDELLE", "SCORPAENA", "SCUP PAUG", "SEA BREAM",
    "SEAHORSE", "SEA PERCH", "SEA RAVEN", "SEA ROBIN",
    "SEA SNAIL", "SKIPJACK", "STINGRAY", "STURGEON",
    "SUCKFISH", "TOADFISH", "TUNA FISH", "WEAKFISH", "WOLFFISH",
    "AMBER-FISH", "AMBERJACK", "ANGELFISH", "ARGENTINE",
    "BARRACUDA", "BLACK BASS", "BLACKFISH", "BLINDFISH",
    "CARANGOID", "CHAMELEON", "CLINGFISH", "CONGEE-EEL",
    "CORAL-FISH", "CORYPHENE", "CRAB-EATER", "CRAMP-FISH",
    "DOVER SOLE", "GLOBE FISH", "GOLDSINNY",
    "GOLELI YKA", "GRENADIER", "GOLOMYANKA",
    "HORNYHEAD", "HOTTENTOT", "JELLYFISH", "KILLIFISH",
    "LEMON SOLE", "MUD MINNOW", "NEON TETRA", "PILOTE FISH",
    "PIPEFISH", "RED MULLET", "RED SALMON", "ROCK PERCH",
    "ROUND FISH", "SAND SMELT", "SCORPAENA", "SCORPION",
    "SEAHORSE", "STONEFISH", "SWEETFISH", "SWEETLIPS",
    "SWORDFISH", "SWORDTAIL", "TIGER FISH", "TOADFISH",
    "TRUNKFISH", "TRUMPETER", "TUNNY FISH", "WHITEBAIT",
    "WHITEBASS", "WRECK FISH",
    "ANGLER FISH", "ARCHERFISH", "ARCTIC CHAR", "BARRACOUTAL",
    "BARRAMUNDI", "BITTERLING", "BOMBAY DUCK", "BOTTLE-FISH",
    "CANDLEFISH", "COCKABUILLY", "COFFER-FISH", "CRAIGFLUKE",
    "CUTTLEFISH", "CYCLOSTOMA", "DAMSELFISH", "DEMOISELLE",
    "DRAGON FISH", "FLATEMOUTH", "FLYING FISH", "GOLDEN ORFE",
    "GREY MULLET", "GROUNDLING", "HAMMERFISH", "HAMMERHEAD",
    "PITTED EEL", "LANCET FISH", "LARGEMOUTH", "LIZARDFISH",
    "LAMPSUCLER", "MAORI CHIEF", "MOSSBUNKER", "NEEDLE-FISH",
    "OCEAN PERCH", "PADDLEFISH", "PIKEMINNOW",
    "PUFFER FISH", "RED SNAPPER", "RIBBONFISH",
    "ROCK SALMON", "ROCK TURBOT", "RUDDERFISH",
    "SACRED FISH", "SCORPAENID", "SEA SWALLOW", "SERRASALMO",
    "SHEATFISH", "SHEEP'S-HEAD", "SILVERFISH",
    "SMALLMOUTH", "SQUEATEAGUE",
    "STONE LOACH", "SUCKERFISH", "TIGER SHARK",
    "TORPEDO RAY", "WHALE SHARK",
    "ANEMONE FISH", "BELLOWS-FISH", "BLUE WHITING",
    "BUFFALO FISH", "CUTLASS FISH", "DOLLY VARDEN",
    "DOLPHIN FISH", "EUROPEAN EEL", "ELECTRIC RAY",
    "FROGFISH", "LAKE HERRING", "LANTERN FISH",
    "BAKING SHARK", "FATHER-LASHER", "FIGHTING FISH",
    "FOUR-EYED FISH", "MOSQUITO FISH", "ORANGE ROUGHY",
    "PARADISE FISH", "PARROT-WRASSE", "RAINBOW TROUT",
    "SCABBARD FISH", "SCORPION FISH", "SEA PORCUPINE",
    "SERGEANT FISH", "SIGANID", "SMOOTH BLENNY",
    "SQUIRREL FISH", "ST PETER'S FISH", "WHIPTAIL HAKE",
    "ARMED BULLHEAD", "BUTTERFLY FISH", "CHINOOK SALMON",
    "CLIMBING PERCH", "HORSE MACKEREL", "LABYRINTH FISH",
    "NORTHERN PORGY", "PORCUPINE FISH",
    "SERGEANT MAJOR", "SOCKEYE SALMON", "YELLOW FIN TUNA",
    "BLUEBACK SALMON", "GREAT WHITE SHARK",
    "HAMMERHEAD SHARK", "SPANISH MACKEREL",
]

all_fish = fish_2 + fish_3 + fish_5 + fish_6 + fish_7 + fish_8plus
add(["fish", "sea creature"], all_fish)

# ============ AQUARIUM FISH ============
aquarium_fish = [
    "BARB", "DANIO", "GUPPY", "LOACH", "TETRA", "ZEBRA",
    "DISCUS", "GOURAMI", "NEON TETRA", "TIGER BARB",
    "CATFISH", "CICHLID", "CRUSIAN", "FANTAIL",
    "GOURAMI", "KOI CARP", "PIRANHA", "GOLDFISH",
    "ANGELFISH", "CLOWNFISH", "GOLDEN BARB", "ZEBRA DANIO",
    "JACK DEMPSEY", "CARDINAL FISH", "DWARF CICHLID", "DWARF GOURAMI",
    "SUCKING LOACH", "CORNET GOLDFISH", "COMMON HATCHET",
    "MALAWI CICHLID", "COMMON GOLDFISH", "KISSING GOURAMI",
    "RED-FINNED SHARK", "WALKING CATFISH", "FANTAIL GOLDFISH",
]
add(["aquarium fish", "tropical fish"], aquarium_fish)

# ============ FISHING FLIES ============
fishing_flies = [
    "BOB", "DRY", "WET", "HARL", "SEDGE", "DOCTOR",
    "HACKLE", "PALMER", "SALMON", "WATCHET",
    "HAIRY MARY", "JOCK SCOTT", "COCK-A-BONDY",
]
add(["fishing fly", "fly"], fishing_flies)

# ============ FISHING TERMS ============
fishing_terms = [
    "DUB", "FLY", "GIG", "JIG", "NET", "ROD", "SET", "TAG", "TIE",
    "BAIT", "BARB", "BITE", "BOAT", "BUNT", "CAST", "DRAG", "GIMP",
    "HAAF", "HOOK", "LEAD", "LINE", "LURE", "REEL", "SEAN", "TROT", "WHIP",
    "ALDER", "ANGLE", "BAKER", "CATCH", "CHUM", "CREEL", "DRAIT",
    "DRESS", "POACH", "SEINE", "TRACE",
    "ANGLER", "BOBBER", "BULLET", "COARSE", "DIBBLE", "DIP NET",
    "DRY-FLY", "FLY-ROD", "GENTLE", "GILLIE", "KEEPER",
    "MONOEIL", "PISCARY", "PLUMMET",
    "POOL NET", "SETLINE", "SPINNER",
    "BACKCAST", "BUZZBART", "DRIFT NET", "GIVE LINE", "HAND LINE",
    "ROLL CAST", "TROTLINE",
    "BRANDLING", "DRABBLING", "EGG SINKER", "FALSE CAST",
    "HALIEUTIC", "INDICATOR", "LEGER BAIT", "LEGER LINE",
    "NIGHT-LINE", "PRECATORY", "PROPELLER",
    "BAIT BUCKET", "BAITRUNNER", "CASTING ARC", "CASTING-NET",
    "DOUBLE HAUL", "FLY CASTING", "FLY FISHING", "HALIEUTICS",
    "LANDING NET", "LEDGER-BAIT", "LEDGER LINE", "LINE GREASE",
    "MULTIPLIER", "NET-FISHING", "SEA-FISHING", "TREBLE HOOK",
    "WEIGHING SCALE",
]
add(["fishing term", "angling term"], fishing_terms)

conn.commit()
print(f"\nTotal inserted: {inserted}")
print(f"Total skipped: {skipped}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"Total crossword_companion entries: {c.fetchone()[0]}")
conn.close()
