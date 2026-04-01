"""Import Tier 2c: Analgesics, Anatomy, Angels, Animals extras, Antelopes, etc."""
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

# ============ ANALGESICS ============
analgesics = [
    "ASPIRIN", "CODEINE", "SALICIN", "MORPHINE", "KETAMINE",
    "IBUPROFEN", "PANADOL", "PETHIDINE", "DICLOFENAC",
    "METHADONE", "NAPROXEN", "ASPIRIN", "PARACETAMOL",
    "PHENACETIN",
]
add(["analgesic", "painkiller", "pain relief"], analgesics)

# ============ ANATOMY TERMS ============
anatomy_terms = [
    "ARM", "EAR", "EYE", "GUM", "HIP", "JAW", "LEG", "RIB", "TOE",
    "ANUS", "BACK", "BONE", "CHIN", "CRUS", "FOOT", "HAND", "HEAD",
    "HEEL", "KNEE", "LIMB", "LOBE", "LUNG", "NAIL", "NECK", "NOSE",
    "SHIN", "VEIN", "WOMB",
    "AORTA", "AURAL", "ILEUM", "JOINT", "LIVER", "LUNGS", "MOUTH",
    "NASAL", "NAVEL", "NERVE", "OPTIC", "OVARY", "PEDAL", "PENIS",
    "RENAL", "SPINE", "THUMB", "TRUNK", "UVULA", "VULVA", "WRIST",
    "ARTERY", "ATRIUM", "AXILLA", "BICEPS", "BREAST", "CARPAL",
    "CARPUS", "CRURAL", "DENTAL", "DERMAL", "DISTAL", "FINGER",
    "FLEXOR", "GENIAL", "GULLET", "KIDNEY", "LINGUA", "LUMBAR",
    "MUSCLE", "NEURAL", "NEURON", "OCULAR", "PENILE", "PLEURA",
    "RECTUM", "ROTULA", "SACRAL", "SEPTUM", "SOLEUS", "SPINAL",
    "SPLEEN", "TEMPLE", "TENDON", "TENSOR", "TESTIS", "TIBIAE",
    "THROAT", "THYMIC", "THYMUS", "TONGUE", "TONSIL", "TRAGUS",
    "UTERUS", "VULVAR", "VAGINA",
    "ABDOMEN", "ALVEARY", "ALVEOLI", "AURICLE", "BLADDER", "CARDIAC",
    "COCHLEA", "CRANIUM", "GASTRIC", "GENITAL", "GLOTTIS", "GRISTLE",
    "HEPATIC", "JEJUNUM", "JUGULAR", "KNEECAP", "LEVATOR", "LINGUAL",
    "MAMMARY", "MARITAL", "MEMBRAL", "NEURONE", "OSSEOUS", "PATELLA",
    "PHALANX", "PLEURAL", "PYLORIC", "PYLORUS", "RIBCAGE", "ROTATOR",
    "STERNUM", "STOMACH", "SUBLIMG", "THYROID",
    "APPENDIX", "AXILLARY", "BRACHIAL", "BRACHIUM", "BRONCHUS",
    "CEREBRAL", "CERVICAL", "COCHLEAR", "CORONARY", "DUODENAL",
    "DUODENUM", "EXTENSOR", "FORESKIN", "GENITALS", "GINGIVAL",
    "LARYNGAL", "LIGAMENT", "MANDIBLE", "MUSCULAR", "OPPONENT",
    "PANCREAS", "PARIETAL", "PECTORAL", "PERINEAL", "PERINEUM",
    "PROXIMAL", "SHOULDER", "THORACIC", "VENA CAVA", "VERTEBRA",
    "VOICE-BOX", "WINDPIPE",
]
add(["anatomy term", "body part", "part of the body"], anatomy_terms)

# ============ ANCHORS ============
anchors = [
    "CQR", "GRAPNEL", "KILLICK", "KILLOCK", "MUSHROOM", "SHEET",
    "YACHTSMAN", "DROGUE", "PLOUGH", "DANFORTH",
    "KEDGE", "BOWER", "DRIFT", "BRIDGE", "SEA ANCHOR",
    "DOUBLE FLUKED",
]
add(["anchor", "mooring"], anchors)

# ============ ANGELS ============
angels = [
    "ARIEL", "EBLIS", "IBLIS", "SATAN", "URIEL",
    "ABDIEL", "ARIOCH", "AZRAEL", "BELIAL", "MAMMON",
    "GABRIEL", "ISRAFEI", "LUCIFER", "MICHAEL", "RAPHAEL", "ZADKIEL",
    "ITHURIEL", "BEELZEBUB", "MOLOCH", "ZEPHON",
]
add(["angel"], angels)

# Orders of angel
angel_orders = [
    "ANGEL", "POWER", "CHERUB", "SERAPH", "THRONE", "VIRTUE",
    "DOMINION", "ARCHANGEL", "DOMINATION", "PRINCIPALITY",
]
add(["order of angel", "angelic order"], angel_orders)

# ============ ANGLE TYPES ============
angle_types = [
    "ACUTE", "RIGHT", "OBTUSE", "STRAIGHT", "REFLEX", "CONJUGATE",
    "HOUR-ANGLE", "COMPLEMENTARY", "SUPPLEMENTARY",
]
add(["angle", "angle type"], angle_types)

# ============ ANGLE MEASUREMENTS ============
angle_measurements = [
    "HOUR", "GRADE", "POINT", "DEGREE", "MINUTE", "RADIAN", "SECOND",
    "ARCMINUTE", "ARCSECOND", "STERADIAN",
    "REVOLUTION", "DEGREE OF ARC", "MINUTE OF ARC", "SECOND OF ARC",
    "RADIAN PER SECOND",
]
add(["angle measurement", "unit of angle"], angle_measurements)

# ============ COLLECTIVE NOUNS FOR ANIMALS ============
collective_nouns = {
    "HERD": ["cattle", "deer", "elephants", "goats"],
    "FLOCK": ["birds", "geese", "sheep"],
    "PACK": ["dogs", "wolves", "hounds"],
    "PRIDE": ["lions"],
    "MURDER": ["crows"],
    "SCHOOL": ["dolphins", "fish", "whales"],
    "SHOAL": ["fish"],
    "GAGGLE": ["geese"],
    "POD": ["seals", "whales"],
    "SWARM": ["ants", "bees"],
    "COLONY": ["ants", "bees", "penguins"],
    "LITTER": ["kittens", "pigs"],
    "BROOD": ["chickens", "hens"],
    "TROOP": ["baboons", "kangaroos", "monkeys"],
    "CRASH": ["rhinoceros"],
    "BLOAT": ["hippopotami"],
    "KINDLE": ["kittens"],
    "LABOUR": ["moles"],
    "SLEUTH": ["bears"],
    "TOWER": ["giraffes"],
    "PARLIAMENT": ["owls"],
    "WEDGE": ["swans"],
    "CHARM": ["finches", "goldfinches"],
    "SKULK": ["foxes"],
    "STREAK": ["tigers"],
    "SHREWDNESS": ["apes"],
    "AMBUSH": ["tigers"],
    "KENNEL": ["dogs"],
    "MUSTER": ["peacocks"],
    "POUNCE": ["cats"],
    "RAFTER": ["turkeys"],
    "SCURRY": ["squirrels"],
    "SPRING": ["teal"],
    "STABLE": ["horses"],
    "STRING": ["horses", "ponies"],
    "VOLERY": ["birds"],
    "COVEY": ["partridges", "quail"],
    "BRACE": ["ducks"],
    "DROVE": ["cattle", "oxen", "sheep"],
    "EXALTATION": ["larks"],
    "DESCENT": ["woodpeckers"],
    "BOUQUET": ["pheasants"],
    "CLOWDER": ["cats"],
    "COMPANY": ["parrots"],
    "PRICKLE": ["porcupines"],
    "ROOKERY": ["rooks", "seals"],
    "SOUNDER": ["swine"],
    "TURMOIL": ["porpoises"],
}
# Add as "collective noun" entries
for noun, animals in collective_nouns.items():
    add(["collective noun", "group of animals", "group"], [noun])

# ============ MALE ANIMALS ============
male_animals = {
    "COB": "swan", "DOG": "dog/fox/wolf", "RAM": "sheep",
    "TOM": "cat", "TUP": "sheep", "BOAR": "pig",
    "BUCK": "deer/goat/hare/rabbit", "BULL": "cattle/elephant/moose/walrus/whale",
    "COCK": "chicken/crab/lobster/salmon/sparrow", "HART": "deer",
    "JACK": "ass/donkey", "STAG": "deer", "BILLY": "goat",
    "DRAKE": "duck", "DRONE": "honey bee", "GANDER": "goose",
    "MUSKET": "sparrowhawk", "TIERCEL": "hawk",
    "STALLION": "horse", "BLACKCOCK": "black grouse",
    "TURKEY COCK": "guinea fowl/turkey",
}
add(["male animal", "male"], list(male_animals.keys()))

# ============ FEMALE ANIMALS ============
female_animals = {
    "COW": "cattle/elephant/elk/whale", "DOE": "antelope/deer/hare/kangaroo/rabbit",
    "EWE": "sheep", "HEN": "chicken", "PEN": "swan", "REE": "sandpiper",
    "RUFF": "pig/badger",
    "GILL": "ferret", "HIND": "deer", "JILL": "ferret",
    "JENNY": "zho", "MARE": "horse",
    "BITCH": "dog/fox/wolf", "NANNY": "goat",
    "QUEEN": "cat", "PEAHEN": "peacock",
    "LIONESS": "lion", "TIGRESS": "tiger",
    "VIXEN": "fox", "GREYHEN": "black grouse",
    "LEOPARDESS": "leopard",
}
add(["female animal", "female"], list(female_animals.keys()))

# ============ ANIMAL HOMES ============
animal_homes = {
    "DEN": "bear/lion", "NID": "pheasant", "STY": "pig",
    "BIKE": "wasp/wild bee", "BINK": "wasp/wild bee",
    "DREY": "squirrel", "FOLD": "sheep",
    "EARTH": "fox", "EYRIE": "eagle", "LODGE": "beaver",
    "BURROW": "rabbit", "WARREN": "rabbit",
    "KENNEL": "dog", "STABLE": "horse",
    "HOLT": "otter", "SETT": "badger",
    "LAIR": "wild animal", "NEST": "bird",
    "COOP": "fowl", "CAGE": "squirrel",
    "BYRE": "cow", "HUTCH": "rabbit",
    "FORMICARY": "ants", "TERMITARIUM": "termites",
}
add(["animal home", "lair", "den", "nest"], list(animal_homes.keys()))

# ============ ANIMAL SOUNDS ============
animal_sounds = [
    "BAA", "BAY", "COO", "COW", "LOW", "MEW", "MOO", "YAP",
    "BARK", "BLAT", "BRAY", "CROW", "HISS", "HOOT", "HOWL",
    "PURR", "WOOF", "YELP",
    "BLEAT", "CHEEP", "CHIRP", "CLUCK", "CROAK", "GROWL", "GRUNT",
    "NEIGH", "QUACK", "SNARL", "TWEET",
    "BELLOW", "CACKLE", "GABBLE", "GOBBLE", "HEEHAN", "SQUAWK",
    "CHIRRUP", "GRUNTLE", "SCREECH", "TRUMPET", "TWITTER", "WHICKER",
    "CATERWAUL",
]
add(["animal sound", "cry", "call"], animal_sounds)

# ============ YOUNG ANIMALS ============
young_animals = {
    "CUB": "bear/lion/wolf", "ELF": "eel", "FRY": "fish",
    "KID": "antelope/goat", "KIT": "ferret/fox/polecat",
    "NIT": "louse",
    "BRIT": "herring/sprat", "CALF": "cattle/elephant/whale",
    "COLT": "horse", "EYAS": "hawk", "FAWN": "deer",
    "FOAL": "horse", "GILT": "female pig", "GRIG": "eel",
    "JOEY": "kangaroo", "LAMB": "sheep",
    "MAID": "skate", "PARR": "salmon", "PEAL": "sea trout",
    "SPAT": "cow", "YELT": "female pig",
    "BUNNY": "rabbit", "CHICK": "chicken", "ELVER": "eel",
    "OWLET": "owl", "POULT": "chicken",
    "ALEVIN": "fish", "CUDDLE": "coalfish", "EAGLET": "eagle",
    "EYELET": "swan", "PIGLING": "pig",
    "PULLET": "chicken", "SMALET": "salmon",
    "WHELP": "dog", "PUPPY": "dog",
    "SHOAT": "pig", "PIGLET": "pig",
    "KITTEN": "cat", "GOSLING": "goose",
    "LEVERET": "hare", "DUCKLING": "duck",
    "COCKEREL": "cock", "YEARLING": "horse",
}
add(["young animal", "offspring", "young"], list(young_animals.keys()))

# ============ WEDDING ANNIVERSARIES ============
anniversaries = {
    "PAPER": "1st", "TIN": "10th", "CRYSTAL": "15th",
    "CHINA": "20th", "SILVER": "25th", "PEARL": "30th",
    "CORAL": "35th", "RUBY": "40th", "GOLD": "50th",
    "DIAMOND": "60th", "PLATINUM": "70th",
    "FUR": "13th", "COTTON": "2nd", "LEATHER": "3rd",
    "LINEN": "4th", "WOOD": "5th", "IRON": "6th",
    "WOOL": "7th", "IVORY": "14th", "LACE": "13th",
    "SILK": "12th", "STEEL": "11th",
    "COPPER": "7th", "BRONZE": "8th",
    "EMERALD": "55th",
    "JADE": "35th",
}
add(["wedding anniversary", "anniversary"], list(anniversaries.keys()))

# ============ ANTS ============
ants = [
    "RED", "ARMY", "FIRE", "LEAF", "WOOD",
    "BLACK", "CRAZY", "AMAZON", "DRIVER", "WEAVER",
    "BULLDOG", "FORAGER", "PHARAOH", "SOLDIER",
    "HONEYDEW", "BLACK LAWN", "CARPENTER", "HARVESTER",
    "LEAF-CUTTER", "RED HARVESTER",
]
add(["ant", "insect"], ants)

# ============ ANTELOPES ============
antelopes = [
    "BOK", "DOE", "GNU", "KID", "KOB",
    "KUDU", "ORYX", "PUKU", "SUNI", "THAR", "TOPI",
    "ADDAX", "BUBAL", "CHIRU", "ELAND", "GORAL", "NAGOR", "NYALA",
    "ORIBI", "SABLE", "SAIGA", "SEROW",
    "BOSBOK", "DIK-DIK", "DUIKER", "DUYKER", "DZEREN", "IMPALA",
    "INYALA", "KOODOO", "LECHWE", "NILGAI", "NILGAU", "PYGARG", "PALAEH",
    "BLAUBOK", "BLESBOK", "BLUEBUCK", "BUBALIS", "CHAMOIS", "CHIKARA",
    "GEMSBOK", "GERENUK", "GRYSBOK", "MADOQUA", "NYIGHAU", "SASSABY",
    "ANTELOPE", "BONTEBOK", "BOSCHBOK", "BUSHBUCK", "BLAUBOK",
    "BLESBOK", "REEDBUCK", "TESSEBE",
    "BLACKBUCK", "SITATUNGA", "SPRINGBOK",
    "STEINBUCK", "TRAGELOPE", "WATERBUCK",
    "ALCELAPHUS", "HARTEBEEST", "OX-ANTELOPE", "WILDEBEEST",
    "ZEBRA DUIKER", "GOAT-ANTELOPE", "KLIPSPRINGER",
    "SABLE ANTELOPE",
]
add(["antelope", "animal"], antelopes)

# ============ ANTIBIOTICS ============
antibiotics = [
    "CIPRO", "ALLICIN", "NEOMYCIN", "NYSTATIN",
    "AMPICARIN", "KANAMYCIN", "POLYMYXIN", "AUREOMYCIN",
    "BACITRACIN", "GRAMICIDIN", "LINCOMYCIN", "METHICILLIN",
    "PENICILLIN", "POLYMYXIN B", "RIFAMPICIN", "TERRAMYCIN",
    "VANCOMYCIN", "AMOXICILLIN", "AMOXYCILLIN", "CLINDAMYCIN",
    "CREASOLIINE", "DOXYCYCLINE", "FUSIDIC ACID", "METHICILLIN",
    "ERYTHROMYCIN", "GRISEOFULVIN", "STREPTOMYCIN",
    "TETRACYCLINE", "TRIMETHOPRIM", "CIPROFLOXACIN",
    "OXYTETRACYCLINE", "CHLORAMPHENICOL",
]
add(["antibiotic", "drug", "medicine"], antibiotics)

# ============ ANTIQUES TERMS ============
antiques_terms = [
    "GOSS", "MING", "RING", "TANG",
    "GLAZE", "IVORY",
    "BASALT", "DEALER", "EMPIRE", "GOTHIC", "LUSTRE", "PATINA", "PERIOD", "ROCOCO",
    "ART DECO", "FILIGREE", "GEORGIAN", "JACOBEAN", "MAJOLICA", "SHERATON",
    "TRECENTO", "BAROCCO", "BAROQUE", "CERAMIC", "FEDERAL", "IMPASTO",
    "OPALINE", "PILGRIM", "POTTERY",
    "BONE CHINA", "COLLECTOR", "DELFTWARE", "EDWARDIAN", "PORCELAIN",
    "QUEEN ANNE", "SOFT PASTE", "STONEWARE", "VALUATION", "VICTORIAN",
    "ART NOUVEAU", "MILLEFIORI", "CHINOISERIE", "CHIPPENDALE",
    "CINQUECENTO", "RESTORATION", "HEPPLEWHITE", "PERIOD PIECE",
    "ARTS AND CRAFTS", "WILLOW PATTERN", "CHURRIGUERESQUE",
]
add(["antique", "antiques term"], antiques_terms)

# ============ ANTISEPTICS ============
antiseptics = [
    "TCP", "SAVLON", "THYMOL", "EUPAD", "EUSOL",
    "BENZOIN", "IODINE", "CRESOL", "CREOSOTE", "FORMALIN", "IODOFORM",
    "DETTOL", "FLAVIN", "FORMOL", "PHENOL",
    "CASSAREEP", "CASSINIPE", "CERMICIDE", "GERMOLENE", "LISTERINE",
    "METHREMIN", "ZINC OXIDE", "ACRIFLAVIN",
    "CARBOLIC ACID", "METHYL VIOLET", "CHLORHEXIDINE", "CRYSTAL VIOLET",
    "GENTIAN VIOLET", "SILVER NITRATE",
]
add(["antiseptic", "disinfectant"], antiseptics)

# ============ APES ============
apes = [
    "CHIMP", "ORANG", "PONGO", "BONOBO", "GIBBON", "GORILLA",
    "ORANG-UTAN", "CHIMPANZEE", "ORANG-OUTANG",
    "PYGMY CHIMPANZEE",
]
add(["ape", "primate"], apes)

# ============ APOSTLES ============
apostles = [
    "JOHN", "JAMES", "JUDAS", "PETER", "SIMON",
    "ANDREW", "PHILIP", "THOMAS", "MATTHEW", "MATTHIAS", "THADDEUS",
    "BARTHOLOMEW", "JUDAS ISCARIOT", "SIMON THE ZEALOT",
    "JAMES OF ALPHAEUS", "SIMON THE CANAANITE",
]
add(["apostle", "disciple"], apostles)

# ============ APPLE VARIETIES ============
apples = [
    "COX", "CRAB", "SNOW", "EATER", "BIFFIN", "CODLIN", "COOKER",
    "EATING", "IDARED", "PIPPIN", "RUSSET",
    "BALDWIN", "BRAMLEY", "CODLING", "COOKING", "COSTARD",
    "CRISPIN", "RIBSTON", "STURMER", "WINE-SAP",
    "BRAEBURN", "JONATHAN", "MCINTOSH", "PEARMAIN", "PINK LADY",
    "QUEENING", "RIBSTONE", "SWEETING",
    "DELICIOUS", "JENNETING", "KING APPLE", "NONPAREIL", "ROYAL GALA",
    "GRANNY SMITH", "RED DELICIOUS", "RIBSTON PIPPIN",
    "GOLDEN DELICIOUS",
]
add(["apple", "apple variety", "fruit"], apples)

# ============ ARAB LEAGUE MEMBERS ============
arab_league = [
    "IRAQ", "OMAN", "EGYPT", "LIBYA", "QATAR", "SUDAN", "SYRIA", "YEMEN",
    "JORDAN", "KUWAIT", "ALGERIA", "BAHRAIN", "LEBANON", "MOROCCO", "SOMALIA",
    "TUNISIA", "DJIBOUTI", "PALESTINE", "MAURITANIA", "SAUDI ARABIA",
    "UNITED ARAB EMIRATES",
]
add(["Arab League member", "Arab country"], arab_league)

# ============ ARCHES ============
arches = [
    "KEEL", "OGEE", "SKEW", "ROUND", "TUDOR",
    "CONVEX", "CORBEL", "GOTHIC", "LANCET", "NORMAN", "TREFOIL",
    "POINTED", "STILTED",
    "INVERTED", "REVERSED", "PARABOLIC", "HORSESHOE", "EQUILATERAL",
    "FOUR-CENTRED", "SHOULDERED",
    "BASKET HANDLE", "THREE-CENTRED",
    "OGIVAL", "TRIUMPHAL",
]
add(["arch", "arch type"], arches)

# ============ ARCHAEOLOGY TERMS ============
archaeology_terms = [
    "AXE", "CUP", "POT", "DIG", "BARROW", "CAIRN", "FLINT", "MOUND",
    "SHARD", "AUGER", "BLADE", "BURIN", "CAIRN", "DOLMEN",
    "IRON AGE", "MENHIR", "MOSAIC", "NEOLITH", "ROCK ART",
    "SONDAGE", "STRATUM", "TUMULUS",
    "AMULET", "ARTIFACT", "CAPSTONE", "CHELLEAM",
    "CITADEL", "DOLMEN", "EXCAVATE", "FLINT AXE",
    "HILL FORT", "KITCHEN", "KEY HOLE", "MEGALITH",
    "ANGLO-SAXON", "HERITAGE", "CAIRNFIELD",
    "CLACTONIAN", "CUP-AND-RING", "FIRE-PLOUGH",
    "GEOPHYSICS", "CROP MARK", "HIEROGLYPH",
    "PALAEOLITH", "MESOLITHIC", "MOUSTERIAN",
    "KITCHEN MIDDEN", "STANDING STONE", "TREASURE TROVE",
    "ARCHAEOLOGY", "CLEARANCE CAIRN",
]
add(["archaeology term", "archaeological term"], archaeology_terms)

# ============ TERMS TO DO WITH ANIMALS ============
animal_terms = [
    "WAR", "EGG", "FUR", "PAW", "HIDE", "WING", "WILD", "CREST",
    "FERAL", "MOUNT", "POUCH",
    "BEAK", "BILL", "CLAW", "FANG", "HOOF", "HORN", "MANE",
    "PELT", "SNOUT", "UDDER",
    "ANTLER", "BARREL", "DEWLAP", "WHISKERS",
    "DIDELPHIC", "GASTROEUM", "MARSUPIUM",
    "OVIPAROUS", "PROBOSCIS",
]
add(["animal term", "animal feature"], animal_terms)

conn.commit()
print(f"\nTotal inserted: {inserted}")
print(f"Total skipped: {skipped}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"Total crossword_companion entries: {c.fetchone()[0]}")
conn.close()
