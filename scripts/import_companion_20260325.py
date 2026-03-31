"""Import 2026-03-25 scans: Astrology, Astronauts, Beliefs/Believers, Berries,
Cactus, Cakes/Pastries/Puddings, Cattle breeds, Cathedrals, Capital cities,
and many other sections from the Crossword Companion book."""
import sqlite3

DB_PATH = "data/cryptic_new.db"
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

c.execute("SELECT LOWER(definition), UPPER(answer) FROM definition_answers_augmented")
existing = set((r[0], r[1]) for r in c.fetchall())
print(f"Existing pairs: {len(existing)}")

inserted = 0
skipped = 0
section_counts = {}

def add(definitions, words, section=None):
    global inserted, skipped, existing
    count = 0
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
            count += 1
    if section:
        section_counts[section] = section_counts.get(section, 0) + count


# =====================================================================
# SCAN 1: Scanned_20260325-1433 (Astrology, Astronauts + bonus sections)
# =====================================================================

# ============ ASTROLOGY TERMS ============
astrology_terms = [
    # 02
    "PI",
    # 03
    "ARC", "RAM", "SUN",
    # 04
    "AURA", "CAST", "CUSP", "FATE", "FIRE", "GOAT", "MARS", "MOON", "NODE",
    "WOOD",
    # 05
    "ARIES", "HORSE", "HOUSE",
    # 06
    "APPEAR", "ASTRAL", "CANCER", "DRAGON", "GEMINI", "MONKEY", "OCCULT",
    "ORACLE", "PISCES", "RABBIT", "SATURN", "SPIRIT", "TAURUS", "TRIGON",
    "URANUS", "ZODIAC",
    # 07
    "ADMETOS",
    # 08
    "AQUARIUS", "EPHEMERIS", "EQUINOX",
    # 09
    "ASCENDANT", "ASPECTUAL", "CAPRICORN", "CELESTIAL", "HOROSCOPE",
    "IMUM COELI", "INFLUENCE",
    # 10
    "OPPOSITION", "PREDICTION",
    # 11
    "ASTROLOGER", "ASTROLOGIST", "CONJUNCTION", "MEDIUM COELI",
    "PROGRESSION", "SAGITTARIUS", "SATELLITIUM",
    # 12
    "ASTRONAVALI", "DEGREE OF FATE", "PLANET-STRUCK", "SIGNIFICATOR",
    # 13
    "CONSTELLATION",
    # 14
    "ACRONYCAL PLACE", "PLANET-STRICKEN",
]
# Additional terms from the page
astrology_signs = [
    "ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO",
    "LIBRA", "SCORPIO", "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES",
]
astrology_animals = [
    "RAT", "OX", "TIGER", "RABBIT", "DRAGON", "SNAKE",
    "HORSE", "GOAT", "MONKEY", "ROOSTER", "DOG", "PIG",
]
add(["star sign", "zodiac sign", "astrological term", "sign of the zodiac"], astrology_signs, "Astrology signs")
add(["astrological term", "astrology term", "horoscope term"], astrology_terms, "Astrology terms")
add(["Chinese zodiac animal", "Chinese year animal"], astrology_animals, "Astrology Chinese animals")

# Birth symbols / zodiac associations
birth_symbols = [
    "RAM", "BULL", "TWINS", "CRAB", "LION", "VIRGIN",
    "SCALES", "SCORPION", "ARCHER", "GOAT", "WATER BEARER", "FISHES",
]
add(["birth symbol", "zodiac symbol"], birth_symbols, "Astrology birth symbols")

# ============ ASTRONAUTS ============
astronauts = [
    # 04
    "BEAN", "RIDE",
    # 05
    "FOALE", "GLENN", "IRWIN", "TITOV", "WHITE",
    # 06
    "ALDRIN", "CONRAD", "LEONOV", "LOVELL",
    # 07
    "CHAFFEE", "COLLINS", "COLLINS", "GAGARIN", "GRISSOM", "SCHIRRA",
    "SHARMAN", "SHEPARD",
    # 08
    "MITCHELL", "WILLIAMS",
    # 09
    "ARMSTRONG",
    # 10
    "TERESHKOVA",
]
add(["astronaut", "spaceman", "space traveller", "cosmonaut"], astronauts, "Astronauts")

# ============ ASTROPHYSICS / ASTRONOMY ============
astronomy_terms = [
    # 03
    "GAS", "NEO", "SUN",
    # 04
    "CORE", "FLUX", "MOON", "MODE", "NOVA", "STAR", "VAPI",
    # 05
    "COMET", "COUDE", "EPOCH", "GIANT", "ORBIT", "RINGS", "UMBRA",
    # 06
    "APOGEE", "BLAZAR", "CORONA", "COSMOS", "GALAXY", "JANSKY", "LANDER",
    "METEOR", "NEBULA", "PARSEC", "PLANET", "PULSAR", "QUASAR", "SPINAR",
    "SYZYGY",
    # 07
    "ALMANAC", "ANOMALY", "AZIMUTH", "BIG BANG", "ECLIPSE", "METONIC",
    # 08
    "APHELION", "ASTEROID", "CONICAL", "ECLIPTIC", "EMERSION", "EVECTION",
    "GAS GIANT", "INFERIOR", "INFRARED", "MILKY WAY", "MUTATION", "PARALLAX",
    "PROGRADE", "RED DWARF", "RED GIANT", "RED SHIFT", "SUBGIANT", "SUNSPOTS",
    "TOTALITY", "TYCHONIC", "UNIVERSE",
    # 09
    "AIR SHOWER", "ASTRODOME", "BLACK BODY", "BLACK HOLE", "COCKTAIL",
    "COLLAPSAR", "EPHEMERIS", "EXOPLANET", "GREAT YEAR", "HOUR-ANGLE",
    "HYPERNOVA", "IMMERSION", "MAGNITUDE", "OORT CLOUD", "POLAR AXIS",
    "PROTOSTAR", "RADIO STAR", "ROCHE LOBE", "SATELLITE", "SHELL STAR",
    "SUPERNOVA", "TELESCOPE",
    # 10
    "ABERRATION", "ALMANCANTAR", "ASTROMETRY", "BINARY STAR", "BLACK DWARF",
    "BROWN DWARF", "COPERNICAN", "COSMIC RAYS", "DARK ENERGY", "DARK MATTER",
    "DOUBLE STAR", "INEQUALITY", "KUIPER BELT", "LOCAL GROUP", "LUMINOSITY",
    "PERIASTRON", "PERIHELION", "RETROGRADE", "ROCHE LIMIT", "SUPERGIANT",
    "WHITE DWARF",
    # 11
    "BAILY'S BEADS", "BRIGHT GIANT", "DECLINATION", "GEGENSCHEIN",
    "HELIUM FLASH", "MAGNETOSTAR", "NEUTRON STAR", "OBSERVATORY",
    "OCCULTATION", "SINGULARITY", "SOLAR SYSTEM",
    # 12
    "BINARY PULSAR", "COSMIC STRING", "DOPPLER SHIFT", "EVENT HORIZON",
    "GALACTIC HALO", "HELIOCENTRIC", "MAIN SEQUENCE", "METEOR SHOWER",
    "METONIC CYCLE", "PERTURBATION", "SHEPHERD MOON", "SPECTRAL TYPE",
    "SPIRAL GALAXY", "SUPERCLUSTER",
    # 13
    "ACCRETION DISK", "CELESTIAL BODY", "CONSTELLATION", "OLBERS' PARADOX",
    "SEYFERT GALAXY", "SOLAR CONSTANT", "SYNODIC PERIOD", "X-RAY ASTRONOMY",
    # 14
    "CELESTIAL POLES", "CHANDLER WOBBLE", "CLOSED UNIVERSE", "EQUATION OF TIME",
    "HUBBLE CONSTANT", "RADAR ASTRONOMY", "RIGHT ASCENSION", "SPACE TELESCOPE",
    # 15
    "ARMILLARY SPHERE", "CELESTIAL SPHERE", "CEPHEID VARIABLE",
    "ECLIPSING BINARY", "GLOBULAR CLUSTER", "HUBBLE TELESCOPE",
    "NEAR-EARTH OBJECT",
]
add(["astronomy term", "astronomical term"], astronomy_terms, "Astronomy")

# ============ ATHLETICS ============
athletics_terms = [
    "PIT", "SB", "SKIP", "STEP", "WALK",
    "BATON", "BOARD", "BREAK", "FIELD", "PRIZE", "RELAY", "TRACK",
    "BLOCKS", "CIRCLE", "NO JUMP",
    "RECORD", "RUNWAY", "SPIKES", "TARTAN",
    "BRAVO IN",
    "QUALIFY", "RED FLAG", "SHOT PUT", "STUTTER",
    "DESELECT", "DISTANCE", "GYMNASIUM", "STRAIGHT",
    "HITCH KICK", "PACEMAKER", "PANCAKE", "POLE VAULT", "WATER JUMP",
    "WHITE FLAG",
    "DISQUALIFY", "FALSE START", "FINISH LINE",
    "PLASTICINE", "FOSBURY FLOP", "HEPTATHLETE", "PHOTO FINISH",
    "STEEPLECHASE",
    "TRACK RECORD", "WESTERN ROLL",
    "HOME STRAIGHT", "LONG DISTANCE", "PERSONAL BEST", "REACTION TIME",
    "TAKE-OVER ZONE", "WIND-ASSISTED",
    "FOLLOWING WIND",
    "CROUCHING START", "MIDDLE DISTANCE", "STAGGERED START",
]
athletics_events = [
    "100M", "200M", "400M", "800M", "1500M",
    "3000M", "5000M", "10000M",
    "SPRINT", "HURDLES", "RELAY",
    "50KM WALK",
    "MARATHON", "LONG JUMP", "HIGH JUMP", "TRIPLE JUMP",
    "SHOT PUT", "DISCUS", "JAVELIN", "HAMMER",
    "DECATHLON", "HEPTATHLON", "PENTATHLON",
    "POLE VAULT", "STEEPLECHASE",
]
add(["athletics term", "athletics event", "track and field"], athletics_terms, "Athletics terms")
add(["athletics event", "Olympic event", "track event"], athletics_events, "Athletics events")

# ============ ATMOSPHERE LAYERS ============
atmosphere_layers = [
    "EXOSPHERE", "IONOSPHERE", "MESOSPHERE", "OZONE LAYER",
    "TROPOPAUSE", "STRATOPAUSE", "TROPOSPHERE",
    "PLASMASPHERE", "STRATOSPHERE", "THERMOSPHERE",
]
add(["atmospheric layer", "layer of atmosphere"], atmosphere_layers, "Atmosphere")

# ============ SUBATOMIC PARTICLES ============
subatomic_particles = [
    "W", "X", "Z",
    "PSI",
    "UPSI", "KAON", "MUON", "PION",
    "BOSON", "GLUON", "MESON", "OMEGA",
    "PROTON", "QUARK", "SIGMA",
    "BARYON", "B-MESON", "HADRON", "LAMBDA", "LEPTON", "PARTON", "PHOTON",
    "PROTON", "NEUTRON", "ANTIQUARK",
    "NUCLEON", "PI MESON", "UP QUARK", "UPSILON",
    "ELECTRON", "NEUTRINO", "NEUTRINO", "NEUTRON",
    "DOWN QUARK", "GRAVITON", "TAU LEPTON",
    "ANTI-LEPTON", "GAUGE BOSON", "TRUTH QUARK",
    "ANTI-NEUTRON", "BEAUTY QUARK", "BOTTOM QUARK",
    "ANTI-NEUTRINO", "CHARMED QUARK", "STRANGE QUARK",
]
add(["subatomic particle", "particle", "elementary particle"], subatomic_particles, "Subatomic particles")

# ============ BADMINTON ============
badminton_terms = [
    "NET", "SET", "BIRD", "KILL", "CLEAR",
    "COURT", "DRIVE", "FLICK", "RALLY", "SERVE",
    "RACKET", "DOUBLES", "SINGLES",
    "DROP SHOT", "SHUTTLECOCK", "SERVICE COURT",
    "UNDERARM CLEAR", "WOOD SHOT",
    "SMASH",
]
add(["badminton term", "badminton"], badminton_terms, "Badminton")

# ============ BAKING / CAKES (from scan 1433 page 4 — partial) ============
baking_items_1433 = [
    "BUN", "PIE", "OAT", "PAN", "TEA",
    "BABA", "FLAN", "FOOL", "PLUM", "ROCK", "SEED", "TART",
    "ANGEL", "BOMBE", "BRIDE", "BUNDT", "CREAM", "CREPE", "FAIRY",
    "FRUIT", "FUDGE", "GENOA", "JELLY", "LARDY", "LAYER", "POUND",
    "QUEEN", "SCONE", "SHORT", "SWEET", "TIPSY", "TORTE", "YEAST",
    "BANANA", "CARROT", "CHEESE", "COFFEE", "DUNDEE", "ECCLES",
    "ECLAIR", "GATEAU",
    "COOKIE", "DANISH", "STOLLEN", "STRUDEL", "TARTLET",
    "MUFFIN", "PARKIN", "SPONGE", "TRIFLE", "WAFFLE",
    "BAKLAVA", "BANBURY", "BANNOCK", "BATH BUN", "BRIOCHE",
    "BROWNIE", "CRUMBLE", "CRUMPET", "CURRANT", "FIG ROLL",
    "GINGER", "GIRDLE", "JUNKET", "MARBLE", "MOUSSE",
    "SIMNEL", "YUM-YUM",
]
# Don't double-add these if the cake section from scan 1444 covers them too
# We'll add them all — dedup handles it
add(["cake", "pastry", "pudding", "sweet", "confection", "baked item"], baking_items_1433, "Baking (scan 1)")

# ============ BEANS ============
beans = [
    "DAL", "PEA", "SOY", "SUGAR",
    "DHAL", "FAVA", "GRAM", "LIMA", "MUNG", "OKRA", "SNAP",
    "ADUKI", "BLACK", "BROAD",
    "CAROB", "GREEN", "PINTO", "SOYA",
    "ADZUKI", "BOSTON", "BUTTER", "CHILLI", "COWPEA", "FRENCH",
    "LEGUME", "LENTIL", "RUNNER",
    "ALFALFA", "EDAMAME", "FLAGEOLET",
    "STRING", "PETIT POIS", "PIGEON PEA",
    "RED KIDNEY", "RED LENTIL",
    "BEANSPROUT", "GOLDEN GRAM",
    "DWARF GRAM", "GARBANZO", "GREEN LENTIL",
    "BLACK-EYED", "BORLOTTI", "GARBANZO", "SPLIT PEA", "TOOR PEA",
    "BLACK GRAM", "MARROWFAT PEA",
    "SCARLET RUNNER", "WATER CHESTNUT",
    "MANGETOUT", "CHICKPEA",
]
add(["bean", "pulse", "legume", "vegetable"], beans, "Beans")

# ============ BEARS ============
bears = [
    "BRUIN", "GREAT", "HONEY", "KOALA", "NANDI", "POLAR",
    "YOGI",
    "BALOO", "BLACK", "BROWN",
    "WATER", "WHITE",
    "LITTLE", "NATIVE", "RUPERT", "WOOLLY",
    "GRIZZLY", "MALAYAN",
    "CINNAMON", "URSA MAJOR", "URSA MINOR",
    "GIANT PANDA", "PADDINGTON",
    "KODIAK",
    "IOREK BYRNISON", "TEDDY ROBINSON",
    "WINNIE THE P",
    "SLOTH", "SOOTY",
    "SUN", "SEA",
]
add(["bear", "animal"], bears, "Bears")

# ============ BEDS ============
beds = [
    "Z", "BOX", "COT", "DAY", "BOAT", "BUNK", "CAMP", "CRIB",
    "SOFA", "TWIN", "BERTH",
    "FUTON", "WATER",
    "SLEIGH", "FOLDING", "HAMMOCK", "TRESTLE", "TRUCKLE", "TRUNDLE",
    "BASSINET", "FOLDAWAY",
    "KING-SIZE", "MATTRESS", "PLATFORM",
    "COUCHETTE", "KING-SIZED", "LIT BATEAU",
    "ADJUSTABLE", "FOUR-POSTER", "MID SLEEPER", "QUEEN-SIZED",
    "HIGH SLEEPER", "CHAISE LONGUE",
    "CRADLE", "DOUBLE", "PALLET", "PUT-UP", "SETTEE", "SINGLE",
    "TRUNDLE",
]
add(["bed", "type of bed", "place to sleep"], beds, "Beds")

# ============ BEDCLOTHES ============
bedclothes = [
    "DOONA", "VALANCE",
    "DUVET", "COVERLET",
    "QUILT", "BED CANOPY", "BEDSPREAD",
    "SHEET", "COMFORTER", "EIDERDOWN", "THROWOVER",
    "PILLOW", "DUVET COVER",
    "BEDROLL", "BLANKET", "BOLSTER",
    "PILLOWCASE", "PILLOW SHAM", "PILLOWSLIP",
    "QUILT COVER", "COUNTERPANE", "FITTED SHEET", "SLEEPING BAG",
    "MATTRESS COVER",
    "VALANCED SHEET", "WITNEY BLANKET",
    "PATCHWORK QUILT", "CELLULAR BLANKET", "ELECTRIC BLANKET",
]
add(["bedclothes", "bedding", "bed linen"], bedclothes, "Bedclothes")

# ============ BEER ============
beers = [
    "ALE", "DRY", "ICE", "IPA", "KEG",
    "BOCK", "MILD", "PILS", "RICE",
    "ABBEY", "BLACK", "FRUIT", "GREEN", "GUEST", "HEAVY", "HONEY",
    "KVASS", "LAGER", "MARCH", "PLAIN", "SAINT", "SIXTY", "STEAM",
    "STONE", "STOUT", "WHEAT",
    "BITTER", "DUNKEL", "EIGHTY",
    "EXPORT", "GUEUZE", "HELLES", "KOLSCH", "LAMBIC", "MARZEN",
    "OLD ALE", "PORTER", "RED ALE", "SHANDY", "VIENNA",
    "ALTBIER", "BOTTLED", "DRAUGHT", "FLASK", "PALE ALE", "PILSNER",
    "REAL ALE", "SEVENTY",
    "AMBER ALE", "BROWN ALE", "CREAM ALE", "GUINNESS",
    "HOME BREW", "IRISH ALE", "LIGHT ALE", "PILSENER", "TRAPPIST",
    "FRAMBOISE", "FRAMBOZEN",
    "KING-SIZE", "MILK STOUT", "RAUCHBIER", "SNAKEBITE", "WEISSBIER",
    "WINTER ALE",
    "BARLEY WINE", "BLACK LAGER", "HARVEST ALE", "HEFEWEIZEN",
    "LOW-ALCOHOL", "MALT LIQUOR", "SWEET STOUT", "WEISSE BIER",
    "WEIZENBIER",
    "BLACK-AND-TAN",
    "BIERE DE GARDE", "CHRISTMAS ALE", "INDIA PALE ALE", "OATMEAL STOUT",
    "HEFE-WEISSBIER", "SIXTY SHILLING",
    "BERLINER WEISSE", "EIGHTY SHILLING", "KRISTALL-WEIZEN",
    "CASK-CONDITIONED", "SEVENTY SHILLING",
    "MICROBREW",
]
add(["beer", "ale", "brew", "drink"], beers, "Beer")

# ============ BEETLES ============
beetles = [
    "DOR", "STAG", "OIL",
    "BARK", "DUNG", "FLEA", "GOLD",
    "BLACK", "CLOCK",
    "LEAF", "ROSE", "STAG", "WATER",
    "SHARD", "SNOUT",
    "TIGER",
    "BURYING", "CARPET", "CHAFER",
]
add(["beetle", "insect"], beetles, "Beetles")

# ============ BREWING TERMS ============
brewing_terms = [
    "FIX", "LEG", "HOP",
    "BALE", "BARLEY", "COPPER", "LAGER",
    "CASK", "LAST", "HEAD", "BRISK",
    "HOPS", "BREW",
    "MALT", "STUM", "WORT",
    "DRAFF", "WIDGET",
    "GOURD", "GRAIN", "GRIST", "NAPPY", "ROUND", "STAVE",
    "BEVERAGE", "BUNNOCK", "DRAUGHT", "EXTRACT", "FERMENT",
    "FLOWERS", "GRAVITY",
    "YEAST", "BARREL", "LAUREL",
    "MALT", "MEDIUM", "STANK", "SWEETS", "TROUGH",
    "MASH", "MASH TUN",
    "PUREE",
    "BLACKING", "HOGSHEAD", "HORSE-BREW", "MALTSTER",
    "MOLASSES", "MUCILAGE", "PALE MALT", "PUNCHEON",
    "MALT BEER", "SPARGING",
    "AMBER MALT", "CHOCOLATE MALT",
    "SET MASH", "MALT ALE",
    "SYNERGIS",
    "BROWN MALT",
]
add(["brewing term", "beer-making term"], brewing_terms, "Brewing")


# =====================================================================
# SCAN 2: Scanned_20260325-1440 (Beliefs, Believers, Berries + bonus)
# =====================================================================

# ============ BELIEFS ============
beliefs = [
    # 06
    "HOLISM", "NUDISM", "RACISM",
    # 07
    "ANIMISM", "ABLEISM", "ELITISM",
    # 08
    "DEMONISM", "HEDONISM", "HUMANISM", "NIHILISM",
    # 09
    "PANTHEISM", "PHYSICISM", "TRITHEISM", "SATANISM",
    # 10
    "LIBERALISM", "MARCIONISM", "MONOTHEISM", "POLYTHEISM",
    # 11
    "AGNOSTICISM", "PARALLELISM",
    # 12
    "MANICHAEANISM", "SUPERNATURALISM",
    # 13
    "ETHNOCENTRISM", "INDIVIDUALISM", "STRUCTURALISM",
    # 14
    "FUNDAMENTALISM", "TRADITIONALISM", "SUPRANATURALISM",
    # 15
    "SUPERNATURALISM",
]
add(["belief", "religion", "faith", "creed", "-ism", "doctrine"], beliefs, "Beliefs")

# ============ BELIEVERS ============
believers = [
    # 03
    "JEW",
    # 04
    "BABI", "JAIN", "SIKH", "SUFI",
    # 05
    "BABEE", "HINDU", "JAINA", "SALESIAN",
    # 06
    "HOLIST", "MUSLIM", "PARSEE",
    # 07
    "ALANITE", "ANIMIST", "BAHAIST", "CATHARI", "LOLLARD", "SCOTIST",
    # 08
    "ARMINIAN", "BUDDHIST", "CALVINIST", "CATHOLIC",
    # 09
    "ANIMALIST", "CALIXTINE", "CHRISTIAN", "CONFUCIAN", "EUTYCHIAN",
    "GREGORIAN", "METHODIST", "NESTORIAN", "ORIGENIST", "PANTHEIST",
    "SABELLIAN",
    # Demonist, Erasmian, Giaouree, Humanist
    "DEMONIST", "ERASMIAN", "HUMANIST",
    # Lutheran, Nazarene, Nazirite, Salesian, Satanist, "Wesleyan"
    "LUTHERAN", "NAZARENE", "NAZIRITE",
    # 10
    "EVANGELIST",
    # 11
    "SANDEMANIAN", "SOUTHCOTTIAN",
    # 12
    "APOLLINARIAN", "HUTCHINSONIAN", "ROMAN CATHOLIC", "SWEDENBORGIAN",
    # 13
    "SIMEONITE", "WYCLIFFITE",
    # 14
    "FUNDAMENTALIST", "THE OXFORD GROUP",
    # 15
    "SUPERNATURALIST",
    # Additional
    "BERGOMASK", "BERKELEIAN", "CAMERONIAN", "CAPERNORTE", "HOLY ROLLER",
    "MARONITE", "POLYTHEIST", "WYCLIFFITE",
]
add(["believer", "follower", "devotee", "adherent"], believers, "Believers")

# ============ BERRIES ============
berries = [
    # 04
    "GOJI",
    # 05
    "LITCHI",
    # 06
    "LICHEE", "LYCHEE", "LITCHI",
    # 07
    "BRAMBLE", "LEECHEE",
    # 08
    "BILBERRY", "DEWBERRY", "GOOSEOOG",
    # 09
    "BLAEBERRY", "BLUEBERRY", "CRANBERRY", "RASPBERRY", "SHADBERRY",
    # 10
    "BLACKBERRY", "CLOUDBERRY", "ELDERBERRY", "GOOSEBERRY", "LOGANBERRY",
    "SALAL BERRY", "STRAWBERRY",
    # 11
    "BOYSENBERRY", "HUCKLEBERRY", "SALAL BERRY",
    # 12
    "BLACKCURRANT", "SERVICEBERRY", "WHITECURRANT", "WHORTLEBERRY",
    "MULBERRY", "TAYBERRY",
]
add(["berry", "fruit", "soft fruit"], berries, "Berries")

# ============ BIBLE — VERSIONS ============
bible_versions = [
    "AV", "RV", "NIV",
    "DOUAI", "DOUAY", "ITALA",
    "REIMS", "GENEVA", "GIDEON", "ITALIC", "WYCLIF",
    "MATTHEW", "PESHITO", "TYNDALE", "VULGATE",
    "BREECHES", "PESHITTA", "PESHITTO",
    "COVERDALE", "KING JAMES", "NEW ENGLISH", "SEPTUAGINT",
    "WYCLIFFE", "REVISED VERSION",
]
add(["Bible version", "version of the Bible", "Bible translation"], bible_versions, "Bible versions")

# ============ BIBLE — OLD TESTAMENT BOOKS ============
ot_books = [
    "JOB", "AMOS", "EZRA", "JOEL", "RUTH",
    "HOSEA", "JONAH", "KINGS", "MICAH", "NAHUM",
    "DANIEL", "ESTHER", "EXODUS", "HAGGAI", "ISAIAH", "JOSHUA",
    "JUDGES", "PSALMS", "SAMUEL",
    "EZEKIEL", "GENESIS", "MALACHI", "NUMBERS", "OBADIAH",
    "HABAKKUK", "JEREMIAH", "NEHEMIAH", "PROVERBS",
    "LEVITICUS", "ZECHARIAH", "ZEPHANIAH",
    "CHRONICLES",
    "DEUTERONOMY",
    "ECCLESIASTES", "LAMENTATIONS",
    "SONG OF SOLOMON",
]
add(["Old Testament book", "book of the Bible", "Bible book"], ot_books, "Bible OT books")

# ============ BIBLICAL FIGURES ============
biblical_figures = [
    "GOLIATH", "ISHMAEL", "JAPHETH", "JEZEBEL", "LAZARUS", "MALACHI",
    "MATTHEW", "MICHAEL", "OBADIAH", "REBEKAH", "SOLOMON", "STEPHEN",
    "SUSANNA", "TABITHA", "TIMOTHY", "ZEBEDEE", "ZEBULUN",
    "BARABBAS", "BARNABAS", "CAIAPHAS", "HABAKKUK", "ISSACHAR",
    "JEREMIAH", "JONATHAN", "MATTHIAS", "MORDECAI", "NAPHTALI",
    "NEHEMIAH", "THADAEUS", "ZEDEKIAH",
    "BATHSHEBA", "NATHANAEL", "NICODEMUS",
    "ZACHARIAH", "ZACCHAEUS", "ZEPHANIAH",
    "ADAM AND EVE", "BARTHOLOMEW", "METHUSELAH",
    "SIMON PETER", "THEOPHILUS",
    "BARTHOLOMEW", "GOG AND MAGOG",
    "SIMON MAGUS",
]
add(["biblical figure", "Bible character", "biblical character"], biblical_figures, "Biblical figures")

# ============ BIBLICAL PLACES ============
biblical_places = [
    "NOD", "GAZA", "ROME", "BABEL", "EGYPT", "JUDAH", "SODOM",
    "CANAAN",
    "CYRENE", "ISRAEL", "JUDAEA", "MT ZION", "RED SEA",
    "BABYLON", "CALVARY", "JERICHO", "MT SINAI",
    "BETHESDA", "DALMATIA", "DAMASCUS", "GOLGOTHA", "GOMORRAH",
    "MT ARARAT", "NAZARETH",
    "BETHLEHEM", "JERUSALEM", "PALESTINE",
    "ALEXANDRIA", "GETHSEMANE",
    "RIVER JORDAN",
    "GARDEN OF EDEN", "SEA OF GALILEE",
    "NINEVEH",
]
add(["biblical place", "Bible place", "place in the Bible"], biblical_places, "Biblical places")

# ============ BUTTERFLIES ============
butterflies = [
    "MAP", "BLUE", "WALL",
    "ARGUS", "COMMA", "ELFIN", "HEATH", "SATYR", "WHITE",
    "APOLLO", "COPPER", "HERMIT", "MORPHO", "PIERID", "PSYCHE",
    "ADMIRAL", "CABBAGE", "MONARCH", "PAPILIO", "PEACOCK", "RINGLET",
    "SATYRID", "SKIPPER",
    "BIRDWING", "CARDINAL", "GRAYLING", "HESPERID", "MILKWEED",
    "BRIMSTONE", "CLEOPATRA", "HESPERIAN", "HOLLY BLUE", "METALMARK",
    "NYMPHALID", "ORANGE-TIP", "WALL BROWN", "WOOD WHITE",
    "BROWN ARGUS", "COMMON BLUE", "FRITILLARY", "GATEKEEPER",
    "HAIRSTREAK", "RED ADMIRAL",
    "LARGE COPPER", "MEADOW BROWN", "PAINTED LADY",
    "SCOTCH ARGUS", "SMALL COPPER", "WALL BROWN",
    "CABBAGE WHITE", "DINGY SKIPPER",
    "LARGE SKIPPER", "MARBLED WHITE",
    "WHITE ADMIRAL",
    "CHALKHILL BLUE", "CLOUDED YELLOW", "MOURNING CLOAK",
    "PURPLE EMPEROR", "TORTOISESHELL",
    "BLACK HAIRSTREAK", "BROWN HAIRSTREAK", "GREEN HAIRSTREAK",
    "GRIZZLED SKIPPER", "HEATH FRITILLARY", "LULWORTH SKIPPER",
    "MARSH FRITILLARY", "MOUNTAIN RINGLET",
    "THISTLE",
]
add(["butterfly", "insect", "lepidopteran"], butterflies, "Butterflies")


# =====================================================================
# SCAN 3: Scanned_20260325-1444 (Cactus, Cakes, and more C sections)
# =====================================================================

# ============ CACTUS ============
cacti = [
    # 04
    "CRAB", "TOAD",
    # 05
    "DILDO", "NOPAL",
    # 06
    "BARREL", "CEREUS", "CHOLLA", "EASTER", "MESCAL", "OLD MAN",
    "ORCHID", "PEANUT", "PEYOTE",
    # 07
    "JOINTED", "OLD LADY", "OPUNTIA",
    # 08
    "DUMPLING", "GOLD LACE", "HEDGEHOG", "RAT TAIL", "SNOWBALL",
    "STARFISH", "TURK'S CAP",
    # 09
    "BUNNY EARS", "CHRISTMAS", "GOAT'S HORN", "GOLD CHARM",
    "INDIAN FIG", "MISTLETOE", "SEA-URCHIN",
    "RAINBOW", "SAGUARO",
    # 10
    "COTTON-POLE", "SAND DOLLAR", "SILVER BALL", "STRAWBERRY",
    "ZYGOCACTUS",
    # 11
    "GRIZZLY BEAR", "MAMMILLARIA", "PRICKLY PEAR", "SCARLET BALL",
    "SILVER TORCH",
    # 12
    "GOLDEN BARREL",
    # 13
    "BRISTOL BEAUTY", "SCHLUMBERGERA",
    # 14
    "DRUNKARD'S DREAM",
    # 15
    "QUEEN OF THE NIGHT", "SNOWBALL CUSHION",
]
add(["cactus", "plant", "desert plant", "succulent"], cacti, "Cactus")

# ============ CAKES, PASTRIES AND PUDDINGS (full list from scan 1444) ============
cakes_full = [
    "BUN", "CUP", "FIG", "OAT", "PAN", "PIE", "TEA",
    "BABA", "FLAN", "FOOL", "PLUM", "ROCK", "SEED", "TART",
    "ANGEL", "BOMBE", "BRIDE", "BUNDT", "CREAM", "CREPE", "FAIRY",
    "FRUIT", "FUDGE", "GENOA", "JELLY", "LARDY", "LAYER", "POUND",
    "QUEEN", "SCONE", "SHORT", "SWEET", "TIPSY", "TORTE", "YEAST",
    "BANANA", "CARROT", "CHEESE", "COFFEE", "DUNDEE", "ECCLES",
    "ECLAIR", "GATEAU", "COOKIE", "DANISH", "MUFFIN", "PARKIN",
    "SPONGE",
    "BAKLAVA", "BANBURY", "BANNOCK", "BATH BUN", "BRIOCHE",
    "BROWNIE", "CRUMBLE", "CRUMPET", "CURRANT", "FIG ROLL",
    "GINGER", "GIRDLE", "JUNKET", "MARBLE", "MOUSSE",
    "SIMNEL", "YUM-YUM",
    # 08
    "APPLE PIE", "BIRTHDAY", "BLACK BUN", "DATE ROLL", "DOUGHNUT",
    "FLUMMERY", "MACAROON", "MALT LOAF", "MERINGUE", "MINCE PIE",
    "MOONCAKE", "PECAN PIE", "SANDWICH", "SYLLABUB", "TIRAMISU",
    "TURNOVER", "WHEAT WHAM",
    # 09
    "ANGEL FOOD", "CHERRY PIE",
    "CHOCOLATE", "CHRISTMAS", "CREAM HORN", "CREAM PUFF",
    "DROP SCONE", "FRUIT TART", "LAMINGTON", "LEMON TART",
    "MADELEINE", "PANETTONE",
    # 10
    "BANANA LOAF", "BATTENBERG", "CHELSEA BUN",
    "KEY LIME PIE", "PANNA COTTA", "PONTEFRACT",
    "PUMPKIN PIE", "SHOOFLY PIE", "TARTE TATIN",
    "TOASTED TEA", "UPSIDE-DOWN",
    # 11
    "BAKED ALASKA", "BANANA BREAD", "BANOFFEE PIE",
    "CHOUX PASTRY", "CINNAMON BUN", "CREME BRULEE",
    "CUSTARD TART", "GINGERBREAD", "HOT CROSS BUN",
    # 12
    "BREAD PUDDING", "DANISH PASTRY", "FRUIT COBBLER",
    "FRUIT CRUMBLE", "PAIN AU RAISIN",
    # 13
    "SPONGE PUDDING",
    # 14
    "PAIN AU CHOCOLAT",
    # Additional
    "SOUFFLE", "STOLLEN", "STRUDEL", "TARTLET", "TEA CAKE",
    "TEA LOAF",
    "BLACKCAP", "EN CROUTE", "FLAPJACK", "MERINGUE",
    "BARN BRITH", "BATCH LOAF", "CLAFOUTI", "CLAPBREAD",
    "CROISSANT", "DROP SCONE",
    "BAKED APPLE", "BRANDY SNAP", "BROWN BETTY", "CHEESECAKE",
    "FLORENTINE", "SHORTBREAD",
    "BAKED ALASKA", "GINGERBREAD", "HOT CROSS BUN",
    "BREAD PUDDING", "DANISH PASTRY",
    "TREACLE TART",
    "BAVAROIS", "PAVLOVA", "STRUDEL",
    "TRIFLE", "WAFFLE",
    # additional from page
    "BLANCMANGE", "SUGAR PLUM", "SUGAR PASTE",
    "FRUIT",
    "MADEIRA", "PARKIN", "STREUSEL",
    "SHORTCAKE",
    "PANDOWDY",
    "JAM ROLY-POLY",
    "FIGGY PUDDING", "HASTY PUDDING", "MILLE-FEUILLE",
    "PLUM PORRIDGE", "TARTE AU SUCRE",
    "APPLE DUMPLING", "APPLE TURNOVER", "SCOTCH PANCAKE",
    "SPONGE PUDDING", "SUMMER PUDDING",
    "APPLE CHARLOTTE", "CHARLOTTE RUSSE", "CHOCOLATE FUDGE",
    "STEAMED PUDDING", "VICTORIA SPONGE",
    "BLACK CAP PUDDING", "CHOCOLATE ECLAIR",
    "QUEEN OF PUDDINGS", "STRAWBERRY SHORT",
]
add(["cake", "pastry", "pudding", "sweet", "confection"], cakes_full, "Cakes/Pastries/Puddings")

# ============ CALENDAR ============
calendars = [
    "BAHA'I", "HINDU", "LUNAR", "ROMAN",
    "COPTIC", "HEBREW", "JEWISH",
    "CHINESE", "ISLAMIC", "PERSIAN",
    "JULIAN", "SOLAR",
    "ARBITRAZY", "GREGORIAN", "LUNISOLAR",
]
add(["calendar", "calendar type"], calendars, "Calendars")

# ============ CANADA — cities ============
canada_cities = [
    "OTTAWA", "QUEBEC", "REGINA",
    "CALGARY", "HALIFAX", "TORONTO",
    "EDMONTON", "MONTREAL",
    "SASKATOON", "VANCOUVER", "WINNIPEG",
]
add(["Canadian city", "city in Canada"], canada_cities, "Canada cities")

# ============ CARNIVORES ============
carnivores = [
    "CAT", "DOG", "OWL",
    "BEAR", "FROG", "HAWK", "KITE", "LION", "NEWT", "ORCA", "PUMA",
    "SEAL", "WOLF",
    "ADDER", "CIVET", "COBRA", "DINGO", "EAGLE", "HERON", "HYENA",
    "MAMBA", "OTTER", "SHARK", "STOAT", "STORK", "TIGER", "VIPER",
    "WHALE",
    "CONDOR", "COYOTE", "FALCON", "FERRET", "HYAENA", "JACKAL",
    "JAGUAR", "LIZARD", "OSPREY", "PYTHON", "TAIPAN", "WALRUS",
    "WEASEL",
    "BARN OWL", "BUZZARD", "CHEETAH", "DOLPHIN", "KESTREL",
    "LEOPARD", "PANTHER", "PELICAN", "PENGUIN", "POLECAT", "VULTURE",
    "WILDCAT",
    "ANACONDA", "BROWN OWL", "EAGLE-OWL",
    "ALLIGATOR", "BALD EAGLE", "BLACK BEAR", "BLUE WHALE", "BROWN BEAR",
    "CROCODILE", "POLAR BEAR",
    "COPPERHEAD", "SALAMANDER", "SCREECH OWL", "SPERM WHALE",
    "TIGER SHARK",
    "ELECTRIC EEL", "GOLDEN EAGLE", "GRIZZLY BEAR", "KILLER WHALE",
    "RATTLESNAKE", "SPARROWHAWK",
    "BOA CONSTRICTOR",
    "GREAT WHITE SHARK", "HAMMERHEAD SHARK", "PEREGRINE FALCON",
    "SNOWY OWL", "TAWNY OWL",
]
add(["carnivore", "predator", "meat-eater"], carnivores, "Carnivores")

# ============ CARPETS ============
carpets = [
    "RAG", "RED", "RYA",
    "KALI", "DUTCH", "KELIM", "KILIM", "MAGIC",
    "PITCH", "STAIR", "THROW",
    "HEARTH", "HOOKED", "KHILIM", "KIRMAN", "NUMDAH",
    "PRAYER", "TURKEY", "WILTON",
    "BERGAMA", "BOKATI", "PERSIAN", "TURKISH",
    "BERGAMOT", "BRUSSELS", "AXMINSTER", "SHEEPSKIN",
    "TRAVELLING", "BESSARABIAN", "BUFFALO ROBE",
    "KIDDERMINSTER",
]
add(["carpet", "rug", "floor covering"], carpets, "Carpets")

# ============ CARRIAGES ============
carriages = [
    "GIG", "FLY",
    "ARAB", "BIGA", "GABY", "SHAY",
    "BUGGY", "COUPE",
    "ARABA", "AROBA", "BRAKE",
    "BERLIN", "CALASH", "DROSKY", "FIACRE", "HANSOM", "LANDAU",
    "BERLIN", "TROIKA",
    "BRISKA", "BRITSKA", "CALICHE",
    "BAROUCHE", "BROUGHAM", "CABRIOLET", "CARRIAGE",
    "CURRICLE", "DORMOUSE", "EQUIPAGE",
    "STAGECOACH", "DILIGENCE",
    "CHARABANC",
    "PHAETON", "SURREY", "SULKY", "TONGA",
    "SPRING", "TANDEM",
    "TILBURY",
    "CLARENCE", "VICTORIA",
    "LANDAULET",
    "SPIDER PHAETON",
]
add(["carriage", "horse-drawn vehicle", "vehicle"], carriages, "Carriages")

# ============ CASTLE PARTS ============
castle_parts = [
    "BERM", "KEEP", "MOAT", "WARD",
    "DITCH", "FOSSE", "MOTTE", "MOUND", "SCARP", "TOWER",
    "BAILEY", "CHAPEL",
    "CORBEL", "CRENEL", "DONJON", "MERLON", "TURRET",
    "BASTION", "DUNGEON", "PARADOS", "PARAPET", "POSTERN", "RAMPART",
    "APPROACH",
    "BARBICAN", "BARTIZAN", "BRATTICE", "BUTTRESS", "CROSSLET",
    "LOOPHOLE", "STOCKADE", "WALL WALK",
    "ARROW-SLIT", "COURTYARD", "EMBRASURE", "GATEHOUSE",
    "INNER WALL", "DRAWBRIDGE", "MURDER HOLE", "PORTCULLIS", "WATCHTOWER",
    "BATTLEMENTS", "CURTAIN WALL", "OUTER BAILEY",
    "CRENELLATION", "LOOKOUT TOWER",
    "ENCLOSURE WALL",
]
add(["castle part", "castle feature", "fortification"], castle_parts, "Castle parts")

# ============ CAT TYPES ============
cat_types = [
    "BOB", "LION", "LYNX", "PUMA",
    "FERAL", "TIGER",
    "COUGAR", "JAGUAR", "KODKOD", "MARGAY", "OCELOT", "PAMPAS",
    "CHEETAH", "LEOPARD",
    "DOMESTIC", "MOUNTAIN",
    "GEOFFROY'S", "JAGUARUNDI", "SNOW LEOPARD",
    "MOUNTAIN LION", "SCOTTISH WILD",
    "LITTLE SPOTTED", "CLOUDED LEOPARD",
]
add(["cat", "feline", "big cat"], cat_types, "Cat types")

# ============ CAT BREEDS ============
cat_breeds = [
    "REX",
    "MANX",
    "KORAT", "TABBY",
    "ANGORA", "BENGAL", "BIRMAN", "BOMBAY", "CYMRIC", "HAVANA",
    "LAPERM", "OCICAT", "SOMALI",
    "BURMESE", "PERSIAN", "RAGDOLL", "SIAMESE", "TIFFANY",
    "BALINESE", "BURMILLA", "DEVON REX", "SNOWSHOE", "TIFFANIE",
    "HIMALAYAN", "MAINE COON", "SINGAPURA", "TONKINESE",
    "ABYSSINIAN", "CARTHUSIAN", "CHINCHILLA", "CORNISH REX",
    "SELKIRK REX", "TURKISH VAN",
    "EGYPTIAN MAU", "FOREIGN BLUE", "RUSSIAN BLUE", "SILVER TABBY",
    "FOREIGN WHITE", "SCOTTISH FOLD",
    "DOMESTIC TABBY", "TORTOISESHELL", "TURKISH ANGORA",
    "BRITISH LONGHAIR", "EXOTIC SHORTHAIR", "JAPANESE BOBTAIL",
    "NORWEGIAN FOREST",
]
add(["cat breed", "breed of cat"], cat_breeds, "Cat breeds")


# =====================================================================
# SCAN 4: Scanned_20260325-1451 (Cattle, Cathedrals + bonus)
# =====================================================================

# ============ CATHEDRAL CITIES (UK) ============
cathedral_cities_uk = [
    "ELY", "BATH", "DERRY", "HULL",
    "DERBY", "RIPON", "TRURO", "WELLS",
    "BANGOR", "BRECON", "DUNDEE", "DURHAM", "EXETER",
    "LINCOLN", "NEWPORT", "NORWICH", "SALFORD", "ST ASAPH",
    "ST JOHN'S", "ST MARY'S", "ST PAUL'S", "SWANSEA", "WREXHAM",
    "ABERDEEN", "BRADFORD", "CARLISLE",
    "BLACKBURN", "BRENTWOOD", "EDINBURGH", "GUILDFORD",
    "BIRMINGHAM", "CANTERBURY", "CHELMSFORD", "LANCASTER", "LEICESTER",
    "CHICHESTER", "GLOUCESTER", "HEREFORD", "LICHFIELD", "LIVERPOOL",
    "NEWCASTLE", "ROCHESTER", "SALISBURY", "SHEFFIELD",
    "SOUTHWARK", "ST ANDREWS", "WAKEFIELD",
    "NORTHAMPTON", "YORK MINSTER",
    "CHRIST CHURCH", "PETERBOROUGH",
    "MIDDLESBROUGH", "ST EDMUNDSBURY",
    "COVENTRY", "LLANDAFF", "PLYMOUTH", "ST DAVIDS",
    "ST ALBANS",
    "MANCHESTER", "NOTTINGHAM", "PORTSMOUTH", "SHREWSBURY", "WINCHESTER",
    "WESTMINSTER",
]
add(["cathedral city", "see", "city with cathedral"], cathedral_cities_uk, "Cathedral cities UK")

# ============ CATHEDRALS WORLDWIDE ============
cathedrals_worldwide = [
    "LUND",
    "DUOMO", "MILAN",
    "AACHEN", "RHEIMS",
    "COLOGNE", "CORDOBA", "ORVIETO", "ST MARK'S",
    "NOTRE-DAME", "ST PETER'S",
    "STRASBOURG",
    "HAGIA SOPHIA",
    "CHARTRES", "FLORENCE",
    "ST BASIL'S",
]
add(["cathedral", "cathedral worldwide", "famous cathedral"], cathedrals_worldwide, "Cathedrals worldwide")

# ============ CATTLE BREEDS ============
cattle_breeds = [
    # 02
    "ZO",
    # 03
    "DZO", "GYR",
    # 04
    "ANKOLE", "DEXTER", "DURHAM", "JERSEY", "SALERS", "SUSSEX",
    # 05
    "BORAN", "WAGYU", "WHITE",
    # 06
    "ANGUS", "BLACK", "DEVON", "DROMO", "DEONI", "KERRY", "LUENG", "SANGA",
    # 07
    "BEEFALO", "BRAHMAN", "BRANGUS", "CATTALO", "LATVIAN", "RED POLL",
    # 08
    "ALDERNEY", "AYRSHIRE", "CHIANINA",
    "FRIESIAN", "GALLOWAY", "GELBVIEH", "GUERNSEY", "HEREFORD",
    "HIGHLAND", "HOLSTEIN", "ILLAWARA", "LIMOUSIN", "LONGHORN", "SHETLAND",
    # 09
    "AFRICANDER", "AFRIKANER", "BRAUNVIEH", "CHAROLAIS", "CORRIENTE",
    "ROMAGNOLA", "SHORTHORN", "SIMMENTAL", "TEESWATER", "UKRAINIAN",
    "WHITE PARK",
    # 10
    "AFRICANDER", "BEEFMASTER", "BROWN SWISS", "CANADIENNE",
    "LINCOLN RED", "MURRAY GREY", "PIEMONTESE", "SIMMENTHAL",
    "SOUTH DEVON", "TARENTAISE", "WELSH BLACK",
    # 11
    "BELGIAN BLUE", "CHILLINGHAM", "PIEDMONTESE",
    # 12
    "BRITISH WHITE", "SIMMENTHALER",
    # 13
    "ABERDEEN ANGUS", "DROUGHTMASTER", "TEXAS LONGHORN",
    # 14
    "BELTED GALLOWAY", "SANTA GERTRUDIS",
    # Additional from page
    "DCHO", "JOMO", "TULI", "ZEBU",
    "ZOBO", "ZOBU",
]
add(["cattle breed", "cow", "bovine", "bull", "breed of cattle"], cattle_breeds, "Cattle breeds")

# ============ CEMETERY ============
cemeteries = [
    "MT HOLLY", "NUNHEAD",
    "BROMPTON", "HIGHGATE", "MT OLIVER", "PANTHEON",
    "ABNEY PARK", "ARLINGTON",
    "EL ESCORIAL", "LA ALMUDENA", "MONTMARTRE",
    "SAN MICHELE",
    "WEISSENSEE",
    "KENSAL GREEN", "WEST NORWOOD",
    "GOLDERS GREEN", "LES INVALIDES",
    "MONTPARNASSE", "PERE LACHAISE", "TOWER HAMLETS",
    "MOUNT OF OLIVES",
    "ISLAND OF THE DEAD",
]
add(["cemetery", "burial place", "graveyard"], cemeteries, "Cemeteries")

# ============ CEREAL ============
cereals = [
    "OAT", "RYE", "ZEA",
    "CORN", "OATS", "RICE", "SAGO",
    "BAJRA", "EMMER", "MAIZE",
    "TEFF", "WHEAT", "SPELT", "BARLEY", "BULGUR",
    "MILLET", "SORGHUM", "SEMOLINA",
    "BUCKWHEAT",
    "INDIAN CORN",
    "COMMON MILLET", "FOXTAIL MILLET", "ITALIAN MILLET",
]
add(["cereal", "grain", "crop"], cereals, "Cereals")

# ============ BREAKFAST CEREALS ============
breakfast_cereals = [
    "ALPEN", "MUESLI", "WEETABIX", "ALL-BRAN",
    "CHEERIOS", "CLUSTERS", "COCO POPS", "FROSTIES",
    "FRUITFUL", "PORRIDGE", "RICE CRISPIES", "SPECIAL K",
    "BRAN FLAKES", "CORNFLAKES", "QUAKER OATS",
    "RAISIN BRAN", "SUGAR PUFFS",
    "COMMON SENSE", "FRUIT'N'FIBRE",
    "GRAPE NUTS", "JUST RIGHT", "READY BREK", "SHREDDIES",
    "PULLED WHEAT", "SULTANA BRAN",
    "COUNTRY CRISP", "RAISIN WHEATS", "RICE KRISPIES",
    "FROSTED FLAKES", "FRUIT AND FIBRE",
    "GOLDEN GRAHAMS", "HONEY NUT LOOPS",
    "NESTLE CLUSTERS",
    "CINNAMON GRAHAMS",
    "SHREDDED WHEAT",
]
add(["breakfast cereal", "cereal brand", "cereal"], breakfast_cereals, "Breakfast cereals")

# ============ CEREMONY ============
ceremonies = [
    "AMRIT", "DOETH", "TANGI",
    "MAUNDY", "NIPTER",
    "BAPTISM",
    "CAPPING", "CHANNOI", "CHUPPAH", "MUATAN", "WEDDING",
    "MARRIAGE",
    "COMMITTAL", "MATRIMONY", "NUPTIALS",
    "BAR MITZVAH", "BAT MITZVAH", "GRADUATION",
    "CHRISTENING", "FIRE-WALKING",
    "CONFIRMATION", "INITIATION",
]
add(["ceremony", "rite", "ritual"], ceremonies, "Ceremonies")

# ============ CHAIRS ============
chairs = [
    "ARM", "JUG", "PEW",
    "CLUB", "CAMP", "CANE", "DECK", "EASY", "FORM", "HEAD", "HIGH",
    "PUSH", "WING",
    "BENCH", "ELBOW",
    "KING'S", "NIGHT", "POTTY", "SEDAN", "STOOL", "WHEEL",
    "BASKET", "CARVER", "CANDLE", "DINING", "ESTATE", "JAMPAN",
    "MORRIS", "POUFFE", "ROCKER",
    "SAG BAG", "SLEDGE", "SWIVEL", "THRONE", "WICKER",
    "BEANBAG", "BENTWOOD", "BERGERE",
    "COMMODE", "GARDEN", "KITCHEN", "LOUNGER", "NURSING", "ROCKING",
    "WINDSOR",
    "CAPTAIN'S", "ELECTRIC", "FAUTEUIL", "PRIE-DIEU", "RECLINER", "WAINSCOT",
    "DIRECTOR'S",
    "BOATSWAIN'S", "FIDDLE-BACK", "FIRTHSTOOL",
    "LADDERBACK",
    "CROMWELLIAN", "GESTATORIAL",
    "DUCKING-STOOL",
]
add(["chair", "seat", "type of chair"], chairs, "Chairs")

# ============ CHEMISTRY — ELEMENTS ============
# (These may overlap with existing entries but dedup handles it)
elements = [
    "TIN", "GOLD", "IRON", "LEAD", "NEON", "ZINC",
    "ARGON", "BORON", "RADON",
    "BARIUM", "CARBON", "CERIUM", "COBALT", "COPPER", "CURIUM",
    "ERBIUM", "HELIUM",
    "XENON",
    "INDIUM", "IODINE", "NICKEL", "OSMIUM", "OXYGEN", "RADIUM",
    "SILVER", "SODIUM",
    "ARSENIC",
    "BISMUTH", "BOHRIUM", "BROMINE", "CADMIUM", "CAESIUM", "CALCIUM",
    "DUBNIUM", "FERMIUM", "GALLIUM", "HAFNIUM", "HASSIUM", "HOLMIUM",
    "IRIDIUM", "KRYPTON", "LITHIUM", "MERCURY", "NIOBIUM", "RHENIUM",
    "RHODIUM", "SILICON", "SULPHUR", "TERBIUM", "THORIUM", "THULIUM",
    "URANIUM", "YTTRIUM",
    "ACTINIUM", "ANTIMONY",
    "ASTATINE", "CHLORINE", "CHROMIUM", "EUROPIUM", "FLUORINE",
    "FRANCIUM", "HYDROGEN", "LUTETIUM", "NITROGEN", "NOBELIUM",
    "PLATINUM", "POLONIUM", "POTASSIUM",
    "RUBIDIUM", "SAMARIUM", "SCANDIUM", "SELENIUM", "TANTALUM",
    "THALLIUM", "TITANIUM", "TUNGSTEN", "VANADIUM",
    "ALUMINIUM", "AMERICIUM", "BERKELIUM", "BERYLLIUM", "GERMANIUM",
    "LANTHANUM", "MAGNESIUM",
    "MANGANESE", "NEODYMIUM", "NEPTUNIUM", "PALLADIUM", "PLUTONIUM",
    "STRONTIUM", "TELLURIUM", "YTTERBIUM", "ZIRCONIUM",
    "DYSPROSIUM", "GADOLINIUM", "LAWRENCIUM", "MEITNERIUM",
    "MOLYBDENUM", "PHOSPHORUS", "PRASEODYMIUM", "PROMETHIUM",
    "ROENTGENIUM", "SEABORGIUM", "TECHNETIUM",
    "CALIFORNIUM", "EINSTEINIUM",
    "MENDELEVIUM",
    "PROTACTINIUM",
    "RUTHERFORDIUM",
]
add(["element", "chemical element"], elements, "Chemical elements")

# ============ CHURCH PARTS ============
church_parts = [
    "PEW",
    "APSE", "ARCH", "FONT", "NAVE", "ROOD", "TOMB",
    "AISLE", "ALTAR", "CHOIR", "CRYPT", "PORCH", "SLYPE",
    "SPIRE", "STALL", "STOUP", "TOWER", "VAULT",
    "ADYTUM", "ARCADE", "ATRIUM", "BELFRY", "CHAPEL", "CHEVET",
    "CORONA", "PARVIS",
    "PORTAL", "PULPIT", "SEDILE", "SHRINE", "SQUINT", "VESTRY",
    "ALMONRY", "CHANCEL", "FRONTAL", "GALLERY", "LECTERN", "LUCARNE",
    "NARTHEX", "PISCINA", "REREDOS", "STEEPLE", "TAMBOUR",
    "CLOISTER", "CREDENCE", "CROSSING", "KEYSTONE", "PARCLOSE",
    "PINNACLE", "PREDELLA", "SACELLUM", "SACRISTY",
    "TRANSEPT",
    "ANTECHOIR", "BELL TOWER", "GRAVEYARD", "ORGAN LOFT", "SACRARIUM",
    "SANCTUARY", "SEPULCHRE", "STASIDION", "TRIFORIUM",
    "AMBULATORY", "BAPTISTERY", "BELL SCREEN", "CLERESTORY",
    "DIACONICON", "FENESTELLA", "FRITHSTOOL", "MISERICORD",
    "PRESBYTERY", "RETROCHOIR", "ROOD SCREEN",
    "CHAPTERHOUSE", "CONFESSIONAL", "DEAMBULATORY",
    "RINGING CHAMBER", "SCHOLA CANTORUM",
]
add(["church part", "part of a church", "cathedral feature"], church_parts, "Church parts")

# ============ CHICKENS ============
chickens = [
    "ANCONA", "BANTAM", "COCHIN", "HOUDAN", "SULTAN",
    "DORKING",
    "HAMBURG", "LEGHORN", "MENORCA",
    "HAMBURGH", "LANGSHAN",
    "ORPINGTON",
    "WELSUMMER", "WYANDOTTE",
    "ANDALUSIAN", "AUSTRALORP", "CHITTAGONG", "JUNGLE FOWL",
    "SPANISH FOWL",
    "PLYMOUTH ROCK",
    "RHODE ISLAND RED",
]
add(["chicken", "chicken breed", "fowl", "poultry"], chickens, "Chickens")

# ============ CINEMAS / THEATRES ============
cinemas = [
    "ABC", "MGM", "REX", "RIO", "UCI", "UGC",
    "GALA", "IMAX", "RITZ", "ROXY",
    "BYRON", "CAMEO", "FORUM", "GRAND", "KINGS", "METRO", "ODEON",
    "ORION", "PLAZA", "REGAL", "ROYAL", "SCALA", "TOWER",
    "ALBANY", "APOLLO", "CANNON", "CASINO", "CURZON", "EMPIRE",
    "GAIETY", "MARINA", "PALACE", "QUEENS",
    "ARCADIA", "ASTORIA", "CAPITOL", "CARLTON", "CENTRAL", "CENTURY",
    "CIRCUIT", "CLASSIC", "CORONET", "EMBASSY", "ESSOLDO", "GAUMONT",
    "GRANADA", "LA SCALA", "LOCARNO", "MAYFAIR", "ORPHEUM", "PARAGON",
    "PHOENIX", "PICCADY",
    "ALHAMBRA", "BROADWAY", "CHARLTON", "CINEPLEX", "CITIZENS",
    "COLONIAL", "DOMINION", "ELECTRIC",
    "EVERYMAN", "FESTIVAL", "IMPERIAL", "LANDMARK", "MAJESTIC",
    "MEMORIAL", "PAVILION", "THE CARMEN", "WINDMILL",
    "ALEXANDRA", "CINEWORLD", "FILMHOUSE", "HOLLYWOOD", "PALLADIUM",
    "PARAMOUNT", "PLAYHOUSE",
    "AMBASSADOR", "HIPPODROME", "LIGHTHOUSE", "VUE CINEMAS",
    "HER MAJESTY'S", "HIS MAJESTY'S", "NEW VICTORIA", "STAR CENTURY",
    "METROPOLITAN", "PICTUREDROME", "PICTUREHOUSE",
    "THEFILMWORKS",
    "PICTURE PALACE", "WARNER VILLAGE",
    "ELECTRIC PALACE", "SCREEN ON THE HILL",
    "REGENT", "RIALTO", "ROBINS", "TIVOLI", "VIRGIN",
]
add(["cinema", "theatre", "picture house"], cinemas, "Cinemas")


# =====================================================================
# SCAN 5: Scanned_20260325-1452 (Capital cities)
# =====================================================================

# ============ CAPITAL CITIES ============
# Capital city -> Country pairs (we store the city as answer, with "capital" as definition,
# and also "capital of X" for specific countries)
capital_cities = [
    # 04
    "BAKU", "DOHA", "LOME", "MALE", "NIUE", "ROME", "SUVA",
    # 05
    "ACCRA", "AMMAN", "BERNE", "CAIRO", "DACCA", "DHAKA", "HANOI",
    "KABUL", "KYOTO", "LAGOS", "MINSK", "PARIS", "PRAIA", "QUITO",
    "RABAT", "RIGA", "SANAA", "SEOUL", "SUCRE", "TOKYO", "VADUZ",
    # 06
    "ANKARA", "ASMARA", "ATHENS", "BANGUI", "BANJUL", "BEIRUT",
    "BERLIN", "BISSAU", "BOGOTA", "DUBLIN", "HARARE", "KIGALI",
    "LISBON", "LONDON", "LUANDA", "LUSAKA", "MADRID", "MAJURO",
    "MALABO", "MANAMA", "MANILA", "MAPUTO", "MASERU", "MEXICO",
    "MORONI", "MOSCOW", "MUSCAT", "NASSAU", "NIAMEY", "OTTAWA",
    "PANAMA", "PRAGUE", "RIYADH", "ROSEAU", "SKOPJE", "TAIPEI",
    "TARNOW", "TEHRAN", "TIRANA", "VIENNA", "WARSAW", "YAOUNDE",
    "ZAGREB",
    # 07
    "ALGIERS", "BAGHDAD", "BANGKOK", "BEIJING", "BISHKEK", "CARACAS",
    "COLOMBO", "CONAKRY", "COTONOU", "MANAGUA", "NAIROBI", "NICOSIA",
    "PODGORICA", "SAN JOSE", "SAO TOME", "ST JOHN'S", "TALLINN",
    "TBILISI", "THIMPHU", "TRIPOLI", "VILNIUS", "YAOUNDE",
    # 08
    "ABU DHABI", "ASHGABAT", "ASUNCION", "BELGRADE", "BELMOPAN",
    "BRASILIA", "BUDAPEST", "CANBERRA", "CAPE TOWN", "CASTRIES",
    "CHISINAU", "DJIBOUTI", "FREETOWN", "FUNAFUTI", "GABORONE",
    "HELSINKI", "KHARTOUM", "KINGSTON", "KINSHASA",
    "LAAYOUNE", "MONROVIA", "NUKU'ALOFA",
    # 09
    "AMSTERDAM", "BUCHAREST", "BUJUMBURA", "ISLAMABAD",
    "KATHMANDU", "KINGSTOWN", "LJUBLJANA", "MOGADISHU",
    "NUKU'ALOFA", "PHNOM PENH", "PORT LOUIS", "PYONGYANG",
    "PORTO NOVO", "REYKJAVIK", "SAN MARINO", "SINGAPORE",
    "STOCKHOLM", "VIENTIANE",
    # 10
    "ADDIS ABABA", "BASSE-TERRE", "BRIDGETOWN", "COPENHAGEN",
    "GEORGETOWN", "KUWAIT CITY", "LIBREVILLE",
    "LUXEMBOURG", "MEXICO CITY", "MONTEVIDEO",
    "NOUAKCHOTT", "PARAMARIBO", "WASHINGTON", "WELLINGTON",
    # 11
    "BRAZZAVILLE", "BUENOS AIRES", "KUALA LUMPUR",
    "OUAGADOUGOU",
    "PORT MORESBY",
    "PORT OF SPAIN",
    "SAN SALVADOR", "TEGUCIGALPA",
    "VATICAN CITY",
    # 12
    "ANTANANARIVO", "BLOEMFONTEIN",
    "PORT-AU-PRINCE", "SANTO DOMINGO",
    "TEL AVIV-JAFFA", "YAMOUSSOUKRO",
    # 13
    "GUATEMALA CITY",
    # 14
    "ANDORRA LA VELLA",
    # Additional
    "TASHKENT", "YEREVAN",
    "NEW DELHI",
    "PORT VILA",
    "PRETORIA",
    "PRISTINA",
    "SANTIAGO",
    "VICTORIA",
    "WINDHOEK",
    "N'DJAMENA",
    "VALLETTA",
]
add(["capital", "capital city"], capital_cities, "Capital cities")

# Also add specific country pairings for the most crossword-relevant ones
capital_country_pairs = {
    "PARIS": "France", "LONDON": "England", "ROME": "Italy",
    "BERLIN": "Germany", "MADRID": "Spain", "LISBON": "Portugal",
    "ATHENS": "Greece", "CAIRO": "Egypt", "TOKYO": "Japan",
    "BEIJING": "China", "MOSCOW": "Russia", "DUBLIN": "Ireland",
    "OTTAWA": "Canada", "CANBERRA": "Australia", "WELLINGTON": "New Zealand",
    "BRASILIA": "Brazil", "LIMA": "Peru", "BOGOTA": "Colombia",
    "SANTIAGO": "Chile", "BUENOS AIRES": "Argentina",
    "NAIROBI": "Kenya", "PRETORIA": "South Africa",
    "NEW DELHI": "India", "BANGKOK": "Thailand", "HANOI": "Vietnam",
    "SEOUL": "South Korea", "TEHRAN": "Iran", "KABUL": "Afghanistan",
    "RIYADH": "Saudi Arabia", "ANKARA": "Turkey", "WARSAW": "Poland",
    "PRAGUE": "Czech Republic", "BUDAPEST": "Hungary",
    "VIENNA": "Austria", "BERNE": "Switzerland", "BRUSSELS": "Belgium",
    "AMSTERDAM": "Netherlands", "COPENHAGEN": "Denmark",
    "STOCKHOLM": "Sweden", "OSLO": "Norway", "HELSINKI": "Finland",
    "REYKJAVIK": "Iceland", "TIRANA": "Albania", "BELGRADE": "Serbia",
    "ZAGREB": "Croatia", "BUCHAREST": "Romania", "SOFIA": "Bulgaria",
    "TALLINN": "Estonia", "RIGA": "Latvia", "VILNIUS": "Lithuania",
    "MINSK": "Belarus", "KYIV": "Ukraine",
    "TBILISI": "Georgia", "BAKU": "Azerbaijan",
    "HAVANA": "Cuba", "NASSAU": "Bahamas", "KINGSTON": "Jamaica",
    "ACCRA": "Ghana", "DAKAR": "Senegal",
    "ISLAMABAD": "Pakistan", "KATHMANDU": "Nepal",
    "MANILA": "Philippines", "SINGAPORE": "Singapore",
    "KUALA LUMPUR": "Malaysia", "JAKARTA": "Indonesia",
}
for city, country in capital_country_pairs.items():
    add([f"capital of {country}"], [city], "Capital city pairs")

# ============ BREAD AND ROLLS ============
breads = [
    "HAP", "COB", "NAN", "RYE", "TSA",
    "AZYM", "CAKE", "CORN", "FARL", "LOAF", "MILK", "NAAN", "PITA",
    "PONE", "PURI", "ROTI", "SODA",
    "AREPA", "AZYME", "BAGEL", "BLACK", "BROWN", "CHEAT", "FANCY",
    "HORSE", "MATZA",
    "DAMPER", "FRENCH", "GARLIC", "GRAHAM", "INDIAN", "INJERA", "LAVASH",
    "MATZO", "PITTA", "PLAIT", "POORI", "RAVEL", "WHITE",
    "BANNOCK", "BLOOMER", "BRIOCHE", "BROWNIE", "BUTTERY",
    "CHALLAH", "CHAPATI", "CURRANT", "FICELLE", "GRANARY",
    "MANCHET", "PARATHA", "PRETZEL", "STOTTIE", "WHEATEN",
    "CIABATTA", "CORN PONE", "FOCACCIA", "GRISSINI", "LEAVENED",
    "RAVELLED", "RYEBREAD", "SCHNECKE", "STANDARD", "TORTILLA",
    "BAGUETTE", "BARM CAKE", "CHAPATTI",
    "BARA BRITH", "BARMBRACK", "BATCH LOAF",
    "BREADSTICK", "BRIDGE ROLL", "FINGER ROLL",
    "BURGER BUN", "CORNBREAD", "CROISSANT", "FLATBREAD",
    "PETIT PAIN", "SCHNECKEN", "SHEWBREAD", "SHOWBREAD",
    "SOURDOUGH", "WHOLEMEAL",
    "FRENCH LOAF", "MULTIGRAIN", "STOTTY CAKE", "UNLEAVENED",
    "VIENNA LOAF", "WHOLEWHEAT",
    "COTTAGE LOAF", "FRENCH STICK", "MORNING ROLL",
    "POTATO BREAD", "POTATO SCONE",
    "PUMPERNICKEL", "FARMHOUSE LOAF",
    "MATZO", "MATZOH", "PANINI", "PANINO", "SIMNEL", "STOTTY", "WASTEL",
]
add(["bread", "bread roll", "type of bread", "loaf"], breads, "Bread")

# ============ BRIDGE TYPES ============
bridge_types = [
    "AIR", "FLY",
    "ARCH", "BEAM", "DECK", "FOOT", "OVER", "RAFT",
    "CHAIN", "PIVOT", "SWING", "BAILEY", "FLYING",
    "ROPE", "ROAD", "TOLL", "WIRE",
    "BASCULE", "FLYOVER", "LATTICE", "LIFTING", "PONTOON", "RAILWAY",
    "THROUGH", "VIADUCT",
    "AQUEDUCT",
    "FLOATING", "HUMPBACK", "OVERPASS",
    "BOW GIRDER",
    "CANTILEVER", "SUSPENSION", "TRAVERSING",
    "CABLE-STAYED", "TRANSPORTER",
]
add(["bridge", "bridge type", "type of bridge"], bridge_types, "Bridge types")

# ============ FAMOUS BRIDGES ============
famous_bridges = [
    "TAY", "SKYE", "TYNE",
    "FORTH", "SIGHS", "TOWER",
    "HUMBER", "KINTAI", "LONDON", "RIALTO", "SEVERN",
    "BIFROST",
    "RAINBOW", "TSING MA", "YICHANG",
    "BOSPORUS", "BROOKLYN", "JIANGYIN", "MACKINAC", "WATERLOO",
    "EVERGREEN", "FORTH ROAD",
    "RIVER KWAI",
    "BOSPORUS II", "GOLDEN GATE", "HOGA KUSTEN", "HONSBRIDGE",
    "KURUSHIMA-2", "KURUSHIMA-3", "MILLENNIUM",
    "PONT DU GARD", "STOREBALT",
    "BROCADE SASH",
    "AKASHI-KAIKYO", "PONT D'AVIGNON", "PONTE VECCHIO",
    "GREAT BELT EAST", "KII BISAN-SETO", "MILLAU VIADUCT",
    "SYDNEY HARBOUR",
    "PONTE 25 DE ABRIL", "QUEBEC RAILROAD",
    "MINAMI BISAN-SETO",
]
add(["bridge", "famous bridge"], famous_bridges, "Famous bridges")

# ============ CINQUE PORTS ============
cinque_ports = [
    "RYE", "DEAL", "LYDD",
    "BARON", "DOVER", "HYTHE",
    "MARGATE", "HASTINGS", "HEAD PORT",
    "PORTSMEN", "RAMSGATE", "SANDWICH",
    "FAVERSHAM",
    "NEW ROMNEY", "TENTERDEN",
    "FOLKESTONE", "LORD WARDEN", "WINCHELSEA",
    "CONFEDERATION",
    "CORPORATE MEMBER",
]
add(["Cinque Port", "port"], cinque_ports, "Cinque Ports")

# ============ CHEMICAL COMPOUNDS ============
compounds = [
    "PVC", "DEET", "SOAP", "UREA",
    "PHENOL", "AMMONIA", "BORAZON", "CHLORAL", "ETHANOL", "STYRENE",
    "ATRAZINE", "KEROSENE", "METHANOL", "PARAFFIN",
    "BENZOLIUM", "CARBAZOLE",
    "CHLORAMINE", "CHLOROFORM",
    "BENZALDEHYDE",
    "TOLUENE", "BOROSILICATE",
    "CARBON DIOXIDE", "CHLORHEXIDINE", "CHLOROBROMIDE",
    "CARBON MONOXIDE", "CHLORAL HYDRATE",
    "ORGANOPHOSPHATE", "SODIUM HYDROXIDE",
]
add(["chemical compound", "compound", "chemical"], compounds, "Chemical compounds")


# =====================================================================
# COMMIT AND REPORT
# =====================================================================
conn.commit()

print(f"\nTotal inserted: {inserted}")
print(f"Total skipped (already existed): {skipped}")
print(f"\n--- Section breakdown ---")
for section, count in sorted(section_counts.items()):
    print(f"  {section}: {count}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"\nTotal crossword_companion entries: {c.fetchone()[0]}")
conn.close()
