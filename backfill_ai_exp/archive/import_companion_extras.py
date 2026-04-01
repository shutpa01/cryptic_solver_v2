"""Import extra sections from crossword companion scan pages."""
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

# ============ FRENCH REGIONS ============
french_regions = [
    "CORSE", "ALSACE", "CENTRE", "GUYANE", "CORSICA", "PICARDY",
    "AUVERGNE", "BRETAGNE", "BRITTANY", "BURGUNDY", "LIMOUSIN", "LORRAINE", "PICARDIE",
    "AQUITAINE", "BOURGOGNE", "LA REUNION",
    "GUADELOUPE", "MARTINIQUE", "RHONE-ALPES",
    "ILE DE FRANCE", "FRANCHE-COMTE", "MIDI-PYRENEES",
    "LOWER NORMANDY", "UPPER NORMANDY",
    "BASSE-NORMANDIE", "HAUTE-NORMANDIE", "NORD-PAS-DE-CALAIS", "POITOU-CHARENTES",
    "CHAMPAGNE-ARDENNE", "LANGUEDOC-ROUSSILLON",
    "PROVENCE-ALPES-COTE D'AZUR", "PAYS DE LA LOIRE",
]
add(["French region", "region of France", "French province"], french_regions)

# ============ FRENCH LANDMARKS ============
french_landmarks = [
    "ALPS", "JURA", "LOIRE", "MEUSE", "RHONE", "SAONE", "SEINE", "SOMME",
    "CARNAC", "LANDES", "VOSGES", "GARONNE", "LASCAUX",
    "AUVERGNE", "CEVENNES", "DORDOGNE", "PROVENCE", "PYRENEES",
    "MONT BLANC", "NOTRE DAME", "MER DE GLACE", "PONT DU GARD", "VERSAILLES",
    "CANAL DU MIDI", "CHENONCEAUX", "EIFFEL TOWER", "GRAND TRIANON", "LES INVALIDES",
    "MONT ST MICHEL", "PETIT TRIANON", "ARC DE TRIOMPHE", "FONTAINEBLEAU",
    "FONTENAY ABBEY", "LYON CATHEDRAL", "MASSIF CENTRAL", "MILLAU VIADUCT",
    "AIGUILLE DU MIDI", "SUISSE NORMANDE", "AMIENS CATHEDRAL", "REIMS CATHEDRAL",
]
add(["French landmark", "landmark in France"], french_landmarks)

# ============ FRENCH BOYS' NAMES ============
french_boys = [
    "LUC", "JEAN", "LEON", "REMI", "REMY", "RENE", "YVES",
    "ALAIN", "ANDRE", "DENIS", "EMILE", "HENRI", "JULES", "LOUIS", "SERGE",
    "CLAUDE", "DIDIER", "GASTON", "GERARD", "HONORE", "JEROME", "MARCEL",
    "ANTOINE", "EDOUARD", "ETIENNE", "GEORGES", "GUSTAVE", "JACQUES", "LAURENT",
    "MICHEL", "OLIVIER", "PASCAL", "PATRICE", "PIERRE", "THIBAUT", "THIERRY", "VINCENT", "XAVIER",
    "FREDERIC", "MATTHIEU", "PHILIPPE", "STEPHANE", "THIBAULT", "GUILLAUME",
]
add(["French name", "French boy's name", "French man's name"], french_boys)

# ============ FRENCH GIRLS' NAMES ============
french_girls = [
    "FIFI", "GIGI", "AIMEE", "FLEUR", "MARIE",
    "AMELIE", "ARIANE", "DENISE", "ELOISE", "EVETTE", "EVONNE", "HELENE",
    "JANINE", "JEANNE", "NICOLE", "SIMONE", "YVETTE", "YVONNE",
    "BLANCHE", "CAMILLE", "CHANTAL", "COLETTE", "MARGAUX", "MONIQUE", "RACQUEL", "SIDONIE",
    "BERTILLE", "BRIGITTE", "CHARLIZE", "DANIELLE", "FRANCINE", "JULIETTE", "MICHELLE", "VILLETTE",
    "ANGELIQUE", "CHARMAINE", "CLAUDETTE", "DOMINIQUE", "FRANCOISE", "GABRIELLE",
    "GENEVIEVE", "GHISLAINE", "MADELEINE", "MODESTINE", "VERONIQUE",
    "ANTOINETTE", "JACQUELINE",
]
add(["French name", "French girl's name", "French woman's name"], french_girls)

# ============ GERMAN CITIES ============
german_cities = [
    "BONN", "ESSEN", "TRIER", "BERLIN", "MUNICH", "WEIMAR",
    "COLOGNE", "HAMBURG", "LEIPZIG", "WURZBURG", "FRANKFURT", "NUREMBERG",
    "STUTTGART", "DUSSELDORF", "FRANKFURT AM MAIN",
]
add(["German city", "city in Germany", "German town"], german_cities)

# ============ GERMAN BOYS' NAMES ============
german_boys = [
    "JAN", "MAX", "UWE", "DIRK", "ERIC", "ERIK", "JENS", "JORG", "RALF",
    "SVEN", "SWEN", "BERND", "ERICH", "FRITZ", "JONAS", "KLAUS", "LUKAS", "RALPH",
    "DIETER", "JURGEN", "MARKUS", "NIKLAS", "STEFAN", "TOBIAS", "ULRICH",
    "ANDREAS", "DOMINIK", "KRISTIAN", "MATTHIAS", "THORSTEN", "WOLFGANG",
]
add(["German name", "German boy's name", "German man's name"], german_boys)

# ============ GERMAN GIRLS' NAMES ============
german_girls = [
    "ELKE", "IRMA", "LILI", "BERTA", "ERIKA", "GERDA", "HEIDI", "HELGA",
    "HILDE", "LOTTI", "PETRA", "TRUDI",
    "ANGELA", "ASTRID", "BIRGIT", "DAGMAR", "FRIEDA", "INGRID", "LIESEL", "MONIKA",
    "SIGRUN", "STEFFI", "ULRIKA", "URSULA",
    "BETTINA", "JOLANDA", "KRISTIN", "ADELHEID", "ANGELIKA", "BIRGITTA", "BRUNHILD",
    "CHRISTIN", "GRETCHEN", "BRUNHILDE", "ELISABETH", "FRANZISKA", "HILDEGARD", "WILHELMINA",
]
add(["German name", "German girl's name", "German woman's name"], german_girls)

# ============ GERMAN LANDMARKS ============
german_landmarks = [
    "ELBA", "ELBE", "MAIN", "ODER", "RHINE", "DACHAU",
    "DANUBE", "BROCKEN", "MOSELLE", "RATHAUS", "RESIDENZ",
    "HELIGOLAND", "LINDERHOF", "REICHENAU", "REICHSTAG", "STARNBERG",
    "BUCHENWALD", "HEIDELBERG", "TIERGARTEN", "WIES CHURCH",
    "BERLINER DOM", "BLACK FOREST", "FERNSEHTURM", "KONIGSPLATZ", "MARIENSAULE",
    "RHINE VALLEY", "TROSTBRUCKE", "BAVARIAN ALPS",
    "FRAUENKIRCHE", "MUSEUMSINSEL",
    "BONN CATHEDRAL", "COLDITZ CASTLE", "FESTSPIELHAUS", "HARZ MOUNTAINS", "MOSELLE VALLEY",
    "ALEXANDERPLATZ", "ESSEN CATHEDRAL", "GEMALDEGALERIE", "HERRENCHIEMSEE",
    "NEUSCHWANSTEIN", "POTSDAMER PLATZ", "TRIER CATHEDRAL", "UNTER DEN LINDEN", "WARTBURG CASTLE",
    "AACHEN CATHEDRAL", "AUERBACH'S KELLER", "BRANDENBURG GATE", "EAST SIDE GALLERY",
    "MUNICH CATHEDRAL", "SPEYER CATHEDRAL",
]
add(["German landmark", "landmark in Germany"], german_landmarks)

# ============ GIANTS (mythology) ============
giants = [
    "GOG", "ONI", "BALI", "BANA", "BRES", "CACA", "CORB", "ERIU", "GAIA", "GERD",
    "GORM", "GRID", "HROD", "KARI", "LOKI", "OTUS", "RHEA", "YMIR",
    "AEGIR", "ARGES", "ARGUS", "ATLAS", "BALOR", "BANBA", "BAUGI", "CACUS",
    "FODLA", "GJALP", "GREIP", "GYMIR", "HYMIR", "JOTUN", "MAGOG", "ORION",
    "PAN GU", "SKADI", "TALOS", "THEIA", "THRYM",
    "ALBION", "ANAKIM", "BESTLA", "CRONUS", "ECHION", "ELATHA", "FACHAN", "FAFNIR",
    "FASOLT", "GERYON", "HAGRID", "PHOEBE", "TETHRA", "TETHYS", "THEMIS", "THIAZI",
    "TITANS", "TITYUS", "TYPHON",
    "ANTAEUS", "ASHURAS", "BRONTES", "CYCLOPS", "DAITYAS", "ETHLINN", "GEIRROD",
    "GILLING", "GOLIATH", "IAPETUS", "KLYTIUS", "OCEANUS", "OLVALDI", "PURUSHA",
    "SUTTUNG", "TELEMOS", "TELEMUS", "WINDIGO", "ZIPACNA",
    "ANGRBODA", "BOLTHORN", "BRIAREUS", "CETHLENN", "CYCLOPES", "EURYTION",
    "FIRBOLGS", "GIGANTES", "GOGMAGOG", "HRUNGNIR", "HYPERION", "JARNSAXA",
    "MORGANTE", "NEPHILIM", "PANOPTES", "STEROPES", "UPELLURI",
    "ANGERBODA", "AURGELMIR", "BERGELMIR", "ENCELADUS", "FOMORIANS", "GANDAREVA",
    "GARGANTUA", "GRANTORTO", "MENOETIUS", "MNEMOSYNE", "OLENTZERO",
    "ANGERBOTHA", "BUARAINECH", "EPIMETHEUS", "PANTAGRUEL", "PAUL BUNYAN",
    "POLYPHEMUS", "PROMETHEUS", "YSBADDADEN",
    "FINN MACCOOL", "GALLIGANTUS", "GOG AND MAGOG",
]
add(["giant", "mythical giant", "legendary giant"], giants)

# ============ ITALIAN CITIES ============
italian_cities = [
    "PISA", "ROME", "GENOA", "MILAN", "TURIN", "NAPLES", "VENICE",
    "BOLOGNA", "PALERMO", "FLORENCE",
]
add(["Italian city", "city in Italy", "Italian town"], italian_cities)

# ============ ITALIAN LANDMARKS ============
italian_landmarks = [
    "PO", "ARNO", "COMO", "ETNA", "LIDO", "DAVID", "FORUM", "GARDA", "TIBER",
    "MT ETNA", "LA SCALA", "POMPEII", "LAKE COMO", "MAGGIORE", "PANTHEON",
    "ST PETER'S", "VESUVIUS", "APPIAN WAY", "CAMPANILE", "COLOSSEUM", "DOLOMITES",
    "LAKE GARDA", "GRAND CANAL", "MT VESUVIUS", "DOGE'S PALACE", "HERCULANEUM",
    "VATICAN CITY", "LAKE MAGGIORE", "LEANING TOWER", "PONTE VECCHIO", "RIALTO BRIDGE",
    "BRIDGE OF SIGHS", "SISTINE CHAPEL", "ST MARK'S SQUARE", "UFFIZI GALLERY",
    "VATICAN PALACE", "PALAZZO VECCHIO", "PIAZZA SAN MARCO", "ST PETER'S SQUARE",
    "VIA APPIA ANTICA",
]
add(["Italian landmark", "landmark in Italy"], italian_landmarks)

# ============ FOSSILS ============
fossils = [
    "BONE", "CAST", "AMBER", "SHELL", "BURROW", "BIVALVE", "CRINOID",
    "AMMONITE", "BACULITE", "DINOSAUR", "ECHINOID", "NAUTILUS", "SKELETON",
    "STEINKERN", "TRILOBITE", "BELEMNITE", "COCCOLITH", "COPROLITE", "FISH TEETH",
    "CAST FOSSIL", "GASTROLITH", "SNAKESTONE", "SHARKS TEETH", "TRACE FOSSIL",
    "ICHNOFOSSIL", "MICROFOSSIL", "MOULD FOSSIL", "RESIN FOSSIL",
    "BURGESS SHALE", "PALEONTOLOGY", "STRATIGRAPHY", "STROMATOLITE",
    "PETRIFICATION",
]
add(["fossil", "ancient remains"], fossils)

# ============ CURRIES ============
curries = [
    "BALTI", "BHUNA", "KORMA", "CEYLON", "MADRAS", "MASALA", "PATHIA",
    "PENANG", "BIRIANI", "BIRYANI", "DHANSAK", "DOPIAZA", "HANGLAY", "MALAYAN",
    "BIRIYANI", "JALFREZI", "KASHMIRI", "MASSAMAN", "PASANDA", "RED THAI",
    "RENDANG", "TANDOORI", "VINDALOO",
    "CHETTINAD", "GREEN THAI", "ROGAN JOSH", "YELLOW THAI", "TIKKA MASALA",
]
add(["curry", "Indian dish", "spicy dish"], curries)

# ============ CUTLERY ============
cutlery = [
    "FORK", "KNIFE", "LADLE", "SPOON", "SPORK",
    "FISH FORK", "TEASPOON", "FISH KNIFE", "FISH SLICE", "SALT SPOON", "SOUP SPOON",
    "BREAD KNIFE", "CADDY SPOON", "CAKE SERVER", "CHOPSTICKS", "PICKLE FORK",
    "STEAK KNIFE", "SUGAR TONGS", "TABLESPOON",
    "BUTTER KNIFE", "CARVING FORK", "CHEESE KNIFE",
    "APOSTLE SPOON", "CARVING KNIFE", "DESSERTSPOON", "SALAD SERVERS",
    "VEGETABLE KNIFE",
]
add(["cutlery", "eating utensil", "kitchen utensil"], cutlery)

# ============ CUTTERS ============
cutters = [
    "AXE", "SAW", "SAX", "ADZE", "BILL", "CELT",
    "BLADE", "KNIFE", "MOWER", "PLANE", "RAZOR", "SWORD",
    "CHISEL", "COLTER", "CULTER", "DAGGER", "ICE AXE", "JIGSAW", "LABRYS",
    "LOPPER", "PIOLET", "POLEAX", "SCYTHE", "SHEARS", "SICKLE",
    "CHOPPER", "CLEAVER", "COULTER", "CUTLASS", "FRETSAW", "GISARME",
    "HACKSAW", "HALBERD", "HATCHET", "MEAT-AXE", "POLEAXE",
    "BATTLE-AX", "BILLHOOK", "CHAINSAW", "CLIPPERS", "SCISSORS", "SHREDDER",
    "TOMAHAWK", "BATTLE-AXE", "LAWNMOWER", "SECATEURS",
    "GUILLOTINE",
]
add(["cutter", "cutting tool", "blade"], cutters)

# ============ LAWYER TYPES ============
lawyers = [
    "QC", "AVOUE", "BRIEF", "JUDGE",
    "AVOCAT", "JURIST", "BENCHER", "CORONER", "COUNSEL", "JUSTICE", "MUKHTAR",
    "SHERIFF", "SHYSTER", "ADVOCATE", "ATTORNEY", "BARRISTER", "LAWMONGER", "SOLICITOR",
    "LEGAL EAGLE", "CONVEYANCER",
]
add(["lawyer", "legal professional", "legal practitioner"], lawyers)

# ============ SPIDERS/ARACHNIDS ============
spiders = [
    "RED", "BIRD", "MITE", "TICK", "WOLF",
    "BOLAS", "MONEY", "WATER", "ZEBRA",
    "DIADEM", "EPEIRA", "KATIPO", "MYGALE", "VIOLIN",
    "ARANEID", "HARVEST", "HUNTING", "JUMPING", "REDBACK",
    "ATTERCOP", "HUNTSMAN", "SCORPION", "TRAPDOOR",
    "FUNNEL-WEB", "HARVESTER", "PHALANGID", "TARANTULA",
    "BLACK WIDOW", "HARVESTMAN", "SALTIGRADE",
    "HARVEST MITE", "HARVEST TICK", "BIRD-CATCHING", "BOOK-SCORPION",
    "MONEY-SPINNER", "WHIP SCORPION",
]
add(["spider", "arachnid"], spiders)

# ============ MUSIC TYPES ============
music_types = [
    "AOR", "MOR", "POP", "RAP", "SKA",
    "FOLK", "FUNK", "JAZZ", "JIVE", "MOOD", "ROCK", "SOUL",
    "BEBOP", "BLUES", "CAJUN", "CRUNK", "DANCE", "DISCO", "HOUSE", "INDIE",
    "KRUNK", "MUZAK", "SALSA", "SAMBA", "SWING", "WORLD",
    "ATONAL", "BALLET", "CHORAL", "GARAGE", "GOSPEL", "GRUNGE", "HIP-HOP",
    "JUNGLE", "LOUNGE", "REGGAE", "SACRED", "TECHNO", "TRANCE",
    "AMBIENT", "BAROQUE", "BHANGRA", "BIG BEAT", "CALYPSO", "CHAMBER", "COUNTRY",
    "GAMELAN", "GANGSTA", "JAZZ-POP", "KARAOKE", "NU-METAL",
    "RAGTIME", "SKIFFLE", "TRIP-HOP",
    "ACID JAZZ", "BALLROOM", "FOLK ROCK", "GLAM ROCK", "HARDCORE", "HARD ROCK",
    "JAZZ-FUNK", "JAZZ-ROCK", "OPERATIC", "ORATORIO", "PUNK ROCK", "ROMANTIC",
    "SOFT ROCK",
    "ACID HOUSE", "BLUEGRASS", "CLASSICAL", "DIXIELAND",
    "ELECTRONIC", "HEAVY METAL", "ROCK AND ROLL", "RHYTHM AND BLUES",
]
add(["music type", "type of music", "genre of music", "musical genre"], music_types)

# ============ SPANISH LANDMARKS ============
spanish_landmarks = [
    "EBRO", "IBIZA", "PRADO", "ALHAMBRA", "CANARIES", "PYRENEES", "TENERIFE",
    "BALEARICS", "LANZAROTE", "PARC GUELL", "GUGGENHEIM", "MONTSERRAT",
    "PICO DE TEIDE", "GUADALQUIVIR", "CANARY ISLANDS", "MUSEO DEL PRADO",
    "SAGRADA FAMILIA", "BALEARIC ISLANDS",
]
add(["Spanish landmark", "landmark in Spain"], spanish_landmarks)

# ============ CURRENCY ABBREVIATIONS ============
currency_abbrevs = {
    "AUD": "dollar", "CAD": "dollar", "CHF": "franc", "CNY": "yuan",
    "DKK": "krone", "EUR": "euro", "GBP": "pound", "HKD": "dollar",
    "HUF": "forint", "INR": "rupee", "JPY": "yen", "MXN": "peso",
    "NOK": "krone", "NZD": "dollar", "RUB": "rouble", "SEK": "krona",
    "SGD": "dollar", "USD": "dollar", "ZAR": "rand",
}
for abbr in currency_abbrevs:
    for defn in ["currency abbreviation", "currency code"]:
        key = (defn.lower(), abbr)
        if key not in existing:
            c.execute("INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
                      (defn, abbr, "crossword_companion"))
            existing.add(key)
            inserted += 1
        else:
            skipped += 1

conn.commit()
print(f"\nTotal inserted: {inserted}")
print(f"Total skipped: {skipped}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"Total crossword_companion entries: {c.fetchone()[0]}")
conn.close()
