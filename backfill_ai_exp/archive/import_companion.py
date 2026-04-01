"""Import crossword companion data into definition_answers_augmented."""
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

# ============ GREEK ALPHABET ============
greek = ["MU", "NU", "PI", "XI", "CHI", "ETA", "PHI", "PSI", "RHO", "SAN", "TAU", "VAU",
         "BETA", "IOTA", "ZETA", "ALPHA", "DELTA", "GAMMA", "KAPPA", "KOPPA", "OMEGA",
         "SIGMA", "THETA", "LAMBDA", "DIGAMMA", "EPSILON", "OMICRON", "UPSILON", "YPSILON"]
add(["Greek letter", "letter", "Greek character"], greek)

# ============ HEBREW ALPHABET ============
hebrew = ["FE", "HE", "PE", "BET", "HEH", "HET", "KAF", "MEM", "NUN", "PEH", "QOF", "SIN",
          "TAV", "TAW", "TET", "VAV", "WAW", "YOD", "ALEF", "AYIN", "BETH", "CHAF", "HETH",
          "KAPH", "KHAF", "KOPH", "RESH", "SADE", "SHIN", "TETH", "YODH", "ALEPH", "CHETH",
          "DALET", "GIMEL", "LAMED", "SADHE", "TSADI", "TZADE", "ZAYIN", "DALETH", "LAMEDH",
          "SADDHE", "SAMECH", "SAMEKH"]
add(["Hebrew letter", "letter", "Hebrew character"], hebrew)

# ============ NATO PHONETIC ALPHABET ============
nato = ["ECHO", "GOLF", "KILO", "LIMA", "MIKE", "PAPA", "XRAY", "ZULU",
        "ALPHA", "BRAVO", "DELTA", "HOTEL", "INDIA", "OSCAR", "ROMEO", "TANGO",
        "JULIET", "QUEBEC", "SIERRA", "VICTOR", "YANKEE", "CHARLIE", "FOXTROT",
        "UNIFORM", "WHISKEY", "NOVEMBER"]
add(["NATO alphabet", "phonetic letter", "code word"], nato)

# ============ CHEMICAL ELEMENTS ============
elements = [
    "TIN", "GOLD", "IRON", "LEAD", "NEON", "ZINC", "ARGON", "BORON", "RADON", "XENON",
    "BARIUM", "CARBON", "CERIUM", "COBALT", "COPPER", "CURIUM", "ERBIUM", "HELIUM",
    "INDIUM", "IODINE", "NICKEL", "OSMIUM", "OXYGEN", "RADIUM", "SILVER", "SODIUM",
    "ARSENIC", "BISMUTH", "BROMINE", "CADMIUM", "CAESIUM", "CALCIUM", "CHROMIUM",
    "FERMIUM", "GALLIUM", "HAFNIUM", "HOLMIUM", "IRIDIUM", "KRYPTON", "LITHIUM",
    "MERCURY", "NIOBIUM", "RHENIUM", "RHODIUM", "SILICON", "SULPHUR", "TERBIUM",
    "THORIUM", "THULIUM", "URANIUM", "YTTRIUM", "ACTINIUM", "ANTIMONY", "NITROGEN",
    "CHLORINE", "FLUORINE", "HYDROGEN", "TITANIUM", "TUNGSTEN", "VANADIUM",
    "PLATINUM", "SELENIUM", "SCANDIUM", "TANTALUM", "ALUMINIUM", "AMERICIUM",
    "BERKELIUM", "BERYLLIUM", "GERMANIUM", "LANTHANUM", "MAGNESIUM", "MANGANESE",
    "NEPTUNIUM", "PALLADIUM", "PLUTONIUM", "POTASSIUM", "STRONTIUM", "TELLURIUM",
    "ZIRCONIUM", "PHOSPHORUS", "PROMETHIUM", "MOLYBDENUM", "TECHNETIUM", "NEODYMIUM",
    "PRASEODYMIUM", "PROTACTINIUM", "RUTHERFORDIUM",
]
add(["element", "chemical element"], elements)

# ============ MUSICAL NOTES (SOL-FA) ============
solfa = ["DO", "RE", "MI", "FA", "SO", "LA", "SI", "TI", "TE", "UT", "DOH", "FAH", "LAH", "RAY", "SOH", "SOL"]
add(["musical note", "note", "note of the scale"], solfa)

# ============ MUSICAL INSTRUMENTS ============
instruments = [
    "SAX", "BASS", "BELL", "DRUM", "ERHU", "FIFE", "GONG", "HARP", "HORN", "KORA",
    "KOTO", "LUTE", "LYRE", "OBOE", "PIPE", "TUBA", "VIOL", "ZEZE",
    "AMATI", "BANJO", "BELLS", "BONGO", "BUGLE", "CELLO", "CHIME", "FLUTE",
    "GUSLA", "GUSLE", "GUSLI", "KAZOO", "MBIRA", "ORGAN", "PIANO", "PIPES",
    "REBEC", "SHALM", "SHAWM", "SITAR", "TABLA", "TABOR", "VIBES", "VIOLA",
    "ZANZE", "ZIRNA", "ZURNA",
    "CITHER", "CORNET", "CYMBAL", "FENDER", "FIDDLE", "GUITAR", "RATTLE", "SPINET",
    "TABOUR", "VIOLIN", "ZITHER",
    "ALPHORN", "BAGPIPE", "BARYTON", "BASSOON", "BODHRAN", "BUCCINA", "CELESTE",
    "CEMBALO", "CITHARA", "CITHERN", "CITTERN", "CLARION", "CLAVIER", "COWBELL",
    "HAUTBOY", "LYRICON", "MARACAS", "MARIMBA", "OCARINA", "PANDORA", "PICCOLO",
    "SACKBUT", "SAMBUCA", "SAXHORN", "SERPENT", "SISTRUM", "TAMBURA", "THEORBO",
    "TIMPANI", "TRUMPET", "UKULELE", "VIHUELA", "WHISTLE",
    "ANGKLUNG", "BAGPIPES", "BARYTONE", "BASS DRUM", "BOUZOUKI", "CALLIOPE",
    "CARILLON", "CIMBALOM", "CLAPPERS", "CLARINET", "CLARSACH", "CORNPIPE",
    "CRUMHORN", "DULCIMER", "HANDBELL", "HORNPIPE", "HUMSTRUM", "KEYBOARD",
    "MANDOLIN", "MANZELLO", "MELODEON", "PAN-PIPES", "POLYPHON", "RECORDER",
    "SIDE-DRUM", "SPINETTE", "STEINWAY", "THERAMIN", "THEREMIN", "TIMBALES",
    "TRIANGLE", "TROMBONE", "VIRGINAL", "VOCALION", "ZAMBOMBA",
    "ACCORDION", "ALPENHORN", "BALALAIKA", "BANJOLELE", "BUGLE-HORN", "CASTANETS",
    "CHIME BARS", "DECACHORD", "EUPHONIUM", "FLAGEOLET", "HARMONICA", "HARMONIUM",
    "POLYPHONE", "SAXOPHONE", "SNARE-DRUM", "TENOR-DRUM", "WOOD BLOCK", "XYLOPHONE",
    "BASS GUITAR", "BIRD-SCARER", "BONGO-DRUMS", "BULLROARER", "CLAVICHORD",
    "CONCERTINA", "COR ANGLAIS", "DIDGERIDOO", "DOUBLE BASS", "EOLIAN HARP",
    "FLUGELHORN", "FRENCH HORN", "GRAND PIANO", "HURDY-GURDY", "KETTLE-DRUM",
    "MOUTH ORGAN", "PENTACHORD", "PIANOFORTE", "SOUSAPHONE", "SQUEEZE-BOX",
    "TAMBOURINE", "THUMB PIANO", "TIN WHISTLE", "VIBRAPHONE",
    "AEOLIAN HARP", "BARREL ORGAN", "HARPSICHORD", "PHONOFIDDLE", "PLAYER-PIANO",
    "SLEIGH BELLS", "SYNTHESIZER", "VIOLONCELLO",
    "GLOCKENSPIEL", "HARMONICHORD", "PENNY WHISTLE",
]
add(["musical instrument", "instrument"], instruments)

# ============ FRENCH WORDS/EXPRESSIONS ============
french_all = [
    "ELAN", "A DEUX", "ADIEU", "BLASE", "COUPE", "DOYEN", "ENNUI", "OUTRE",
    "AU FAIT", "AU PAIR", "CLICHE", "DEJA VU", "DE TROP", "RISQUE",
    "AFFAIRE", "A LA MODE", "ATELIER", "CHAGRIN", "CHAMBRE", "EN ROUTE",
    "ENTENTE", "FAUX PAS", "PELOTON", "VIS-A-VIS",
    "A LA CARTE", "AMBIANCE", "APRES-SKI", "BARRETTE", "CUL-DE-SAC",
    "DERRIERE", "FILM NOIR", "IDEE FIXE", "MOT JUSTE", "PRIX FIXE",
    "AU COURANT", "AU NATUREL", "BANQUETTE", "BEAU MONDE", "BEL ESPRIT",
    "BETE NOIRE", "BON VIVANT", "BON VOYAGE", "BOURGEOIS", "SANGFROID",
    "VOLTE-FACE", "VIN DU PAYS", "DE RIGUEUR", "GRAND PRIX", "HAUT MONDE",
    "RECHERCHE",
    "AIDE-DE-CAMP", "AVANT-GARDE", "CORDON BLEU", "DESHABILLE", "PIED A TERRE",
    "AIDE-MEMOIRE", "AMUSE-BOUCHE", "AU CONTRAIRE", "BELLE EPOQUE", "BILLETS-DOUX",
    "CHEF D'OEUVRE", "COUP DE GRACE", "FEMME FATALE",
    "ANCIEN REGIME", "CARTE BLANCHE", "CAUSE CELEBRE", "FAIT ACCOMPLI",
    "FORCE MAJEURE", "HAUTE COUTURE", "LAISSEZ-FAIRE", "NOUVEAU RICHE",
    "TROMPE L'OEIL", "TOUR DE FORCE", "SAVOIR FAIRE", "RAISON D'ETRE",
    "PRET-A-PORTER", "FIN DE SIECLE",
    "NOBLESSE OBLIGE", "NOUVELLE CUISINE", "DOUBLE ENTENDRE", "ENFANT TERRIBLE",
    "CREME DE LA CREME",
]
add(["French expression", "French phrase", "French"], french_all)

# ============ GERMAN WORDS/EXPRESSIONS ============
german_words = [
    "ECHT", "FLAK", "UBER", "ANGST", "BLITZ", "GEIST", "KAPUT", "REICH", "STASI", "U-BOOT",
    "ABSEIL", "ERSATZ", "FUHRER", "KAISER", "KITSCH", "LANDAU", "PANZER", "UMLAUT",
    "ACHTUNG", "BAUHAUS", "GESTALT", "GESTAPO", "PILSNER", "PRETZEL", "STRUDEL",
    "AUTOBAHN", "DUMMKOPF", "HAUSFRAU", "KOHLRABI", "PINSCHER", "SCHNAPPS", "SPRITZER", "ZUGZWANG",
    "ALPENHORN", "ANSCHLUSS", "BRATWURST", "BUNDESTAG", "DACHSHUND", "EDELWEISS", "HAMBURGER",
    "LEITMOTIV", "LUFTWAFFE", "REICHSTAG", "SCHNAUZER", "SCHNITZEL", "WEHRMACHT", "ZEITGEIST",
    "ALPENSTOCK", "BLITZKRIEG", "FRANKFURTER", "KULTURKAMPF", "POLTERGEIST",
    "KINDERGARTEN", "WANDERLUST", "WUNDERKIND",
]
add(["German word", "German expression", "German"], german_words)

# ============ ITALIAN WORDS/EXPRESSIONS ============
italian_words = [
    "CIAO", "PREGO", "SALVE", "GRAZIE", "STUCCO",
    "AL DENTE", "BARISTA", "AL FRESCO", "INTAGLIO", "SERAGLIO",
    "A CAPPELLA", "ANTIPASTO", "CRESCENDO", "PAPARAZZO", "SGRAFFITO",
    "SOTTO VOCE", "BUONGIORNO", "COSA NOSTRA", "PRIMA DONNA",
    "COSI FAN TUTTE", "LINGUA FRANCA", "CHE SARA SARA", "CHIAROSCURO",
    "GRAN TURISMO", "LA DOLCE VITA", "ARRIVEDERCI",
]
add(["Italian word", "Italian expression", "Italian"], italian_words)

# ============ LATIN WORDS/EXPRESSIONS ============
latin_words = [
    "SIC", "IDEM", "PACE", "STET", "AD HOC", "CIRCA", "ID EST", "PER SE",
    "GRATIS", "IBIDEM", "PASSIM", "ALUMNUS", "A PRIORI", "DE FACTO", "ERRATUM",
    "FLORUIT", "IN VITRO", "SUB ROSA", "AB INITIO", "ADDENDUM", "EMERITUS",
    "ET CETERA", "EX GRATIA", "GRAVITAS", "INFRA DIG", "MEA CULPA", "NOTA BENE",
    "SUBPOENA", "AD NAUSEAM", "ALMA MATER", "CARPE DIEM", "ET TU BRUTE",
    "EX OFFICIO", "INTER ALIA", "IPSO FACTO", "PER CAPITA", "STATUS QUO",
    "SUB JUDICE", "VOX POPULI", "ANNO DOMINI", "ANTE-BELLUM", "EX CATHEDRA",
    "IN ABSENTIA", "IN EXTREMIS", "MAGNUM OPUS", "POST MORTEM", "PRIMA FACIE",
    "QUID PRO QUO", "SINE QUA NON", "TABULA RASA", "AD INFINITUM", "MEMENTO MORI",
    "NON SEQUITUR", "TEMPUS FUGIT", "ANTE MERIDIEM", "CAVEAT EMPTOR", "COMPOS MENTIS",
    "HABEAS CORPUS", "POST MERIDIEM", "CAMERA OBSCURA", "DEUS EX MACHINA",
    "EXEMPLI GRATIA", "MODUS OPERANDI", "ANNUS MIRABILIS", "IN LOCO PARENTIS",
    "PRO BONO PUBLICO", "TERRA INCOGNITA", "CURRICULUM VITAE",
    "DELIRIUM TREMENS", "PERSONA NON GRATA",
]
add(["Latin expression", "Latin phrase", "Latin"], latin_words)

# ============ SPANISH WORDS/EXPRESSIONS ============
spanish_words = [
    "OLE", "ADOBE", "COSTA", "TAPAS", "GUANO", "JUNTA", "PLAYA",
    "BARRIO", "BODEGA", "BOLERO", "EL NINO", "GAUCHO", "GITANA", "GRINGO",
    "HOMBRE", "MANANA", "PELOTA", "CHICANO", "CORRIDA", "INFANTA",
    "MATADOR", "PICADOR", "VAQUERO", "COMPADRE", "EL DORADO", "HACIENDA",
    "FRIJOLES", "HABANERA", "MARIACHI", "TOREADOR", "AY CARAMBA",
    "BANDOLERO", "GUERRILLA", "AFICIONADO", "CARABINERO", "PECCADILLO",
    "EMBARCADERO", "HASTA LA VISTA", "INCOMMUNICADO",
]
add(["Spanish word", "Spanish expression", "Spanish"], spanish_words)

# ============ FORMER CURRENCIES ============
former_currencies = [
    "LIRA", "MARK", "PUNT", "FRANC", "KROON", "POUND", "SUCRE", "TOLAR", "ZAIRE",
    "ESCUDO", "KORUNA", "MARKKA", "PESETA", "DRACHMA", "GUILDER", "SCHILLING", "DEUTSCHMARK",
]
add(["former currency", "old currency", "currency", "money"], former_currencies)

# ============ CURRENT CURRENCIES ============
current_currencies = [
    "BAHT", "DRAM", "EURO", "KINA", "LARI", "LIRA", "RAND", "REAL", "RIAL", "YUAN",
    "DINAR", "FRANC", "KRONE", "KRONA", "MANAT", "NAIRA", "POUND", "RIYAL", "RUPEE",
    "DOLLAR", "FORINT", "SHEKEL", "TUGRIK",
    "AFGHANI", "BOLIVAR", "CORDOBA", "GUARANI", "METICAL", "QUETZAL", "RINGGIT",
    "SHILLING", "STERLING",
]
add(["currency", "money", "unit of currency"], current_currencies)

# ============ MUSICAL TERMS ============
music_terms = [
    "BAR", "BIS", "CUE", "KEY", "TIE",
    "ALTO", "ARCO", "BASS", "BEAT", "CLEF", "CODA", "FINE", "FLAT", "FRET",
    "HOLD", "MODE", "MUTE", "NOTE", "PART", "REST", "SLUR", "SOLO", "TONE", "TUNE",
    "CHORD", "DOLCE", "DRONE", "FORTE", "GRAVE", "LARGO", "LENTO", "LYRIC",
    "MAJOR", "METRE", "MINIM", "MINOR", "MOLTO", "OUTRO", "PAUSE", "PIANO",
    "PIECE", "PITCH", "SCALE", "SCORE", "SENZA", "SHAKE", "SHARP", "STAFF",
    "STAVE", "SWELL", "TACET", "TANTO", "TEMPO", "TENOR", "THEME", "TRIAD",
    "TRILL", "TUTTI",
    "ADAGIO", "DA CAPO", "DUPLET", "ENCORE", "FINALE", "LEGATO", "MANUAL",
    "MEDLEY", "MELODY", "OCTAVE", "PHRASE", "PRESTO", "QUAVER", "RHYTHM",
    "SEMPRE", "SUBITO", "TENUTO", "TIMBRE", "TREBLE", "UNISON", "UPBEAT", "VIVACE",
    "AGITATO", "ALLEGRO", "AMOROSO", "ANDANTE", "ANIMATO", "ATTACCA", "CADENCE",
    "CON BRIO", "CONCERT", "CON MOTO", "DESCANT", "HARMONY", "MARCATO", "MORDENT",
    "NATURAL", "RECITAL", "REFRAIN", "SOPRANO", "TREMOLO", "TRIPLET", "VIBRATO",
    "ACOUSTIC", "ALTO CLEF", "ARPEGGIO", "BARITONE", "BASS CLEF", "CON FUOCO",
    "CROTCHET", "DIATONIC", "DOMINANT", "DOWNBEAT", "ENSEMBLE", "INTERVAL",
    "MAESTOSO", "MODERATO", "STACCATO", "VIRTUOSO",
    "ALLA BREVE", "ALTISSIMO", "CANTABILE", "CANTILENA", "CHROMATIC", "CONTRALTO",
    "CRESCENDO", "GLISSANDO", "HARMONICS", "LARGHETTO", "MEZZA VOCE", "PIZZICATO",
    "SEMIBREVE", "SFORZANDO", "SMORZANDO", "SOSTENUTO", "SOTTO VOCE", "TABLATURE",
    "TENOR CLEF",
    "ACCIDENTAL", "DIMINUENDO", "DISSONANCE", "FORTISSIMO",
    "RALLENTANDO", "DECRESCENDO", "ACCELERANDO",
]
add(["musical term", "music term", "musical direction"], music_terms)

# ============ MUSIC COMPOSITIONS ============
compositions = [
    "JIG", "LAY", "RAG", "ARIA", "HYMN", "LIED", "MASS", "OPUS", "RAGA", "REEL",
    "SONG", "TUNE", "CANON", "CAROL", "ETUDE", "FUGUE", "GIGUE", "MARCH", "MOTET",
    "OPERA", "PIECE", "POLKA", "RONDO", "ROUND", "SUITE", "TANGO", "TRACK", "WALTZ",
    "ANTHEM", "AUBADE", "BALLAD", "BOLERO", "CHORUS", "LAMENT", "LIEDER", "MASQUE",
    "MINUET", "NUMBER", "PAVANE", "SHANTY", "SONATA",
    "BALLADE", "BOURREE", "CANTATA", "CHORALE", "FANFARE", "GAVOTTE", "MAZURKA",
    "PARTITA", "PRELUDE", "REQUIEM", "SCHERZO", "TOCCATA",
    "BERCEUSE", "CAVATINA", "CHACONNE", "CONCERTO", "FANDANGO", "FANTASIA",
    "GALLIARD", "HORNPIPE", "MADRIGAL", "NOCTURNE", "OPERETTA", "OVERTURE",
    "RHAPSODY", "SARABAND", "SERENADE", "SONATINA", "SYMPHONY", "ZARZUELA",
    "ALLEMANDE", "ARABESQUE", "BAGATELLE", "CABALETTA", "CAPRICCIO", "ECOSSAISE",
    "FARANDOLE", "IMPROMPTU", "INVENTION",
    "BARCAROLLE", "BERGAMASCA", "CONCERTINO", "HUMORESQUE", "INTERMEZZO",
    "OPERA BUFFA", "TARANTELLA",
]
add(["musical composition", "composition", "piece of music", "musical work"], compositions)

# ============ FRENCH CITIES ============
french_cities = [
    "CAEN", "LYON", "NICE", "ARLES", "DIJON", "LILLE", "LYONS", "PARIS",
    "REIMS", "ROUEN", "TOURS", "AMIENS", "CALAIS", "CANNES", "LE MANS", "NANTES",
    "RENNES", "RHEIMS", "AVIGNON", "DUNKIRK", "LIMOGES", "ORLEANS", "BORDEAUX",
    "TOULOUSE", "CHERBOURG", "MARSEILLES", "STRASBOURG", "CARCASSONNE", "MONTPELLIER",
]
add(["French city", "city in France", "French town"], french_cities)

conn.commit()
print(f"\nTotal inserted: {inserted}")
print(f"Total skipped: {skipped}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"Total crossword_companion entries: {c.fetchone()[0]}")
conn.close()
