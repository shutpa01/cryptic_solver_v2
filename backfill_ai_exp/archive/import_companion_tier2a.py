"""Import Tier 2a: Rivers, Birds, Trees, Roads, and extras from scan."""
import sqlite3
import re

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

# ============ RIVER/WATERCOURSE TYPES ============
river_types = [
    "EA", "CUT", "POW", "SOY", "BECK", "BURN", "FLOW", "KILL", "LAKE",
    "NALA", "RILL", "WADI", "WADY", "BOURN", "BROOK", "CANAL", "CREEK",
    "DELTA", "FIRTH", "FLUSH", "INLET", "MOUTH", "NALLA", "NULLA",
    "AFFLUENT", "BROOKLET", "EFFLUENT", "INFLUENT", "WATERWAY",
    "ANABRANCH", "BILLABONG", "CONFLUENCE", "HEADWATER",
    "STREAMLET", "TRIBUTARY", "BACKWATER",
    "RILLET", "RUNNEL", "SOURCE", "STREAM", "CHANNEL", "ESTUARY",
    "FRESHET", "RIVERET", "TORRENT", "RIVERBEND",
    "HEAD-STREAM", "MILLSTREAM", "STREAMING",
    "TROUT STREAM", "WATER SPLASH", "DISTRIBUTARY", "EMBRANCHMENT",
    "MOUNTAIN STREAM",
]
add(["watercourse", "river type", "waterway", "stream"], river_types)

# ============ RIVERS (2-letter) ============
rivers_2 = ["EA", "LI", "OB", "PO"]

# ============ RIVERS (3-letter) ============
rivers_3 = [
    "AHR", "AIN", "AXE", "BUG", "CAM", "DAL", "DEE", "DON", "EMS",
    "ESK", "EXE", "FAL", "FLY", "HSI", "ILL", "LEE", "LEK", "LIM",
    "LOT", "LYS", "MUN", "MUR", "NIT", "OLI", "OUI", "RED", "SAU",
    "SAN", "TAW", "TAY", "TUA", "UME", "UNA", "USK", "VAH", "VEY",
    "WYE",
]

# ============ RIVERS (4-letter) ============
rivers_4 = [
    "AARE", "AIRE", "ALTA", "AMUR", "ARNO", "ARUN", "AUBE", "AUDE",
    "AVON", "BANN", "BENI", "CHER", "COCO", "DART", "DRIN", "EBRO",
    "EDEN", "ELBE", "ENNS", "GAIL", "GERS", "GILA", "HONG", "IBAR",
    "KAMP", "KEMI", "KRKA", "KUPA", "KYMI", "LABE", "LAHN", "LECH",
    "LELE", "LIMA", "LULE", "LUNE", "MAAS", "MAIN", "MINO", "MIRA",
    "MOLE", "MONO", "MURG", "NAAB", "NAMO", "NAPA", "NENE", "NILE",
    "NITH", "ODER", "OISE", "OMME", "ORNE", "OUSE", "OXUS", "PING",
    "PRUT", "RAAB", "RABA", "RAMU", "RAVI", "RENO", "RUHR", "SAAR",
    "SADO", "SAVA", "SAVE", "SOAR", "SPEY", "SURE", "SWIT", "SWAN",
    "TAFF", "TAJO", "TANA", "TANO", "TARN", "TARO", "TEES", "TEJO",
    "TEME", "TEST", "TONY", "TYNE", "TYVI", "URAL", "VAAL", "VALK",
    "VALI", "YARE",
]

# ============ RIVERS (5-letter) ============
rivers_5 = [
    "ADIGE", "ADOUR", "AGOUT", "AISNE", "ALDAN", "ARGUN", "BENUE",
    "BLACK", "BOYNE", "CAUCA", "CHARI", "CLERT", "CLYDE", "CONWY",
    "DEMIR", "DESNA", "DONAU", "DOUBS", "DRAVA", "DROME", "DUERO",
    "EIDER", "FLEET", "FORTH", "FULDA", "GANGA", "GENIL", "GILMA",
    "HUNTE", "INDUS", "ISERE", "ISHIM", "ISKAR", "JALON", "JICAR",
    "JUMNA", "KALPA", "KOLON", "KUBAN", "LAGEN", "LAGAN", "LEINA",
    "LEMPA", "LOING", "LOIRE", "LUANG", "MANAS", "MARCH", "MARNE",
    "MEMEL", "MERIC", "MEUSE", "MINHO", "MOSEL", "MULDE",
    "NAREW", "NEGRO", "NEMAN", "NEWRY", "NIGER", "NOTER",
    "OMEGA", "OTTER", "OUNAS", "PEACE", "PEARL", "PECOS", "PERAK",
    "PIAVE", "PLATE", "PURUS", "RHEIN", "RHINE", "RHONE", "RISLE",
    "SAALE", "SAONE", "SARRE", "SAURE", "SEGRE", "SEINE", "SERET",
    "SOMES", "SOMME", "SPEER", "STOUR", "TAGUS", "TAMAR", "TEIFI",
    "THAME", "TIBER", "TINTO", "TISZA", "TOBOL", "TRAUN", "TRAVE",
    "TRENT", "TUMEN", "TURIA", "TWEED", "VARDE", "VOLGA", "VOLTA",
    "VOUGA", "VRBAS", "WARTA", "WERRA", "WESER", "WISLA",
    "XINGU", "YONNE", "YUKON", "ZUJAR",
]

# ============ RIVERS (6-letter) ============
rivers_6 = [
    "ALIAKOS", "ALLIER", "AMAZON", "ANGARA", "ANIENE", "BARROW",
    "BELIZE", "BRAZOS", "CALDER", "CLUTHA", "COOPER", "CREUSE",
    "DANUBE", "DENDER", "DEVOLL", "DNIEST", "DONETS",
    "ESCAUT", "FRASER", "GAMBIA", "GAMBIE", "GANDAK", "GANGES",
    "GLOMMA", "GRANDE", "GUAYAS", "GUDENA", "HEBRON", "HUDSON",
    "IIJOKI", "IMATRA", "IRTYSH", "JANTRA", "JHELUM", "JORDAN",
    "KAVERI", "KHABAR", "KISTNA", "KOLYMA", "LEITHA", "LIFFEY",
    "LJUNGAN", "LOGONE", "MARICA", "MEKONG", "MEPING", "MERSEY",
    "MOLDAU", "MORAVA", "MURRAY", "NECKAR", "NELSON",
    "OGOOUE", "ORANGE", "OTTAWA", "OURTHE", "PARANA", "PIANCA",
    "PRIPET", "REJANG", "RIBBLE", "RIDEAU", "ROUUNA",
    "RUDALL", "RUVUMA", "SALADO", "SAMBRE", "SANAGA", "SARTHE",
    "SEGURA", "SEVERN", "STRUMA", "SUTLEJ", "TAMEGA",
    "TICINO", "TIGRIS", "TUMMEL", "UBANGI",
    "VIENNA", "VIJOSE", "VITAVA", "VYATKA", "WABASH", "WARTHE",
    "WIEPRZ", "WITHAM", "YAKIMA", "YAMUNA", "YANTRA", "YELLOW",
]

# ============ RIVERS (7+ letter) ============
rivers_7plus = [
    "ALPHEUS", "ARDECHE", "AVEYRON", "BIREMES", "CABRIAL",
    "CALEDON", "CHARLES", "CUBANGO", "DARLING", "DERWENT",
    "DNIEPER", "DNIESTER", "DRAMMEN", "DURANEE", "GARONNE",
    "GLOMMEN", "GUAYAPE", "HANGANG", "HELMAND", "HERAULT",
    "HOOGHLY", "HUANG HE", "HUANG HO", "ILMENAU", "KAVANGO",
    "KEMIJOKI", "KRISHNA", "LACHLAN", "LIMPOPO",
    "LJUNGAN", "LUALABA", "MADEIRA", "MARANON", "MARITSA",
    "MATAPAU", "MAYENNE", "MEURTHE", "MONDEGO", "MOSELLE",
    "MORAGNA", "NARBADA", "NARMADA", "NERETVA", "NIAGARA",
    "ORINOCO", "ORONTES", "PARRETT", "PECHORA", "POTOMAC",
    "SALWEEN", "SALZACH", "SAN JUAN", "SCHELDE", "SCHELDT",
    "SEGOVIA", "SELENGA", "SHALGAR", "SHANNON", "SITTANG",
    "SLANEY", "STANNON", "SUKHONA", "TAPAJDS",
    "THAMES", "UCAYALI", "URUGUAY", "VISTULA", "WAIKATO",
    "WAVENEY", "WELLAND", "YANGTZE", "YENISEI", "YENISEY",
    "ZAMBEZI",
    "ABUKUMA", "ACHELOQS", "AKIKOMBO", "ALTAMAHA", "AMAZONAS",
    "AMU DARYA", "ARAGUAIA", "ARKANSAS", "ATHABASCA", "BLUE NILE",
    "CANADIAN", "CHARENTE", "CHU JIANG", "CLARENCE", "COLORADO",
    "COLUMBIA", "CUYAHOGA", "DALALVEN", "DELAWARE", "DEMERARA",
    "DNIESTER", "DNEISTER", "DORDOGNE", "DRYSDALE",
    "GHAGHARA", "GODAVARI", "GUADIANA", "IALOMICA",
    "KEMIJOKI", "KLONDIKE", "MAHANADI", "MAHAWELI", "MAZARUNI",
    "MISSOURI", "OKANANGO", "ORKHON", "PARAGUAY", "PARRIBA",
    "PSURUGA", "RIO BRACO", "RIO NEGRO", "SANTA ANA",
    "SHKUMBIN", "SOLIMOES", "SYR DARYA",
    "TUNGUSKA", "VYSHEQRA", "WANGANUI", "ZHU JIANG",
    "ALBERT NILE", "CHURCHILL", "CROCODILE",
    "EUPHRATES", "GREAT OUSE", "IRRAWADDY", "KATHERINE",
    "KUSKOKWIM", "MACKENZIE", "MAGDALENA",
    "RIO GRANDE", "SAN CARLOS", "SCHWECHAT", "ST LAURENT",
    "TOCANTINS", "WHITE NILE",
    "CHAO PHRAYA", "DES PLAINES", "HACKENSACK",
    "RANGITIKEI", "RIO URUGUAY", "SAN ANTONIO", "SAN JOAQUIN",
    "SHENANDOAH", "ST LAWRENCE", "WALLA WALLA",
    "ASSINIBOINE", "MISSISSIPPI", "RIO PARAGUAY",
    "GUADALQUIVIR", "SAO FRANCISCO", "SASKATCHEWAN",
]

all_rivers = rivers_2 + rivers_3 + rivers_4 + rivers_5 + rivers_6 + rivers_7plus
add(["river", "waterway", "watercourse"], all_rivers)

# ============ ROAD TYPES ============
road_types = [
    "A", "B", "C", "WAY", "DRAG", "HIGH", "LANE", "MEWS", "PASS",
    "RING", "SIDE", "SLIP", "TOLL", "ALLEY", "BYWAY", "CLOSE", "GATED",
    "ROMAN", "ROUTE", "STRIP", "TRACK", "TRUNK",
    "AVENUE", "BYPASS", "PARADE", "RELIEF",
    "BELLWAY", "DEAD END", "FLYOVER", "FREEWAY", "HIGHWAY",
    "OFF RAMP", "PARKWAY", "THROUGH",
    "ALLEYWAY", "CAUSEWAY", "CLEARWAY", "CRESCENT", "CUL-DE-SAC",
    "METALLED", "MOTORWAY", "OVERPASS", "RED ROUTE", "SHORT CUT",
    "SPEEDWAY", "TRACKWAY", "TURNPIKE",
    "AUTOROUTE", "BOULEVARD", "BRIDLEWAY", "CART TRACK", "DIRT TRACK",
    "ESPLANADE", "GREEN LANE", "PROMENADE", "UNADOPTED", "UNDERPASS",
    "AUTOSTRADA", "BRIDLEPATH", "CLOVERLEAF", "EXPRESSWAY", "INTERSTATE",
    "UNMETALLED",
    "GRAVEL TRACK", "SCENIC ROUTE", "SINGLE TRACK",
    "HOLIDAY ROUTE", "MOUNTAIN PASS", "SUPERHIGHWAY", "THOROUGHFARE",
    "UNCLASSIFIED",
    "DIVIDED HIGHWAY", "GYRATORY SYSTEM",
    "DUAL CARRIAGEWAY", "ELEVATED SECTION", "EUROPEAN HIGHWAY",
]
add(["road", "road type", "route", "way"], road_types)

# ============ BIRDS (3-letter) ============
birds_3 = [
    "ANI", "AUK", "COB", "DAW", "EMU", "HEN", "JAY", "KEA", "MEW",
    "MOA", "OWL", "PEN", "PIE", "ROC", "TIT", "TUI",
]

# ============ BIRDS (4-letter) ============
birds_4 = [
    "CHAT", "COOT", "CROW", "DODO", "DOVE", "DUCK", "ERNE", "GAGE",
    "GULL", "HAWK", "IBIS", "JACK", "KAGU", "KITE", "KNOT", "LARK",
    "LOON", "MINA", "MYNA", "NENE", "PERN", "RAIL", "RHEA", "ROOK",
    "RUFF", "SHAG", "SKUA", "SMEW", "SORA", "SWAN", "TEAL", "TERN",
    "WREN",
]

# ============ BIRDS (5-letter) ============
birds_5 = [
    "BOOBY", "CAPON", "COLIN", "CRAKE", "CRANE", "DIVER", "EAGLE",
    "EGRET", "EIDER", "FINCH", "FLEET", "GALAH", "GOOSE", "GREBE",
    "HERON", "HOBBY", "JUNCO", "MACAW", "MAVIS", "MERLE", "MONAL",
    "MURRE", "MYNAH", "NANDU", "NODDY", "OUSEL", "OUZEL", "PEWIT",
    "PIPIT", "PITTA", "POKER", "QUAIL", "RAVEN", "ROBIN", "SERIN",
    "SNIPE", "STILT", "STORK", "SWIFT", "VEERY", "VIREO",
]

# ============ BIRDS (6-letter) ============
birds_6 = [
    "AVOCET", "BANDIT", "BARBET", "BISHOP", "BULBUL", "CANARY",
    "CHOUGH", "CONDOR", "CUCKOO", "CURLEW", "CYGNET", "DIPPER",
    "DRONGO", "DUNLIN", "FALCON", "FULMAR", "GANNET", "GODWIT",
    "GROUSE", "HOOPOE", "JABIRU", "KAKADO", "LINNET", "MAGPIE",
    "MARTIN", "MERLIN", "MOONAL", "ORIOLE", "OSPREY", "PARROT",
    "PEAHEN", "PETREL", "PIGEON", "PLOVER", "PUFFIN", "QUELEA",
    "REDCAP", "ROLLER", "SHRIKE", "SISKIN", "THRUSH", "TOUCAN",
    "TURACO", "TURKEY", "WAGTAIL",
]

# ============ BIRDS (7-letter) ============
birds_7 = [
    "ANTBIRD", "BITTERN", "BLUEJAY", "BUNTING", "BUSTARD", "BUZZARD",
    "CATBIRD", "COALIT", "COURSER", "COWBIRD", "CREEPER", "DOTTREL",
    "FANTAIL", "FIGBIRD", "GADWALL", "GOBBLER", "GOSHAWK", "GRACKLE",
    "HALCYON", "HARRIER", "JACAMAR", "JACKDAW", "KESTREL", "KINGLET",
    "LAPWING", "LIMPKIN", "MALLARD", "MANAKIN", "MARTLET", "MOORHEN",
    "OILBIRD", "ORTOLAN", "PEACOCK", "PELICAN", "PENGUIN", "PHOENIX",
    "PINTAIL", "POCHARD", "REDPOLL", "REDWING", "ROOSTER", "ROSELLA",
    "RUDDOCK", "SAKERET", "SEAGULL", "SKYLARK", "SPARROW", "SUNBIRD",
    "SWALLOW", "TANAGER", "TINAMOU", "TITLARK", "VULTURE", "WAXWING",
    "WRYNECK",
]

# ============ BIRDS (8+ letter) ============
birds_8plus = [
    "ACCENTOR", "ADJUTANT", "AIGRETTE", "ALCATRAZ", "ARAPUNGA",
    "AVADAVAT", "BELLBIRD", "BLACKCAP", "BLUEBIRD", "BOATBILL",
    "BOBWHITE", "BUNTLARK", "COCKATOO", "COCKEREL", "CURASSOW",
    "DABCHICK", "DOTTEREL", "FIRECBIRD", "FISH-HAWK", "FLAMINGO",
    "GNATCREN", "GREAT TIT", "GROSBEAK", "GUACHARO", "HAWFINCH",
    "HERNSHAW", "HOATZIN", "HORNBILL", "KINGBIRD",
    "LANDRAIL", "LAVEROCK", "LEAFBIRD", "LORIEET", "LOVEBIRD",
    "LYREBIRD", "MANNAKIN", "MARABOUT", "MEGAPODE", "MYNA BIRD",
    "NIGHTJAR", "NUTHATCH", "OVENBIRD", "OXPECKER", "PALMCHAT",
    "PARAKEET", "PERCOLIN", "PHEASANT",
    "PRUNELLA", "REDSHANK", "REDSTART", "REEDBIRD", "REEDLING",
    "RINGTAIL", "ROCK BIRD", "ROCK DOVE", "SAND LARK",
    "SEA EAGLE", "SHOEBILL", "STARLING", "SURFBIRD",
    "TICK BIRD", "TITMOUSE", "TRAGOPAN", "WHEATEAR", "WHIMBREL",
    "WHINCHAT", "WHITE-EYE", "WIDEBIRD", "WOODCOCK", "WOODLARK",
    "WOODPECKER",
    "ALBATROSS", "BALD EAGLE", "BALTIMORE", "BEE-EATER", "BLACKBIRD",
    "BRAMBLING", "BROADBILL", "BULLFINCH", "CAMPANERO", "CASSOWARY",
    "CHAFFINCH", "CHICKADEE", "CORMORANT", "CORNCRAKE", "CROSSBILL",
    "EIDER DUCK", "FAIRY TERN", "FIELDFARE", "FIRECREST", "FLYCATCHER",
    "FRANCOLIN", "FROGMOUTH", "GALLINULE", "GERFALCON", "GOLDCREST",
    "GOLDENEYE", "GOLDFINCH", "GOOSANDER", "GYRFALCON",
    "GUILLEMOT", "GUINEA FOWL",
    "HAPPY EAGLE", "HARRIS HAWK", "HEN HARRIER", "SCREECH OWL",
    "SPOTTED OWL", "TAWNY EAGLE",
    "HONEY GUIDE", "INDIGO BIRD",
    "KINGFISHER", "KOOKABURRA", "MAGPIE LARK", "MEADOW PIPIT",
    "MUTTON BIRD", "NIGHT HERON", "NIGHTHAWK",
    "PRATINCOLE", "QUAKER BIRD", "SANDERLING", "SANDGROUSE",
    "SAND MARTIN", "SHEARWATER", "SECRETARY BIRD",
    "SPARROWHAWK", "STONECHAT", "TREE CREEPER", "WOODPECKER",
    "YELLOWHAMMER",
]

# ============ BIRDS OF PREY ============
birds_of_prey = [
    "OWL", "ERNE", "HAWK", "KITE", "PERN", "EAGLE", "HOBBY",
    "FALCON", "MERLIN", "OSPREY", "RAPTOR",
    "BARN OWL", "HAWK OWL", "KESTREL", "HARRIER", "BUZZARD", "GOSHAWK",
    "BATELEUR", "DUCK-HAWK", "EAGLE OWL", "FISH-HAWK",
    "BALD EAGLE", "BLACK KITE", "EAGLE-HAWK", "FISH EAGLE",
    "GYRFALCON", "STONE FALCON",
    "BOOTED EAGLE", "CHICKEN HAWK", "COOPER'S HAWK", "GOLDEN EAGLE",
    "SPARROWHAWK",
    "GREAT GREY OWL", "HONEY BUZZARD", "LONG-EARED OWL",
    "MARSH HAWK", "MARSH HARRIER",
    "HAPPY EAGLE", "HARRIS HAWK", "HEN HARRIER",
    "IMPERIAL EAGLE", "LESSER KESTREL",
    "MONTAGU'S HARRIER", "PEREGRINE FALCON", "RED-FOOTED FALCON",
    "SHORT-EARED OWL",
]
add(["bird of prey", "raptor", "hawk"], birds_of_prey)

# ============ SEABIRDS ============
seabirds = [
    "AUK", "COB", "GUGA", "GULL", "SHAG", "SKUA", "TERN",
    "CAHOW", "SOLAN", "FULMAR", "GANNET", "PETREL", "PUFFIN",
    "SEAGULL", "GUILLEMOT", "KITTIWAKE", "RAZORBILL",
    "ARCTIC SKUA", "ARCTIC TERN", "COMMON GULL", "COMMON TERN",
    "LITTLE GULL", "HERRING GULL", "ICELAND GULL", "ROSEATE TERN",
    "LITTLE TERN", "STORM PETREL", "SANDWICH TERN",
    "GLAUCOUS GULL", "LEACH'S PETREL", "POMARINE TERN",
    "SABINE'S GULL", "BERMUDA PETREL", "BLACK GUILLEMOT",
    "LONG-TAILED SKUA", "MANX SHEARWATER", "BLACK-HEADED GULL",
    "GREAT SKUA", "CORMORANT",
]
add(["seabird", "sea bird"], seabirds)

# ============ WADING BIRDS ============
wading_birds = [
    "REE", "HERN", "IBIS", "KNOT", "RUFF", "CRANE", "CRAKE", "EGRET",
    "HERON", "SNIPE", "STILT", "STORK", "AVOCET", "CURLEW", "DUNLIN",
    "GODWIT", "PLOVER", "BITTERN", "BUSTARD", "LAPWING", "DOTTEREL",
    "FLAMINGO", "REDSHANK", "SANDPIPER", "WHIMBREL", "WOODCOCK",
    "DOWITCHER", "GREY HERON", "PHALAROPE", "SANDPIPER",
    "GREENSHANK", "SANDERLING", "LITTLE STINT", "STONE CURLEW",
    "GOLDEN PLOVER", "GREAT BUSTARD", "RINGED PLOVER",
    "LITTLE BUSTARD", "OYSTER-CATCHER",
]
add(["wading bird", "wader"], wading_birds)

# ============ FLIGHTLESS BIRDS ============
flightless_birds = [
    "EMU", "DODO", "KIWI", "RHEA", "WEKA", "KAKAPO", "OSTRICH",
    "PENGUIN", "GREAT AUK", "TAKAHE", "NOTORNIS", "CASSOWARY",
    "MOA", "RATITE",
]
add(["flightless bird"], flightless_birds)

# All birds combined for general "bird" definition
all_birds = birds_3 + birds_4 + birds_5 + birds_6 + birds_7 + birds_8plus
add(["bird"], all_birds)

# ============ BIRTH FLOWERS ============
birth_flowers = [
    "ROSE", "ASTER", "DAISY", "HOLLY", "POPPY", "COSMOS", "VIOLET",
    "JONQUIL", "HAWTHORN", "LARKSPUR", "PRIMROSE", "SNOWDROP",
    "SWEET PEA", "CALENDULA", "CARNATION", "GLADIOLUS", "NARCISSUS",
    "WATER LILY", "POINSETTIA", "HONEYSUCKLE", "MORNING GLORY",
    "CHRYSANTHEMUM", "LILY OF THE VALLEY",
]
add(["birth flower", "flower"], birth_flowers)

# ============ BIRTH STONES ============
birth_stones = [
    "OPAL", "RUBY", "PEARL", "TOPAZ", "GARNET", "ZIRCON",
    "DIAMOND", "EMERALD", "PERIDOT", "AMETHYST", "SAPPHIRE", "SARDONYX",
    "MOONSTONE", "TURQUOISE", "AQUAMARINE", "BLOODSTONE", "TOURMALINE",
    "ALEXANDRITE",
]
add(["birth stone", "gemstone", "precious stone", "gem"], birth_stones)

# ============ TREE TYPES ============
tree_types = [
    "NUT", "PALM", "CITRON", "CITRUS", "FOREST", "TIMBER",
    "CONIFER", "DWARFED", "HARDWOOD", "SOFTWOOD",
    "BROAD-LEAF", "CHRISTMAS", "DECIDUOUS", "EVERGREEN", "ORNAMENTAL",
    "BONSAI", "COWAN", "FRUIT",
]
add(["tree type", "type of tree"], tree_types)

# ============ TREES (2-4 letter) ============
trees_2_4 = [
    "BO", "LI", "ASH", "BAY", "BEL", "BEN", "BOX", "ELM", "FIG",
    "FIR", "GUM", "JAK", "KOA", "MAY", "NIM", "OAK", "SAL", "TEA",
    "ULE", "YEW", "ACER", "AKEE", "AMLA", "ARAR", "BAEL", "BHEL",
    "BITO", "COCO", "COLA", "DALI", "DHAK", "DIKA", "DITA", "HOLM",
    "HULE", "ILEX", "JACK", "KINA", "KOLA", "LIME", "MAKO", "NEEM",
    "NIMB", "OMBU", "PALM", "PEAR", "PINE", "PLUM", "POON", "RATA",
    "RIMU", "SHEA", "SORB", "TAWA", "TEAK", "TITI", "TOON", "UPAS",
]

# ============ TREES (5-letter) ============
trees_5 = [
    "ABELE", "ABIES", "ACKEN", "ALAMO", "ALDER", "APPLE", "ARECA",
    "ARGAN", "ASPEN", "BAHAN", "BALSA", "BEECH", "BELAH", "BIRCH",
    "BODHI", "BUNYA", "CACAO", "CARAB", "CAROB", "CEDAR", "CEIBA",
    "CHINA", "EBONY", "ELDER", "FICUS", "GENIP", "GUAVA",
    "HAZEL", "HEVEA", "HOLLY", "IROKO", "JAMBU", "JAOUL",
    "KARRI", "KAURI", "KHAYA", "KIAAT", "KOKUM", "LARCH",
    "LEMON", "LICHI", "LILAC", "LOTUS", "MAHOE", "MAHUA",
    "MAHWA", "MAMEE", "MAMEY", "MANGO", "MAPLE", "MARRI",
    "MATAI", "MVULE", "NEEMB", "NGAIO", "NIKAU", "NYSSA",
    "OLIVE", "OSIER", "PALAS", "PALAY", "PANAX", "PAPAW",
    "PEACH", "PECAN", "PINON", "PIPAL", "PIPUL", "PLANE",
    "QUINA", "RAMIN", "RAOUL", "ROBLE", "ROWAN", "SALIX",
    "SAMAN", "THUJA", "THUYA", "TOYON", "WAHOO", "WICKY",
    "WILGA", "ZAMAN",
]

# ============ TREES (6-letter) ============
trees_6 = [
    "ACACIA", "ACAJOU", "ALMOND", "ANGICO", "ANTIAR", "AROLLA",
    "BALATA", "BAMBOO", "BANANA", "BANIAN", "BANYAN", "BAOBAB",
    "BILLAN", "BOMBAX", "BO TREE", "CASHEW", "CASSIA", "CEMBRA",
    "CERRIS", "CHENAR", "CHERRY", "CHICHA", "CORNEL", "DAMMAK",
    "DEODAR", "DURIAN", "DURION", "EMBLIC", "FEIJOA", "GINKGO",
    "GINNGO", "GUANGO", "GURJUN", "JARRAH", "JUJUBE", "KAMALA",
    "KAMELA", "KARAKA", "KARITE", "KERMES", "KOWHAI", "LAUREL",
    "LEBBEK", "LICHEE", "LINDEN", "LITCHI", "LOCUST", "LONGAN",
    "LOQUAT", "LUCUMA", "LYCHEE", "MALLEE", "MAMMEE", "MANUKA",
    "MASTIC", "MEDLAR", "MIMOSA", "MOPANE", "MORANI", "MYRTLE",
    "NUTMEG", "OBECHE", "PADAUK", "PADOUK", "PAPAYA", "PAWPAW",
    "PEEPAL", "PLATAN", "POMELO", "POPLAR", "PRUNUS", "REDBUD",
    "RED FIR", "RED GUM", "RED OAK", "SALLOW", "SAMAAN", "SANDAL",
    "SAPELE", "SAPOTA", "SAXAUL", "SHE-OAK", "SISSOO", "SOUARI",
    "SPRUCE", "TAMANU", "TINDIL", "TI TREE", "TOTARA", "TUPELO",
    "WAHROOM", "WALNUT", "WICKEN", "WILLOW", "WITGAT", "ZAMANG",
]

# ============ TREES (7+ letter) ============
trees_7plus = [
    "AILANTO", "APRICOT", "AVOCADO", "BANKSIA", "BEBEERU",
    "BIG TREE", "BILIMBI", "BLUE GUM", "BROWNIA", "BUBINGA",
    "BUCKEYE", "BULLACE", "CAMELIA", "CATALPA", "CHAMPAC",
    "CHAMPAK", "CORK OAK", "CORYLUS", "CONIFER", "CUMQUAT",
    "CYPRESS", "DADDALA", "DOGWOOD", "DURRMAST", "FIG TREE",
    "FIR TREE", "GEELBUNG", "GENIPAP", "GUM TREE", "HEMLOCK",
    "HOG-PLUM", "HOLM-OAK", "HOOP-ASH", "JAMBOOL", "JIPYAPA",
    "KUMQUAT", "LEECHEE", "LENTISK", "LIVE OAK", "LOGWOOD",
    "LUMBANG", "MADRONA", "MADRONO", "MANJACK", "MANGOSEA",
    "MASTICH", "MAY TREE", "MESQUIT", "NUT PINE", "OIL TREE",
    "PEREIRA", "PIMENTO", "PLATANE", "QUASSIA", "QUICKEN",
    "QUILLAI", "RADIATA", "RED PINE", "REDWOOD", "ROCK ELM",
    "SAKSAUL", "SASAURI", "SATSUMA", "SEQUOIA", "SERINGA",
    "SERVICE", "SHITTAH", "SNOW GUM", "SOURSOP", "SUNDARI",
    "BASSWOOD", "BAUHINIA", "BEAD TREE", "BEAN TREE", "BERGAMOT",
    "BOORTREE", "BOURTREE", "BUDDLEIA", "CALABASH", "CECROPIA",
    "CHESTNUT", "CIDER GUM", "CINCHONA", "CINNAMON", "COCO-PALM",
    "COCOPLUM", "COCO TREE", "COOK PINE", "COOLABAH", "COOLIBAH",
    "COOLEBAR", "COPROSMA", "CRAB TREE", "DATE PALM", "DHAK TREE",
    "DIVIDIVI", "DUTCH ELM", "EUONYMUS", "GARCINIA", "GARDENIA",
    "GHOST GUM", "GOLD TREE", "GUAIACUM", "HAWTHORN", "HIBISCUS",
    "HOLLY OAK", "HORNBEAM", "HUON PINE", "IRONBARK", "IRONWOOD",
    "JACK PINE", "JAMBOLAN", "JELUTONG", "KALUMPIT",
    "KINGWOOD", "LABURNUM", "LOBLOLLY", "MAGNOLIA", "MAHOGANY",
    "MAKO MAKO", "MANDARIN", "MANGROVE", "MANNA ASH",
    "MESQUITE", "MILK TREE", "MULBERRY", "NEEM TREE", "OLEASTER",
    "PALM TREE", "PANDANUS", "PEAR TREE", "PIASSABA", "PIASSAVA",
    "PINASTER", "PINE TREE", "PLANTAIN", "PODOCARP",
    "QUANDONG", "QUANTONG", "RAIN TREE", "RAMBUTAN",
    "SACK TREE", "SCOTS FIR", "SEBESTEN", "SHADDOCK", "SHADBARK",
    "SHEA TREE", "SILKY OAK", "SIMARUBA", "SOAPBARK",
    "SOAP TREE", "SOURWOOD", "SWAMP OAK", "SYCAMINE",
    "SYCAMORE", "TAMARIND", "TEAK TREE", "TUNG TREE",
    "ZIZYPHUS",
    "AILANTHUS", "ANGOPHORA", "ARAUCARIA", "ARROWWOOD",
    "BALSAM FIR", "BERG-CEDAR", "BLACK BEAN",
    "BLACK BUTT", "BLACKWOOD", "BODHI TREE", "BOLLETTIE",
    "BRAZIL NUT", "BREAD TREE", "BULLY TREE", "BUTTERNUT",
    "CARAMBOLA", "CASSARINA", "CHILE PINE", "CHINKAPIN",
    "CHINAPIN", "CIGAR TREE", "CLOVE-TREE", "COACHWOOD",
    "CONIFERUM", "COMMON ASH", "COMMON OAK", "COMMON YEW",
    "CORAL TREE", "CORDYLINE", "COURTARIL", "COWDIE-GUM",
    "CRAB APPLE", "CURRAJONG", "CURRY-LEAF", "DOORNBOOM",
    "EAGLEWOOD", "EUCRYPTIA", "FEVER TREE", "FLAME TREE",
    "FOREST-OAK", "GRAPETREE", "GREENGAOE", "HACKBERRY",
    "INDIAN FIG", "IVORY TREE", "JACARANDA", "JACKFRUIT",
    "JAMBULANA", "JUDAS TREE", "KAHIKATEA", "KAPOK TREE",
    "KAURI PINE", "KERMES OAK", "KUNRAJONG", "LANCEWOOD",
    "LEYLANDII", "MACADAMIA", "MALIWA TREE", "MELALEUCA",
    "MIRABELLE", "MOCKERNUT", "MONKEY POD",
]

all_trees = trees_2_4 + trees_5 + trees_6 + trees_7plus
add(["tree"], all_trees)

# ============ RHYMES ============
rhymes = [
    "END", "EYE", "HALF", "HEAD", "MOLE", "NEAR", "RICH", "TAIL",
    "SLANT", "VOWEL", "FEMALE", "RIDING", "TAILED", "FEMININE",
    "INTERNAL", "IDENTICAL", "MASCULINE",
    "PARARHYME", "RIME RICHE", "APOCOPATED", "CYNGHANEDD",
    "RHYME ROYAL", "RIME SUFFISANTE",
]
add(["rhyme", "rhyme type"], rhymes)

# ============ TRIANGLES ============
triangles = [
    "RIGHT", "ETERNAL", "PASCAL'S", "COCKED HAT",
    "SCALENE", "SIMILAR", "WARNING", "CONGRUENT", "ISOSCELES",
    "SPHERICAL", "ACUTE-ANGLED", "RIGHT-ANGLED",
    "EQUILATERAL", "OBTUSE-ANGLED",
]
add(["triangle", "triangle type"], triangles)

conn.commit()
print(f"\nTotal inserted: {inserted}")
print(f"Total skipped: {skipped}")

c.execute("SELECT COUNT(*) FROM definition_answers_augmented WHERE source='crossword_companion'")
print(f"Total crossword_companion entries: {c.fetchone()[0]}")
conn.close()
