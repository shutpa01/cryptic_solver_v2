"""Find valid 5x5 crossword grids using only common everyday words."""
import sqlite3
from collections import defaultdict

conn = sqlite3.connect("data/clues_master.db")
cur = conn.cursor()
cur.execute("""
    SELECT UPPER(answer), COUNT(*) as cnt
    FROM clues
    WHERE LENGTH(answer) = 5
      AND answer GLOB '[A-Za-z][A-Za-z][A-Za-z][A-Za-z][A-Za-z]'
    GROUP BY UPPER(answer)
    HAVING cnt >= 2
""")
db_words = set(r[0] for r in cur.fetchall())
conn.close()

# Only common everyday words that any adult would know
everyday = {
    "ABOUT","ABOVE","ACTOR","ADAPT","ADMIT","ADOPT","AFTER","AGAIN","AGREE",
    "ALARM","ALERT","ALIVE","ALLOW","ALONE","ALONG","ALTER","AMAZE","AMPLE",
    "ANGEL","ANGER","ANGLE","ANGRY","APART","APPLE","APPLY","ARENA","ARGUE",
    "ARISE","ASSET","AVOID","AWAKE","AWARD",
    "BADGE","BASIC","BEGIN","BELOW","BENCH","BLACK","BLADE","BLAME","BLAND",
    "BLANK","BLAST","BLAZE","BLEED","BLEND","BLESS","BLIND","BLOCK","BLOOD",
    "BLOOM","BLOWN","BOARD","BOOST","BOUND","BRAIN","BRAND","BRAVE","BREAD",
    "BREAK","BREED","BRIEF","BRING","BROAD","BROKE","BRUSH","BUILD","BUNCH",
    "BURST","BUYER",
    "CABLE","CANDY","CARRY","CATCH","CAUSE","CHAIN","CHAIR","CHALK","CHEAP",
    "CHECK","CHEER","CHEST","CHIEF","CHILD","CHINA","CLAIM","CLASS","CLEAN",
    "CLEAR","CLERK","CLIMB","CLING","CLOCK","CLOSE","CLOTH","CLOUD","COACH",
    "COAST","COUNT","COURT","COVER","CRACK","CRAFT","CRANE","CRASH","CRAZY",
    "CREAM","CRIME","CROSS","CROWD","CRUDE","CRUSH","CURVE",
    "DAILY","DANCE","DEATH","DECAY","DELAY","DENSE","DEPTH","DEVIL","DIRTY",
    "DOUBT","DOUGH","DRAFT","DRAIN","DRAMA","DRAWN","DREAM","DRESS","DRIED",
    "DRIFT","DRILL","DRINK","DRIVE","DROWN","DYING",
    "EAGER","EARLY","EARTH","EIGHT","ELDER","ELECT","EMPTY","ENEMY","ENJOY",
    "ENTER","ENTRY","EQUAL","ERROR","EVENT","EVERY","EXACT","EXTRA",
    "FAINT","FAITH","FALSE","FANCY","FAULT","FEAST","FENCE","FIELD","FIFTY",
    "FIGHT","FINAL","FLAME","FLASH","FLESH","FLOAT","FLOOD","FLOOR","FLOUR",
    "FLUID","FLUSH","FORCE","FORGE","FORTY","FOUND","FRAME","FRANK","FRAUD",
    "FRESH","FRONT","FROST","FRUIT","FULLY",
    "GHOST","GIANT","GIVEN","GLARE","GLASS","GLOBE","GLORY","GOOSE","GRACE",
    "GRADE","GRAIN","GRAND","GRANT","GRAPE","GRASP","GRASS","GRAVE","GREAT",
    "GREEN","GREET","GRIEF","GRILL","GRIND","GROSS","GROUP","GROWN","GUARD",
    "GUESS","GUEST","GUIDE","GUILT",
    "HAPPY","HARSH","HEART","HEAVY","HORSE","HOTEL","HOUSE","HUMAN","HURRY",
    "IDEAL","IMAGE","IMPLY","INDEX","INNER","INPUT","ISSUE",
    "JOINT","JUDGE","JUICE",
    "KNIFE","KNOCK",
    "LABEL","LARGE","LASER","LATER","LAUGH","LAYER","LEARN","LEASE","LEAST",
    "LEAVE","LEVEL","LIGHT","LIMIT","LINEN","LIVER","LOCAL","LOGIC","LOOSE",
    "LOVER","LOWER","LUCKY","LUNAR","LUNCH",
    "MAGIC","MAJOR","MAKER","MARCH","MATCH","MAYOR","MEDAL","MEDIA","MERCY",
    "METAL","METER","MIGHT","MINOR","MINUS","MODEL","MONEY","MONTH","MORAL",
    "MOUNT","MOUSE","MOUTH","MOVIE","MUSIC",
    "NERVE","NEVER","NIGHT","NOBLE","NOISE","NORTH","NOTED","NOVEL","NURSE",
    "OCEAN","OFFER","OFTEN","OLIVE","ORDER","OTHER","OUTER","OWNER",
    "PAINT","PANEL","PANIC","PAPER","PARTY","PATCH","PAUSE","PEACE","PEARL",
    "PENNY","PHASE","PHONE","PIANO","PIECE","PILOT","PITCH","PLACE","PLAIN",
    "PLANE","PLANT","PLATE","PLEAD","PLUMB","POINT","POLAR","POUND","POWER",
    "PRESS","PRICE","PRIDE","PRIME","PRINT","PRIOR","PRIZE","PROOF","PROUD",
    "PROVE","PUNCH","PUPIL",
    "QUEEN","QUERY","QUEUE","QUIET","QUOTE",
    "RADAR","RADIO","RAISE","RALLY","RANCH","RANGE","RAPID","RATIO","REACH",
    "REACT","READY","REALM","REBEL","REIGN","RELAX","REPLY","RIDER","RIDGE",
    "RIGHT","RIGID","RIVAL","RIVER","ROBOT","ROUGH","ROUND","ROUTE","ROYAL",
    "RULER","RURAL",
    "SAINT","SALAD","SCALE","SCARE","SCENE","SCOPE","SCORE","SCOUT","SERVE",
    "SEVEN","SHADE","SHALL","SHAME","SHAPE","SHARE","SHARK","SHARP","SHEER",
    "SHEET","SHELF","SHELL","SHIFT","SHIRT","SHOCK","SHOOT","SHORT","SHOUT",
    "SIGHT","SINCE","SIXTY","SKIRT","SLATE","SLEEP","SLICE","SLIDE","SLOPE",
    "SMALL","SMART","SMELL","SMILE","SMOKE","SOLVE","SOUTH","SPACE","SPARE",
    "SPEAK","SPEED","SPELL","SPEND","SPENT","SPICE","SPLIT","SPOKE","SPORT",
    "SPRAY","SQUAD","STAFF","STAGE","STAIN","STAIR","STAKE","STALE","STALL",
    "STAMP","STAND","STARE","START","STATE","STEAK","STEAL","STEAM","STEEL",
    "STEEP","STEER","STICK","STILL","STOCK","STONE","STORE","STORM","STORY",
    "STOVE","STRIP","STUCK","STUFF","STYLE","SUGAR","SUITE","SUPER","SURGE",
    "SWEAR","SWEEP","SWEET","SWIFT","SWING","SWORD",
    "TABLE","TASTE","TEACH","TEETH","THANK","THEFT","THEME","THICK","THIEF",
    "THING","THINK","THIRD","THOSE","THREE","THROW","THUMB","TIGER","TIGHT",
    "TITLE","TOKEN","TOTAL","TOUCH","TOUGH","TOWER","TOXIC","TRACE","TRACK",
    "TRADE","TRAIL","TRAIN","TRAIT","TRASH","TREAT","TREND","TRIAL","TRIBE",
    "TRICK","TRIED","TROOP","TRUCK","TRULY","TRUNK","TRUST","TRUTH","TUTOR",
    "TWICE",
    "ULTRA","UNCLE","UNDER","UNION","UNITY","UNTIL","UPPER","UPSET","URBAN",
    "USAGE","USUAL","UTTER",
    "VAGUE","VALID","VALUE","VERSE","VIDEO","VIRUS","VISIT","VITAL","VIVID",
    "VOCAL","VOICE","VOTER",
    "WAGES","WASTE","WATCH","WATER","WEARY","WEIGH","WEIRD","WHEAT","WHEEL",
    "WHERE","WHICH","WHILE","WHITE","WHOLE","WHOSE","WIDER","WOMAN","WORLD",
    "WORSE","WORST","WORTH","WOULD","WOUND","WRITE","WRONG","WROTE",
    "YACHT","YIELD","YOUNG","YOUTH",
}

words = sorted(everyday & db_words)
word_set = set(words)
print(f"Common everyday words in DB: {len(words)}")

# Build prefix -> possible next letters
prefix_next = defaultdict(set)
for w in words:
    for i in range(5):
        prefix_next[w[:i]].add(w[i])

results_found = []

def search(rows, depth):
    if depth == 5:
        cols = ["".join(rows[r][c] for r in range(5)) for c in range(5)]
        if all(c in word_set for c in cols):
            results_found.append((list(rows), cols))
            return len(results_found) >= 200
        return False

    col_pfx = ["".join(rows[r][c] for r in range(depth)) for c in range(5)]
    possible = [prefix_next.get(col_pfx[c], set()) for c in range(5)]
    if any(not p for p in possible):
        return False

    for w in words:
        if all(w[c] in possible[c] for c in range(5)):
            rows.append(w)
            if search(rows, depth + 1):
                rows.pop()
                return True
            rows.pop()
    return False

# Search starting from every word
for start in words:
    search([start], 1)

print(f"\nFound {len(results_found)} grids\n")
for rows, cols in results_found[:30]:
    print(f"  {' | '.join(rows)}  //  {' | '.join(cols)}")
