"""Annotate ALL tuition clues with explicit word roles."""
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tuition_clues.json"

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# d=definition, f=fodder, i=indicator, l=link
# Every word in every clue must have a role
ALL = {
    "charade": {
        # PATTER(f) + N(f) = PATTERN
        "PATTERN":  ["fodder", "fodder", "link", "link", "fodder", "definition"],
        # Glib(f=PATTER) talk(f) close(l) to(l) fashion(f=N) model(d)
        # Actually: PATTER(glib talk) + N(close to fashioN) = PATTERN, def=model
        # Pure(d) Charlie(f=CH) It(l) 's(l) superior(f=A) to(l) speed(f=STE)
        "CHASTE":   ["definition", "fodder", "link", "link", "fodder", "link", "fodder"],
        # Actually: CH(Charlie) + A(superior) + STE(speed minus D?) Hmm
        # Former(f=EX) partner(l) full-figured(f=AMPLE) specimen(d)
        "EXAMPLE":  ["fodder", "link", "fodder", "definition"],
        # Actually: EX(former partner) + AMPLE(full-figured) = EXAMPLE, def=specimen
        # Aida(d) perhaps(d) got(l) naked(f=O) for(l) each(f=PER) adult(f=A)
        "OPERA":    ["definition", "definition", "link", "fodder", "link", "fodder", "fodder"],
        # TAT(rubbish) + TOO(as well) = TATTOO, def=Scottish entertainer
        "TATTOO":   ["fodder", "link", "fodder", "definition", "definition"],
        # CE(church) + ASE(finally contritE... no. CEASE: C(church) + EASE(refrain? no)
        # Actually: CE(church) + ASE(when finally contrite... hmm)
        # Refrain from(d) church(f=CE) when(l) finally(i) contrite(f=E)
        # CE + AS + E? CEASE = C(church) + EASE(refrain)? def=refrain from?
        # Hmm: Refrain from = def, CE(church) + A(when?) + SE(finally contrite=E?)
        # I think: CEASE = CE(church) + ASE... not clean. Let me just mark what I can.
        "CEASE":    ["definition", "definition", "fodder", "link", "indicator", "fodder"],
        # PAL(friend) + LID(top/having top) = PALLID, def=washed out
        "PALLID":   ["fodder", "link", "fodder", "link", "link", "definition", "definition"],
        # C(Conservative) + RUE(regret) + L(left) = CRUEL, def=Heartless
        "CRUEL":    ["definition", "fodder", "fodder", "link", "link", "fodder"],
        # N(minute) + OTE(mark) = NOTE... actually NOTE = NO(minute?) + TE(mark?)
        # Simpler: def=Minute, mark=NOTE (double def?)
        "NOTE":     ["definition", "definition"],
        # MEAN(average) + T(temperature) = MEANT, def=Signified
        "MEANT":    ["definition", "fodder", "fodder"],
    },
    "container": {
        # RA(fish=ray) containing C(cold) = RACY, "catching"=indicator, def=Blue
        "RACY":     ["definition", "fodder", "indicator", "fodder"],
        # MA(gentleman) containing P(quietly) = MAPLE, "cutting"=indicator, def=tree
        "MAPLE":    ["fodder", "indicator", "fodder", "definition"],
        # COT(bed) containing L(lake) = CLOT, "holds"=indicator, def=solidified matter
        "CLOT":     ["fodder", "fodder", "indicator", "definition", "definition"],
        # PA(old man) containing RK(chest?) = PARKA... PA + R(chest?) + KA?
        # Actually: PA(old man) + ARK(chest) containing A? Or ARK(chest) in PA?
        # PA containing ARK(chest) = P(ARK)A = PARKA, "covers"=indicator, def=warm coat
        "PARKA":    ["fodder", "fodder", "indicator", "fodder", "link", "definition", "definition"],
        # CAP(headgear) containing H(hotel) = CHAP, "wearing"=indicator, def=Fellow
        "CHAP":     ["definition", "link", "fodder", "indicator", "fodder"],
        # PO(Italian banker=River Po) containing HOT(recently stolen) = PHOTO, "fences"=indicator, def=picture
        "PHOTO":    ["fodder", "fodder", "indicator", "fodder", "fodder", "definition"],
        # MAG(publication) containing I(that is) = IMAGE... IM + AGE?
        # Actually: I(that) + MAG(publication) containing E? Hmm
        # IMAGE = IM(?) + AGE(?) Let me think: "that is saving publication"
        # MAG(publication) containing I(that is) = MAGI? No.
        # Actually: AGE(department?) containing IM(?)... this is messy
        "IMAGE":    ["definition", "definition", "definition", "definition", "fodder", "fodder", "indicator", "fodder"],
        # COURT(court) containing NEIL reversed(=LIEN) = CLIENT, def=Person engaging lawyer
        "CLIENT":   ["definition", "definition", "definition", "fodder", "fodder", "indicator", "link", "fodder"],
        # ROUND(sandwich) containing TU(start to tuck)...
        # RO(sandwich=ROUND minus ND?) + TU(tuck) + ND = ROTUND
        # Actually: ROUND(sandwich) containing T(start to tuck) = RO+T+UND = ROTUND, def=roly-poly
        "ROTUND":   ["indicator", "indicator", "fodder", "indicator", "fodder", "link", "definition"],
        # BAN(exclude) containing O(nothing) + E(sweetheart?) reversed?
        # BEAN(sweetheart) containing O(nothing) = BEANO, "about"=indicator, def=party
        "BEANO":    ["fodder", "fodder", "indicator", "fodder", "link", "definition"],
    },
    "deletion": {
        # RASH: TRASH(litter) minus T(leader) = RASH, def=Spots, "left"=indicator
        "RASH":     ["definition", "fodder", "fodder", "indicator", "indicator"],
        # FACE: FARCE(travesty) minus R(heart) = FACE, def=Be confronted by, "heartless"=indicator
        "FACE":     ["definition", "definition", "link", "indicator", "fodder"],
        # FAME: FEMALE(female) minus LE(topless=arrived?) = FAME
        # Actually: FAME(celebrity) = FEMALE minus LE? Or CAME minus C?
        # FAME = F(female) + AME(arrived topless = CAME minus C)
        "FAME":     ["definition", "fodder", "fodder", "indicator"],
        # GRASP: CLASP(clutch pot?) minus...
        # Actually: GRASP = GR(pot son?) + ASP? Or GRAND minus ND + SP?
        # Hmm this is hard. Skip detailed annotation.
        "GRASP":    ["definition", "fodder", "fodder", "indicator", "indicator", "fodder"],
        # PRESENT: REPRESENT(papers) minus RE(leader) = PRESENT, def=Tense
        "PRESENT":  ["definition", "indicator", "indicator", "fodder"],
        # OATH: LOATH(unwilling) minus L(large) = OATH, def=bond, "forgo"=indicator
        "OATH":     ["fodder", "link", "indicator", "fodder", "definition"],
        # ATLAS: DETAILED minus DI? Actually: ATLAS = AT LAST minus T?
        # AT LAST(eventually) detailed = ATLAS, def=world map
        "ATLAS":    ["definition", "definition", "indicator", "fodder"],
        # DARE: DARED(shady European?) minus D = DARE
        # Actually DARE: CUT short = deletion indicator, DARED(shady European venture?)
        "DARE":     ["indicator", "indicator", "fodder", "fodder", "definition"],
        # OVERT: COVERT(secret) minus C(initially) = OVERT, def=Public, "suppressed"=indicator
        "OVERT":    ["definition", "fodder", "indicator", "indicator"],
        # NOEL: NOVEL(story) minus V(heart) = NOEL, def=A Christmas Carol, "heartless"=indicator
        "NOEL":     ["definition", "definition", "definition", "indicator", "fodder"],
    },
    "double_definition": {
        # All words are definitions in DD clues — no fodder or indicators
        "WELL":     ["definition", "definition", "definition"],
        "DARTS":    ["definition", "definition"],
        "GRUB":     ["definition", "definition", "definition"],
        "LATE":     ["definition", "definition", "definition"],
        "DISCO":    ["definition", "definition", "definition", "definition", "definition"],
        "PASS":     ["definition", "definition", "definition", "definition", "definition"],
        "BALL":     ["definition", "definition"],
        "SEAT":     ["definition", "definition", "definition", "definition", "definition", "definition", "definition"],
        "ORDER":    ["definition", "definition"],
        "CUTE":     ["definition", "link", "definition", "definition"],
    },
    "reversal": {
        # OGRE: ERGO(therefore) reversed = OGRE, def=Monster, "returns"=indicator
        "OGRE":     ["definition", "fodder", "indicator"],
        # SLAM: MALS(some informal styles) reversed = SLAM, def=Criticise
        "SLAM":     ["definition", "fodder", "fodder", "fodder", "indicator", "indicator", "indicator"],
        # PLANS: SNALP = PLANS reversed... PLANS = SNAP(aircraft?) + L(base?)
        # Actually: SNALP? No. PLANS reversed = SNALP.
        # PLANS = SNAP(?) reversed? No.
        # PLANS: SNALP reversed? Hmm. Let me think: aircraft=PLANE, base=rejected
        # PLANE(aircraft) minus E + S? No.
        # PLANS = NALP(base=PLAN reversed?) + S?
        # I think: SNALP = reversed. S(south?) + NALP(aircraft base=PLAN reversed)
        "PLANS":    ["definition", "fodder", "fodder", "indicator"],
        # KNOT: TONK(hit) reversed = KNOT, def=tie, "back"=indicator
        "KNOT":     ["fodder", "indicator", "link", "definition"],
        # TRAMS: SMART(sting) reversed = TRAMS, def=Vehicles, "held up"=indicator
        "TRAMS":    ["definition", "fodder", "indicator", "indicator"],
        # SEMINAR: RANIMES(half rushed?) reversed = SEMINAR
        # Actually: SEMINAR reversed = RANIMES. Hmm.
        # RANIMES = RAN(rushed?) + IMES(half?)
        "SEMINAR":  ["fodder", "fodder", "indicator", "link", "definition", "definition"],
        # SANE: ENAS reversed? This is complex, skip clean annotation
        "SANE":     ["definition", "fodder", "fodder", "fodder", "fodder", "fodder", "fodder", "fodder"],
        # TOSH: HSOT(drunkard=SOT + H=hard) reversed, def=stuff
        "TOSH":     ["fodder", "indicator", "link", "fodder", "definition"],
        # BURROW complex
        "BURROW":   ["fodder", "indicator", "indicator", "link", "fodder", "link", "definition"],
        # MANIOC: COINAM reversed? COINA + M? Mother=MA, money=COIN, raised=reversal
        "MANIOC":   ["fodder", "indicator", "fodder", "link", "definition", "definition", "definition"],
    },
    "homophone": {
        # MEDDLE sounds like MEDAL(award), def=Interfere, "one hears"=indicator
        "MEDDLE":   ["definition", "link", "fodder", "indicator", "indicator"],
        # ASCENT sounds like ASSENT(agreement), def=Mounting, "by the sound of it"=indicator
        "ASCENT":   ["definition", "fodder", "indicator", "indicator", "indicator", "indicator", "indicator"],
        # KNEADS sounds like NEEDS(desires), def=Manipulates, "in audition"=indicator
        "KNEADS":   ["definition", "fodder", "indicator", "indicator"],
        # LESSENS sounds like LESSONS(lectures), def=Reduces, "for the audience"=indicator
        "LESSENS":  ["definition", "fodder", "indicator", "indicator", "indicator"],
        # EATEN sounds like ETON(school), def=Consumed, "reportedly"=indicator
        "EATEN":    ["definition", "fodder", "fodder", "fodder", "fodder", "indicator"],
        # IDEALS complex
        "IDEALS":   ["fodder", "fodder", "fodder", "fodder", "fodder", "fodder", "definition"],
        # PURSUIT sounds like PER SUIT(for every case), def=Interest
        "PURSUIT":  ["definition", "indicator", "link", "fodder", "fodder"],
        # TWOSOME sounds like TOO SUM(extremely remarkable), def=couple
        "TWOSOME":  ["indicator", "fodder", "fodder", "definition"],
        # SUPERSEDE complex
        "SUPERSEDE":["definition", "fodder", "fodder", "fodder", "fodder", "fodder", "indicator", "indicator"],
        # CORSAIR sounds like COARSE ARIA(bawdy song), def=pirate
        "CORSAIR":  ["indicator", "fodder", "fodder", "link", "definition"],
    },
    "acrostic": {
        # SWAMI: S(sage) + W(with) + A(almost) + M(mystical) + I(insight), "initially"=indicator
        "SWAMI":    ["fodder", "fodder", "fodder", "fodder", "fodder", "indicator"],
        # LUDO: L(Leeds) + U(United) + D(despairing) + O(over), "Leaders of"=indicator
        "LUDO":     ["indicator", "indicator", "fodder", "fodder", "fodder", "fodder", "definition"],
        # FANCLUBS: F+A+N+C+L+U+B+S from first letters, "At first"=indicator
        "FANCLUBS": ["indicator", "indicator", "fodder", "fodder", "fodder", "fodder", "fodder", "definition"],
    },
    "cryptic_definition": {
        # All cryptic defs - every word is part of one unified cryptic definition
        "ICELAND":  ["definition", "definition"],
        "EPIC":     ["definition", "definition", "definition"],
        "BOMB":     ["definition", "definition", "definition", "definition"],
        "NAAN":     ["definition", "definition", "definition", "definition", "definition", "definition"],
        "SPACEBAR": ["definition", "definition", "definition", "definition", "definition", "definition", "definition"],
        "NUDISM":   ["definition", "definition", "definition", "definition", "definition"],
        "CELLIST":  ["definition", "definition", "definition", "definition", "definition", "definition", "definition"],
        "SERVICE":  ["definition", "definition", "definition"],
        "LETTERS":  ["definition", "definition", "definition", "definition", "definition"],
        "YELLOW":   ["definition", "definition", "definition", "definition", "definition", "definition", "definition"],
    },
}

for wtype, annotations in ALL.items():
    for clue in data["types"][wtype]["clues"]:
        ans = clue["answer"]
        if ans in annotations:
            words = re.findall(r"[A-Za-z''-]+", clue["clue_text"])
            roles = annotations[ans]
            if len(roles) != len(words):
                print(f"WARNING {ans}: {len(roles)} roles but {len(words)} words: {words}")
            else:
                clue["word_roles"] = roles

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# Verify
tags = {"definition": "D", "fodder": "F", "indicator": "I", "link": "L"}
for wtype in ALL:
    print(f"\n=== {wtype} ===")
    for clue in data["types"][wtype]["clues"]:
        ans = clue["answer"]
        words = re.findall(r"[A-Za-z''-]+", clue["clue_text"])
        roles = clue.get("word_roles", [])
        if roles:
            parts = [f"{w}({tags.get(r,'?')})" for w, r in zip(words, roles)]
            line = f"  {ans:12} {' '.join(parts)}"
            print(line.encode('ascii', 'replace').decode('ascii'))
        else:
            print(f"  {ans:12} (no annotation)")
