"""Annotate anagram clues with explicit word roles."""
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tuition_clues.json"

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Hand-annotated anagram clues
# d=definition, f=fodder, i=indicator, l=link
ANNOTATIONS = {
    "ESTIMATE": {
        # Guess(d) teatimes(f) varied(i)
        "word_roles": ["definition", "fodder", "indicator"],
    },
    "PHEASANT": {
        # Heats(f) pan(f) to(l) cook(i) game(d) bird(d)
        "word_roles": ["fodder", "fodder", "link", "indicator", "definition", "definition"],
    },
    "GENERATE": {
        # Create(d) concoction(i) of(l) green(f) tea(f)
        "word_roles": ["definition", "indicator", "link", "fodder", "fodder"],
    },
    "BALANCE": {
        # Ban(f) lace(f) pants(i) to(l) create(l) stability(d)
        "word_roles": ["fodder", "fodder", "indicator", "link", "link", "definition"],
    },
    "AUTHORISE": {
        # Hires(f) out(f) a(f) buggy(i) producing(l) permit(d)
        "word_roles": ["fodder", "fodder", "fodder", "indicator", "link", "definition"],
    },
    "ELBOW": {
        # Joint(d) below(f) or(l) bowel(f) given(i) treatment(i)
        "word_roles": ["definition", "fodder", "link", "fodder", "indicator", "indicator"],
    },
    "USEFUL": {
        # Reconstruct(i) flue(f) with(l) us(f) That'll(l) be(l) handy(d)
        "word_roles": ["indicator", "fodder", "link", "fodder", "link", "link", "definition"],
    },
    "PROGNOSIS": {
        # Terribly(i) poor(f) signs(f) seeing(l) medical(d) forecast(d)
        "word_roles": ["indicator", "fodder", "fodder", "link", "definition", "definition"],
    },
    "OTHERWISE": {
        # Their(f) woes(f) compounded(i) if(d) not(d)
        "word_roles": ["fodder", "fodder", "indicator", "definition", "definition"],
    },
    "SOMETIMES": {
        # Crackers(i) seem(f) moist(f) on(d) occasion(d)
        "word_roles": ["indicator", "fodder", "fodder", "definition", "definition"],
    },
}

for clue in data["types"]["anagram"]["clues"]:
    ans = clue["answer"]
    if ans in ANNOTATIONS:
        clue["word_roles"] = ANNOTATIONS[ans]["word_roles"]

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Annotated anagram clues:")
for clue in data["types"]["anagram"]["clues"]:
    ans = clue["answer"]
    words = re.findall(r"[A-Za-z''-]+", clue["clue_text"])
    roles = clue.get("word_roles", [])
    if roles:
        tags = {"definition": "D", "fodder": "F", "indicator": "I", "link": "L"}
        parts = [f"{w}({tags[r]})" for w, r in zip(words, roles)]
        print(f"  {ans:12} {' '.join(parts)}")
    else:
        print(f"  {ans:12} (no annotation)")
