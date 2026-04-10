"""Annotate hidden word clues with explicit word roles for the learn page colour map."""
import json
import re
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tuition_clues.json"
ORIGINAL_PATH = Path(__file__).resolve().parent.parent / "data" / "tuition_clues_backup.json"

with open(DATA_PATH, encoding="utf-8") as f:
    data = json.load(f)

# Hand-annotated hidden word clues with word roles
# Each word in the clue gets: "definition", "fodder", "indicator", or "link"
HIDDEN_CLUES = [
    {
        "answer": "LYING",
        "clue_text": "Whitefly in greenhouse to some extent remaining",
        "definition": "remaining",
        "enumeration": "5",
        "word_roles": ["fodder", "fodder", "fodder", "indicator", "indicator", "indicator", "definition"],
        # Whitefly(f) in(f) greenhouse(f) to(i) some(i) extent(i) remaining(d)
    },
    {
        "answer": "RATHER",
        "clue_text": "Somewhat in clover at hers",
        "definition": "Somewhat",
        "enumeration": "6",
        "word_roles": ["definition", "indicator", "fodder", "fodder", "fodder"],
        # Somewhat(d) in(i) clover(f) at(f) hers(f)
    },
    {
        "answer": "ETHOS",
        "clue_text": "Greet host, holding spirit",
        "definition": "spirit",
        "enumeration": "5",
        "word_roles": ["fodder", "fodder", "indicator", "definition"],
        # Greet(f) host(f) holding(i) spirit(d)
    },
    {
        "answer": "PHANTOM",
        "clue_text": "Triumphant ombudsman's hiding illusion",
        "definition": "illusion",
        "enumeration": "7",
        "word_roles": ["fodder", "fodder", "indicator", "definition"],
        # Triumphant(f) ombudsman's(f) hiding(i) illusion(d)
    },
    {
        "answer": "RINGLET",
        "clue_text": "Answering letter enclosing lock of hair",
        "definition": "lock of hair",
        "enumeration": "7",
        "word_roles": ["fodder", "fodder", "indicator", "definition", "definition", "definition"],
        # Answering(f) letter(f) enclosing(i) lock(d) of(d) hair(d)
    },
    {
        "answer": "OVERSPILL",
        "clue_text": "Discovers pillagers pinching surplus",
        "definition": "surplus",
        "enumeration": "9",
        "word_roles": ["fodder", "fodder", "indicator", "definition"],
        # Discovers(f) pillagers(f) pinching(i) surplus(d)
    },
    {
        "answer": "ARDENT",
        "clue_text": "Passionate, somewhat wayward entrepreneur",
        "definition": "Passionate",
        "enumeration": "6",
        "word_roles": ["definition", "indicator", "fodder", "fodder"],
        # Passionate(d) somewhat(i) wayward(f) entrepreneur(f)
    },
    {
        "answer": "TESS",
        "clue_text": "Hardy character demonstrates stamina, to an extent",
        "definition": "Hardy character",
        "enumeration": "4",
        "word_roles": ["definition", "definition", "fodder", "fodder", "indicator", "indicator", "indicator"],
        # Hardy(d) character(d) demonstrates(f) stamina(f) to(i) an(i) extent(i)
    },
    {
        "answer": "IMPEL",
        "clue_text": "Drive forward as part of grim peloton",
        "definition": "Drive forward",
        "enumeration": "5",
        "word_roles": ["definition", "definition", "indicator", "indicator", "indicator", "fodder", "fodder"],
        # Drive(d) forward(d) as(i) part(i) of(i) grim(f) peloton(f)
    },
    {
        "answer": "CUMIN",
        "clue_text": "Type of spice and some capsicum inside",
        "definition": "Type of spice",
        "enumeration": "5",
        "word_roles": ["definition", "definition", "definition", "link", "indicator", "fodder", "fodder"],
        # Type(d) of(d) spice(d) and(link) some(i) capsicum(f) inside(f)
    },
]

# Replace the hidden clues and carry over components from originals where available
original_hidden = {c["answer"]: c for c in data["types"]["hidden"]["clues"]}

new_hidden = []
for annotated in HIDDEN_CLUES:
    ans = annotated["answer"]
    # Start from original if it exists, to keep components
    if ans in original_hidden:
        clue = dict(original_hidden[ans])
    else:
        clue = {}

    # Override with our annotations
    clue["answer"] = annotated["answer"]
    clue["clue_text"] = annotated["clue_text"]
    clue["definition"] = annotated["definition"]
    clue["enumeration"] = annotated["enumeration"]
    clue["word_roles"] = annotated["word_roles"]
    clue.setdefault("wordplay_type", "hidden")

    new_hidden.append(clue)

data["types"]["hidden"]["clues"] = new_hidden

with open(DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"Updated {len(new_hidden)} hidden word clues with word_roles")
for c in new_hidden:
    words = re.findall(r"[A-Za-z''-]+", c["clue_text"])
    roles = c["word_roles"]
    print(f"  {c['answer']:10}", end=" ")
    for w, r in zip(words, roles):
        tag = {"definition": "D", "fodder": "F", "indicator": "I", "link": "L"}[r]
        print(f"{w}({tag})", end=" ")
    print()
