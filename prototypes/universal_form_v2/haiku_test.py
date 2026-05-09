"""Quick test of Haiku as a blog reader.

Send (clue, answer, blog) → ask for role-tagged pieces → see what comes
back. No verification yet — just inspect the raw LLM output to gauge
whether Haiku has the power for this task.
"""
import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are a cryptic-crossword analyst. You will be given:
- a cryptic clue
- the known answer
- a TFTT blog explanation that decomposes the wordplay

Your job: tag every word of the clue with its role.

Roles:
  - definition: part of the definition phrase that defines the answer
  - piece: a wordplay piece that contributes letters to the answer
  - indicator: a word that signals an operation (anagram, container,
              reversal, deletion, hidden, homophone, positional)
  - link: a connective word (a, the, of, in, with, etc.) that contributes
          nothing
  - unaccounted: a word the blog doesn't account for (rare; usually a flag)

For each piece, also give:
  - value: the UPPERCASE letters the piece contributes
  - mechanism: synonym / abbreviation / literal / positional / homophone

For each indicator, also give:
  - operation: anagram / container / reversal / deletion / hidden / homophone /
              acrostic
  - subtype: optional (head/tail/outer/heart for deletion; first/last/outer/
            middle/alternate for positional)

Output ONLY a JSON object with this shape:
{
  "tags": [
    {"words": ["the", "actual", "clue", "words"], "role": "...",
     ...other fields per role above...}
  ]
}

Tags must cover every word of the clue, in order, no overlaps. Multi-word
glosses get one tag with the full word list.
"""


CLUES = [
    # 1. The clean compound — should be straightforward
    ("First thing honest outlaw admits, at great cost", "BRIGHTANDEARLY",
     "BAN (outlaw) around (admits) RIGHT (honest) + DEARLY (at great cost)"),
    # 2. Terse format — pieces with no glosses
    ("Some letters in top of shoe box", "UPPERCASE",
     "UPPER + CASE."),
    # 3. Terse compound with parens-as-insert
    ("Beer guzzling job for true believer", "APOSTLE",
     "A(POST)LE, a chestnut for sure."),
    # 4. Prose-described positional + container
    ("Bones gathered by Jack visiting exposed basin", "METATARSI",
     "MET (gathered) then TAR (Jack) inside the interior letters of bASIn"),
    # 5. Pure prose homophone (the hard one)
    ("Where trial is broadcast and heard", "CAUGHT",
     "A trial takes place in a 'court', which when broadcast sounds like our answer, "
     "which I have done rather better since getting my new hearing aids."),
]


def ask_haiku(clue, answer, blog):
    user_msg = (f"Clue: {clue}\n"
                f"Answer: {answer}\n"
                f"Blog: {blog}\n"
                "Output the JSON now.")
    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        temperature=0,
        system=SYSTEM,
        messages=[
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "{"},
        ],
    )
    raw = "{" + response.content[0].text.strip()
    return raw, response.usage.input_tokens, response.usage.output_tokens


def main():
    for clue, ans, blog in CLUES:
        print(f"=== {ans} ===")
        print(f"Clue: {clue}")
        print(f"Blog: {blog}")
        try:
            raw, ti, to = ask_haiku(clue, ans, blog)
            print(f"(tokens in={ti} out={to})")
            try:
                parsed = json.loads(raw)
                print(json.dumps(parsed, indent=2))
            except json.JSONDecodeError:
                print("(not valid JSON)")
                print(raw)
        except Exception as e:
            print(f"(error: {e!r})")
        print()


if __name__ == "__main__":
    main()
