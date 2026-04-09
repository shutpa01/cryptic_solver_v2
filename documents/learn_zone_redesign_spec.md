# Learn Zone Redesign Spec

## Principles

1. **Wordplay types come first.** Reorder so the user learns wordplay concepts before being shown any tools.

2. **Simplify clue type explanations.** Drop jargon. Use Cordelia's no-nonsense voice — direct, practical, encouraging. Example tone: "Hidden words are the easiest type to spot — and a great place to start."

3. **Introduce wordplay building blocks simply.** FODDER = the word(s) that contribute letters. INDICATOR = a word that tells you what to do with the fodder. Use ultra-simple teaching examples that make the mechanism unmistakable (e.g. "Trainee acted strangely" = CADET, "Uncovers sleep over" = PEELS).

4. **10 examples max per type, all common words.** No obscure vocabulary — no TENCH, no CREEL, no NEAP. If the user doesn't know the answer word, the teaching moment is lost.

5. **Cordelia's tone is inviting, not prescriptive.** Never "learn each one" — instead "take a look, you'll already know a lot of 'em." She's a companion, not a teacher giving homework.

6. **No separate practice mode.** Cordelia transitions the user into a real puzzle: "I don't know about you but I learn better by doing, so let's have a go at a puzzle. And don't worry because you're not on your own. We'll be using a range of tools that are available for every clue on the site. They'll become your best friend, but maybe not forever..."

7. **Cordelia's First Puzzle: hand-crafted 5x5 mini crossword.** Uses real solve mode with all real tools. The puzzle IS the practice.

8. **Crossing letters are a key feature.** As clues are solved, pattern finder choices narrow. Cordelia points this out — the user sees it happen live.

9. **The puzzle is a gateway, not a course.** Cordelia makes clear this isn't comprehensive — the whole site is the learning resource. The hints, explanations, and tools are on every puzzle. Once you've got the basics, you just solve with Cordelia.

10. **Hints button always there.** Every clue on the site has progressive hints (Definition, Wordplay type, Explanation, Answer). The tutorial points this out as a safety net — you're never stuck.

---

## The Grid

5x5 word square — every answer appears both across and down, clued with different wordplay types.

```
F E A S T
E A R T H
A R G U E
S T U F F
T H E F T
```

### Across Clues (real published clues)

| # | Answer | Type | Clue | Source |
|---|--------|------|------|--------|
| 1a | FEAST | Anagram | "Twist of fate's causing blow-out" | Telegraph #28211 |
| 2a | EARTH | Anagram | "Hide a broken heart" | Guardian #24828 |
| 3a | ARGUE | Hidden | "Popular guest admitting row" | Telegraph #28087 |
| 4a | STUFF | Double def | "Pack things" | Telegraph #3282 |
| 5a | THEFT | Charade | "Stealing the newspaper" | Telegraph #30691 |

### Down Clues (same words, different mechanisms)

| # | Answer | Type | Clue | Source |
|---|--------|------|------|--------|
| 1d | FEAST | Hidden | "Provided in cafe, a stonking big meal" | Telegraph-toughie #2989 |
| 2d | EARTH | Hidden | "Discovered near the ground" | Telegraph #3262 |
| 3d | ARGUE | Anagram | "A wild urge to quarrel" | Telegraph #30180 |
| 4d | STUFF | Double def | "Material things" | Times #5016 |
| 5d | THEFT | Double def | "Taking offence" | Telegraph #28123 |

### Type coverage
- 3 anagrams (showcases anagram solver tool)
- 2 hidden words (showcases pattern finder + scanning technique)
- 3 double definitions (teaches the concept, no tool needed — just think)
- 1 charade (abbreviation + synonym, introduces the building-block concept)

### Teaching objectives
- **Two objectives per puzzle:** learning how clue types work AND showcasing the tools
- Same word clued two different ways reinforces that there's always more than one angle
- Crossing letters demonstrated live — as you solve, choices narrow
- All answers are everyday words every adult knows

---

## Tools to Showcase

The guided puzzle must introduce every tool available on the site:

1. **Click a clue word** — shows synonyms, abbreviations, whether it's an indicator (and what type), homophones
2. **Click (i) on any suggestion** — shows the word's meaning/definition
3. **Click a synonym** — puts it straight into the answer box as a guess
4. **Anagram solver** — click fodder words in the clue to add their letters; add extra letters manually if needed; auto-solves as you build
5. **Pattern finder** — crossing letters auto-populate from solved clues; narrows choices as you solve more
6. **Check button** — immediate right/wrong feedback on your guess
7. **Add to grid** — places the answer, updates crossing letters for all intersecting clues
8. **Show grid** — see your visual progress
9. **Hints button** — progressive reveal (Definition → Wordplay type → Explanation → Answer). Available on EVERY clue on the site — this is the safety net, you're never truly stuck

---

## Interactive Walkthrough Plan

The puzzle uses the real solve mode infrastructure. Cordelia provides contextual nudges at natural moments — not a rigid script, but tips that trigger based on what the user does.

### Suggested clue order for guided solving

Cordelia suggests a starting point but the user can go anywhere. The ideal teaching order:

1. **Start with 1a FEAST (anagram)** — "Twist of fate's causing blow-out"
   - Cordelia: "Let's start here. Click on the words in the clue — see what comes up."
   - User clicks "fate's" → sees it flagged as anagram fodder
   - User clicks "Twist" → sees it's an anagram indicator
   - Cordelia: "See? 'Twist' tells you to rearrange, and 'fate's' gives you the letters. Open the Tools and click those letters into the anagram solver."
   - User opens Tools → clicks fodder words → anagram solver finds FEAST
   - User clicks FEAST → goes into answer box → Check → Correct!
   - Cordelia: "Now hit Add to grid — watch what happens to the other clues."

2. **Next: 2d EARTH (hidden)** — "Discovered near the ground"
   - Now there's an E from FEAST as a crossing letter
   - Cordelia: "See that letter that's appeared? That's from the clue you just solved. Try the pattern finder — type ?E??? and hit Find."
   - Pattern finder narrows results
   - Cordelia: "Hidden words are spelled out right inside the clue. Can you spot EARTH hiding in 'nEAR THe'?"

3. **Then: 5a THEFT (charade)** — "Stealing the newspaper"
   - Cordelia: "Click on 'the' — and then 'newspaper'. See what abbreviations come up."
   - User sees THE = literal, FT = Financial Times (newspaper abbreviation)
   - Cordelia: "THE + FT = THEFT. That's a charade — pieces join together."
   - Cordelia: "And don't forget — if you're ever stuck, the Hints button is right there. It's on every clue on the whole site."

4. **Continue with remaining clues** — less hand-holding, more nudges
   - Cordelia: "You're getting the hang of this. Try the rest — use the tools, use the hints. I'm not going anywhere."

5. **On completion:**
   - Cordelia: "You've done it! Those tools and hints are on every single puzzle on the site. Pick one and have a go — I'll be right here."

### Implementation approach

- Store clues in the database as source="cordelia", puzzle_number=1
- Serve through normal puzzle route with a tutorial flag
- Cordelia tips triggered by user actions (first word click, first tool open, first correct answer, first hint reveal, completion)
- Existing CordeliaTips system can handle this with page-specific tips
