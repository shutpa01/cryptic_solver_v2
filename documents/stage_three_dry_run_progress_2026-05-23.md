# Stage Three Dry-Run Progress - 2026-05-23

## Position

Stage Three remains read-only. It now produces:

- a strict PASS/REVIEW proof;
- queue-shaped human/admin review items;
- concrete pending-enrichment rows for DB facts only;
- a whole-puzzle dry-run report.

No Stage Three code writes to the database.

## Runner

Dry-run command:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_three_puzzle.py --source dailymail --puzzle-number 17883
```

Action-detail command:

```powershell
.\.venv\Scripts\python.exe scripts\run_stage_three_puzzle.py --source dailymail --puzzle-number 17883 --actions
```

The runner groups clues into:

- `mechanical_review`: letter derivation / source / definition / span proof is not yet sound;
- `db_enrichment_review`: mechanical proof is sound, but a concrete DB fact is missing;
- `purpose_review`: mechanical proof is sound, but word-purpose / grammar evidence is still only candidate or unresolved;
- `final_pass`: all Stage Three checks pass.

## Daily Mail 17883 Snapshot

Latest dry run after queue cleanup:

- Final PASS: 0/32
- Mechanical proof pass: 7/32
- Mechanical proof review: 25/32
- Purpose-only next action: 6/32
- DB-enrichment next action: 1/32
- Review items: 38
- Pending DB enrichments: 4

Remaining pending DB enrichments:

- 11A ROC: `Large reptile -> CROC`
- 13A HASBEEN: `One no longer relevant -> HASBEEN`
- 13A HASBEEN: `north London suburb -> HENDON`
- 24A VITAMIN: `French friend -> AMI`

## Quality Tightening Done

False or weak phrase-widening enrichments are now blocked when:

- the added phrase word is an operation/link indicator;
- the added phrase word is another answer-source candidate;
- the output value is only one or two letters;
- the added phrase word is adverbial.

This removed bad DB candidates such as:

- `disturbed bird -> ...`
- `Quiet trip -> OUTING`
- `European struggles -> VIE/VIES`
- `first point -> NE/PT`
- `college sink -> GO`
- `carefully symbol -> ICON`
- `overly solemn ring -> DIAL`
- `This boy dropping book -> TOYISH`

Definition-gap enrichments now require a substantial first working pair.

Stage Three also now distinguishes a candidate mechanism word from a candidate
definition separator when the word sits exactly on the definition/wordplay
boundary. This changed cases such as:

- 18D LEVIES: `giving` is now a definition-separator evidence request;
- 28A NARROWS: `in` is now a definition-separator evidence request.

These still block PASS. They are not silently accepted.

## Tests Run

Passing:

- `signature_solver/test_stage_two_casefile.py`
- `signature_solver/test_stage_three_proof.py`
- `signature_solver/test_stage_three_review_queue.py`
- `signature_solver/test_wfw_display_adapter.py`
- `scripts/test_stage_three_puzzle_runner.py`
- `web/test_clue_wfw_render_contract.py`
- `python -m compileall -q signature_solver scripts web`

## Next Sensible Step

Do not add writes yet.

Next, inspect the six `purpose_review` clues in Daily Mail 17883 and decide which purpose categories can be promoted by grammar rules, if any. The key danger is making words like `with`, `to`, `when`, `taken`, and `on` pass just because they look common. They should only pass when their grammatical/cryptic job is demonstrable.
