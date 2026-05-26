# Phase 1 Regression Harness — Codex Instruction

Task: Write a standalone script that runs the pipeline on a fixed set of
test clues and prints a result table.

File to create: documents/regression_harness.py

Do not modify any other file.

This is a Phase 1 no-write behavioural harness. It does not test
persistence or wfw_proof rows. Those are Phase 2.


---

## Entry point

Call run_signature_clue_pipeline from sonnet_pipeline/clue_pipeline.py.
Do not call run_clue_pipeline. Do not call solve_clue directly.

run_signature_clue_pipeline calls solve_clue internally, which calls
_attach_gt2_evidence, which is where Stage Two and Stage Three are
populated. That is the path Phase 1 changed and the path to test.

Always pass write_db=False. The harness must not modify the database.


---

## DB connections

Open these once in the main thread before the test loop.

Import sqlite3 and RefDB from signature_solver.db.

CLUES_DB path:
    C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db

Open a sqlite3 connection to CLUES_DB with timeout=30.
Instantiate RefDB with no arguments.

Note: with write_db=False, run_signature_clue_pipeline does not write
to the sqlite connection. However, because SQLite connections are not
safe to use across threads, do not pass the sqlite connection into
worker threads. Instead, look up all clue data in the main thread
before spawning any threads, so the worker only receives plain values
(clue_id, source, puzzle_number, clue_text, answer) plus ref_db.
Pass a fresh sqlite3 connection opened inside the worker thread if the
pipeline requires one, or pass None and let write_db=False ensure it
is never used.


---

## Test clues

Schema note: the column name is clue_text, not clue. Use clue_text
in all queries.

For each named answer, look up one row using:
    SELECT id, clue_text, source, puzzle_number
    FROM clues
    WHERE answer = ?
    ORDER BY id
    LIMIT 1

Named test clues:

    SLAVISHLY    expected=PASS    mechanism=charade
    STIPULATION  expected=PASS    mechanism=anagram
    AINTREE      expected=PASS    mechanism=wfw_review
    TIJUANA      expected=PASS    mechanism=phrase_sources
    TOTALLY      expected=PASS    mechanism=charade
    ROC          expected=PASS    mechanism=deletion
    HASBEEN      expected=PASS    mechanism=compound

Additional test clues — pick the first match found for each:

    hidden word:
        SELECT id, clue_text, answer, source, puzzle_number
        FROM clues WHERE wordplay_type = 'hidden'
        ORDER BY id LIMIT 1

    double definition:
        SELECT id, clue_text, answer, source, puzzle_number
        FROM clues WHERE wordplay_type = 'double_definition'
        ORDER BY id LIMIT 1

    homophone:
        SELECT id, clue_text, answer, source, puzzle_number
        FROM clues WHERE wordplay_type = 'homophone'
        ORDER BY id LIMIT 1

For the three additional clues the expected result is NO_HANG only.
The signature solver does not solve hidden, DD, or homophone clues.
The only thing being verified is that the pipeline completes without
hanging.

Do all DB lookups in the main thread before the test loop starts.
Store the results as a list of plain dicts so no DB access is needed
inside worker threads.


---

## Calling the pipeline

For each test clue, call run_signature_clue_pipeline with:
    conn, clue_id, source, puzzle_number, clue_text, answer, ref_db
and keyword argument write_db=False.

Where conn is concerned: open a fresh sqlite3 connection to CLUES_DB
inside the worker thread rather than passing the main thread connection.
This avoids SQLite cross-thread errors even though write_db=False means
the connection should never actually be written to.

After the call, read solve_result from the returned CluePipelineResult.


---

## Timeout handling

Use a daemon thread and a queue for each pipeline call.

The pattern is:
- Create a queue
- Start a daemon thread that calls the pipeline and puts the result
  in the queue
- In the main thread, call queue.get(timeout=30)
- If queue.get raises queue.Empty, the call has timed out — record
  timed_out=True and move on
- Because the thread is a daemon, it will not prevent the script from
  exiting even if it is still running

Do not use ThreadPoolExecutor. Its shutdown waits for all threads to
finish, which defeats the timeout.


---

## What to record

After each call record:

    answer        the answer string
    mechanism     from the test set definition above
    high_conf     True if solve_result is not None and confidence >= 80
    s2_path       see detection rules below
    s3_status     solve_result.stage_three_proof.status if present, else "none"
    runtime       wall clock seconds
    timed_out     True if runtime exceeded 30 seconds

s2_path detection rules:
    If solve_result is None: "none"
    If stage_two_casefile is absent or None: "none"
    If stage_two_casefile.status == "answer_fit" and high_conf is True:
        "from_solve_result"
    If high_conf is False and stage_two_casefile is present:
        "grammar_only"
    Otherwise: "unknown"


---

## Pass criteria

For named clues (the seven answers above):

    PASS if high_conf=True and s2_path=from_solve_result
            and s3_status != none and timed_out=False
    WARN if the PASS conditions are met and s3_status=REVIEW
         (REVIEW is expected in Phase 1 — the definition check will
         not pass for legacy_solver boundary_status and that is correct)
    FAIL if high_conf=False or s2_path != from_solve_result
            or timed_out=True

For additional clues (hidden, double_definition, homophone):

    PASS if timed_out=False
    FAIL if timed_out=True


---

## Output format

Print a plain text table with one row per clue and columns:
    ANSWER, MECHANISM, CONF, S2_PATH, S3_STATUS, TIME, RESULT

After the table print:
    X/Y passed (Z warnings, W failures)

where Y is total clues, X is passed or warned, Z is warned, W is failed.


---

## After writing the script

Do not run it. Paste the complete script for Claude to audit before
it is run.
