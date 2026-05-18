# Telegraph 31242 Clean Current Pipeline Baseline

Date: 2026-05-18

This baseline was created after:

1. snapshotting the existing Telegraph 31242 report, gaps, DB rows, and full `clues_master.db`;
2. deleting only Telegraph 31242 puzzle-scoped rows from `clues_master.db`;
3. rescraping Telegraph 31242 from the Telegraph daily scraper;
4. rerunning the current live pipeline with the pipeline interpreter.

## Commands

Rescrape:

```powershell
.\.venv\Scripts\python.exe scraper\telegraph\telegraph_daily.py
```

Pipeline:

```powershell
& 'C:\Users\shute\PycharmProjects\AI_Solver\.venv\Scripts\python.exe' -m sonnet_pipeline.run 31242 --source telegraph --write-db --mode 1 --no-review --output-dir documents\baselines\telegraph_31242\2026-05-18_082443_clean_current_pipeline_baseline
```

## Result

Current reproducible baseline:

```text
Total clues: 30
Assembled: 23/30
High: 23/23 assembled
Failed: 7/30
Cost: $0.0000
```

Pipeline phase evidence from stdout:

```text
Phase 0 hidden/double-definition: 3 solves
Phase 0.5 mechanical solvers: 12 solves
Phase 1 signature solver: 8 HIGH, 0 medium
Phase 2 production solver: 7 clues left unsolved
```

Saved artifacts:

- `puzzle_report_telegraph_31242.txt`
- `pending_gaps_telegraph_31242.json`
- `post_run_db_counts.json`

## Failed Clues

```text
1A FORAGE
10A KNEADING
14A OWE
16D ESTIMATES
17D GAME SHOWS
7D TRIPOLI
9A RASCAL
```

## Important Interpretation

This is the baseline to compare future GT2 code against from this point forward.

It is not the same as the historical reviewed/reverified `2/30` observation. That earlier state is no longer exactly reproducible because review activity changed reference data and/or pending enrichment state.

Also, `23/30 high` is a parse-rate baseline, not a guarantee that every high explanation is clean. For example, `8D ELGAR` still exposes a representation problem: the report currently shows `Article`, `in`, and `Madrid` each producing `EL`, rather than preserving the real phrase span `Article in Madrid -> EL`.

So the GT2 target is not simply "increase 23". It is:

1. keep or improve the 23/30 parse rate;
2. recover some of the 7 remaining failures, starting with `FORAGE`;
3. improve explanation/evidence quality so cases like `ELGAR` and `IMAM` are represented as graph atoms rather than lossy enrichment rows.
