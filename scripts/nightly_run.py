"""Nightly automated run — the PUBLISH-FIRST chain (rewired 2026-07-10, phase 6).

Runs at 2am UTC via Windows Task Scheduler (CrypticSolver_NightlyRun).

Flow:
  1. Scrape ALL five papers (corpus). Danword answer-scraping is OFF
     (puzzle_scraper --danword is opt-in; prize puzzles are hand-solved
     by the user in the morning, never scraped).
  2. WFW cascade on today's SERVING papers (telegraph/times/guardian):
     every new clue WITH an answer through the engines, zero AI calls.
  3. Headless Claude — post-publish DIAGNOSIS of the frozen manual solves
     the user committed yesterday (pending-only signatures + engine
     worklist; prompt: scripts/prompts/nightly_diagnosis.md).
  4. Headless Claude — PREFILL today's FAIL/PENDING clues into the
     hand-solver grid (prompt: scripts/prompts/nightly_prefill.md).
Morning: the user walks each puzzle in /solver/hs (Commit/Uncommit only)
and publishes. Nothing waits for blogs; nothing is invoked by hand.

The old Sonnet pipeline / TFTT / mash-up steps are gone (2026-07-10) —
see git history for the pre-rewire version.

Usage:
    python scripts/nightly_run.py                 # full run
    python scripts/nightly_run.py --dry-run       # show plan, don't execute
    python scripts/nightly_run.py --skip-scraper  # skip scraper step
    python scripts/nightly_run.py --skip-claude   # mechanical steps only
"""

import argparse
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
PYTHON_V2 = str(ROOT / ".venv" / "Scripts" / "python.exe")
PYTHON_SCRAPER = PYTHON_V2
SCRAPER_SCRIPT = str(ROOT / "scraper" / "orchestrator" / "puzzle_scraper.py")
CASCADE_SCRIPT = str(ROOT / "scripts" / "nightly_cascade.py")
CLAUDE_EXE = r"C:\Users\shute\.local\bin\claude.exe"
# Pin the headless model: the CLI otherwise inherits the user's saved default,
# which changes whenever they /model in an interactive session (2026-07-14).
CLAUDE_MODEL = "claude-fable-5"
PROMPT_DIR = ROOT / "scripts" / "prompts"
LOG_DIR = ROOT / "logs"

# Scrape every night: all five (the corpus). Cascade + prefill: SERVING papers
# only (the publish-first stock) — see scripts/nightly_cascade.py.
SCRAPE_SOURCES = ["telegraph", "dailymail", "times", "guardian", "independent"]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    # Replace chars that Windows cp1252 can't encode
    safe_msg = str(msg).encode('cp1252', errors='replace').decode('cp1252')
    print(f"[{ts}] {safe_msg}", flush=True)


def run_scraper():
    """Scrape today's Telegraph and Daily Mail puzzles.

    puzzle_scraper.py's --only flag takes a single source, so we invoke
    it once per source. Failure of one source doesn't stop the other.
    """
    overall_ok = True
    for source in SCRAPE_SOURCES:
        log(f"Step 1: Scraping {source}...")
        try:
            result = subprocess.run(
                [PYTHON_SCRAPER, SCRAPER_SCRIPT, "--only", source],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
        except subprocess.TimeoutExpired:
            log(f"  {source} scraper TIMEOUT (30 min) — continuing")
            overall_ok = False
            continue
        except Exception as e:
            log(f"  {source} scraper ERROR: {e} — continuing")
            overall_ok = False
            continue

        if result.returncode != 0:
            log(f"  {source} scraper failed (exit {result.returncode})")
            if result.stderr:
                log(f"  stderr: {result.stderr[-500:]}")
            overall_ok = False
            continue

        # Show summary lines from output
        for line in (result.stdout or "").splitlines():
            if any(k in line.lower() for k in
                   ("clues", "saved", "skip", "indexing", "submitted", "err ")):
                log(f"  {line.strip()}")
        log(f"  {source} scraper completed")
    return overall_ok


def run_prize_toughie_ingest():
    """Ingest the newest scraped Telegraph Prize Toughie into the `clues` table.

    The main scraper downloads the Sunday Prize Toughie JSON, but its promote-to-`clues`
    step is broken for embargoed prizes (it demands an `explanation` that is never scraped),
    so the puzzle was silently left un-ingested until run by hand. This wires the standalone
    ingest (scripts/ingest_prize_toughie.py --commit) into the nightly. It is IDEMPOTENT —
    "ALREADY PRESENT, nothing to do" and exit 0 when the newest puzzle is already in the DB —
    so it is safe every night and only writes when a fresh Sunday puzzle has appeared.
    Never fails the nightly."""
    log("Step 1b: Ingest Telegraph Prize Toughie (newest JSON, --commit)...")
    try:
        result = subprocess.run(
            [PYTHON_V2, "-m", "scripts.ingest_prize_toughie", "--commit"],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
    except Exception as e:
        log(f"  Prize Toughie ingest ERROR: {e} — continuing")
        return
    for line in (result.stdout or "").strip().splitlines():
        log(f"  {line}")
    if result.returncode != 0 and result.stderr:
        log(f"  stderr: {result.stderr[-300:]}")


def run_cascade(target_date):
    """Run the WFW cascade on today's serving-paper clues (nightly_cascade.py)."""
    log("Step 2: WFW cascade (serving papers, answerless clues skipped)...")
    try:
        result = subprocess.run(
            [PYTHON_V2, CASCADE_SCRIPT, "--date", target_date],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        log("  Cascade TIMEOUT (60 min)")
        return False
    except Exception as e:
        log(f"  Cascade ERROR: {e}")
        return False
    for line in (result.stdout or "").splitlines():
        log(f"  {line.strip()}")
    if result.returncode != 0:
        log(f"  Cascade failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def run_claude(prompt_name, label, timeout=5400):
    """Run a headless Claude Code session on a prompt file, logging its output.

    Unattended, so permissions are skipped — the prompt files carry the hard
    write-scope rules (pending-only signatures / hs assignments / logs ONLY),
    and CLAUDE.md + the memory index load automatically in this directory."""
    prompt_file = PROMPT_DIR / prompt_name
    log(f"{label}: claude -p {prompt_name} ...")
    if not prompt_file.exists():
        log(f"  Prompt file missing: {prompt_file}")
        return False
    prompt = prompt_file.read_text(encoding="utf-8")
    # Strip any inherited ANTHROPIC_API_KEY so claude bills the subscription,
    # never prepaid API credits (see run_prefill.py, 2026-07-13). The task
    # scheduler env is clean today; this guards against future launch contexts.
    claude_env = os.environ.copy()
    claude_env.pop("ANTHROPIC_API_KEY", None)
    try:
        result = subprocess.run(
            [CLAUDE_EXE, "-p", prompt, "--model", CLAUDE_MODEL,
             "--dangerously-skip-permissions"],
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=claude_env,
        )
    except subprocess.TimeoutExpired:
        log(f"  {label} TIMEOUT ({timeout}s)")
        return False
    except Exception as e:
        log(f"  {label} ERROR: {e}")
        return False
    tail = (result.stdout or "").strip().splitlines()[-25:]
    for line in tail:
        log(f"  {line}")
    if result.returncode != 0:
        log(f"  {label} failed (exit {result.returncode})")
        if result.stderr:
            log(f"  stderr: {result.stderr[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Nightly publish-first chain")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-scraper", action="store_true", help="Skip scraper step")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip cascade step")
    parser.add_argument("--skip-claude", action="store_true", help="Skip the Claude steps")
    args = parser.parse_args()

    target_date = args.date or date.today().isoformat()
    target_day = date.fromisoformat(target_date).strftime('%A')

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)

    log("=" * 60)
    log(f"NIGHTLY RUN — {target_date} ({target_day})")
    log("=" * 60)

    # Step 1: Scrape all five papers (corpus). Danword is opt-in inside
    # puzzle_scraper and NOT passed here — answerless prizes wait for the user.
    if not args.skip_scraper:
        if args.dry_run:
            log("[DRY RUN] Would scrape: " + ", ".join(SCRAPE_SOURCES))
        else:
            run_scraper()

    # Step 1b: ingest the Sunday Prize Toughie (idempotent; only writes on a fresh one).
    # Runs regardless of --skip-scraper so a scraped-but-un-ingested puzzle still lands;
    # gated only by --dry-run.
    if args.dry_run:
        log("[DRY RUN] Would ingest newest Prize Toughie (scripts.ingest_prize_toughie --commit)")
    else:
        run_prize_toughie_ingest()

    # Step 2: WFW cascade — today's serving-paper clues through the engines.
    if not args.skip_cascade:
        if args.dry_run:
            log(f"[DRY RUN] Would cascade {target_date} (serving papers)")
        else:
            run_cascade(target_date)

    # Steps 3+4: headless Claude. Diagnosis FIRST (yesterday's committed manual
    # solves -> pending-only signatures + engine worklist), then prefill
    # (today's FAIL/PENDING clues -> hand-solver readings for the morning walk).
    if args.skip_claude:
        log("Steps 3+4: Claude diagnosis + prefill: SKIPPED (--skip-claude)")
    elif args.dry_run:
        log("[DRY RUN] Would run claude -p nightly_diagnosis.md, then nightly_prefill.md")
    else:
        run_claude("nightly_diagnosis.md", "Step 3: post-publish diagnosis")
        run_claude("nightly_prefill.md", "Step 4: prefill")

    log("")
    log("=" * 60)
    log("NIGHTLY RUN COMPLETE")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(f"  {line}")
        sys.exit(1)
