"""Nightly automated run — the PUBLISH-FIRST chain (rewired 2026-07-10, phase 6).

Runs at 2am UTC via Windows Task Scheduler (CrypticSolver_NightlyRun).

Flow:
  1. Scrape ALL five papers (corpus). Danword answer-scraping is OFF
     (puzzle_scraper --danword is opt-in; prize answers are never scraped).
  1c. Fill today's answerless PRIZE grids with headless Claude, each grid
     checked against its published solution hash (scripts/fill_prize_grids.py),
     so steps 2-3 run on them like any other puzzle (user, 2026-10-11).
  2. WFW cascade on today's SERVING papers (telegraph/times/guardian):
     every new clue WITH an answer through the engines, zero AI calls.
  3. Headless Claude — PREFILL today's FAIL/PENDING clues into the
     hand-solver grid (prompt: scripts/prompts/nightly_prefill.md).
  3b. Headless Claude — REVIEW today's engine passes, pass->pending only
     (prompt: scripts/prompts/nightly_pass_review.md).

The post-publish DIAGNOSIS is deliberately NOT here — it is on demand from
the dashboard (scripts/run_diagnosis.py). It improves future puzzles, so
nothing in the morning waits on it, and overnight it was taking 37% of the
run's tokens (user, 2026-09-11).
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
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import ctypes
import urllib.error
import urllib.request

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


def uk_now():
    """The time in the UK (GMT, or BST from 01:00 UTC on the last Sunday of March
    to 01:00 UTC on the last Sunday of October), whatever zone the laptop is in.

    The puzzles are dated in UK days, so "today" must be the UK day. On 2026-11-01
    the laptop was on a ship in Atlantic Standard Time (UTC-4): the run fired at
    20:05 local = 00:05 UK, took the LOCAL date, re-ran 11-01 and never cascaded
    11-02. Computed by hand because the venv has no tzdata for zoneinfo."""
    utc = datetime.now(timezone.utc)

    def last_sunday_0100(month):
        last_day = datetime(utc.year, month + 1, 1, 1, tzinfo=timezone.utc) - timedelta(days=1)
        return last_day - timedelta(days=(last_day.weekday() + 1) % 7)

    in_bst = last_sunday_0100(3) <= utc < last_sunday_0100(10)
    return utc + timedelta(hours=1 if in_bst else 0)


def preflight(target_date):
    """Check the conditions the run depends on and say plainly, at the top of the
    log, if any is wrong. Never stops the run: a warning that is read beats a run
    that did not happen. Problems are prefixed '!!' so they stand out."""
    log("PRE-FLIGHT")

    # 1. Clock. The run uses the UK date; say so when the laptop's own date differs.
    local_now = datetime.now().astimezone()
    zone = local_now.tzname() or "?"
    offset = local_now.strftime("%z")
    uk = uk_now()
    if local_now.date().isoformat() != uk.date().isoformat():
        log(f"  !! Clock: laptop is on {zone} (UTC{offset}), its date is "
            f"{local_now.date()} but the UK date is {uk.date()}. Using the UK date.")
    elif offset not in ("+0000", "+0100"):
        log(f"  !! Clock: laptop is on {zone} (UTC{offset}), not UK time. Dates agree "
            f"right now ({uk.date()}), and the run always uses the UK date.")
    else:
        log(f"  Clock: OK ({zone}, UTC{offset}); UK date {uk.date()}")
    if target_date != uk.date().isoformat():
        log(f"  Target date {target_date} was given by hand (UK date is {uk.date()}).")

    # 2. Power. A laptop on battery can die mid-run (2026-10-30 it slept and the
    # run never happened). This cannot catch a run that never started; it warns
    # when this one is at risk.
    class _SPS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    try:
        sps = _SPS()
        ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sps))
        pct = "unknown" if sps.BatteryLifePercent == 255 else f"{sps.BatteryLifePercent}%"
        if sps.ACLineStatus == 1:
            log(f"  Power: OK (mains, battery {pct})")
        elif sps.ACLineStatus == 0:
            log(f"  !! Power: ON BATTERY ({pct}). Plug in, or the run may stop part-way.")
        else:
            log(f"  !! Power: mains status unknown (battery {pct})")
    except Exception as e:
        log(f"  !! Power: could not read ({e})")

    # 3. Network. Any HTTP answer, even an error page, means the site is reachable;
    # only a failure to connect counts as down.
    for name, url in [("telegraph", "https://www.telegraph.co.uk/"),
                      ("times", "https://www.thetimes.com/"),
                      ("guardian", "https://www.theguardian.com/")]:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            urllib.request.urlopen(req, timeout=20).close()
            log(f"  Network: {name} reachable")
        except urllib.error.HTTPError as e:
            log(f"  Network: {name} reachable (HTTP {e.code})")
        except Exception as e:
            log(f"  !! Network: {name} UNREACHABLE ({e}). Its puzzle will not be scraped.")
    log("")


def run_scraper():
    """Scrape today's Telegraph and Daily Mail puzzles.

    puzzle_scraper.py's --only flag takes a single source, so we invoke
    it once per source. Failure of one source doesn't stop the other.
    """
    overall_ok = True
    for source in SCRAPE_SOURCES:
        log(f"Step 1: Scraping {source}...")
        try:
            # --no-email: the scraper's own per-paper email is replaced by the ONE
            # summary this script sends at the end (send_summary, user 2026-11-06).
            result = subprocess.run(
                [PYTHON_SCRAPER, SCRAPER_SCRIPT, "--only", source, "--no-email"],
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


def run_prize_fill(target_date):
    """Step 1c: fill the day's answerless prize grids (scripts/fill_prize_grids.py).
    Never fails the nightly — a grid it cannot fill is left for the morning, as before."""
    log("Step 1c: fill answerless prize grids (Claude, checked against the solution hash)...")
    try:
        result = subprocess.run(
            [PYTHON_V2, "-m", "scripts.fill_prize_grids", "--date", target_date],
            cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=7200,
        )
    except Exception as e:
        log(f"  Prize fill ERROR: {e} — continuing")
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


def run_claude(prompt_name, label, target_date, timeout=5400):
    """Run a headless Claude Code session on a prompt file, logging its output.

    Unattended, so permissions are skipped — the prompt files carry the hard
    write-scope rules (pending-only signatures / hs assignments / logs ONLY),
    and CLAUDE.md + the memory index load automatically in this directory."""
    prompt_file = PROMPT_DIR / prompt_name
    log(f"{label}: claude -p {prompt_name} ...")
    if not prompt_file.exists():
        log(f"  Prompt file missing: {prompt_file}")
        return False
    # The prompts say "today"; left alone, Claude takes the laptop's date, which is
    # wrong whenever the laptop is outside UK time (2026-11-01, on a ship in UTC-4,
    # prefill worked on 11-01 and reported 11-02 as "tomorrow's puzzles").
    prompt = (f"TODAY'S DATE FOR THIS RUN: {target_date} (the UK date). Wherever this "
              f"prompt says \"today\" or \"today's\", it means publication_date = "
              f"'{target_date}'. Do not use the system clock's date.\n\n"
              + prompt_file.read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# ONE EMAIL A NIGHT (user, 2026-11-06): "all I want is one email at the end of
# the process that tells me in the subject whether all puzzles worked". Until
# then the scraper emailed once per paper, and the 02:00 catch-up run repeated
# all five — about ten a night. The scraper is now called with --no-email and
# this sends the only one.
#
# The SUBJECT is the verdict. "Worked" = every puzzle the schedule expects today
# is in the DB, the serving papers' clues all have answers, and every step of
# the run succeeded. Clues left for the morning walk are normal work, not a
# failure: they are counted in the body, not the subject.
#
# The 02:00 catch-up runs this same script. It sends only when the 00:05 run
# sent nothing for the date (it crashed, or never ran) or the verdict CHANGED —
# so a normal night is one email. The record is logs/nightly_email_<date>.txt.
# ---------------------------------------------------------------------------

def _summary(target_date, steps):
    """(subject, body) for the night."""
    sys.path.insert(0, str(ROOT / "scraper" / "orchestrator"))
    sys.path.insert(0, str(ROOT))
    import puzzle_scraper as ps
    from web.serving import SERVED_SOURCES

    day = date.fromisoformat(target_date)
    problems = []

    recon, found, expected = ps.reconcile(day)
    for label, status, detail in recon:
        if status != "OK":
            problems.append(f"{label} missing")

    conn = sqlite3.connect(str(CLUES_DB))
    try:
        ph = ",".join("?" for _ in SERVED_SOURCES)
        rows = conn.execute(
            "SELECT c.source, c.puzzle_number, COUNT(*), "
            "  SUM(c.answer IS NULL OR c.answer = ''), "
            "  SUM(s.status = 'pass'), SUM(s.status = 'pending'), "
            "  SUM(s.status = 'invalid'), SUM(s.status = 'continuation'), "
            "  SUM(s.status IS NULL OR s.status = 'fail') "
            "FROM clues c LEFT JOIN wfw_solve s ON s.clue_id = c.id "
            "WHERE c.publication_date = ? AND c.source IN (%s) "
            "GROUP BY c.source, c.puzzle_number ORDER BY c.source" % ph,
            (target_date, *SERVED_SOURCES)).fetchall()
    finally:
        conn.close()

    puzzle_lines = []
    for src, pnum, n, no_ans, p, pend, inv, cont, fail in rows:
        if no_ans:
            problems.append(f"{src.title()} {pnum}: {no_ans} answers missing")
        puzzle_lines.append(
            f"  {src.title():10} {pnum:>6}  {n:>2} clues | pass {p or 0}, pending "
            f"{pend or 0}, invalid {inv or 0}, continuation {cont or 0}, "
            f"not done {fail or 0}" + (f" | {no_ans} WITHOUT ANSWERS" if no_ans else ""))

    for name, ok, detail in steps:
        if not ok:
            problems.append(f"{name} failed")

    label = day.strftime("%a %d %b")
    if problems:
        subject = f"Nightly {label}: PROBLEM - " + "; ".join(problems)
        if len(subject) > 180:
            subject = subject[:177] + "..."
    else:
        subject = f"Nightly {label}: ALL {expected} PUZZLES OK"

    body = [f"Nightly run for {target_date}, finished "
            f"{uk_now().strftime('%H:%M')} UK time.", ""]
    if problems:
        body += ["PROBLEMS", "-" * 40] + [f"  {p}" for p in problems] + [""]
    body += ["EXPECTED PUZZLES", "-" * 40]
    body += [f"  [{'+' if s == 'OK' else 'X'}] {lab:28} {d}" for lab, s, d in recon]
    body += [f"  {found}/{expected} captured", "", "SERVING PUZZLES", "-" * 40]
    body += puzzle_lines or ["  none in the DB for this date"]
    body += ["", "STEPS", "-" * 40]
    body += [f"  {'OK    ' if ok else 'FAILED'} {name}" + (f" - {d}" if d else "")
             for name, ok, d in steps]
    body += ["", f"Full log: logs\\nightly_{date.today().isoformat()}.log"]
    return subject, "\n".join(body)


def send_summary(target_date, steps, dry_run=False):
    """Build and send the night's one email (see the block comment above)."""
    log("")
    log("Summary email...")
    sys.path.insert(0, str(ROOT / "scraper" / "orchestrator"))   # notify, puzzle_scraper
    try:
        subject, body = _summary(target_date, steps)
    except Exception as e:
        subject = f"Nightly {target_date}: PROBLEM - summary could not be built"
        body = f"The run finished but building the summary failed: {e}\nSee the nightly log."
    log(f"  subject: {subject}")
    if dry_run:
        log("  [DRY RUN] not sent. Body:")
        for line in body.splitlines():
            log(f"    {line}")
        return
    sent_file = LOG_DIR / f"nightly_email_{target_date}.txt"
    if sent_file.exists() and sent_file.read_text(encoding="utf-8").strip() == subject:
        log("  same verdict already emailed for this date - not sending again")
        return
    from notify import _send_email
    if _send_email(subject, body):
        sent_file.write_text(subject + "\n", encoding="utf-8")
        log("  sent")
    else:
        log("  !! NOT sent (see the [notify] line above)")


def main():
    parser = argparse.ArgumentParser(description="Nightly publish-first chain")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--skip-scraper", action="store_true", help="Skip scraper step")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip cascade step")
    parser.add_argument("--skip-claude", action="store_true", help="Skip the Claude steps")
    args = parser.parse_args()

    # The UK date, never the laptop's: see uk_now().
    target_date = args.date or uk_now().date().isoformat()
    target_day = date.fromisoformat(target_date).strftime('%A')

    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)
    # (name, ok, detail) per step, for the one summary email (send_summary).
    steps = []
    _RUN["target_date"], _RUN["steps"], _RUN["dry_run"] = target_date, steps, args.dry_run

    log("=" * 60)
    log(f"NIGHTLY RUN — {target_date} ({target_day})")
    log("=" * 60)
    preflight(target_date)

    # Step 1: Scrape all five papers (corpus). Danword is opt-in inside
    # puzzle_scraper and NOT passed here — answerless prizes wait for the user.
    if not args.skip_scraper:
        if args.dry_run:
            log("[DRY RUN] Would scrape: " + ", ".join(SCRAPE_SOURCES))
        else:
            steps.append(("Step 1 scrape", run_scraper(), ""))

    # Step 1b: ingest the Sunday Prize Toughie (idempotent; only writes on a fresh one).
    # Runs regardless of --skip-scraper so a scraped-but-un-ingested puzzle still lands;
    # gated only by --dry-run.
    if args.dry_run:
        log("[DRY RUN] Would ingest newest Prize Toughie (scripts.ingest_prize_toughie --commit)")
    else:
        run_prize_toughie_ingest()

    # Step 1c: fill today's answerless PRIZE grids with Claude, checked against the
    # puzzle's published solution hash (scripts/fill_prize_grids.py), so the cascade and
    # prefill below treat them like any other puzzle and they are ready to check at 05:00
    # (user, 2026-10-11). An answer that cannot be checked stays blank and is skipped.
    if args.skip_claude:
        log("Step 1c: fill prize grids: SKIPPED (--skip-claude)")
    elif args.dry_run:
        log(f"[DRY RUN] Would fill answerless prize grids for {target_date} (scripts.fill_prize_grids)")
    else:
        run_prize_fill(target_date)

    # Step 2: WFW cascade — today's serving-paper clues through the engines.
    if not args.skip_cascade:
        if args.dry_run:
            log(f"[DRY RUN] Would cascade {target_date} (serving papers)")
        else:
            steps.append(("Step 2 cascade", run_cascade(target_date), ""))

    # Steps 3+3b: headless Claude. Prefill (today's FAIL/PENDING clues ->
    # hand-solver readings for the morning walk), then a READ of today's engine
    # passes.
    #
    # PASS REVIEW RUNS LAST, AFTER PREFILL, deliberately. A demoted clue keeps the
    # parse the engine claimed, so the morning walk shows what was said and why it
    # was held. Run before prefill, a demotion would put the clue in prefill's
    # fail/pending work list and its reading would replace the evidence.
    #
    # POST-PUBLISH DIAGNOSIS IS NO LONGER RUN HERE (user, 2026-09-11). Measured on
    # the 09-11 run it was 8.24M of the night's 22.0M cache-read tokens — 37% — and
    # nothing before lunchtime depends on it: it improves FUTURE puzzles, while
    # prefill and pass review are what the morning actually needs. Spending the
    # overnight budget on it left too little for the early-morning work. It is now
    # on demand from the dashboard (Diagnosis page -> scripts/run_diagnosis.py),
    # which is where the publish-first process always said it belonged: "step 4,
    # post-publish, in non-critical time".
    if args.skip_claude:
        log("Steps 3+3b: Claude prefill + pass review: SKIPPED (--skip-claude)")
    elif args.dry_run:
        log("[DRY RUN] Would run claude -p nightly_prefill.md, nightly_pass_review.md")
    else:
        steps.append(("Step 3 prefill",
                      run_claude("nightly_prefill.md", "Step 3: prefill", target_date), ""))
        steps.append(("Step 3b pass review",
                      run_claude("nightly_pass_review.md",
                                 "Step 3b: review today's engine passes", target_date), ""))

    # Step 5: draft the usage phrases a double-definition SHORT needs, hours before
    # anyone is publishing. They are filed UNAPPROVED and the narrator speaks only
    # approved ones, so this cannot put words in Cordelia's mouth — it just means the
    # review is a glance at publish time instead of writing copy under pressure
    # (user, 2026-09-07). Never fatal: a missed draft costs a plainer sentence.
    log("")
    log("Step 5: draft double-definition usage phrases (for your approval)...")
    try:
        # --pending: prefilled DDs are pending at this hour; without it they were
        # invisible (2026-11-03 ROAST, FIDDLESTICKS got no phrases).
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "draft_senses.py"),
                            "--pending"],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900, cwd=str(ROOT))
        for line in (r.stdout or "").strip().splitlines()[-12:]:
            log(f"  {line}")
        steps.append(("Step 5 usage phrases", r.returncode == 0,
                      "" if r.returncode == 0 else f"exit {r.returncode}"))
        if r.returncode != 0:
            log(f"  draft_senses failed (exit {r.returncode}) — not fatal")
            # Log WHY. Without this, 2026-11-03's crash read only "exit 1".
            for line in (r.stderr or "").strip().splitlines()[-15:]:
                log(f"  stderr: {line}")
    except Exception as e:
        log(f"  draft_senses ERROR: {e} — not fatal")
        steps.append(("Step 5 usage phrases", False, str(e)[:120]))

    # Step 6: draft the clue pages' prose blocks. LAST, and with --pending, because
    # by now prefill has filed its readings and the user accepts the vast majority of
    # them unchanged (user, 2026-09-24) — so drafting from the PENDING reading means
    # the prose is already sitting beside it at 05:00, in the same glance as the tick,
    # instead of a day late. Measured on telegraph 31353: prefill filed 15 readings
    # and all 15 are exactly the clues the user went on to commit.
    #
    # Only a reading the user CHANGES needs drafting again, and that happens by
    # itself: each draft stores a fingerprint of the record it was written from, and
    # the /hs Commit button re-drafts only when that fingerprint no longer matches.
    #
    # Filed UNAPPROVED, like the senses above, so this cannot put a word on a page or
    # in Cordelia's mouth. Never fatal: a missed draft just means no prose block.
    log("")
    log("Step 6: draft clue-page prose, incl. pending prefills (for your approval)...")
    # Deliberately NOT like Step 5, which runs for real under --dry-run. A dry run
    # that spends model calls is not a dry run, and this step calls the CLI once per
    # puzzle.
    #
    # ONE PUZZLE AT A TIME, named explicitly for target_date. Two failures on
    # 2026-09-28 made the bare `--pending` call useless:
    #   - its default day is SQLite date('now'), which is UTC, so at 00:05 BST it
    #     drafted YESTERDAY's puzzles (all already drafted: "0 to draft");
    #   - a whole day in one call (84 clues) passes the prompt as a command-line
    #     argument longer than Windows allows, and nothing at all was filed.
    # One puzzle is ~14k characters, well inside the limit.
    try:
        sys.path.insert(0, str(ROOT))
        from web.serving import SERVED_SOURCES
        conn = sqlite3.connect(str(CLUES_DB))
        ph = ",".join("?" for _ in SERVED_SOURCES)
        puzzles = conn.execute(
            "SELECT DISTINCT source, puzzle_number FROM clues WHERE source IN (%s) "
            "AND publication_date = ? ORDER BY source" % ph,
            (*SERVED_SOURCES, target_date)).fetchall()
        conn.close()
    except Exception as e:
        puzzles = []
        log(f"  could not list {target_date}'s puzzles: {e} — not fatal")
    for source, pnum in puzzles:
        log(f"  --- {source} {pnum}")
        if args.dry_run:
            log(f"  [DRY RUN] Would run draft_prose.py --pending --source {source} --puzzle {pnum}")
            continue
        try:
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "draft_prose.py"), "--pending",
                 "--source", source, "--puzzle", str(pnum)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=1800, cwd=str(ROOT))
            # Generous tail: draft_prose prints one line per clue and ENDS with the
            # refusal summary, which is the only part worth reading in the morning.
            # A 14-line tail lost 11 of 12 refusals on the first real run.
            for line in (r.stdout or "").strip().splitlines()[-120:]:
                log(f"  {line}")
            steps.append((f"Step 6 prose {source} {pnum}", r.returncode == 0,
                          "" if r.returncode == 0 else f"exit {r.returncode}"))
            if r.returncode != 0:
                log(f"  draft_prose failed (exit {r.returncode}) — not fatal")
                # Log WHY. Without this, 2026-11-03's crash read only "exit 1".
                for line in (r.stderr or "").strip().splitlines()[-15:]:
                    log(f"  stderr: {line}")
        except Exception as e:
            log(f"  draft_prose ERROR: {e} — not fatal")
            steps.append((f"Step 6 prose {source} {pnum}", False, str(e)[:120]))

    send_summary(target_date, steps, args.dry_run)

    log("")
    log("=" * 60)
    log("NIGHTLY RUN COMPLETE")
    log("=" * 60)


# What main() has done so far, so a crash can still send the night's email.
_RUN = {}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        for line in traceback.format_exc().splitlines():
            log(f"  {line}")
        if _RUN.get("target_date"):
            _RUN["steps"].append(("Run CRASHED", False, str(e)[:150]))
            try:
                send_summary(_RUN["target_date"], _RUN["steps"], _RUN["dry_run"])
            except Exception as e2:
                log(f"  summary email also failed: {e2}")
        sys.exit(1)
