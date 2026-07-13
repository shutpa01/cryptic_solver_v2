"""On-demand prefill for ONE puzzle (or one date) — the prize-morning chain.

Prize puzzles get their answers when the USER solves the grid, so waiting for
the next nightly means publishing explanations a day late (user, 2026-07-12).
This script runs the same chain as the nightly, scoped and on demand:

  1. cascade any un-cascaded clues (scripts/nightly_cascade.py — idempotent,
     same guards: answer required, no existing parse, never a frozen manual
     solve), so the prefill has a work list;
  2. headless Claude prefill (scripts/prompts/nightly_prefill.md + a scope
     override) — files each validated reading as a PENDING commit via
     core.prefill_commit (never pass; review = Confirm on the clue page).

Invoked by the puzzle page's "Prefill now" button (web/routes/admin.py, runs
this DETACHED so the site never blocks on it) or by hand:

    python scripts/run_prefill.py --source telegraph --pnum 3377
    python scripts/run_prefill.py --date 2026-07-11        # catch-up a day
    python scripts/run_prefill.py --source telegraph --pnum 3377 --plan
                                                           # print prompt only
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON_V2 = str(ROOT / ".venv" / "Scripts" / "python.exe")
CASCADE_SCRIPT = str(ROOT / "scripts" / "nightly_cascade.py")
CLAUDE_EXE = r"C:\Users\shute\.local\bin\claude.exe"
PROMPT_FILE = ROOT / "scripts" / "prompts" / "nightly_prefill.md"
LOG_DIR = ROOT / "logs"


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def build_prompt(source=None, pnum=None, date=None, report_name=None):
    """The production prefill prompt with an explicit SCOPE OVERRIDE prepended —
    the prompt file itself scopes to "today", which is wrong for an on-demand
    or catch-up run."""
    if source and pnum:
        scope = ("- Scope: %s puzzle %s ONLY (on-demand run from the puzzle "
                 "page), NOT \"today\"." % (source, pnum))
    else:
        scope = ("- Scope: publication_date %s, serving papers "
                 "(telegraph/times/guardian), NOT \"today\"." % date)
    override = (
        "# ON-DEMAND SCOPE OVERRIDE (scripts/run_prefill.py)\n\n"
        "This is an ON-DEMAND invocation of the standard nightly prefill "
        "prompt below.\nOverride ONLY the scope; every other rule applies "
        "unchanged:\n\n%s\n- Write the summary report to logs/%s — do NOT "
        "overwrite any nightly report.\n\n----\n\n" % (scope, report_name))
    return override + PROMPT_FILE.read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="On-demand prefill chain")
    ap.add_argument("--source", default=None)
    ap.add_argument("--pnum", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (catch-up mode)")
    ap.add_argument("--skip-cascade", action="store_true")
    ap.add_argument("--plan", action="store_true", help="print the prompt, run nothing")
    args = ap.parse_args()
    if not ((args.source and args.pnum) or args.date):
        ap.error("need --source + --pnum, or --date")

    stamp = time.strftime("%Y%m%d_%H%M")
    tag = ("%s_%s" % (args.source, args.pnum)) if args.pnum else args.date
    report_name = "prefill_ondemand_%s_%s.md" % (tag, stamp)
    prompt = build_prompt(args.source, args.pnum, args.date, report_name)
    if args.plan:
        print(prompt)
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    # 1. cascade the remainder (idempotent; skips clues that already have a parse)
    if not args.skip_cascade:
        cargs = [PYTHON_V2, CASCADE_SCRIPT]
        cargs += (["--source", args.source, "--pnum", str(args.pnum)]
                  if args.pnum else ["--date", args.date])
        log("cascade: %s" % " ".join(cargs[2:]))
        c = subprocess.run(cargs, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(ROOT),
                           timeout=3600)
        for line in (c.stdout or "").splitlines():
            log("  " + line.strip())
        if c.returncode != 0:
            log("cascade failed (exit %d): %s" % (c.returncode,
                                                  (c.stderr or "")[-300:]))
            return 1

    # 2. headless Claude prefill (same invocation as the nightly)
    # The web server's environment carries ANTHROPIC_API_KEY (loaded from .env
    # for the ai_* helpers). claude.exe prefers an API key over the claude.ai
    # login, so a button-launched run would bill prepaid API credits — strip it
    # so headless runs ALWAYS bill the subscription (2026-07-13).
    claude_env = os.environ.copy()
    claude_env.pop("ANTHROPIC_API_KEY", None)
    log("prefill: claude -p (scope %s) ..." % tag)
    r = subprocess.run([CLAUDE_EXE, "-p", prompt, "--dangerously-skip-permissions"],
                       cwd=str(ROOT), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=5400, env=claude_env)
    for line in (r.stdout or "").strip().splitlines()[-25:]:
        log("  " + line)
    if r.returncode != 0:
        log("prefill failed (exit %d): %s" % (r.returncode, (r.stderr or "")[-500:]))
        return 1
    log("prefill complete — report: logs/%s" % report_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
