"""On-demand prefill for ONE puzzle (or one date) — the prize-morning chain.

Prize puzzles get their answers when the USER solves the grid, so waiting for
the next nightly means publishing explanations a day late (user, 2026-07-12).
This script runs the same chain as the nightly, scoped and on demand:

  1. cascade any un-cascaded clues (scripts/nightly_cascade.py — idempotent,
     same guards: answer required, no existing parse, never a frozen manual
     solve), so the prefill has a work list;
  2. headless Claude prefill (scripts/prompts/nightly_prefill.md + a scope
     override) — files each validated reading as a PENDING commit via
     core.prefill_commit (never pass; review = Confirm on the clue page);
  3. headless Claude pass review of the puzzle's engine passes
     (nightly_pass_review.md + the same scope override);
  4. double-definition usage phrases (draft_senses.py), filed unapproved;
  5. clue-page prose from the pending readings (draft_prose.py), filed unapproved.

  --skip-prefill runs steps 3-5 alone (e.g. after a prefill that ran without them).

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
# Pin the headless model: the CLI otherwise inherits the user's saved default,
# which changes whenever they /model in an interactive session (2026-07-14).
CLAUDE_MODEL = "claude-fable-5"
PROMPT_FILE = ROOT / "scripts" / "prompts" / "nightly_prefill.md"
REVIEW_PROMPT_FILE = ROOT / "scripts" / "prompts" / "nightly_pass_review.md"
LOG_DIR = ROOT / "logs"

# This script's own stdout is a LOG FILE, never a console, so Python picks the
# locale encoding (cp1252) for it. Claude's summary routinely contains an arrow
# or an em dash, and echoing one killed the run AFTER the readings were already
# filed — so the log never reached "prefill complete" and the puzzle page's
# status box sat polling until it timed out and reported that no prefill was
# running, on a run that had in fact worked. Silent since at least 2026-08-30
# (guardian 4167, times 5232); found 2026-09-10.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):      # already-wrapped or closed stream
        pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def build_prompt(source=None, pnum=None, date=None, report_name=None):
    """The production prefill prompt with an explicit SCOPE OVERRIDE prepended —
    the prompt file itself scopes to "today", which is wrong for an on-demand
    or catch-up run."""
    if source and pnum:
        scope = ("- Scope: %s puzzle %s ONLY (on-demand run from the puzzle "
                 "page), NOT \"today\"." % (source, pnum))
        # The prompt below opens "today's SERVING-PAPER puzzles". A source
        # outside telegraph/times/guardian contradicts that wording, so say
        # plainly that the named source is the scope — otherwise a headless
        # run can read the body and skip the puzzle it was asked to do.
        # `custom` is the user's Reddit clue-writing threads (2026-09-10);
        # a clue there is an ordinary cryptic clue and is read as one.
        if source not in ("telegraph", "times", "guardian"):
            scope += ("\n- The source above is the scope even though it is not "
                      "a serving paper: ignore the \"serving papers\" wording "
                      "below. Every OTHER rule — the validation gate, the "
                      "prefill-discipline rules, PENDING only, never a verdict "
                      "— applies exactly as written.")
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


def build_review_prompt(source=None, pnum=None, date=None, report_name=None):
    """The nightly pass-review prompt, scoped the same way as the prefill."""
    if source and pnum:
        scope = ("- Scope: the engine passes of %s puzzle %s ONLY. Run the audit "
                 "command as written and read only that puzzle's blocks; leave "
                 "every other puzzle's passes alone." % (source, pnum))
    else:
        scope = ("- Scope: the engine passes of publication_date %s, serving "
                 "papers (telegraph/times/guardian)." % date)
    override = (
        "# ON-DEMAND SCOPE OVERRIDE (scripts/run_prefill.py)\n\n"
        "This is an ON-DEMAND invocation of the standard nightly pass review "
        "below, run straight after this puzzle's prefill.\nOverride ONLY the "
        "scope; every other rule applies unchanged:\n\n%s\n- Write the summary "
        "report to logs/%s — do NOT overwrite any nightly report.\n\n----\n\n"
        % (scope, report_name))
    return override + REVIEW_PROMPT_FILE.read_text(encoding="utf-8")


def run_claude(prompt, label, env):
    """Headless Claude, the same invocation as the nightly. Returns the exit code."""
    log("%s: claude -p ..." % label)
    r = subprocess.run([CLAUDE_EXE, "-p", prompt, "--model", CLAUDE_MODEL,
                        "--dangerously-skip-permissions"],
                       cwd=str(ROOT), stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=5400, env=env)
    for line in (r.stdout or "").strip().splitlines()[-25:]:
        log("  " + line)
    if r.returncode != 0:
        log("%s failed (exit %d): %s" % (label, r.returncode, (r.stderr or "")[-500:]))
    return r.returncode


def run_step(label, pargs):
    """A non-Claude follow-on step. Never fatal."""
    log("%s: %s" % (label, " ".join(pargs[2:])))
    try:
        p = subprocess.run(pargs, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(ROOT),
                           timeout=1800)
        for line in (p.stdout or "").strip().splitlines()[-120:]:
            log("  " + line)
        if p.returncode != 0:
            log("%s failed (exit %d) — not fatal: %s"
                % (label, p.returncode, (p.stderr or "")[-300:]))
    except Exception as e:
        log("%s ERROR: %s — not fatal" % (label, e))


def main():
    ap = argparse.ArgumentParser(description="On-demand prefill chain")
    ap.add_argument("--source", default=None)
    ap.add_argument("--pnum", default=None)
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (catch-up mode)")
    ap.add_argument("--skip-cascade", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true",
                    help="run only the follow-on steps (pass review, senses, prose)")
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
    if args.skip_prefill:
        log("prefill: SKIPPED (--skip-prefill)")
    else:
        if run_claude(prompt, "prefill (scope %s)" % tag, claude_env) != 0:
            return 1
        log("prefill complete — report: logs/%s" % report_name)

    # The rest of the nightly's per-puzzle chain (scripts/nightly_run.py Steps 3b,
    # 5, 6), so a button or catch-up run leaves the puzzle exactly as a good night
    # would (user, 2026-10-19: "they should all be part of prefill otherwise the
    # manual version is pointless"). Same order as the nightly: pass review AFTER
    # prefill, so a demotion keeps the engine's claim instead of being re-prefilled.
    # None of them is fatal.
    scope_args = (["--source", args.source, "--puzzle", str(args.pnum)]
                  if args.pnum else ["--day", args.date])

    # 3. read this puzzle's engine passes and hold back the indefensible.
    review_name = "pass_review_ondemand_%s_%s.md" % (tag, stamp)
    if run_claude(build_review_prompt(args.source, args.pnum, args.date, review_name),
                  "pass review (scope %s)" % tag, claude_env) == 0:
        log("pass review complete — report: logs/%s" % review_name)

    # 4. double-definition usage phrases, filed UNAPPROVED (nightly Step 5).
    run_step("senses", [PYTHON_V2, str(ROOT / "scripts" / "draft_senses.py")] + scope_args)

    # 5. draft the clue-page prose from the PENDING readings just filed — the
    # nightly's Step 6, scoped to this run. Missing here, the button's puzzles got
    # no prose at all (telegraph 31355, 2026-09-26). Filed UNAPPROVED; never fatal
    # — a missed draft just means no prose block.
    run_step("prose", [PYTHON_V2, str(ROOT / "scripts" / "draft_prose.py"), "--pending"]
             + scope_args)
    log("ALL STEPS COMPLETE (scope %s)" % tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
