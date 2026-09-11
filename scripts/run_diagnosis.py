"""On-demand post-publish diagnosis — step 4 of the publish-first process.

This used to be step 3 of the nightly run. It was moved here (user, 2026-09-11)
because nothing before lunchtime depends on it: it reads the manual solves the
user has COMMITTED and works out why the engines could not do them, which
improves FUTURE puzzles. Prefill and pass review are what the morning needs.

Measured on the 2026-09-11 nightly it was 8.24M of the run's 22.0M cache-read
tokens — 37% — spent first in the queue, leaving too little for early-morning
work. The publish-first process always said this belonged in "non-critical
time"; now it does.

The prompt is UNCHANGED (scripts/prompts/nightly_diagnosis.md), so the work and
its hard write-scope rules are exactly what they were. Only when it runs changed.

    python scripts/run_diagnosis.py              # diagnose, write the report
    python scripts/run_diagnosis.py --plan       # print the prompt, run nothing

Usage is CAPTURED (--output-format json) and logged, because the nightly never
recorded it and the only way to find out what a step cost was to read the
session transcripts afterwards.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLAUDE_EXE = r"C:\Users\shute\.local\bin\claude.exe"
# Pinned, for the same reason run_prefill.py pins it: the CLI otherwise inherits
# whatever the user last chose with /model in an interactive session.
CLAUDE_MODEL = "claude-fable-5"
PROMPT_FILE = ROOT / "scripts" / "prompts" / "nightly_diagnosis.md"
LOG_DIR = ROOT / "logs"

# This script's own stdout is a log file, never a console, so Python picks the
# locale encoding (cp1252) and an arrow in Claude's summary would kill the run
# AFTER the work was done. Same fault, same fix as run_prefill.py (2026-09-10).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def build_prompt():
    """The nightly diagnosis prompt, with a note that this is an on-demand run.

    The prompt scopes itself to solves committed in the last 2 days and skips
    clues it has already diagnosed, so running it by hand is safe to repeat —
    no scope override is needed, unlike the prefill's.
    """
    return ("# ON-DEMAND RUN (scripts/run_diagnosis.py)\n\n"
            "This is the standard post-publish diagnosis, started by hand from "
            "the dashboard instead of by the nightly. Every rule below applies "
            "unchanged.\n\n----\n\n"
            + PROMPT_FILE.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(description="On-demand post-publish diagnosis")
    ap.add_argument("--plan", action="store_true",
                    help="print the prompt and run nothing")
    args = ap.parse_args()

    prompt = build_prompt()
    if args.plan:
        print(prompt)
        return 0

    LOG_DIR.mkdir(exist_ok=True)
    # claude.exe prefers an API key over the claude.ai login, so strip it and
    # always bill the subscription (run_prefill.py, 2026-07-13).
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)

    log("diagnosis: claude -p nightly_diagnosis.md ...")
    started = time.time()
    r = subprocess.run(
        [CLAUDE_EXE, "-p", prompt, "--model", CLAUDE_MODEL,
         "--output-format", "json", "--dangerously-skip-permissions"],
        cwd=str(ROOT), stdin=subprocess.DEVNULL, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=5400, env=env)
    mins = (time.time() - started) / 60.0

    if r.returncode != 0:
        log("diagnosis failed (exit %d): %s" % (r.returncode, (r.stderr or "")[-500:]))
        return 1

    # --output-format json gives ONE json object: the result text plus usage.
    text, usage, cost = (r.stdout or "").strip(), None, None
    try:
        payload = json.loads(r.stdout)
        text = payload.get("result") or text
        usage = payload.get("usage")
        cost = payload.get("total_cost_usd")
    except ValueError:
        log("(could not parse the json envelope — printing raw output)")

    for line in text.strip().splitlines()[-25:]:
        log("  " + line)

    log("diagnosis complete in %.1f min" % mins)
    if usage:
        log("tokens: in=%s cache_write=%s cache_read=%s out=%s%s" % (
            f"{usage.get('input_tokens', 0):,}",
            f"{usage.get('cache_creation_input_tokens', 0):,}",
            f"{usage.get('cache_read_input_tokens', 0):,}",
            f"{usage.get('output_tokens', 0):,}",
            ("  cost=$%.2f" % cost) if cost is not None else ""))
    log("report: logs/diagnosis_%s.md" % time.strftime("%Y-%m-%d"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
