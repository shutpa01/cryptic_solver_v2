"""Form-first solver — emits universal Form directly from a clue + answer.

Calls the Anthropic API (default Haiku, optional Sonnet) with a prompt
that tells the model the schema, the 12 basic operations, and the
verifier's four rules. The model returns a JSON form. We parse it,
validate against the schema, then run the verifier.

On verifier FAIL, one retry — sending the failure detail back to the
model so it can correct the form.

This is the source-fix experiment: instead of producing flat components
that have to be repaired by translator inference, the model produces
forms directly, designed against the verifier's contract.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from .schema import Form, validate as validate_form, FormValidationError
from .verifier import FormVerifier, Verdict


# Model defaults — same naming as the existing helpers in
# signature_solver/haiku_*.py and sonnet_pipeline/tier2_solver.py.
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

_CLIENT = None


def _client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    return _CLIENT


# --- Prompt -----------------------------------------------------------------

PROMPT_PREAMBLE = """You are parsing a single cryptic crossword clue. The
answer is given. Your job is to produce a structured "form" of the
wordplay as JSON.

The form is a recursive operation tree. There are exactly 12 basic
operations:

LEAF operations (terminal — produce letters from one source word):
  - literal       letters are the source word's own letters as-is
                  (e.g. source "nit" -> "NIT")
  - synonym       source word means a word with these letters
                  (e.g. source "fish" -> "IDE")
  - abbreviation  source word maps to letters by cryptic convention
                  (e.g. "Republican" -> "R", "good" -> "G")
  - positional    extracts letter(s) by position from one source word
                  (e.g. first/last/middle/outer/odd/even of the source)

OPERATION nodes (combine or transform one or more children):
  - charade       concatenate children left-to-right (no indicator needed)
  - container     children[0] outer wraps around children[1] inner
                  (needs indicator like "around", "in", "containing")
  - anagram       rearrange the letters of all children
                  (needs indicator like "cooked", "wild", "scrambled")
  - reversal      reverse the letters of one child
                  (needs indicator like "back", "sent over")
  - deletion      remove letters from one child (head/tail/outer/heart)
                  (needs indicator like "lacking", "without", "mostly")
  - hidden        the answer is contiguous letters across a span
                  (needs indicator like "in", "covers", "concealed in")
  - double_definition   two phrases each define the answer (no indicator)
  - acrostic      first/last/middle letter of each child's source word
                  (needs indicator like "at first", "initially")

The form has four fields:

  {
    "tree":        <recursive node>,
    "definition":  {"phrase": "<phrase from clue>", "answer": "<ANSWER>"},
    "link_words":  [<surface words that just join parts, e.g. "the", "of">],
    "is_and_lit":  false
  }

Each tree node is either a leaf:
  {"kind": "leaf", "operation": "literal|synonym|abbreviation|positional",
   "source_word": "<word from clue>", "value": "<LETTERS>",
   "positional_kind": null | "first|last|middle|outer|odd|even|alternate",
   "positional_indicator": null | "<clue word>"}

Or an op:
  {"kind": "op", "operation": "<one of the op types above>",
   "indicator": "<clue word(s)>" | null,
   "sources": [<node>, ...],
   "deletion_kind": null | "head|tail|outer|heart",
   "acrostic_kind": null | "first|last|middle"}

CRITICAL RULES (your form will be checked against these):

1. ASSEMBLY: walking the tree must produce the answer letters.
2. BRIDGE: every non-literal op needs an indicator named from the clue.
   Charade and double_definition are exempt — implicit by structure.
3. RESIDUE: every clue word must have a role — as a leaf source_word,
   an op indicator, the definition phrase, or in link_words. Anything
   not claimed is a failure.
4. OUTER-OP-FIRST: the outermost (last-applied) operation is the root.
   E.g. "REVOLTING (= lover + nit + good, sent over)" is reversal at
   the root, charade as child:
     reversal{indicator:"sent over"}(
       charade(syn"good"->G, lit"nit"->NIT, syn"lover"->LOVER))
   Pieces inside the charade are listed in the ORDER THEY APPEAR
   BEFORE the outer operation applies (so reversed below produces the
   answer).

NAMING: when there are multiple operations, the type name lists the
INNERMOST FIRST (chronologically earliest applied), then outwards.
E.g. "deletion anagram" means deletion happened first, then anagram.

Two worked examples follow. Then your task.

EXAMPLE 1
Clue: "Old man, among others, gets a meal" (6)
Answer: REPAST
Form:
{
  "tree": {"kind": "op", "operation": "container",
    "indicator": "among",
    "sources": [
      {"kind": "leaf", "operation": "synonym",
       "source_word": "others", "value": "REST"},
      {"kind": "leaf", "operation": "synonym",
       "source_word": "old man", "value": "PA"}
    ]},
  "definition": {"phrase": "meal", "answer": "REPAST"},
  "link_words": ["gets", "a"],
  "is_and_lit": false
}

EXAMPLE 2
Clue: "Cook Antonio, mostly pale native of southern Italy" (10)
Answer: NEAPOLITAN
Form:
{
  "tree": {"kind": "op", "operation": "anagram",
    "indicator": "Cook",
    "sources": [
      {"kind": "op", "operation": "deletion",
       "indicator": "mostly", "deletion_kind": "tail",
       "sources": [
         {"kind": "leaf", "operation": "literal",
          "source_word": "Antonio", "value": "ANTONIO"}
       ]},
      {"kind": "leaf", "operation": "literal",
       "source_word": "pale", "value": "PALE"}
    ]},
  "definition": {"phrase": "native of southern Italy",
                 "answer": "NEAPOLITAN"},
  "link_words": [],
  "is_and_lit": false
}

YOUR TASK
Clue: "{{CLUE}}"
Answer: {{ANSWER}}{{ENUM_PART}}

Reply with ONLY the JSON form for the wordplay. No prose, no markdown,
no code fences. Just the JSON object.
"""


RETRY_TEMPLATE = """The previous form failed verification with these
specific failures:

{{FAILURE_LIST}}

Common fixes:
- "X is not a known indicator" -> the form named the wrong word as the
  indicator. Look at the clue surface for an actual indicator word.
- "X unaccounted" -> add the unaccounted word as either a leaf
  source_word, an op indicator, or to link_words.
- "assembly mismatch" -> the tree's letters don't produce the answer.
  Re-check piece order and/or operation choice.

Reply with a CORRECTED JSON form. ONLY JSON, no prose, no code fences."""


def build_prompt(clue_text: str, answer: str,
                 enumeration: Optional[str] = None) -> str:
    enum_part = f" ({enumeration})" if enumeration else ""
    return (PROMPT_PREAMBLE
            .replace("{{CLUE}}", clue_text.replace('"', "'"))
            .replace("{{ANSWER}}", answer)
            .replace("{{ENUM_PART}}", enum_part))


def build_retry_prompt(failure_list: list[str]) -> str:
    return RETRY_TEMPLATE.replace(
        "{{FAILURE_LIST}}",
        "\n".join(f"  - {f}" for f in failure_list))


# --- API call ---------------------------------------------------------------

def _call_api(model: str, system_msg: str, user_msg: str,
              followup: Optional[str] = None) -> tuple[str, dict]:
    """Returns (text, usage_dict). Raises on API errors."""
    client = _client()
    messages = [{"role": "user", "content": user_msg}]
    if followup is not None:
        # Multi-turn: include the previous assistant response as well.
        # For simplicity, just use the followup as a fresh prompt that
        # references the prior failure.
        messages = [{"role": "user", "content":
                     user_msg + "\n\n" + followup}]
    response = client.messages.create(
        model=model, max_tokens=4000, temperature=0,
        messages=messages,
    )
    text = response.content[0].text
    usage = {"input_tokens": response.usage.input_tokens,
             "output_tokens": response.usage.output_tokens}
    return text, usage


# --- JSON extraction --------------------------------------------------------

_JSON_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def extract_json(text: str) -> str:
    """Find the JSON object in the response. Tolerates code fences and
    leading/trailing prose."""
    text = text.strip()
    m = _JSON_FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    # Find the first { and matching last }
    if not text.startswith("{"):
        i = text.find("{")
        if i >= 0:
            text = text[i:]
    if not text.endswith("}"):
        i = text.rfind("}")
        if i >= 0:
            text = text[:i + 1]
    return text


# --- Solver -----------------------------------------------------------------

@dataclass
class SolveResult:
    form: Optional[Form]
    verdict: Optional[Verdict]
    attempts: int               # 1 or 2
    raw_responses: list         # text from each attempt
    parse_errors: list          # JSON / schema errors per attempt
    usage_total: dict           # cumulative token usage
    notes: list

    @property
    def passed(self) -> bool:
        return self.verdict is not None and self.verdict.verdict == "PASS"


def solve_clue(clue_text: str, answer: str,
               enumeration: Optional[str] = None,
               model: str = HAIKU_MODEL,
               verifier: Optional[FormVerifier] = None) -> SolveResult:
    """Form-first solve. Returns a SolveResult."""
    own_verifier = verifier is None
    if own_verifier:
        verifier = FormVerifier()
    notes: list = []
    raw_responses: list = []
    parse_errors: list = []
    usage_total = {"input_tokens": 0, "output_tokens": 0}

    try:
        prompt = build_prompt(clue_text, answer, enumeration)
        # Attempt 1
        text1, u1 = _call_api(model, "", prompt)
        raw_responses.append(text1)
        usage_total["input_tokens"] += u1["input_tokens"]
        usage_total["output_tokens"] += u1["output_tokens"]
        form1, err1 = _parse_form(text1)
        if err1:
            parse_errors.append(err1)
            notes.append(f"attempt_1_parse_error:{err1}")
        verdict1 = None
        if form1:
            verdict1 = verifier.verify(form1, clue_text)
            if verdict1.verdict == "PASS":
                return SolveResult(
                    form=form1, verdict=verdict1, attempts=1,
                    raw_responses=raw_responses,
                    parse_errors=parse_errors,
                    usage_total=usage_total, notes=notes)

        # Attempt 2 — retry with failure detail
        if form1 and verdict1:
            failure_list = [f"{c.name}: {c.detail}"
                            for c in verdict1.failures]
        else:
            failure_list = [f"could not parse JSON: "
                            f"{err1 or 'unknown'}"]
        retry = build_retry_prompt(failure_list)
        text2, u2 = _call_api(model, "", prompt + "\n\n" + retry)
        raw_responses.append(text2)
        usage_total["input_tokens"] += u2["input_tokens"]
        usage_total["output_tokens"] += u2["output_tokens"]
        form2, err2 = _parse_form(text2)
        if err2:
            parse_errors.append(err2)
            notes.append(f"attempt_2_parse_error:{err2}")
        verdict2 = None
        if form2:
            verdict2 = verifier.verify(form2, clue_text)

        # Pick whichever attempt is best
        winner_form = form2 or form1
        winner_verdict = verdict2 or verdict1

        return SolveResult(
            form=winner_form, verdict=winner_verdict, attempts=2,
            raw_responses=raw_responses, parse_errors=parse_errors,
            usage_total=usage_total, notes=notes)
    finally:
        if own_verifier:
            verifier.close()


def _parse_form(text: str) -> tuple[Optional[Form], Optional[str]]:
    """Parse the model output into a Form. Returns (form, error_or_None)."""
    if not text.strip():
        return None, "empty response"
    raw = extract_json(text)
    try:
        d = json.loads(raw)
    except Exception as e:
        return None, f"json parse: {e}"
    try:
        form = Form.from_dict(d)
    except Exception as e:
        return None, f"schema deserialise: {e}"
    problems = validate_form(form)
    if problems:
        return form, "validate: " + "; ".join(problems[:3])
    return form, None


# --- Batch runner -----------------------------------------------------------

def solve_puzzle(source: str, puzzle_number: str,
                 model: str = HAIKU_MODEL,
                 limit: Optional[int] = None) -> list:
    """Solve every clue in a puzzle. Returns list of (clue_row, SolveResult).

    Read-only: doesn't write to the live DBs.
    """
    import sqlite3
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer,
               enumeration
        FROM clues
        WHERE source = ? AND puzzle_number = ?
          AND answer IS NOT NULL AND answer != ''
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (source, str(puzzle_number))).fetchall()
    if limit:
        rows = rows[:limit]

    verifier = FormVerifier()
    results = []
    try:
        for r in rows:
            try:
                sr = solve_clue(r["clue_text"], r["answer"],
                                r["enumeration"], model=model,
                                verifier=verifier)
            except Exception as e:
                sr = SolveResult(
                    form=None, verdict=None, attempts=0,
                    raw_responses=[], parse_errors=[f"exception: {e}"],
                    usage_total={"input_tokens": 0, "output_tokens": 0},
                    notes=[f"exception: {e!r}"])
            results.append((dict(r), sr))
    finally:
        verifier.close()
        conn.close()
    return results


# --- CLI --------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--puzzle", required=True)
    ap.add_argument("--model", default="haiku",
                    choices=["haiku", "sonnet"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    model = HAIKU_MODEL if args.model == "haiku" else SONNET_MODEL
    print(f"Form-first solver — model={model}, "
          f"source={args.source}, puzzle={args.puzzle}")
    print()

    results = solve_puzzle(args.source, args.puzzle,
                           model=model, limit=args.limit)
    n_pass = sum(1 for _, sr in results if sr.passed)
    n_fail = sum(1 for _, sr in results
                 if sr.verdict and sr.verdict.verdict == "FAIL")
    n_no_form = sum(1 for _, sr in results if sr.form is None)
    total_in = sum(sr.usage_total["input_tokens"] for _, sr in results)
    total_out = sum(sr.usage_total["output_tokens"] for _, sr in results)

    for r, sr in results:
        v = sr.verdict.verdict if sr.verdict else "NO_FORM"
        att = sr.attempts
        print(f"{r['clue_number']:>3s}{r['direction'][:1]}  "
              f"{r['answer']:<16s} "
              f"{v:<8s} attempts={att}  notes={sr.notes}")

    print()
    print(f"PASS: {n_pass}  FAIL: {n_fail}  NO_FORM: {n_no_form} "
          f"(of {len(results)})")
    print(f"Tokens: in={total_in:,}  out={total_out:,}")
