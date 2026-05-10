"""Per-puzzle driver — parallel-of-production cascade for one puzzle.

Mirrors `sonnet_pipeline/run.py::run_puzzle` for the prototype
universal-form pipeline. Runs the cascade against every clue in a
single puzzle, writes results to shadow_db, generates a markdown
report.

Per `PARALLEL_SYSTEM_DESIGN.md` §3.6, the puzzle is the runtime unit
of work. First pass walks every clue through the cascade and emits
PASS / FAIL / PENDING. Leftover authoring (separate dashboard step)
sits between the passes — it creates the catalog entry and the
shadow-vocabulary rows a FAIL clue needs to verify. Second pass
(separate invocation with --second-pass) re-runs the cascade against
first-pass FAILs, picking up any catalog or DB rows the leftover
process added.

Production engines act as DETECTORS, not solvers (§3.3): the driver
calls hidden / spoonerism / DD detectors to bias the catalog walk
order; the cascade does the actual word-by-word accounting.

Usage
-----
    python -m prototypes.universal_form_v2.run_puzzle <source> <puzzle>
    python -m prototypes.universal_form_v2.run_puzzle telegraph 31235
    python -m prototypes.universal_form_v2.run_puzzle telegraph 31235 --force
    python -m prototypes.universal_form_v2.run_puzzle telegraph 31235 --second-pass
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from signature_solver.db import RefDB
from signature_solver.solver import extract_definition_candidates
from signature_solver.grammar_triage import (
    grammar_triage as _grammar_triage,
)
from backfill_ai_exp.backfill_dd_hidden import (
    build_graph as _build_dd_graph,
    try_hidden as _detect_hidden,
    generate_dd_hypotheses as _detect_dd,
)
from sonnet_pipeline.solver import (
    try_spoonerism_v2 as _detect_spoonerism, clean,
)
from sonnet_pipeline.sig_adapter import (
    build_ai_pieces as _sig_build_ai_pieces,
    build_assembly_dict as _sig_build_assembly_dict,
)
from backfill_ai_exp.batch_v1_solver import (
    try_anagram as _detect_anagram,
    try_charade as _detect_charade,
    try_container as _detect_container,
    try_deletion as _detect_deletion,
    try_reversal as _detect_reversal,
    try_acrostic as _detect_acrostic,
    try_homophone as _detect_homophone,
)

from prototypes.universal_form_v2.cascade import solve_clue_parallel
from prototypes.universal_form_v2.shadow_db import (
    ensure_shadow, write_solve, write_seed_failure,
)
from prototypes.universal_form_v2.surface import tokenize as _tokenize
from prototypes.universal_form_v2.json_translator import (
    translate_components as _translate_components,
)
from prototypes.universal_form_v2.clipboard_verifier import (
    verify as _clipboard_verify,
)
from prototypes.universal_form_v2.extract_catalog import (
    signature as _form_signature,
)


CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
CRYPTIC_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
CATALOG_PATH = (PROJECT_ROOT / "prototypes" / "universal_form_v2"
                / "runs" / "catalog_v1.json")
REPORT_DIR = PROJECT_ROOT / "documents"

# Production cross-reference skip pattern, matching sonnet_pipeline/run.py.
CROSSREF_RE = re.compile(r'\b\d+\s*(?:across|down|ac|dn)\b', re.IGNORECASE)


# --- Setup ----------------------------------------------------------------

def load_catalog() -> list:
    """Load catalog_v1 entries sorted by frequency descending."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        cat = json.load(f)
    return sorted(cat["entries"], key=lambda e: -e.get("frequency", 0))


def fetch_puzzle_clues(conn: sqlite3.Connection,
                        source: str, puzzle_number: str) -> list:
    """Pull every clue for a puzzle, in clue-number order."""
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer, enumeration
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        ORDER BY clue_number
    """, (source, str(puzzle_number))).fetchall()
    return [dict(r) for r in rows]


# --- Routing hints --------------------------------------------------------

def detect_routing_hints(clue_text: str, answer_clean: str,
                          dd_graph: dict, ref_db: RefDB) -> list:
    """Run production detectors; return mechanism-class hints.

    The hints bias the catalog walk order — they never restrict it
    (per §3.3). An empty list means "no hint, walk by frequency only".

    Per §3.4 step 3 of the design, every production detector class
    runs as a triage signal:
      hidden / spoonerism / double_definition / anagram / charade /
      container / deletion / reversal / acrostic / homophone.

    Each V1 detector is itself a solver in production, but as triage
    we only consume its boolean fire/no-fire signal. The detectors
    are cheap because they pre-filter each word's synonym values to
    those that could possibly fit the answer (the same target-driven
    binding our tree-matcher lacks); that's why they don't recreate
    the per-clue combinatorial explosion the cascade hits.

    Detectors run on the full surface tokens of the clue. For triage
    we don't need definition extraction — over-firing only dilutes
    the bias, never breaks correctness, since the cascade still walks
    the rest of the catalog after hinted templates.
    """
    hints: list = []
    if not answer_clean or len(answer_clean) < 3:
        return hints

    tokens = _tokenize(clue_text)

    # --- Cheap dedicated detectors -------------------------------------

    # Hidden — pure mechanical substring search.
    try:
        if _detect_hidden(clue_text, answer_clean,
                          graph=dd_graph,
                          enumeration=len(answer_clean)):
            hints.append("hidden")
    except Exception:
        pass

    # Spoonerism — gated on "spooner" in clue, as production does.
    if "spooner" in clue_text.lower() and len(answer_clean) >= 4:
        try:
            if _detect_spoonerism(answer_clean, ref_db.is_real_word,
                                   clue_text=clue_text, ref_db=ref_db):
                hints.append("spoonerism")
        except Exception:
            pass

    # Double definition — both halves must independently produce the
    # answer. dd_graph required.
    if dd_graph:
        try:
            if _detect_dd(clue_text, dd_graph,
                          total_len=len(answer_clean),
                          answer=answer_clean):
                hints.append("double_definition")
        except Exception:
            pass

    # --- V1 mechanical detectors (per §3.4 step 3) ---------------------
    #
    # try_anagram takes the raw clue text plus an optional list of
    # definition words to exclude from fodder; passing None means
    # "every word is fodder candidate" — the right call for triage,
    # where we don't yet trust any one definition split.
    try:
        if _detect_anagram(clue_text, answer_clean, ref_db,
                           definition_words=None):
            hints.append("anagram")
    except Exception:
        pass

    # The remaining V1 detectors take a tokenised wordplay window.
    # They require the definition word(s) removed first — they check
    # that every input word is accounted for as piece / link / indicator,
    # and the definition word otherwise blocks the assembly.
    #
    # So we pull definition candidates here and run each detector once
    # per candidate, unioning the hints that fire. Once a hint is in
    # the set we skip re-running its detector.
    try:
        def_candidates = extract_definition_candidates(
            tokens, answer_clean, ref_db)
    except Exception:
        def_candidates = []

    word_window_hints: set = set()
    for _def_phrase, wp_words in def_candidates:
        if not wp_words:
            continue
        if "charade" not in word_window_hints:
            try:
                if _detect_charade(wp_words, answer_clean, ref_db):
                    word_window_hints.add("charade")
            except Exception:
                pass
        if "container" not in word_window_hints:
            try:
                if _detect_container(wp_words, answer_clean, ref_db):
                    word_window_hints.add("container")
            except Exception:
                pass
        if "deletion" not in word_window_hints:
            try:
                if _detect_deletion(wp_words, answer_clean, ref_db):
                    word_window_hints.add("deletion")
            except Exception:
                pass
        if "reversal" not in word_window_hints:
            try:
                if _detect_reversal(wp_words, answer_clean, ref_db):
                    word_window_hints.add("reversal")
            except Exception:
                pass
        if "acrostic" not in word_window_hints:
            try:
                if _detect_acrostic(wp_words, answer_clean, ref_db):
                    word_window_hints.add("acrostic")
            except Exception:
                pass
        if "homophone" not in word_window_hints:
            try:
                if _detect_homophone(wp_words, answer_clean, ref_db):
                    word_window_hints.add("homophone")
            except Exception:
                pass
        # Stop iterating definition candidates once every word-window
        # mechanism has fired at least once — additional candidates
        # cannot add new hints.
        if len(word_window_hints) == 6:
            break

    hints.extend(sorted(word_window_hints))
    return hints


def reorder_catalog_by_hints(catalog_entries: list, hints: list) -> list:
    """Reorder so signatures matching any hint come first (in their
    existing frequency order), then the rest. Hint biases — never
    restricts — per §3.3."""
    if not hints:
        return catalog_entries
    hinted = []
    rest = []
    for e in catalog_entries:
        sig_id = e.get("id", "").lower()
        if any(h in sig_id for h in hints):
            hinted.append(e)
        else:
            rest.append(e)
    return hinted + rest


# --- Grammar-triage routing layer (§3.3a) ---------------------------------

def try_grammar_triage_solve(clue_text: str, answer_clean: str,
                              db: RefDB, shadow_conn) -> tuple:
    """Linguistic-triage solve attempt via signature_solver.grammar_triage.

    Per §3.3a, this is the routing layer in front of the
    tree-matcher. spaCy POS-tags the clue; the resulting grammar
    pattern yields per-word role predictions and a candidate solve.

    No confidence threshold is applied: every grammar_triage result
    that has a SignatureResult is pushed through the conversion
    chain (sig_adapter → json_translator → clipboard verifier).
    The clipboard verifier is the trust anchor — a low-confidence
    triage reading that doesn't verify costs nothing.

    Even when no def candidate yields a verifier-PASSing form, the
    wordplay-type predictions grammar_triage emitted are returned
    as routing hints so the cascade can bias its catalog walk
    toward the predicted family.

    Returns (form, signature_string, hints):
      - form: the verified Form, or None on no PASS;
      - signature_string: the universal-form signature from the
        Form's tree on PASS, else None;
      - hints: list of unique wordplay_type strings grammar_triage
        proposed across def candidates (empty if grammar_triage
        produced nothing).
    """
    hints: list = []
    try:
        tokens = _tokenize(clue_text)
        def_candidates = extract_definition_candidates(
            tokens, answer_clean, db)
    except Exception:
        return None, None, hints

    for def_phrase, wp_words in def_candidates:
        if not wp_words:
            continue
        try:
            gt = _grammar_triage(clue_text, answer_clean, db,
                                  def_phrase=def_phrase,
                                  wp_words=list(wp_words))
        except Exception:
            continue
        if not gt or gt.result is None:
            continue

        # SignatureResult → components JSON dict via sig_adapter.
        try:
            ai_pieces = _sig_build_ai_pieces(gt)
            assembly = _sig_build_assembly_dict(gt)
        except Exception:
            continue
        if not assembly:
            continue
        wordplay_type = assembly.get("op")

        # Record the wordplay_type as a routing hint for the
        # cascade fallback, even if this attempt won't verify.
        if wordplay_type and wordplay_type not in hints:
            hints.append(wordplay_type)

        if not ai_pieces or not wordplay_type:
            continue

        components_json = json.dumps({
            "ai_pieces": ai_pieces,
            "assembly": assembly,
            "wordplay_type": wordplay_type,
        })

        # Components dict → Form via json_translator.
        row = {
            "clue_text": clue_text,
            "answer": answer_clean,
            "components": components_json,
            "definition_text": def_phrase,
        }
        try:
            form, _err = _translate_components(row, db)
        except Exception:
            continue
        if form is None:
            continue

        # Form → clipboard verifier (the trust anchor).
        try:
            verdict = _clipboard_verify(form, clue_text, db, shadow_conn)
        except Exception:
            continue
        if verdict.verdict != "PASS":
            continue

        # Derive the universal-form signature from the Form's tree.
        try:
            sig = _form_signature(form.tree)
        except Exception:
            sig = wordplay_type or "grammar_triage"
        return form, sig, hints

    return None, None, hints


# --- Per-clue processing --------------------------------------------------

def process_clue(clue_row: dict, catalog: list, db: RefDB,
                  dd_graph: dict, shadow: sqlite3.Connection,
                  source: str, puzzle_number: str,
                  run_number: int) -> tuple:
    """Run the cascade on one clue; persist the verdict.

    Returns a (verdict, record) tuple where `verdict` is
    'PASS' | 'PENDING' | 'FAIL' and `record` is a dict carrying
    per-clue summary data for the report.
    """
    clue_id = clue_row["id"]
    clue_text = clue_row["clue_text"]
    answer = clue_row["answer"]
    answer_clean = clean(answer)

    meta = {
        "source": source,
        "puzzle_number": puzzle_number,
        "clue_number": clue_row["clue_number"],
        "direction": clue_row["direction"],
        "clue_text": clue_text,
        "answer": answer,
    }

    # Per §3.3a: try the grammar-triage routing layer first. No
    # confidence threshold — clipboard verifier is the trust anchor.
    # On verifier-PASS we bypass the catalog walk; on miss we still
    # collect the wordplay-type predictions as routing hints.
    gt_form, gt_signature, gt_hints = try_grammar_triage_solve(
        clue_text, answer_clean, db, shadow)
    if gt_form is not None:
        write_solve(
            shadow,
            clue_id=clue_id,
            signature=gt_signature or "",
            verdict="PASS",
            answer=answer_clean,
            form_dict=gt_form.to_dict(),
            run_number=run_number,
        )
        record = {
            "clue_id": clue_id,
            "clue_number": clue_row["clue_number"],
            "direction": clue_row["direction"],
            "clue_text": clue_text,
            "answer": answer,
            "verdict": "PASS",
            "signature": gt_signature,
            "hints": ["grammar_triage"] + gt_hints,
            "n_enrichments": 0,
        }
        return "PASS", record

    # Fall through to the cascade. Combine grammar_triage's
    # wordplay-type predictions (when it produced any) with the V1
    # detectors' mechanism-class hints. Both feed the catalog walk
    # order; the verifier still gates every form.
    v1_hints = detect_routing_hints(clue_text, answer_clean, dd_graph, db)
    hints = list(dict.fromkeys(gt_hints + v1_hints))
    ordered_catalog = reorder_catalog_by_hints(catalog, hints)

    result = solve_clue_parallel(
        clue_id=clue_id,
        clue_text=clue_text,
        answer=answer_clean,
        db=db,
        catalog_entries=ordered_catalog,
        shadow_conn=shadow,
    )

    record = {
        "clue_id": clue_id,
        "clue_number": clue_row["clue_number"],
        "direction": clue_row["direction"],
        "clue_text": clue_text,
        "answer": answer,
        "verdict": result.verdict,
        "signature": result.signature,
        "hints": hints,
        "n_enrichments": len(result.enrichment_candidates),
    }

    if result.verdict in ("PASS", "PENDING"):
        # Both verdicts produce a verified-or-pending Form; persist to
        # solves. PENDING goes via the same path so the dashboard's
        # PENDING tab can pick it up.
        write_solve(
            shadow,
            clue_id=clue_id,
            signature=result.signature or "",
            verdict=result.verdict,
            answer=answer_clean,
            form_dict=result.form.to_dict(),
            run_number=run_number,
        )
    else:
        # FAIL — record in seed_failures with the deduped enrichment
        # candidates. seed_source='cascade' marks the row as coming
        # from the runtime cascade rather than the seeding-pass
        # translators.
        enrichments = [c.to_dict() for c in result.enrichment_candidates]
        write_seed_failure(
            shadow,
            clue_id=clue_id,
            seed_source="cascade",
            failure_kind="verifier_fail",
            failure_detail="cascade walked catalog; no template verified",
            clue_meta=meta,
            enrichments=enrichments,
            run_number=run_number,
        )
        record["enrichments"] = enrichments

    return result.verdict, record


# --- First and second pass ------------------------------------------------

def run_pass(source: str, puzzle_number: str, run_number: int,
              force: bool = False, only_first_pass_fails: bool = False,
              limit: Optional[int] = None
              ) -> dict:
    """Drive one pass over a puzzle.

    `run_number` records which run this is in shadow_db: 1 for first
    pass, 2 for second.

    `force` (first pass only) re-processes clues that are already
    marked PASS.

    `only_first_pass_fails` (second pass only) restricts processing to
    clues that the first pass FAILed on. PENDING clues are not
    re-processed — they are awaiting human review.
    """
    label = "first" if run_number == 1 else "second"
    print(f"=== {label} pass — {source} puzzle {puzzle_number} ===",
          flush=True)

    clues_conn = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    clues_conn.row_factory = sqlite3.Row
    print("  opening RefDB ...", flush=True)
    db = RefDB(str(CRYPTIC_DB))
    shadow = ensure_shadow()

    catalog = load_catalog()
    print(f"  catalog: {len(catalog)} signatures", flush=True)

    print("  building dd_graph for hidden / DD detectors ...", flush=True)
    t0 = datetime.now()
    dd_graph = _build_dd_graph(db)
    print(f"  dd_graph built ({(datetime.now() - t0).total_seconds():.1f}s)",
          flush=True)

    rows = fetch_puzzle_clues(clues_conn, source, puzzle_number)
    if not rows:
        print(f"  no clues found for {source} puzzle {puzzle_number}",
              flush=True)
        return {"stats": {}, "records": []}
    print(f"  found {len(rows)} clues", flush=True)
    if limit is not None:
        rows = rows[:limit]
        print(f"  --limit {limit} → processing first {len(rows)}",
              flush=True)

    stats = {"PASS": 0, "FAIL": 0, "PENDING": 0,
             "skipped_xref": 0,
             "skipped_no_answer": 0,
             "skipped_no_clue_text": 0,
             "skipped_already_pass": 0,
             "skipped_already_pending": 0,
             "skipped_not_first_pass_fail": 0}
    records: list = []

    for r in rows:
        clue_id = r["id"]
        clue_text = r["clue_text"] or ""
        answer = r["answer"] or ""

        if not clue_text.strip():
            stats["skipped_no_clue_text"] += 1
            continue
        if not answer.strip():
            stats["skipped_no_answer"] += 1
            continue

        if CROSSREF_RE.search(clue_text):
            stats["skipped_xref"] += 1
            print(f"  [skip xref] {r['clue_number']}{r['direction'][:1].lower()}: "
                  f"{clue_text[:60]}", flush=True)
            continue

        # Already-PASS skip (any pass): a confirmed PASS doesn't need
        # rework. --force overrides this on the first pass only.
        already_pass = shadow.execute(
            "SELECT 1 FROM solves WHERE clue_id=? AND verdict='PASS' LIMIT 1",
            (clue_id,)).fetchone()
        if already_pass and not (force and run_number == 1):
            stats["skipped_already_pass"] += 1
            continue

        # PENDING items are awaiting human review, never re-run.
        already_pending = shadow.execute(
            "SELECT 1 FROM solves WHERE clue_id=? AND verdict='PENDING' LIMIT 1",
            (clue_id,)).fetchone()
        if already_pending:
            stats["skipped_already_pending"] += 1
            continue

        # Second-pass scope: only clues whose first pass FAILed.
        if only_first_pass_fails:
            had_first_fail = shadow.execute(
                "SELECT 1 FROM seed_failures "
                "WHERE clue_id=? AND run_number=1 LIMIT 1",
                (clue_id,)).fetchone()
            if not had_first_fail:
                stats["skipped_not_first_pass_fail"] += 1
                continue

        cnum = f"{r['clue_number']}{r['direction'][:1].lower()}"
        t_clue = datetime.now()
        print(f"  → {cnum} {answer:<14}  {clue_text[:50]}", flush=True)

        try:
            verdict, record = process_clue(
                r, catalog, db, dd_graph, shadow,
                source, puzzle_number, run_number)
        except Exception as e:  # noqa: BLE001 - keep the run going
            elapsed = (datetime.now() - t_clue).total_seconds()
            print(f"  [ERROR {elapsed:.1f}s] {cnum}: {e}", flush=True)
            stats["FAIL"] += 1  # treat as failure for accounting
            records.append({
                "clue_id": clue_id, "clue_number": r["clue_number"],
                "direction": r["direction"], "clue_text": clue_text,
                "answer": answer, "verdict": "ERROR", "signature": None,
                "hints": [], "n_enrichments": 0, "error": str(e),
            })
            continue

        elapsed = (datetime.now() - t_clue).total_seconds()
        stats[verdict] += 1
        records.append(record)
        marker = {"PASS": "PASS", "FAIL": "FAIL",
                   "PENDING": "PEND"}.get(verdict, verdict)
        sig = record["signature"] or ""
        hint_str = ",".join(record["hints"]) if record["hints"] else "—"
        print(f"  [{marker:>4} {elapsed:5.1f}s] {cnum}: {answer:<14}  "
              f"hints={hint_str:<35}  {sig}",
              flush=True)

    print()
    print(f"  summary: {stats['PASS']} PASS, {stats['FAIL']} FAIL, "
          f"{stats['PENDING']} PENDING")
    print(f"  skipped: {stats['skipped_xref']} xref, "
          f"{stats['skipped_no_answer']} no-answer, "
          f"{stats['skipped_already_pass']} already-PASS, "
          f"{stats['skipped_already_pending']} already-PENDING")
    if only_first_pass_fails:
        print(f"  skipped (second pass): "
              f"{stats['skipped_not_first_pass_fail']} not-first-pass-FAIL")

    write_report(source, puzzle_number, label, stats, records)

    clues_conn.close()
    return {"stats": stats, "records": records}


# --- Reporting ------------------------------------------------------------

def write_report(source: str, puzzle_number: str, pass_label: str,
                  stats: dict, records: list) -> Path:
    """Markdown report mirroring the `documents/puzzle_report_*` shape
    used by production runs."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = (REPORT_DIR
            / f"parallel_puzzle_report_{source}_{puzzle_number}_{pass_label}.md")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Parallel-system puzzle report — {source} {puzzle_number} "
        f"— {pass_label} pass",
        f"_generated {ts}_",
        "",
        "## Counts",
        f"- PASS: {stats.get('PASS', 0)}",
        f"- FAIL: {stats.get('FAIL', 0)}",
        f"- PENDING: {stats.get('PENDING', 0)}",
        f"- skipped (cross-reference): {stats.get('skipped_xref', 0)}",
        f"- skipped (no answer): {stats.get('skipped_no_answer', 0)}",
        f"- skipped (already PASS): "
        f"{stats.get('skipped_already_pass', 0)}",
        f"- skipped (already PENDING): "
        f"{stats.get('skipped_already_pending', 0)}",
        f"- skipped (no clue text): "
        f"{stats.get('skipped_no_clue_text', 0)}",
    ]
    if stats.get("skipped_not_first_pass_fail", 0):
        lines.append(f"- skipped (not first-pass FAIL): "
                     f"{stats['skipped_not_first_pass_fail']}")
    lines.append("")

    pendings = [r for r in records if r["verdict"] == "PENDING"]
    fails = [r for r in records if r["verdict"] == "FAIL"]
    passes = [r for r in records if r["verdict"] == "PASS"]
    errors = [r for r in records if r["verdict"] == "ERROR"]

    if passes:
        lines.append("## PASS")
        for r in passes:
            lines.append(
                f"- **{r['clue_number']}{r['direction'][:1].lower()}** "
                f"{r['answer']} — `{r['signature']}` "
                f"(hints: {', '.join(r['hints']) or 'none'})")
            lines.append(f"  - {r['clue_text']}")
        lines.append("")

    if pendings:
        lines.append("## PENDING — awaiting human review")
        for r in pendings:
            lines.append(
                f"- **{r['clue_number']}{r['direction'][:1].lower()}** "
                f"{r['answer']} — `{r['signature']}`")
            lines.append(f"  - {r['clue_text']}")
        lines.append("")

    if fails:
        lines.append("## FAIL — enrichment candidates")
        for r in fails:
            lines.append(
                f"- **{r['clue_number']}{r['direction'][:1].lower()}** "
                f"{r['answer']} — {r['n_enrichments']} candidate(s)")
            lines.append(f"  - {r['clue_text']}")
            for c in r.get("enrichments") or []:
                op = c.get("operation")
                sub = c.get("subtype")
                kind = c.get("kind")
                tag = kind
                if kind == "indicator" and op:
                    tag = f"indicator({op}{':'+sub if sub else ''})"
                lines.append(
                    f"    - {tag}: `{c.get('source_word')}` → "
                    f"`{c.get('value')}`")
        lines.append("")

    if errors:
        lines.append("## ERROR — processing exceptions")
        for r in errors:
            lines.append(
                f"- **{r['clue_number']}{r['direction'][:1].lower()}** "
                f"{r['answer']}: {r.get('error')}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  report: {path}")
    return path


# --- CLI ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Per-puzzle driver for the parallel-system cascade.")
    p.add_argument("source", help="telegraph | times | guardian | ...")
    p.add_argument("puzzle_number", help="puzzle number (string)")
    p.add_argument("--force", action="store_true",
                   help="(first pass) re-process even already-PASS clues")
    p.add_argument("--second-pass", action="store_true",
                   dest="second_pass",
                   help="re-run the cascade against first-pass FAILs only")
    p.add_argument("--limit", type=int, default=None,
                   help="cap the number of clues processed (smoke test)")
    args = p.parse_args()

    if args.second_pass:
        run_pass(args.source, args.puzzle_number,
                  run_number=2, only_first_pass_fails=True,
                  limit=args.limit)
    else:
        run_pass(args.source, args.puzzle_number,
                  run_number=1, force=args.force,
                  limit=args.limit)


if __name__ == "__main__":
    main()
