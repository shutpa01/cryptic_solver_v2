"""Nightly TRIAGE report generator (READ-ONLY — informs, never resolves).

Implements the output surface for `memory/nightly_triage_process.md`: after the nightly
solve, review every FAIL and PENDING clue on a puzzle and gather the mechanical evidence
needed to classify WHY it did not cleanly solve — missing data / missing signature /
missing engine (one or more per clue). The classification itself is a human/Claude
judgement; this module only GATHERS the evidence and RENDERS a markdown worksheet, then
(optionally) merges a diagnoses file back in to produce the final summary.

Hard boundaries (by design):
  * Writes NOTHING to any database. No enrichment is queued, no signature is filed, no
    clue is re-solved. It reads `wfw_solve` (the verdicts as they stand) + the reference
    DB (via core.diagnose) and writes a markdown file. That is all.
  * ONE pass. It never re-runs the solver to check whether a fix "worked".

Suggested enrichments are printed in the report in the shape `/enrich` consumes
(kind + word + value + answer) so a human can act on them from the dashboard — but this
module does not act on them.

CLI:
  python -m core.triage <source> <puzzle_number>            # evidence worksheet
  python -m core.triage <source> <puzzle_number> <diag.json>  # + merge diagnoses
"""

import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_DB = os.path.join(ROOT, "data", "clues_master.db")
TRIAGE_DIR = os.path.join(ROOT, "documents", "triage")

_STATUS_ORDER = {"fail": 0, "pending": 1, "pass": 2}


def collect_puzzle(source, pnum, con=None):
    """Return (meta, clues) for one puzzle. `clues` is every clue with its verdict, ordered
    across/down then by clue number. `meta` holds the pass/pending/fail counts."""
    own = con is None
    if own:
        con = sqlite3.connect(MASTER_DB)
    try:
        rows = con.execute(
            "SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer, "
            "       s.status, s.solved_by, s.warnings, s.template_id "
            "FROM clues c LEFT JOIN wfw_solve s ON s.clue_id = c.id "
            "WHERE c.source = ? AND c.puzzle_number = ?",
            (source, str(pnum))).fetchall()
    finally:
        if own:
            con.close()

    # across before down, then numeric clue number
    clues = sorted(
        [{"id": r[0], "number": r[1], "direction": r[2] or "", "clue_text": r[3] or "",
          "answer": r[4] or "", "status": (r[5] or "unsolved"), "solved_by": r[6] or "",
          "warnings": r[7] or "", "template_id": r[8]} for r in rows],
        key=lambda c: (0 if c["direction"] == "across" else 1,
                       int(c["number"]) if str(c["number"]).isdigit() else 9999))
    meta = {"source": source, "pnum": str(pnum), "total": len(clues),
            "pass": sum(1 for c in clues if c["status"] == "pass"),
            "pending": sum(1 for c in clues if c["status"] == "pending"),
            "fail": sum(1 for c in clues if c["status"] == "fail")}
    return meta, clues


def _wiring():
    """The same READ-ONLY wiring core.diagnose uses (predicates disabled that need AI)."""
    from core.engine_registry import make_db_wiring
    w = make_db_wiring()
    for k in ("ai_is_definition", "define_fallback", "suggest_piece", "value_check"):
        w[k] = None
    return w


def evidence_for(clue_text, answer, wiring):
    """The mechanical 'is the material present?' evidence for one clue: core.diagnose.pieces
    (definition candidates + per-word/phrase resolution flagged if it lands in the answer)."""
    from core import diagnose
    try:
        return diagnose.pieces(clue_text, answer, wiring)
    except Exception as e:  # never let one clue's drift break the whole report
        return "(diagnose.pieces failed: %s: %s)" % (type(e).__name__, e)


def _clabel(c):
    return "%s %s" % (c["number"], c["direction"])


def _diag_block(d):
    """Render one clue's Claude diagnosis (from the merged diagnoses file), or a placeholder."""
    if not d:
        return ("**Diagnosis:** _to classify — missing data / missing signature / "
                "missing engine._\n")
    out = []
    reasons = d.get("reasons") or []
    out.append("**Diagnosis:** %s" % (", ".join(reasons) if reasons else "_unclassified_"))
    if d.get("gaps"):
        out.append("\n- **Gap:** %s" % d["gaps"])
    for e in (d.get("enrichments") or []):
        parts = [f"`{k}={v}`" for k, v in e.items() if v not in (None, "")]
        out.append("\n- **Suggested enrichment:** " + " ".join(parts))
    sig = d.get("signature")
    if sig:
        out.append("\n- **Missing signature:** shape `%s` — tier **%s**%s"
                   % (sig.get("shape", "?"), sig.get("tier", "?"),
                      (" — " + sig["note"]) if sig.get("note") else ""))
    if d.get("note"):
        out.append("\n- _Note: %s_" % d["note"])
    return "".join(out) + "\n"


def render(source, pnum, meta, clues, diagnoses=None, evidence=None):
    """Full markdown report. `evidence` maps clue_id -> diagnose text; `diagnoses` maps
    str(clue_id) -> Claude's classification dict (optional)."""
    diagnoses = diagnoses or {}
    evidence = evidence or {}
    review = [c for c in clues if c["status"] in ("fail", "pending")]
    L = []
    L.append("# Triage — %s %s" % (source.title(), pnum))
    L.append("")
    L.append("**Solved:** %d pass · %d pending · %d fail  (%d clues)"
             % (meta["pass"], meta["pending"], meta["fail"], meta["total"]))
    L.append("")
    if diagnoses:
        from collections import Counter
        cats = Counter()
        for c in review:
            for r in (diagnoses.get(str(c["id"]), {}).get("reasons") or ["unclassified"]):
                cats[r] += 1
        breakdown = " · ".join("%d %s" % (n, k) for k, n in cats.most_common())
        L.append("**Why the %d unsolved clues didn't solve** (a clue may have more than one "
                 "reason): %s" % (len(review), breakdown))
        L.append("")
    L.append("_Read-only diagnostic pass. Claude diagnoses and informs only — nothing here "
             "has been written to any database, queued, or re-solved. Suggested enrichments "
             "are for a human to accept on the dashboard._")
    L.append("")

    # --- summary table ---
    L.append("## Summary")
    L.append("")
    L.append("| Clue | Answer | Status | Near-miss engine | Diagnosis |")
    L.append("|------|--------|--------|------------------|-----------|")
    for c in review:
        d = diagnoses.get(str(c["id"]))
        cls = ", ".join(d.get("reasons")) if d and d.get("reasons") else "_to classify_"
        L.append("| %s | %s | %s | %s | %s |"
                 % (_clabel(c), c["answer"], c["status"], c["solved_by"] or "—", cls))
    L.append("")

    # --- per-clue evidence + diagnosis ---
    for status in ("fail", "pending"):
        group = [c for c in review if c["status"] == status]
        if not group:
            continue
        L.append("## %s clues (%d)" % (status.upper(), len(group)))
        L.append("")
        for c in group:
            L.append("### %s — %s — \"%s\"" % (_clabel(c), c["answer"], c["clue_text"]))
            L.append("")
            L.append("- status: **%s** · near-miss engine: **%s**"
                     % (c["status"], c["solved_by"] or "—"))
            if c["warnings"]:
                L.append("- near-miss detail: %s" % c["warnings"].replace("\n", " "))
            L.append("")
            L.append("```")
            L.append(evidence.get(c["id"], "(no evidence gathered)"))
            L.append("```")
            L.append("")
            L.append(_diag_block(diagnoses.get(str(c["id"]))))
            L.append("")
    return "\n".join(L)


def build(source, pnum, diagnoses_path=None, out_path=None):
    meta, clues = collect_puzzle(source, pnum)
    review = [c for c in clues if c["status"] in ("fail", "pending")]
    wiring = _wiring()
    evidence = {c["id"]: evidence_for(c["clue_text"], c["answer"], wiring) for c in review}
    diagnoses = None
    if diagnoses_path and os.path.exists(diagnoses_path):
        with open(diagnoses_path, "r", encoding="utf-8") as f:
            diagnoses = json.load(f)
    md = render(source, pnum, meta, clues, diagnoses=diagnoses, evidence=evidence)
    if out_path is None:
        os.makedirs(TRIAGE_DIR, exist_ok=True)
        out_path = os.path.join(TRIAGE_DIR, "TRIAGE_%s_%s.md" % (source, pnum))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    return out_path, meta, evidence


# --- action layer for the /triage review page (apply is called ONLY on a human click) ------

def default_diagnoses_path(pnum):
    return os.path.join(TRIAGE_DIR, "diagnoses_%s.json" % pnum)


def load_diagnoses(pnum):
    p = default_diagnoses_path(pnum)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _rejected_path(pnum):
    return os.path.join(TRIAGE_DIR, "rejected_%s.json" % pnum)


def rejected_keys(pnum):
    p = _rejected_path(pnum)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def mark_rejected(pnum, key, undo=False):
    keys = rejected_keys(pnum)
    keys.discard(key) if undo else keys.add(key)
    os.makedirs(TRIAGE_DIR, exist_ok=True)
    with open(_rejected_path(pnum), "w", encoding="utf-8") as f:
        json.dump(sorted(keys), f)


def _classified_path(pnum):
    return os.path.join(TRIAGE_DIR, "classified_%s.json" % pnum)


def load_classified(pnum):
    """The cached DETERMINISTIC classification for a puzzle: {str(clue_id): {reasons, detail}}.
    Computed by the /triageclassify action (running core.triage_classify), not on every render —
    the classifier runs discover() per clue, which is too slow to repeat on each page load."""
    p = _classified_path(pnum)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_classified(pnum, data):
    os.makedirs(TRIAGE_DIR, exist_ok=True)
    with open(_classified_path(pnum), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)


def enrichment_key(clue_id, enr):
    """Stable identity for one suggested enrichment (for reject-state + form round-trip)."""
    return "|".join(str(x) for x in (
        clue_id, enr.get("type", ""),
        enr.get("word") or enr.get("definition") or "",
        enr.get("value") or enr.get("answer") or "",
        enr.get("indicator_type") or "", enr.get("subtype") or ""))


def enrichment_present(enr):
    """READ-ONLY: is this exact enrichment already in the reference DB? (So an accepted or
    pre-existing entry renders as done, with no Accept button.)"""
    from core import admin_db
    t = enr.get("type", "")
    con = sqlite3.connect(admin_db.CRYPTIC_DB)
    try:
        if t == "synonym":
            return bool(con.execute("SELECT 1 FROM synonyms_pairs WHERE lower(word)=lower(?) "
                        "AND upper(synonym)=upper(?)", (enr.get("word"), enr.get("value"))).fetchone())
        if t == "substitution":
            return bool(con.execute("SELECT 1 FROM wordplay WHERE lower(indicator)=lower(?) "
                        "AND upper(substitution)=upper(?)", (enr.get("word"), enr.get("value"))).fetchone())
        if t == "definition":
            return bool(con.execute("SELECT 1 FROM definition_answers_augmented WHERE "
                        "lower(definition)=lower(?) AND upper(answer)=upper(?)",
                        (enr.get("definition"), enr.get("answer"))).fetchone())
        if t == "indicator":
            return bool(con.execute("SELECT 1 FROM indicators WHERE lower(word)=lower(?) AND "
                        "wordplay_type=?", (enr.get("word"), enr.get("indicator_type"))).fetchone())
        return False
    finally:
        con.close()


def apply_enrichment(enr):
    """WRITE one enrichment to the reference DB via the SAME adders the dashboard uses (dedup
    built in). Called only from the /triageaccept route (a human click). Returns the adder's
    status string."""
    from core import admin_db
    t = enr.get("type", "")
    if t == "synonym":
        return admin_db.add_synonym(enr.get("word"), enr.get("value"))
    if t == "substitution":
        return admin_db.add_substitution(enr.get("word"), enr.get("value"))
    if t == "definition":
        return admin_db.add_definition(enr.get("definition"), enr.get("answer"))
    if t == "indicator":
        return admin_db.add_indicator(enr.get("word"), enr.get("indicator_type"),
                                      enr.get("subtype"))
    return "Unknown enrichment type: %r" % t


def wiring_form(enr):
    """Map an enrichment to the form apply_add_to_wiring() expects (so a just-added piece goes
    live without a full reload — except substitution, which apply_add_to_wiring rebuilds)."""
    t = enr.get("type", "")
    if t == "synonym":
        return {"kind": "synonym", "word": enr.get("word"), "synonym": enr.get("value")}
    if t == "definition":
        return {"kind": "definition", "definition": enr.get("definition"),
                "answer": enr.get("answer")}
    if t == "indicator":
        return {"kind": "indicator", "word": enr.get("word"), "type": enr.get("indicator_type")}
    return {"kind": "substitution"}   # -> apply_add_to_wiring falls back to reload_wiring()


def main():
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m core.triage <source> <puzzle_number> [diagnoses.json]")
        return
    source, pnum = sys.argv[1], sys.argv[2]
    diag = sys.argv[3] if len(sys.argv) > 3 else None
    out, meta, _ = build(source, pnum, diagnoses_path=diag)
    print("wrote %s" % out)
    print("counts:", meta)


if __name__ == "__main__":
    main()
