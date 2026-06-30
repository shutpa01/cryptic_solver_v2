"""Per-clue failure diagnostics (READ-ONLY) — codifies the manual checks we keep doing when a
clue fails: WHAT does each clue word/phrase mean to the DB, and IS the raw material to reach
the answer present? Touches no engine; only reads the wiring's predicates.

Two views:
  pieces(...)  — definition candidates + per word/phrase: synonyms / abbreviations / literal /
                 indicator types / homophones, each flagged if it appears in the answer
                 (forward or reversed) — so a missing piece vs a missing-engine gap is obvious.
  engines(...) — (part 2, separate) run each engine and report how far it got.

CLI:  python -m core.diagnose <clue_id>
"""

import sys


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _hit(value, answer):
    """How `value` relates to the answer: substring (charade/inner), reversed substring
    (reversal), or '' (only useful as a container OUTER or not at all)."""
    v = (value or "").upper().replace(" ", "")
    if not v:
        return ""
    if v in answer:
        return "in-answer"
    if v[::-1] in answer:
        return "in-answer(reversed)"
    # a container OUTER is not a substring; flag values whose letters are all present in order
    # with gaps (i.e. the answer could be this value with something inserted)
    it = iter(answer)
    if all(ch in it for ch in v):
        return "outer? (answer = value + insertion)"
    return ""


def pieces(clue_text, answer, wiring):
    """Structured 'is the material present?' report for one clue."""
    from core.wfw_atoms import build_wfw_atom_context
    ctx = build_wfw_atom_context(clue_text, answer)
    ans = _answer_letters(ctx)
    db = wiring["db"]
    defines = wiring["defines"]
    indicator_types = wiring["indicator_types"]
    words = [t.text for t in ctx.clue_tokens if t.kind == "word"]
    n = len(words)

    lines = [f"clue:   {clue_text!r}", f"answer: {answer}  ({ans})", ""]

    # --- definition candidates: confirmed defines() at either edge, 1..4 words ---
    lines.append("DEFINITION candidates (DB-confirmed defines -> answer):")
    found_def = False
    for k in range(1, min(4, n) + 1):
        head = " ".join(words[:k])
        tail = " ".join(words[n - k:])
        for label, phrase in (("start", head), ("end", tail)):
            try:
                ok = defines(phrase, answer)
            except Exception:
                ok = False
            if ok:
                lines.append(f"   [{label}] {phrase!r}  <- CONFIRMED")
                found_def = True
    if not found_def:
        lines.append("   (none — NO confirmed definition; this alone blocks every engine)")
    lines.append("")

    # --- per word/phrase resolution (1..4-word contiguous runs) ---
    lines.append("PIECE material (value [mechanism] flag), per clue run:")
    seen_runs = set()
    for L in range(1, min(4, n) + 1):
        for i in range(n - L + 1):
            phrase = " ".join(words[i:i + L])
            key = phrase.lower()
            if key in seen_runs:
                continue
            seen_runs.add(key)
            vals = []
            try:
                for s in db.get_synonyms(phrase)[:40]:
                    f = _hit(s, ans)
                    if f:
                        vals.append(f"{s} [syn] {f}")
            except Exception:
                pass
            try:
                for s in db.get_abbreviations(phrase):
                    f = _hit(s, ans)
                    if f:
                        vals.append(f"{s} [abbr] {f}")
            except Exception:
                pass
            from core import literals
            lv = literals.literal_value(phrase)
            if lv:
                f = _hit(lv, ans)
                vals.append(f"{lv} [literal]{(' '+f) if f else ''}")
            inds = set()
            try:
                inds = indicator_types(phrase) or set()
            except Exception:
                inds = set()
            if vals or inds:
                tag = ("  indicator:" + "/".join(sorted(inds))) if inds else ""
                vshow = ("  ->  " + " | ".join(vals[:8])) if vals else ""
                lines.append(f"   {phrase!r}{vshow}{tag}")
    return "\n".join(lines)


def engines(clue_text, answer, wiring):
    """Run each major engine on the clue and report how far it got: status, pieces it placed,
    clue words it left unaccounted, and its own reason/warnings. Each call is wrapped, so a
    signature drift degrades to '(could not run)' rather than breaking the report."""
    from core.wfw_atoms import build_wfw_atom_context
    import importlib
    ctx = build_wfw_atom_context(clue_text, answer)
    w = wiring
    df, dbe = w.get("define_fallback"), w.get("is_dbe")
    out = ["ENGINE attempts (status | pieces placed | unaccounted | reason):"]

    def rec(name, p):
        if p is None:
            out.append(f"   {name:26} abstained (no parse)")
            return
        pcs = ", ".join(f"{s.text}->{s.value}" for s in p.sources) or "-"
        try:
            un = ", ".join(p.unexplained_words(ctx)) or "-"
        except Exception:
            un = "?"
        warn = "; ".join(getattr(p, "warnings", []) or [])
        out.append(f"   {name:26} {p.status:7} | {pcs} | unacct: {un}"
                   + (f" | {warn}" if warn else ""))

    def mod(name):
        return importlib.import_module("core." + name)

    attempts = [
        ("hidden", lambda: mod("hidden_engine").solve_hidden(
            ctx, w["defines"], w["is_link"], w["indicator_types"])),
        ("dd", lambda: mod("dd_engine").solve_dd(
            ctx, w["defines"], w["is_link"], w["indicator_types"], is_dbe=dbe)),
        ("anagram", lambda: mod("anagram_signature_engine").solve_anagram(
            ctx, w["defines"], w["is_link"], w["indicator_types"],
            w.get("anagram_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("charade", lambda: mod("charade_signature_engine").solve_charade(
            ctx, w["defines"], w["lookup"], w["is_link"],
            w.get("charade_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("container", lambda: mod("container_signature_engine").solve_container(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            w.get("container_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("container_charade", lambda: mod("container_charade_signature_engine").solve_container_charade(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            w.get("container_charade_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("reversal", lambda: mod("reversal_signature_engine").solve_reversal(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            w.get("reversal_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("reversal_charade", lambda: mod("reversal_charade_signature_engine").solve_reversal_charade(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            w.get("reversal_charade_templates") or [], define_fallback=df, is_dbe=dbe)),
        ("reverse_charade", lambda: mod("reverse_charade_engine").solve_reverse_charade(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            define_fallback=df, is_dbe=dbe)),
        ("deletion", lambda: mod("signature_verifier").solve_deletion(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["deletion_subtypes"],
            templates=w.get("deletion_templates"), define_fallback=df, is_dbe=dbe,
            loc_rules=w.get("selection_rules"))),
        ("charade_deletion", lambda: mod("charade_deletion_engine").solve_charade_deletion(
            ctx, w["defines"], w["lookup_all"], w["is_link"], w["indicator_types"],
            w["deletion_subtypes"], define_fallback=df, is_dbe=dbe,
            loc_rules=w.get("selection_rules"))),
    ]
    for name, fn in attempts:
        try:
            rec(name, fn())
        except Exception as e:
            out.append(f"   {name:26} (could not run: {type(e).__name__}: {e})")
    return "\n".join(out)


def main():
    if len(sys.argv) < 2:
        print("usage: python -m core.diagnose <clue_id>")
        return
    import os
    import sqlite3
    cid = int(sys.argv[1])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    con = sqlite3.connect(os.path.join(root, "data", "clues_master.db"))
    row = con.execute("SELECT clue_text, answer FROM clues WHERE id=?", (cid,)).fetchone()
    con.close()
    if not row:
        print(f"no clue {cid}")
        return
    clue_text, answer = row
    from core.engine_registry import make_db_wiring
    w = make_db_wiring()
    for k in ("ai_is_definition", "define_fallback", "suggest_piece", "value_check"):
        w[k] = None
    print(pieces(clue_text, answer, w))
    print()
    print(engines(clue_text, answer, w))


if __name__ == "__main__":
    main()
