"""Render the JSON run output as a human-reviewable markdown document.

Usage:
    python -m prototypes.universal_form_v2.format_review > review.md
"""
import json
from pathlib import Path

PUZZLES = ["31132", "31138", "31150"]
RUNS = Path(__file__).resolve().parent / "runs"


def render_form(form: dict) -> str:
    """Pretty-print a form tree compactly."""
    if form is None:
        return "(no form)"
    tree = form["tree"]
    return _node_str(tree, 0)


def _node_str(node, depth):
    op = node.get("operation")
    ind = node.get("indicator")
    val = node.get("value")
    src = node.get("source_word")
    pos_kind = node.get("positional_kind")
    del_kind = node.get("deletion_kind")

    if op in ("literal", "synonym", "abbreviation", "raw", "homophone"):
        bits = [f"{op}({val!r}"]
        if src and src != val:
            bits.append(f", src={src!r}")
        bits.append(")")
        return "".join(bits)

    if op == "positional":
        return f"positional[{pos_kind}]({val!r}, src={src!r})"

    sources = node.get("sources") or []
    indent = "  " * depth
    head = op
    if ind:
        head += f" [{ind!r}]"
    if del_kind:
        head += f" kind={del_kind}"
    parts = [head]
    if sources:
        for s in sources:
            parts.append(indent + "  " + _node_str(s, depth + 1))
    return "\n".join(parts)


def render_check(chk: dict) -> str:
    mark = "+" if chk["status"] == "pass" else "X"
    detail = chk.get("detail") or ""
    if len(detail) > 120:
        detail = detail[:117] + "..."
    return f"  {mark} {chk['name']:<22s} {detail}"


def main():
    print("# v0 wrapper - per-clue review")
    print()
    print("Test bed: DT 31132 / 31138 / 31150 (92 clues, unprocessed or "
          "lightly mechanical-only).")
    print()

    # Aggregate by status across puzzles
    by_status = {"PASS": [], "FAIL": [], "NO_FORM": []}
    for p in PUZZLES:
        d = json.loads((RUNS / f"{p}.json").read_text())
        for c in d["per_clue"]:
            c["puzzle"] = p
            by_status.setdefault(c["status"], []).append(c)

    # Summary
    print("## Headline")
    print()
    print(f"- PASS: {len(by_status['PASS'])}")
    print(f"- FAIL: {len(by_status['FAIL'])}")
    print(f"- NO_FORM: {len(by_status['NO_FORM'])}")
    print()

    # PASS section - short list
    print("## PASS - cases the wrapper got right (10)")
    print()
    print("| Puzzle | Clue # | Answer | Op | Clue text |")
    print("|---|---|---|---|---|")
    for c in sorted(by_status["PASS"], key=lambda x: (x["puzzle"], x["direction"], int(x["clue_number"]))):
        op = (c["form"]["tree"]["operation"]
              if c["form"] else "?")
        clue = c["clue_text"].replace("|", "\\|")
        print(f"| {c['puzzle']} | {c['clue_number']}{c['direction'][:1]} | "
              f"{c['answer']} | {op} | {clue} |")
    print()

    # FAIL section - detailed
    print("## FAIL - wrapper built a form but verifier rejected it (36)")
    print()
    print("Each entry: clue text -> answer, the tree, and the failed checks.")
    print()
    for c in sorted(by_status["FAIL"], key=lambda x: (x["puzzle"], x["direction"], int(x["clue_number"]))):
        clue = c["clue_text"]
        print(f"### {c['puzzle']} {c['clue_number']}{c['direction']} - `{c['answer']}`")
        print()
        print(f"> {clue}")
        print()
        print("**Form:**")
        print()
        print("```")
        print(render_form(c["form"]))
        print("```")
        print()
        if c.get("verdict"):
            print("**Verifier:**")
            for chk in c["verdict"]["checks"]:
                print(render_check(chk))
            print()
        if c.get("flags"):
            print(f"**Flags:** {', '.join(sorted(set(c['flags'])))}")
            print()

    # NO_FORM - short summary
    print("## NO_FORM - solve_clue returned nothing (46)")
    print()
    print("These are clues the production solver itself can't crack - "
          "grammar_triage + catalog matchers + Haiku-enrichment all failed. "
          "Listed for awareness; widening solver coverage is the next plan.")
    print()
    print("| Puzzle | Clue # | Answer | Clue text |")
    print("|---|---|---|---|")
    for c in sorted(by_status["NO_FORM"], key=lambda x: (x["puzzle"], x["direction"], int(x["clue_number"]))):
        clue = c["clue_text"].replace("|", "\\|")
        print(f"| {c['puzzle']} | {c['clue_number']}{c['direction'][:1]} | "
              f"{c['answer']} | {clue} |")


if __name__ == "__main__":
    main()
