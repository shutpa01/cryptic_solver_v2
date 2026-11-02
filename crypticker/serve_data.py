"""Extract Crypticker's servable per-clue data by REUSING the live WFW logic.

No tile logic of our own: we load the exact Parse the clue page renders
(core.store.load_parse) and cut/colour tiles the same way core.wfw_render does
— walk the answer letters, group by each letter's source (via links), colour by
source index from the shared PALETTE. Read-only.
"""
import sqlite3, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.store import load_parse
from core.wfw_render import PALETTE, _first_index

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clues_master.db")


def atoms_for(parse):
    """Tiles for one Parse, using the WFW link-walk. Returns tiles in CLUE order
    plus the assembly order that rebuilds the answer."""
    letters = parse.answer_letters()
    by_pos = {l.answer_pos: l.source_index for l in parse.links}

    # 1. group consecutive same-source answer letters into runs (answer order)
    runs = []  # each: {text, si, first_pos}
    for pos in range(1, len(letters) + 1):
        si = by_pos.get(pos)
        if runs and runs[-1]["si"] == si:
            runs[-1]["text"] += letters[pos - 1]
        else:
            runs.append({"text": letters[pos - 1], "si": si, "first_pos": pos})

    # 2. clue-reading order of each source (same key wfw_render sorts rows by)
    def clue_order(si):
        if si is None or si >= len(parse.sources):
            return 1_000_000
        return _first_index(parse.sources[si].clue_atom_ids)

    # 3. atoms in clue order; split-word parts tie-break by answer position
    order = sorted(range(len(runs)),
                   key=lambda r: (clue_order(runs[r]["si"]), runs[r]["first_pos"]))
    atoms = []
    for r in order:
        run = runs[r]
        si = run["si"]
        src = parse.sources[si] if si is not None and si < len(parse.sources) else None
        atoms.append({
            "true": run["text"],
            "colour": PALETTE[si % len(PALETTE)][0] if si is not None else "#94a3b8",
            "source_word": src.text if src else "?",
            "mechanism": src.mechanism if src else "?",
            "source_index": si,
        })
    # 4. assembly order = the runs in answer order, mapped to their atom index
    run_to_atom = {r: i for i, r in enumerate(order)}
    assembly_order = [run_to_atom[i] for i in range(len(runs))]
    return atoms, assembly_order


def show(conn, clue_id):
    parse = load_parse(conn, clue_id)
    if parse is None:
        print("  (no parse for %s)" % clue_id); return
    atoms, order = atoms_for(parse)
    d = parse.definition.text if parse.definition else "(none)"
    print("clue_id %s  op=%s" % (clue_id, parse.operation))
    print("  clue  :", parse.clue_text)
    print("  answer:", parse.answer_text, " def:", d)
    print("  TILES (clue order):")
    for i, a in enumerate(atoms):
        print("    [%d] %-6s  %-9s from '%s'  colour %s"
              % (i, a["true"], a["mechanism"], a["source_word"], a["colour"]))
    built = "".join(atoms[i]["true"] for i in order)
    print("  assembly_order:", order, " -> rebuilds:", built,
          "OK" if built == "".join(parse.answer_letters()) else "MISMATCH")
    print()


def main():
    conn = sqlite3.connect(DB)
    # one real example of each whitelisted shape
    ids = {}
    for op in ("charade", "container", "container_charade"):
        row = conn.execute(
            "select clue_id from wfw_solve where status='pass' and operation=? "
            "and length(replace(replace(answer_text,' ',''),'-','')) between 5 and 8 "
            "limit 1", (op,)).fetchone()
        if row:
            ids[op] = row[0]
    for op, cid in ids.items():
        show(conn, cid)


if __name__ == "__main__":
    main()
