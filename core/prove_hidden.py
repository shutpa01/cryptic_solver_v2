"""Prove the hidden engine wired to the universal definition engine.

Shows, for each clue: the definition split, the host word(s), each answer letter
mapped to the EXACT clue character, every other word accounted for, and that the
answer is never found hiding inside its own definition.

Run:  python -m core.prove_hidden
"""

from core.wfw_atoms import build_wfw_atom_context
from core.hidden_engine import solve_hidden


def atom_index(ctx, atom_id):
    for a in ctx.clue_atoms:
        if a.atom_id == atom_id:
            return a.index, a.char
    return None, None


def show(db_defines, clue, answer, indicator_types=None, is_link=None):
    ctx = build_wfw_atom_context(clue, answer)
    parse = solve_hidden(ctx, db_defines, indicator_types=indicator_types,
                         is_link=is_link)
    print("=" * 66)
    print("CLUE:   %s" % clue)
    print("ANSWER: %s" % answer)
    if parse is None:
        print("  -> not hidden")
        return None
    print("  operation:", parse.operation, " complete:", parse.is_complete())
    print("  definition:", parse.definition.text if parse.definition else "(none)")
    print("  host word(s):", parse.sources[0].text)
    for link in parse.links:
        idx, ch = atom_index(ctx, link.clue_atom_id)
        ans_letter = parse.answer_letters()[link.answer_pos - 1]
        print("    answer %d (%s)  <-  clue char %d (%s)"
              % (link.answer_pos, ans_letter, idx, ch))
    if parse.annotations:
        for a in parse.annotations:
            print("  %s: %r (%s)" % (a.role, a.text, a.note))
    # completeness of accounting: every clue word has a role
    print("  unexplained words:", parse.unexplained_words(ctx))
    return parse


if __name__ == "__main__":
    from signature_solver.db import RefDB
    db = RefDB()

    def defines(phrase, answer):
        try:
            return db.is_definition_of(phrase, answer)
        except Exception:
            return False

    def indicator_types(word):
        try:
            return {t for t, _, _ in db.get_indicator_types(word)}
        except Exception:
            return set()

    def is_link(word):
        try:
            return db.is_link_word(word)
        except Exception:
            return False

    show(defines, "Cheese hidden in debriefs", "BRIE", indicator_types, is_link)
    # 'in' should come back as a LINK word, 'hidden' as a hidden INDICATOR —
    # by lookup, not by elimination.
    show(defines, "Some cooks are dishonest", "OKSARE", indicator_types, is_link)
    show(defines, "Vessel concealed by carthorse", "SHIP", indicator_types, is_link)
