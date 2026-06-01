"""Prove the universal definition engine.

Two proofs:
  1. Pure, with a controlled fake `defines` — shows the edge split and the exact
     wordplay words/atom ids handed on, no DB needed.
  2. Real RefDB — shows it finds genuine definitions from the live reference data.

Run:  python -m core.prove_definition
"""

from core.wfw_atoms import build_wfw_atom_context
from core.definition_engine import find_definitions


def show(ctx, splits):
    print("CLUE:", ctx.clue_text, " ANSWER:", ctx.answer_text)
    if not splits:
        print("   -> no definition found")
        return
    for s in splits:
        wp = " ".join(t.text for t in s.wordplay_tokens)
        print("   def[%s]: %r   wordplay: %r" % (s.where, s.phrase, wp))


def prove_pure():
    print("=" * 60, "\nPURE (fake predicate)\n", "=" * 60, sep="")
    # Fake: 'to start' defines INSTIGATE; 'popular' does not, etc.
    known = {("to start", "INSTIGATE"), ("record", "CATALOGUE")}

    def defines(phrase, answer):
        return (phrase.lower(), answer.upper()) in known

    for clue, ans in [
        ("Popular street with one barrier to start", "INSTIGATE"),
        ("Record a go at clue being rewritten", "CATALOGUE"),
    ]:
        ctx = build_wfw_atom_context(clue, ans)
        show(ctx, find_definitions(ctx, defines))


def prove_real():
    print("\n" + "=" * 60, "\nREAL RefDB\n", "=" * 60, sep="")
    try:
        from signature_solver.db import RefDB
    except Exception as exc:
        print("   (RefDB unavailable:", exc, ")")
        return
    db = RefDB()

    def defines(phrase, answer):
        try:
            return db.is_definition_of(phrase, answer)
        except Exception:
            return False

    for clue, ans in [
        ("Cheese hidden in debriefs", "BRIE"),
        ("Quiet vessel", "SHIP"),
        ("Flower of London", "THAMES"),
    ]:
        ctx = build_wfw_atom_context(clue, ans)
        show(ctx, find_definitions(ctx, defines))


if __name__ == "__main__":
    prove_pure()
    prove_real()
