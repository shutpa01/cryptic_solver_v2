"""Substrate-persistence test — pins the design rule that a solve is not preserved
until it is on disk, against the CURRENT wfw_model.Parse.

SOLVER_REDESIGN.md §2 "preserve all letter-contributing evidence, even on failure";
§10 the provenance "is the WFW substrate persisted ... nothing may live only in a
Python object that vanishes when the call returns."

Writes with one connection, CLOSES it, then reads back with a FRESH connection —
so it passes only on real durable storage, and checks the Parse round-trips intact.

Run:  python -m core.test_substrate
"""

import os
import tempfile

from core.wfw_atoms import build_wfw_atom_context
from core.hidden_engine import find_hidden
from core import store


def test_hidden_parse_is_durably_persisted():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    clue_id = 999
    try:
        # NEAR is hidden across two words: di[NE AR]ound.
        ctx = build_wfw_atom_context("dine around", "NEAR")
        parse = find_hidden(ctx)
        assert parse is not None, "expected a hidden solve for NEAR in 'dine around'"

        # Persist, then CLOSE the writing connection entirely.
        w = store.connect(db_path)
        store.save_parse(w, clue_id, parse)
        w.close()

        # Read back with a brand-new connection — proves it is on disk.
        r = store.connect(db_path)
        loaded = store.load_parse(r, clue_id)
        r.close()

        assert loaded is not None, "nothing was persisted"
        # Every answer letter has a preserved link (§2/§3.2: every slot sourced).
        assert sorted(l.answer_pos for l in loaded.links) == [1, 2, 3, 4]
        assert loaded.sources and loaded.sources[0].value == "NEAR"
        assert loaded.operation == "hidden"
        assert loaded.clue_text == "dine around" and loaded.answer_text == "NEAR"
        # The exact clue character behind each answer letter survived.
        assert all(l.clue_atom_id for l in loaded.links)
    finally:
        os.remove(db_path)


if __name__ == "__main__":
    test_hidden_parse_is_durably_persisted()
    print("PASS: hidden Parse durably persisted and round-trips intact "
          "(design §2 + §10)")
