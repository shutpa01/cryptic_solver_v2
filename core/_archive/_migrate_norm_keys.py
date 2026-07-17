"""One-time, additive migration: give each reference table a normalized-key column
(populated by the SAME signature_solver.db._normalize_key the preload uses) and index
it, so LiveDB's `WHERE norm_key=?` is faithful to RefDB AND fast.

Reversible: only ADDs columns/indexes (cryptic_new.db backed up to
cryptic_new.db.bak-prenorm-20260610). RefDB ignores the new columns; only LiveDB uses them.
"""
import os
import sqlite3
from signature_solver.db import _normalize_key

DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cryptic_new.db")

# table, key column, new normalized column, index name
SPECS = [
    ("synonyms_pairs", "word", "norm_word", "idx_sp_norm"),
    ("definition_answers_augmented", "definition", "norm_def", "idx_daa_norm"),
    ("wordplay", "indicator", "norm_ind", "idx_wp_norm"),
    ("indicators", "word", "norm_word", "idx_ind_norm"),
    ("homophones", "word", "norm_word", "idx_homo_norm"),
]


def main():
    conn = sqlite3.connect(DB, timeout=60)
    conn.create_function("pynorm", 1, lambda s: _normalize_key(s or ""))
    for table, key, norm, idx in SPECS:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
        if norm not in cols:
            conn.execute("ALTER TABLE %s ADD COLUMN %s TEXT" % (table, norm))
        n = conn.execute("UPDATE %s SET %s = pynorm(%s)" % (table, norm, key)).rowcount
        conn.execute("CREATE INDEX IF NOT EXISTS %s ON %s(%s)" % (idx, table, norm))
        conn.commit()
        print(f"{table}: populated {norm} ({n} rows), index {idx} ready", flush=True)
    # quick proof the gap is closed: 'action?' is now reachable by normalized key 'action'
    r = conn.execute("SELECT answer FROM definition_answers_augmented "
                     "WHERE norm_def=? AND UPPER(answer)='CASE'", ("action",)).fetchall()
    print("verify action->CASE via norm_def:", r)
    conn.close()


if __name__ == "__main__":
    main()
