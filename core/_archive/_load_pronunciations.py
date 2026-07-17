"""One-off loader: CMUdict -> cryptic_new.db `pronunciations` table.

Loads the public-domain CMU Pronouncing Dictionary (data/resources/cmudict.dict)
into a new `pronunciations(word, phonemes, norm_word)` table. Phonemes are stored
RAW (ARPABET with stress digits, exactly as CMU ships them) so the homophone
engine can derive whatever normalised / non-rhotic key it needs without a reload.

Multiple rows per word are kept (CMU alternate pronunciations: word(2), word(3)).
Comments (# ...) and the (N) variant suffix are stripped from the word token.

Idempotent guard: if the table already has rows it refuses to run, so a re-run
cannot duplicate or silently clobber. To force a reload, drop the table first
(with the user's explicit confirmation).

Run:  python -m core._load_pronunciations
"""
import os
import re
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "data", "cryptic_new.db")
DICT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "resources", "cmudict.dict")

# Same key normalisation the rest of the solver uses (signature_solver/db.py).
def _normalize_key(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip()).strip()

_VARIANT = re.compile(r"\(\d+\)$")


def parse(path):
    """Yield (word, phonemes, norm_word) for each dict line."""
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].strip()   # drop trailing comment
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            token = _VARIANT.sub("", parts[0])     # strip (2)/(3) variant suffix
            phonemes = " ".join(parts[1:]).strip()
            norm = _normalize_key(token)
            if not norm or not phonemes:
                continue
            yield token, phonemes, norm


def main():
    if not os.path.exists(DICT_PATH):
        raise SystemExit(f"CMUdict not found at {DICT_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS pronunciations (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   word TEXT NOT NULL,
                   phonemes TEXT NOT NULL,
                   norm_word TEXT
               )"""
        )
        existing = conn.execute("SELECT COUNT(*) FROM pronunciations").fetchone()[0]
        if existing:
            raise SystemExit(
                f"pronunciations already has {existing:,} rows — refusing to reload. "
                "Drop the table first (with confirmation) to force a reload."
            )

        rows = list(parse(DICT_PATH))
        conn.executemany(
            "INSERT INTO pronunciations (word, phonemes, norm_word) VALUES (?, ?, ?)",
            rows,
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pron_norm ON pronunciations(norm_word)"
        )
        conn.commit()

        n_words = conn.execute(
            "SELECT COUNT(DISTINCT norm_word) FROM pronunciations"
        ).fetchone()[0]
        print(f"Loaded {len(rows):,} pronunciation rows "
              f"({n_words:,} distinct words) into {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
