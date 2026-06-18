"""Token definitions for the signature system."""


# Core fodder roles
SYN_F = "SYN_F"        # synonym fodder — look up synonym in DB
ABR_F = "ABR_F"        # abbreviation fodder — look up abbreviation in DB
ANA_F = "ANA_F"        # anagram fodder — raw letters to rearrange
RAW = "RAW"            # raw word used as-is (its own letters)
HID_F = "HID_F"        # hidden word fodder — answer spans these words
HOM_F = "HOM_F"        # homophone fodder — sounds like answer/component
DEL_F = "DEL_F"        # deletion fodder — what gets removed
POS_F = "POS_F"        # positional fodder — word from which letters extracted

# Indicator tokens
ANA_I = "ANA_I"        # anagram indicator
REV_I = "REV_I"        # reversal indicator
CON_I = "CON_I"        # container indicator
DEL_I = "DEL_I"        # deletion indicator
HID_I = "HID_I"        # hidden word indicator
HOM_I = "HOM_I"        # homophone indicator

# Positional indicators — what survives
POS_I_FIRST = "POS_I_FIRST"              # keep first letter
POS_I_LAST = "POS_I_LAST"                # keep last letter
POS_I_OUTER = "POS_I_OUTER"              # keep outer letters
POS_I_MIDDLE = "POS_I_MIDDLE"            # keep middle letter(s)
POS_I_ALTERNATE = "POS_I_ALTERNATE"      # keep alternate letters

# Positional indicators — what gets removed
POS_I_TRIM_FIRST = "POS_I_TRIM_FIRST"    # remove first letter
POS_I_TRIM_LAST = "POS_I_TRIM_LAST"      # remove last letter
POS_I_TRIM_MIDDLE = "POS_I_TRIM_MIDDLE"  # remove middle letter(s)
POS_I_TRIM_OUTER = "POS_I_TRIM_OUTER"    # remove first and last
POS_I_HALF = "POS_I_HALF"                # take half the letters

# Structural tokens
DEF = "DEF"            # definition
LNK = "LNK"            # link word (ignorable)
DBE_MARKER = "DBE_MARKER"  # definition-by-example marker (maybe, perhaps, say, ...)

# Whole-clue types (no wordplay window)
DOUBLE_DEFINITION = "DOUBLE_DEFINITION"
CRYPTIC_DEFINITION = "CRYPTIC_DEFINITION"
AND_LIT = "AND_LIT"

# Rare operations
SUB = "SUB"            # letter substitution

# --- Indicator type mappings ---
# Maps DB indicator wordplay_type values to our indicator tokens
INDICATOR_TYPE_TO_TOKEN = {
    "anagram": ANA_I,
    "reversal": REV_I,
    "container": CON_I,
    "insertion": CON_I,      # insertion = container (same operation)
    "deletion": DEL_I,
    "hidden": HID_I,
    "homophone": HOM_I,
    "acrostic": POS_I_FIRST,
    "alternating": POS_I_ALTERNATE,
    "selection": POS_I_FIRST,  # selection ≈ acrostic
}

# Maps (wordplay_type, subtype) to specific positional token for 'parts' indicators.
# When wordplay_type is 'parts', use the subtype to pick the correct token.
PARTS_SUBTYPE_TO_TOKEN = {
    "first_use": POS_I_FIRST,
    "last_use": POS_I_LAST,
    "outer_use": POS_I_OUTER,
    "center_use": POS_I_MIDDLE,
    "inner_use": POS_I_MIDDLE,
    "first_delete": POS_I_TRIM_FIRST,
    "last_delete": POS_I_TRIM_LAST,
    "tail_delete": POS_I_TRIM_LAST,
    "outer_delete": POS_I_TRIM_OUTER,
    "center_delete": POS_I_TRIM_MIDDLE,
    "alternate": POS_I_ALTERNATE,
    "odd": POS_I_ALTERNATE,
    "even": POS_I_ALTERNATE,
    "outer": POS_I_OUTER,
    "last": POS_I_LAST,
    "last letter": POS_I_LAST,
    "pattern": POS_I_ALTERNATE,
}

# Common link words that connect definition to wordplay or pieces to each other.
# These NOW LIVE IN THE DB — the link_words table in data/cryptic_new.db — so the
# list is editable as data, not code (migrated by core/_migrate_link_words.py). Edit
# the table (e.g. remove "over", which is really an indicator) and restart; no code
# change. LiveDB.is_link_word / RefDB.is_link_word read the table directly; this
# module attribute is loaded from the same table for any remaining importer.
def _load_link_words():
    """The link words, read from the link_words table. Empty set if the table/DB is
    unavailable (callers then simply find no link words — never a crash at import)."""
    import os
    import sqlite3
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "cryptic_new.db")
    try:
        conn = sqlite3.connect(path, timeout=30)
        try:
            return {(w or "").lower().strip()
                    for (w,) in conn.execute("SELECT word FROM link_words")
                    if w and w.strip()}
        finally:
            conn.close()
    except sqlite3.Error:
        return set()


LINK_WORDS = _load_link_words()

# Definition-by-example markers. When one of these is adjacent to a clue
# word, the cryptic convention is that the word stands for an EXAMPLE in a
# category — so the word's RAW letters cannot be used as a wordplay piece;
# only synonyms / category-mates count. Source: data/cryptic_new.db
# wordplay table where category='dbe', plus the conventional 'maybe' and
# 'possibly' which crossword setters use the same way.
DBE_MARKERS_SINGLE = frozenset({
    'maybe', 'perhaps', 'possibly', 'say', 'example',
})
# Multi-word DBE markers (lowercase tuples).
DBE_MARKERS_MULTI = (
    ('for', 'example'),
    ('for', 'one'),
    ('for', 'instance'),
    ('in', 'particular'),
)

# All fodder tokens (contribute letters to the answer)
FODDER_TOKENS = {SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F}

# All indicator tokens (signal operations but don't contribute letters)
INDICATOR_TOKENS = {
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
}
