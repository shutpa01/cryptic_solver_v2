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
# STRICT: only words that are NEVER indicators or fodder belong here.
# Do NOT add: "up/back/out/about" (indicators), "some" (hidden ind),
# "say" (hom ind), "not" (del ind), "into/put/has/have/had" (con ind),
# "one" (abbreviation), "may/will" (ambiguous).
LINK_WORDS = {
    # Prepositions / conjunctions
    "for", "with", "in", "to", "of", "from", "by", "after", "before",
    "on", "at", "and", "but", "then", "that", "or",
    "under", "over", "above", "below", "beside", "between",
    "through", "into", "upon", "across", "among",
    # Articles
    "a", "an", "the",
    # Auxiliary verbs / copula
    "is", "be", "being", "been", "was", "were", "are",
    # Pronouns / demonstratives
    "it", "its", "this", "that's", "thats",
    # Conditionals / connectives
    "if", "once", "while",
    # Conjunctions / adverbs
    "as", "so", "when", "where", "yet", "here", "thus", "hence", "which",
    # Result words (already had most)
    "giving", "making", "producing", "getting", "providing", "creating",
    "showing", "becoming", "yielding",
    # Possibility
    "perhaps", "maybe", "possibly",
    # Words that are also indicators but frequently serve as links
    # (both roles coexist — matcher picks whichever fits the pattern)
    "wanting", "needing", "requiring",
}

# All fodder tokens (contribute letters to the answer)
FODDER_TOKENS = {SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F}

# All indicator tokens (signal operations but don't contribute letters)
INDICATOR_TOKENS = {
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
}
