"""Indicator enrichment — queue a provisional hidden indicator for verification.

The hidden engine reads a missing indicator off the clue's own leftover words (the
grammatically-bound phrase) and flags it source='pending'. This module is the one
step that routes such a phrase to the live enrichment queue for human review, or —
if a reviewer already rejected it — drops it and fails the parse honestly. Nothing
is written to the reference table here; Accept in the dashboard does that.
"""


_NOTE_TYPES = ("anagram", "container", "reversal", "deletion", "hidden",
               "insertion", "homophone", "acrostic")


def _wp_type(note):
    """The wordplay type a provisional indicator should be queued/checked under, read
    off its note ("anagram indicator" -> 'anagram', "container indicator" -> ...).
    Defaults to 'hidden' for an unlabelled note."""
    n = (note or "").lower()
    for t in _NOTE_TYPES:
        if t in n:
            return "container" if t == "insertion" else t
    return "hidden"


def finalize_indicators(parse, ctx, store, source=None, puzzle_number=None):
    """For each PROVISIONAL indicator on the final parse: queue its phrase for
    verification UNDER ITS OWN TYPE (anagram/container/reversal/... not always hidden),
    or drop it (and fail) if a reviewer already rejected it."""
    if store is None or parse is None:
        return
    for ann in list(parse.annotations):
        if ann.role != "indicator" or getattr(ann, "source", "db") != "pending":
            continue
        wp_type = _wp_type(getattr(ann, "note", ""))
        if store.is_rejected_indicator(ann.text, wp_type):
            parse.annotations.remove(ann)
            msg = "%s indicator %r was rejected by a reviewer" % (wp_type, ann.text)
            if msg not in parse.warnings:
                parse.warnings = list(parse.warnings) + [msg]
            parse.status = "fail"
        else:
            store.queue_indicator(ann.text, parse.answer_text, ctx.clue_text,
                                  wp_type, source, puzzle_number)
