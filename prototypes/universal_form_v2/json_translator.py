"""Translator: structured_explanations.components JSON -> Form.

Reads a `structured_explanations` row, parses the components JSON,
and builds a Form per our schema. Used by the seeder (Source A path).

See `CATALOG_SEEDING_DESIGN.md` section 3a for the design and per-op
contract.

Two return shapes:
  * Success: `(form, None)` where form is a Form per `schema.py`.
  * Failure: `(None, {"kind": "translation_error", "detail": "..."})`.

The translator does not run the verifier — that's the seeder's job
once it has the Form. The translator's contract is structural
plausibility plus full word accounting (every clue word in a role
or rejected as translation_error).
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Optional, Tuple

from .schema import (
    Form, Definition, Node,
    lit, syn, abbr, raw, positional,
    charade, anagram, reversal, container, deletion, hidden,
    double_definition, acrostic, homophone_leaf,
)
from .surface import tokenize as _tokenize


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REF_DB_PATH = _PROJECT_ROOT / "data" / "cryptic_new.db"


def _lookup_definition_for_answer(answer: str, clue_text: str) -> str:
    """If structured_explanations.definition_text is missing for this
    clue, look up the answer in `definition_answers_augmented` and
    return the longest phrase that appears contiguously (case-
    insensitive) in the clue text. Empty string if no match.
    """
    answer_clean = re.sub(r"[^A-Za-z]", "", answer or "").upper()
    if not answer_clean:
        return ""
    conn = sqlite3.connect(str(_REF_DB_PATH))
    try:
        rows = conn.execute(
            "SELECT definition FROM definition_answers_augmented "
            "WHERE UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?",
            (answer_clean,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return ""
    clue_low = clue_text.lower()
    candidates = [defn for (defn,) in rows
                   if defn and defn.lower() in clue_low]
    if not candidates:
        return ""
    candidates.sort(key=lambda s: -len(s))
    return candidates[0]


def translate_components(row, db) -> Tuple[Optional[Form], Optional[dict]]:
    """Translate a structured_explanations row into a Form.

    `row` must have: clue_text, answer, components (JSON string),
        definition_text. `db` is a RefDB for lookups (currently used
        only by indicator extraction, which is added in subsequent
        translator slices).

    Returns (form, error). On success, form is set and error is None.
    On failure, form is None and error is a dict with `kind` and
    `detail`.
    """
    clue_text = row["clue_text"]
    answer = (row["answer"] or "").strip()
    components_raw = row["components"] or ""
    definition_text = (row["definition_text"] or "").strip()
    if not definition_text:
        definition_text = _lookup_definition_for_answer(answer, clue_text)

    if not clue_text or not answer or not components_raw:
        return None, _err("translation_error",
                          "missing clue_text / answer / components")

    try:
        comp = json.loads(components_raw)
    except Exception as e:
        return None, _err("translation_error",
                          f"components JSON parse failed: {e}")

    if not isinstance(comp, dict):
        return None, _err("translation_error",
                          f"components is {type(comp).__name__}, not dict")

    pieces = comp.get("ai_pieces") or []
    assembly = comp.get("assembly") or {}
    op = (assembly.get("op")
          or _first_wordplay_type(comp))
    if not op:
        return None, _err("translation_error",
                          "no op in assembly nor wordplay_type")

    answer_clean = re.sub(r"[^A-Za-z]", "", answer).upper()
    if not answer_clean:
        return None, _err("translation_error", "answer is empty after clean")

    # Dispatch per top-level op.
    if op == "charade":
        return _translate_charade(
            pieces, assembly, clue_text, answer_clean, definition_text, db)
    if op == "double_definition":
        return _translate_dd(
            pieces, assembly, clue_text, answer_clean, definition_text)
    if op == "anagram":
        return _translate_anagram(
            pieces, assembly, clue_text, answer_clean, definition_text, db)
    if op in ("hidden", "hidden_reversed"):
        return _translate_hidden(
            pieces, assembly, clue_text, answer_clean, definition_text, db,
            reversed_=(op == "hidden_reversed"))
    if op == "container":
        return _translate_container(
            pieces, assembly, clue_text, answer_clean, definition_text, db)
    if op in ("deletion", "trim"):
        return _translate_deletion(
            pieces, assembly, clue_text, answer_clean, definition_text, db)
    if op == "reversal":
        return _translate_reversal(
            pieces, assembly, clue_text, answer_clean, definition_text, db)
    if op == "homophone":
        return _translate_homophone(
            pieces, assembly, clue_text, answer_clean, definition_text, db)

    return None, _err("translation_error",
                      f"op {op!r} not yet supported in slice-1 translator")


# --- Per-op translators (slice 1) ---------------------------------------

def _merge_consecutive_duplicates(pieces: list) -> list:
    """Merge runs of consecutive ai_pieces sharing the same letters
    AND mechanism into a single compound-source piece.

    Production solve_clue can list multiple clue words as
    alternative sources for the same piece slot — e.g. "First" and
    "Lady" both → EVE for the EVE piece in EVEREST. The pieces are
    listed individually because production's verifier accepts any
    one of them; it is not saying every piece contributes
    separately to the answer.

    Our schema requires each piece to have a single source, so we
    coalesce the run into one piece whose clue_word joins the
    contributing words ("First Lady"). The verifier then looks up
    the compound phrase in synonyms_pairs as one bridge; a missing
    row becomes a precise enrichment candidate rather than silent
    over-counting.
    """
    if not pieces:
        return pieces
    merged: list = []
    cur = dict(pieces[0])
    for p in pieces[1:]:
        cur_letters = (cur.get("letters") or "").strip().upper()
        p_letters = (p.get("letters") or "").strip().upper()
        cur_mech = (cur.get("mechanism") or "").lower()
        p_mech = (p.get("mechanism") or "").lower()
        if cur_letters and cur_letters == p_letters and cur_mech == p_mech:
            cur_cw = (cur.get("clue_word") or "").strip()
            p_cw = (p.get("clue_word") or "").strip()
            cur["clue_word"] = (cur_cw + " " + p_cw).strip()
        else:
            merged.append(cur)
            cur = dict(p)
    merged.append(cur)
    return merged


def _translate_charade(pieces, assembly, clue_text, answer_clean,
                        definition_text, db):
    """Charade: ordered concatenation of leaves built from each piece."""
    if not pieces:
        return None, _err("translation_error",
                          "charade with empty ai_pieces")

    # Coalesce alternative-source pieces (see _merge_consecutive_duplicates).
    pieces = _merge_consecutive_duplicates(pieces)

    children = []
    for p in pieces:
        leaf, err = _build_leaf(p)
        if err:
            return None, err
        children.append(leaf)

    # Mechanical assembly check: child values concat to answer.
    concat = "".join((c.value or "").upper() for c in children)
    if concat != answer_clean:
        return None, _err(
            "translation_error",
            f"charade pieces concat to {concat!r}, expected {answer_clean!r}")

    # Detect runs of acrostic-mergeable leaves and rebuild them as
    # single acrostic operations. Done before attach_leaf_indicators
    # so the merged acrostic node carries its own indicator and the
    # remaining positional leaves are processed against an updated
    # consumed set.
    _detect_acrostic_runs(children, clue_text, definition_text or "", db)

    # Attach indicators to positional leaves that don't already have one.
    err = _attach_leaf_indicators(
        clue_text, definition_text or "", children, db)
    if err:
        return None, err

    tree = charade(*children)

    # Optional charade indicator (e.g. "with", "next to", "collectively
    # display"). Search for any clue phrase typed `charade` in the
    # indicators DB that's not already consumed by definition or
    # children. Attach to the charade node if found.
    charade_indicator = _find_indicator(
        clue_text, definition_text or "", children, db,
        expected_types={"charade"})
    if charade_indicator:
        tree.indicator = charade_indicator

    # Definition phrase + link words
    def_phrase = definition_text or ""
    op_indicators = [charade_indicator] if charade_indicator else []
    link_words = _residual_link_words(
        clue_text, def_phrase, children, indicators=op_indicators)
    if link_words is None:
        return None, _err(
            "translation_error",
            "leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_dd(pieces, assembly, clue_text, answer_clean,
                   definition_text):
    """Double definition: two synonym leaves both yielding the answer.

    For DDs the definition span is the contiguous clue text covering
    both halves (both halves jointly *are* the definition). The
    placeholder `definition_text='Double definition'` is not used —
    it's not in the clue.
    """
    left_def = (assembly.get("left_def") or "").strip()
    right_def = (assembly.get("right_def") or "").strip()
    if not left_def or not right_def:
        # claude_review-style empty assembly. Slice 1 rejects.
        return None, _err(
            "translation_error",
            "double_definition without left_def/right_def in assembly")

    left_leaf = syn(source_word=left_def.lower(), value=answer_clean)
    right_leaf = syn(source_word=right_def.lower(), value=answer_clean)
    tree = double_definition(left_leaf, right_leaf)

    # Definition span = the contiguous clue text from where left_def
    # starts to where right_def ends (inclusive). Both halves are
    # the definition; link words like "and" / "or" between them sit
    # inside the span and are recorded in form.link_words for
    # transparency, but the verifier's residue check sees them as
    # part of the definition span.
    clue_tokens = _tokenize(clue_text)
    left_tokens = [t.lower() for t in _tokenize(left_def)]
    right_tokens = [t.lower() for t in _tokenize(right_def)]
    clue_lower = [t.lower() for t in clue_tokens]

    left_start, left_end = _find_subseq(clue_lower, left_tokens)
    right_start, right_end = _find_subseq(clue_lower, right_tokens)
    if left_start is None or right_start is None:
        return None, _err(
            "translation_error",
            f"DD: left_def {left_def!r} or right_def {right_def!r} "
            f"not contiguously in clue")

    def_lo = min(left_start, right_start)
    def_hi = max(left_end, right_end)
    def_phrase = " ".join(clue_tokens[def_lo:def_hi])

    # Tokens between the two halves but inside def span — these are
    # link words (e.g. "and", "or" between "Cannon" and "Ball"). We
    # check they're allow-listed even though the verifier's residue
    # check won't penalise them (they're inside def span).
    inner_lo = min(left_end, right_end)
    inner_hi = max(left_start, right_start)
    inner_tokens = [t.lower() for t in clue_tokens[inner_lo:inner_hi]]

    from .clipboard_verifier import LINK_WORDS
    if any(t not in LINK_WORDS for t in inner_tokens):
        return None, _err(
            "translation_error",
            f"DD: word(s) between halves not on LINK_WORDS: {inner_tokens}")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=inner_tokens,
    ), None


def _find_subseq(haystack: list, needle: list):
    """Find first contiguous occurrence of needle inside haystack.
    Returns (start, end) or (None, None)."""
    if not needle:
        return None, None
    n = len(needle)
    for i in range(len(haystack) - n + 1):
        if haystack[i:i + n] == needle:
            return i, i + n
    return None, None


def _translate_hidden(pieces, assembly, clue_text, answer_clean,
                       definition_text, db, reversed_=False):
    """Hidden: the answer letters appear contiguously across spanning
    words. Tree shape: hidden(literal,…) — wrapped in reversal if
    hidden_reversed."""
    # The spanning words come from assembly['words'] or from the
    # single ai_pieces entry whose mechanism is 'hidden'.
    span_text = (assembly.get("words") or "").strip()
    if not span_text and pieces:
        for p in pieces:
            if (p.get("mechanism") or "").lower() == "hidden":
                span_text = (p.get("clue_word") or "").strip()
                break
    if not span_text:
        return None, _err("translation_error",
                          "hidden: no spanning words in components")

    span_tokens = _tokenize(span_text)
    if not span_tokens:
        return None, _err("translation_error",
                          "hidden: spanning words tokenise to nothing")

    # Verify the answer is actually hidden in the spanning text.
    span_letters = "".join(re.sub(r"[^A-Za-z]", "", t).upper()
                            for t in span_tokens)
    target = answer_clean[::-1] if reversed_ else answer_clean
    if target not in span_letters:
        return None, _err(
            "translation_error",
            f"hidden: {target} not contiguously in {span_letters!r}")

    # Build literal leaves over each spanning token.
    literals = [lit(source_word=t,
                     value=re.sub(r"[^A-Za-z]", "", t).upper())
                for t in span_tokens]

    # Find hidden indicator. Scan clue (minus def, minus span words)
    # for a word with type 'hidden' in indicators DB.
    indicator_word = _find_indicator(
        clue_text, definition_text, literals, db,
        expected_types={"hidden"})
    if not indicator_word:
        return None, _err(
            "translation_error",
            "hidden: no hidden indicator found in clue")

    inner = hidden(*literals, indicator=indicator_word)
    tree = reversal(inner) if reversed_ else inner

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, literals, indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "hidden: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_anagram(pieces, assembly, clue_text, answer_clean,
                        definition_text, db):
    """Anagram: literal-fodder leaves, anagram indicator extracted from
    the clue's wordplay window."""
    # Fodder source: prefer ai_pieces' anagram_fodder entries; fall
    # back to assembly['fodder'] if present.
    fodder_pieces = [p for p in pieces
                     if (p.get("mechanism") or "").lower()
                     in ("anagram_fodder", "literal")]
    if not fodder_pieces:
        # Assembly may carry fodder list directly
        ass_fodder = assembly.get("fodder") or []
        if ass_fodder:
            fodder_pieces = [{"clue_word": s, "letters": s,
                              "mechanism": "anagram_fodder"}
                             for s in ass_fodder]
    if not fodder_pieces:
        return None, _err("translation_error",
                          "anagram with no fodder pieces")

    # Build literal leaves from each fodder word.
    leaves = []
    fodder_letters = []
    for p in fodder_pieces:
        cw = (p.get("clue_word") or "").strip()
        if not cw:
            return None, _err("translation_error",
                              "anagram fodder piece missing clue_word")
        # If clue_word is multi-word, split into separate literal
        # leaves — the schema's anagram fodder integrity requires
        # literal/raw children only, and multi-word literals are
        # honest if the clue word is space-joined surface text.
        for tok in _tokenize(cw):
            value = re.sub(r"[^A-Za-z]", "", tok).upper()
            if not value:
                continue
            leaves.append(lit(source_word=tok, value=value))
            fodder_letters.append(value)

    if not leaves:
        return None, _err("translation_error",
                          "anagram fodder produced no leaves")

    # Mechanical assembly check: fodder letters anagram to answer
    if sorted("".join(fodder_letters)) != sorted(answer_clean):
        return None, _err(
            "translation_error",
            f"anagram fodder letters {''.join(fodder_letters)} "
            f"don't anagram to {answer_clean!r}")

    # Find anagram indicator. Scan clue tokens (minus def, minus leaf
    # source-words) for one with type 'anagram' in the indicators DB.
    indicator_word = _find_indicator(
        clue_text, definition_text, leaves, db,
        expected_types={"anagram"})
    if not indicator_word:
        return None, _err("translation_error",
                          "anagram: no anagram indicator found in clue")

    tree = anagram(*leaves, indicator=indicator_word)

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, leaves, indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "anagram: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_container(pieces, assembly, clue_text, answer_clean,
                          definition_text, db):
    """Container: outer wraps inner. Indicator from clue (type=container
    or insertion).

    Slice 1 handles the two-piece case: one ai_piece supplies the
    outer letters, another supplies the inner letters, and one
    surface phrase in the clue is a container/insertion indicator.
    Compound shapes (e.g. charade-with-container-piece) require a
    later slice and FAIL here as translation_error.
    """
    inner_letters = (assembly.get("inner") or "").strip().upper()
    outer_letters = (assembly.get("outer") or "").strip().upper()
    if not inner_letters or not outer_letters:
        return None, _err(
            "translation_error",
            "container: assembly missing inner or outer letters")

    # Mechanical assembly check: outer with inner inserted at some
    # position equals the answer.
    insertable = False
    for i in range(len(outer_letters) + 1):
        if (outer_letters[:i] + inner_letters
                + outer_letters[i:]) == answer_clean:
            insertable = True
            break
    if not insertable:
        return None, _err(
            "translation_error",
            f"container: outer {outer_letters!r} with inner "
            f"{inner_letters!r} inserted doesn't yield {answer_clean!r}")

    # Build a leaf for each ai_piece, then match by letters to inner /
    # outer. Any piece whose letters don't match either slot is
    # unaccounted-for — slice 1 doesn't handle compound shapes.
    inner_node: Optional[Node] = None
    outer_node: Optional[Node] = None
    unaccounted: list = []
    for p in pieces:
        leaf, err = _build_leaf(p)
        if leaf is None:
            return None, err or _err(
                "translation_error",
                "container: failed to build leaf from piece")
        if leaf.value == inner_letters and inner_node is None:
            inner_node = leaf
        elif leaf.value == outer_letters and outer_node is None:
            outer_node = leaf
        else:
            unaccounted.append(leaf.value)
    if unaccounted:
        return None, _err(
            "translation_error",
            f"container: extra piece(s) {unaccounted!r} beyond "
            f"inner/outer — compound shape not yet supported")
    if inner_node is None or outer_node is None:
        return None, _err(
            "translation_error",
            f"container: couldn't match pieces to "
            f"inner={inner_letters!r} / outer={outer_letters!r}")

    indicator_word = _find_indicator(
        clue_text, definition_text, [inner_node, outer_node], db,
        expected_types={"container", "insertion"})
    if not indicator_word:
        return None, _err(
            "translation_error",
            "container: no container/insertion indicator found in clue")

    tree = container(outer_node, inner_node, indicator=indicator_word)

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, [inner_node, outer_node],
        indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "container: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_deletion(pieces, assembly, clue_text, answer_clean,
                         definition_text, db):
    """Deletion: a single source piece (synonym/abbreviation/literal)
    has letters removed at one end (head/tail/outer/heart) to yield
    the answer. The clue carries a deletion or parts indicator.

    Slice 1 covers head, tail, outer, heart kinds — the standard
    cryptic positions. Specific-internal-letter deletion (e.g.
    LAUNCH minus the literal A naming the deleted letter) is a
    later slice and FAILs here.
    """
    source_value = (assembly.get("from") or "").strip().upper()
    if not source_value:
        # Some assemblies don't record `from`; try to read it from
        # the lone fodder piece.
        if len(pieces) == 1:
            source_value = (pieces[0].get("letters") or "").strip().upper()
    if not source_value:
        return None, _err(
            "translation_error",
            "deletion: assembly missing 'from' value and no single piece")

    if not source_value.endswith(answer_clean) and \
            not source_value.startswith(answer_clean) and \
            not (source_value[1:-1] == answer_clean) and \
            answer_clean not in source_value:
        # Bail early if the answer can't be carved out at all.
        return None, _err(
            "translation_error",
            f"deletion: source {source_value!r} doesn't contain "
            f"{answer_clean!r} as a head/tail/outer/heart subword")

    # Determine deletion kind by which subword equals the answer.
    kind: Optional[str] = None
    if source_value[len(source_value) - len(answer_clean):] == answer_clean \
            and len(source_value) > len(answer_clean):
        # Trailing portion equals answer → leading letters were dropped
        kind = "head"
    if kind is None and source_value[:len(answer_clean)] == answer_clean \
            and len(source_value) > len(answer_clean):
        kind = "tail"
    if kind is None and len(source_value) >= len(answer_clean) + 2 \
            and source_value[1:1 + len(answer_clean)] == answer_clean:
        # First and last dropped (outer letters removed)
        kind = "outer"
    if kind is None:
        # Heart: middle removed; outer letters of source equal outer
        # letters of answer
        n = len(source_value)
        if n >= len(answer_clean) + 1 and answer_clean and \
                source_value[0] == answer_clean[0] and \
                source_value[-1] == answer_clean[-1]:
            kind = "heart"
    if kind is None:
        return None, _err(
            "translation_error",
            f"deletion: {source_value!r} -> {answer_clean!r} doesn't fit "
            f"head/tail/outer/heart kinds")

    # Build the source leaf from the single fodder piece.
    fodder_pieces = [p for p in pieces
                     if (p.get("letters") or "").strip().upper() == source_value]
    if not fodder_pieces:
        return None, _err(
            "translation_error",
            f"deletion: no piece with letters={source_value!r}")
    if len(fodder_pieces) > 1:
        return None, _err(
            "translation_error",
            "deletion: multiple pieces match source value — "
            "compound shape not yet supported")
    src_leaf, err = _build_leaf(fodder_pieces[0])
    if src_leaf is None:
        return None, err or _err("translation_error",
                                  "deletion: failed to build source leaf")

    indicator_word = _find_indicator(
        clue_text, definition_text, [src_leaf], db,
        expected_types={"deletion", "parts"})
    if not indicator_word:
        return None, _err(
            "translation_error",
            "deletion: no deletion/parts indicator found in clue")

    tree = deletion(src_leaf, indicator=indicator_word, kind=kind)

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, [src_leaf], indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "deletion: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_reversal(pieces, assembly, clue_text, answer_clean,
                        definition_text, db):
    """Reversal: a single source piece's value reversed equals the answer.

    Slice 1 covers the single-piece case. The compound case
    (`reversed_parts`: charade of pieces, then whole reversed) is a
    later slice.
    """
    if assembly.get("reversed_parts"):
        return None, _err(
            "translation_error",
            "reversal of multiple pieces (reversed_parts) not yet supported")
    source_value = (assembly.get("reversed") or "").strip().upper()
    if not source_value and len(pieces) == 1:
        source_value = (pieces[0].get("letters") or "").strip().upper()
    if not source_value:
        return None, _err(
            "translation_error",
            "reversal: assembly missing 'reversed' value")
    if source_value[::-1] != answer_clean:
        return None, _err(
            "translation_error",
            f"reversal: {source_value!r} reversed != {answer_clean!r}")

    fodder_pieces = [p for p in pieces
                     if (p.get("letters") or "").strip().upper() == source_value]
    if not fodder_pieces:
        return None, _err(
            "translation_error",
            f"reversal: no piece with letters={source_value!r}")
    if len(fodder_pieces) > 1:
        return None, _err(
            "translation_error",
            "reversal: multiple pieces match — compound not yet supported")
    src_leaf, err = _build_leaf(fodder_pieces[0])
    if src_leaf is None:
        return None, err or _err("translation_error",
                                  "reversal: failed to build source leaf")

    indicator_word = _find_indicator(
        clue_text, definition_text, [src_leaf], db,
        expected_types={"reversal"})
    if not indicator_word:
        return None, _err(
            "translation_error",
            "reversal: no reversal indicator found in clue")

    tree = reversal(src_leaf, indicator=indicator_word)

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, [src_leaf], indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "reversal: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


def _translate_homophone(pieces, assembly, clue_text, answer_clean,
                         definition_text, db):
    """Homophone (op form): a single child piece yields a value that
    sounds like the answer. The clue carries a homophone indicator
    (e.g. "reportedly", "we hear", "audibly").
    """
    sounds_like = (assembly.get("sounds_like") or "").strip().upper()
    if not sounds_like and len(pieces) == 1:
        sounds_like = (pieces[0].get("letters") or "").strip().upper()
    if not sounds_like:
        return None, _err(
            "translation_error",
            "homophone: assembly missing 'sounds_like' value")

    fodder_pieces = [p for p in pieces
                     if (p.get("letters") or "").strip().upper() == sounds_like]
    if not fodder_pieces:
        return None, _err(
            "translation_error",
            f"homophone: no piece with letters={sounds_like!r}")
    if len(fodder_pieces) > 1:
        return None, _err(
            "translation_error",
            "homophone: multiple pieces match — compound not yet supported")
    child_leaf, err = _build_leaf(fodder_pieces[0])
    if child_leaf is None:
        return None, err or _err("translation_error",
                                  "homophone: failed to build child leaf")

    indicator_word = _find_indicator(
        clue_text, definition_text, [child_leaf], db,
        expected_types={"homophone"})
    if not indicator_word:
        return None, _err(
            "translation_error",
            "homophone: no homophone indicator found in clue")

    # Schema's homophone op is a non-leaf with one child.
    tree = Node(operation="homophone", indicator=indicator_word,
                sources=[child_leaf])

    def_phrase = definition_text or ""
    link_words = _residual_link_words(
        clue_text, def_phrase, [child_leaf], indicators=[indicator_word])
    if link_words is None:
        return None, _err(
            "translation_error",
            "homophone: leftover clue word(s) not on LINK_WORDS allow-list")

    return Form(
        tree=tree,
        definition=Definition(phrase=def_phrase, answer=answer_clean),
        link_words=link_words,
    ), None


# --- Leaf builders ------------------------------------------------------

def _build_leaf(piece: dict) -> Tuple[Optional[Node], Optional[dict]]:
    """Build a leaf Node from one ai_pieces entry."""
    mech = (piece.get("mechanism") or "").lower()
    cw = (piece.get("clue_word") or "").strip()
    # Some upstream parsers wrap the actual source word in quotes and
    # add explanatory commentary (e.g. '"second rate" as in B-movie').
    # If quotes are present, take the first quoted span as the source.
    m = re.search(r'["‘’“”\']([^"‘’“”\']+)["‘’“”\']', cw)
    if m:
        cw = m.group(1).strip()
    letters = (piece.get("letters") or "").strip().upper()
    if not mech or not cw or not letters:
        return None, _err("translation_error",
                          f"piece missing mechanism/clue_word/letters: "
                          f"{piece!r}")

    if mech == "synonym":
        # Honest-relationship rule: if the value is a single letter
        # matching the first letter of the source word, the
        # relationship is structurally an abbreviation (e.g. "good"
        # -> "G", "Time" -> "T"), not a synonym. Build as abbr so
        # the form's UX is truthful, regardless of how the original
        # production solver labelled it.
        cw_first = next((c for c in cw if c.isalpha()), "").upper()
        if len(letters) == 1 and cw_first == letters:
            return abbr(source_word=cw, value=letters), None
        return syn(source_word=cw, value=letters), None
    if mech == "abbreviation":
        return abbr(source_word=cw, value=letters), None
    if mech == "literal":
        # value should equal the letters of cw
        return lit(source_word=cw,
                   value=re.sub(r"[^A-Za-z]", "", cw).upper()), None
    if mech == "anagram_fodder":
        # Anagram fodder is always literal
        return lit(source_word=cw,
                   value=re.sub(r"[^A-Za-z]", "", cw).upper()), None
    if mech == "hidden":
        # Hidden fodder is literal source spanning the answer
        return lit(source_word=cw,
                   value=re.sub(r"[^A-Za-z]", "", cw).upper()), None
    if mech == "first_letter":
        return _build_first_letter_leaf(cw, letters)
    if mech == "last_letter":
        return _build_positional_leaf(cw, letters, kind="last")
    if mech == "outer_letters":
        return _build_positional_leaf(cw, letters, kind="outer")
    if mech == "middle_letters":
        return _build_positional_leaf(cw, letters, kind="middle")
    if mech == "alternate_letters":
        return _build_positional_leaf(cw, letters, kind="alternate")
    if mech == "half_letters":
        return _build_positional_leaf(cw, letters, kind="half")
    if mech == "homophone":
        return homophone_leaf(source_word=cw, value=letters), None

    return None, _err("translation_error",
                      f"mechanism {mech!r} not yet supported")


def _build_positional_leaf(cw: str, letters: str, kind: str
                            ) -> Tuple[Optional[Node], Optional[dict]]:
    """Build a positional leaf of the given kind from a piece whose
    clue_word may carry a bundled indicator (e.g. "occasionally
    loud" with letters="LU" and kind="alternate"). The source word
    is the rightmost token whose positional-extracted letters match
    the piece's letters value; remaining tokens become the leaf's
    bundled indicator phrase, to be reconciled later.
    """
    tokens = _tokenize(cw)
    if not tokens:
        return None, _err(
            "translation_error",
            f"{kind}_letters: clue_word {cw!r} tokenises to nothing")

    # Walk tokens; the source is the one whose positional extract
    # matches `letters` exactly. Prefer rightmost candidate.
    from .clipboard_verifier import _positional_extract
    candidates = []
    for i, tok in enumerate(tokens):
        bare = re.sub(r"[^A-Za-z]", "", tok).upper()
        if not bare:
            continue
        extracted = _positional_extract(bare, kind)
        if extracted == letters.upper():
            candidates.append((i, tok))
    if not candidates:
        return None, _err(
            "translation_error",
            f"{kind}_letters: no token in {cw!r} produces {letters!r}")

    idx, src_tok = candidates[-1]
    leaf = positional(source_word=src_tok, value=letters.upper(), kind=kind)
    indicator_tokens = [t for i, t in enumerate(tokens) if i != idx]
    if indicator_tokens:
        leaf.indicator = " ".join(indicator_tokens)
    return leaf, None


def _build_first_letter_leaf(cw: str, letters: str
                              ) -> Tuple[Optional[Node], Optional[dict]]:
    """Build a positional[first] leaf from a first_letter piece.

    The piece's clue_word may be either a single source word
    ('appreciate') or a phrase that bundles the indicator with the
    source word ('start to appreciate'). We identify the source word
    as the token whose first alpha letter equals the piece's letters
    field. If multiple tokens qualify, we prefer the rightmost (in
    cryptic clues the source word usually follows the indicator:
    'head of state', 'start to appreciate'). Any unused tokens form
    the leaf's bundled indicator phrase; if nothing is bundled, the
    indicator is left None and gets resolved at op-translator time
    via `_attach_leaf_indicators`.
    """
    if len(letters) != 1 or not letters.isalpha():
        return None, _err(
            "translation_error",
            f"first_letter: letters {letters!r} must be a single alpha char")

    tokens = _tokenize(cw)
    if not tokens:
        return None, _err(
            "translation_error",
            f"first_letter: clue_word {cw!r} tokenises to nothing")

    candidates = []
    for i, tok in enumerate(tokens):
        first = next((c for c in tok if c.isalpha()), "").upper()
        if first == letters:
            candidates.append((i, tok))
    if not candidates:
        return None, _err(
            "translation_error",
            f"first_letter: no token in {cw!r} starts with {letters!r}")

    # Prefer rightmost (source word usually follows indicator).
    idx, src_tok = candidates[-1]
    leaf = positional(source_word=src_tok, value=letters, kind="first")
    indicator_tokens = [t for i, t in enumerate(tokens) if i != idx]
    if indicator_tokens:
        leaf.indicator = " ".join(indicator_tokens)
    return leaf, None


# --- Residual link-word check -------------------------------------------

def _collect_consumed_tokens(node, consumed: set) -> None:
    """Recursively collect tokens from a node's indicator and (if it's
    a leaf) its source_word. Used by both `_residual_link_words` and
    `_attach_leaf_indicators` to mark surface tokens already accounted
    for by leaves and indicators across the whole subtree.

    Walks into non-leaf nodes (e.g. an `acrostic` operation produced by
    `_detect_acrostic_runs`) so the children's source words are not
    incorrectly flagged as residual."""
    if node.indicator:
        for tok in _tokenize(node.indicator):
            consumed.add(tok.lower())
    if not node.sources:
        for tok in _tokenize(node.source_word or ""):
            consumed.add(tok.lower())
    else:
        for c in node.sources:
            _collect_consumed_tokens(c, consumed)


def _residual_link_words(clue_text: str, def_phrase: str,
                          leaves: list, indicators: list,
                          extra_consumed_phrase: str = "") -> Optional[list]:
    """Compute link words = clue tokens minus def tokens minus all
    leaf source-word tokens (recursively, into non-leaf children) minus
    indicator tokens minus extra-consumed tokens.

    Returns the list of link words (lowercased) if every residual word
    is on the LINK_WORDS allow-list. Returns None if any residual is
    not allow-listed (translation_error).
    """
    from .clipboard_verifier import LINK_WORDS

    surface = [t.lower() for t in _tokenize(clue_text)]
    consumed = set()

    for tok in _tokenize(def_phrase):
        consumed.add(tok.lower())
    for tok in _tokenize(extra_consumed_phrase):
        consumed.add(tok.lower())
    for node in leaves:
        _collect_consumed_tokens(node, consumed)
    for ind in indicators:
        for tok in _tokenize(ind or ""):
            consumed.add(tok.lower())

    leftovers = [w for w in surface
                  if w not in consumed and any(c.isalpha() for c in w)]
    if any(w not in LINK_WORDS for w in leftovers):
        return None
    return leftovers


def _attach_leaf_indicators(clue_text: str, def_phrase: str,
                             leaves: list, db) -> Optional[dict]:
    """For each positional leaf without an indicator, scan the clue
    for a DB-typed indicator that matches the leaf's positional kind
    and isn't already consumed by another role (definition, leaf
    source word, leaf indicator already attached). Attach in-place.

    Returns None on success, an error dict if any positional leaf
    can't be assigned an indicator.
    """
    from .clipboard_verifier import POS_KIND_INDICATOR_TYPES

    surface = _tokenize(clue_text)
    consumed = set()
    for tok in _tokenize(def_phrase):
        consumed.add(tok.lower())
    for node in leaves:
        _collect_consumed_tokens(node, consumed)

    for leaf in leaves:
        if leaf.operation != "positional" or leaf.indicator:
            continue
        kind = leaf.positional_kind
        expected = POS_KIND_INDICATOR_TYPES.get(kind, set())
        if not expected:
            return _err(
                "translation_error",
                f"positional[{kind}]: no indicator-type mapping known")
        found = None
        for tok in surface:
            low = tok.lower()
            if low in consumed:
                continue
            types = db.get_indicator_types(low)
            if any(t[0] in expected for t in types):
                found = tok
                break
        if not found:
            return _err(
                "translation_error",
                f"positional[{kind}] (value={leaf.value!r}, "
                f"source={leaf.source_word!r}): no matching "
                f"indicator found in clue")
        leaf.indicator = found
        consumed.add(found.lower())
    return None


def _detect_acrostic_runs(children: list, clue_text: str,
                           def_phrase: str, db) -> None:
    """Detect runs of 2+ acrostic-mergeable leaves and replace each
    such run with a single `acrostic` node. Mutates `children` in
    place.

    Mergeability has two tiers:

    * **Primary** — a `positional[first]` piece (built from a
      `first_letter` JSON piece). Strong evidence the cryptic intent
      is first-letter extraction.
    * **Secondary** — an `abbreviation` or `literal` piece whose
      value happens to equal the first letter of its source word.
      Weak signal on its own — could be coincidence — but valid
      acrostic content when surrounded by primary pieces.

    Both tiers also require: value is a single alpha char equal to
    the first alpha char of the source word, AND the piece has no
    bundled indicator already attached (a bundled indicator signals
    singular wordplay context).

    A run is a contiguous span of children where every child is
    primary-or-secondary AND the span contains at least 2 primary
    pieces. The "2+ primary" anchor distinguishes BARBS-style genuine
    acrostics (4-5 first_letter pieces, optionally with one abbr like
    'British→B' folded in) from OWLET-style false positives (one
    first_letter piece sitting next to an unrelated abbr like
    'wife→W'). For each run, all pieces — primary and secondary —
    become literal children of the resulting acrostic.

    For each detected run, scans the clue for an unconsumed indicator
    typed `acrostic` or `parts` in the indicators DB. If one is found,
    builds an `acrostic(literal,...)` node carrying that indicator and
    replaces the run. If none is found, the run is left as-is and the
    downstream attach-leaf-indicators step will handle (or fail) the
    leaves individually.
    """
    def _value_matches_first_letter(leaf):
        if leaf.indicator:
            return False
        val = (leaf.value or "").strip().upper()
        if len(val) != 1 or not val.isalpha():
            return False
        src = leaf.source_word or ""
        first = next((c.upper() for c in src if c.isalpha()), "")
        return first == val

    def _is_primary_mergeable(leaf):
        return (leaf.operation == "positional"
                and leaf.positional_kind == "first"
                and _value_matches_first_letter(leaf))

    def _is_secondary_mergeable(leaf):
        return (leaf.operation in ("abbreviation", "literal")
                and _value_matches_first_letter(leaf))

    def _is_any_mergeable(leaf):
        return _is_primary_mergeable(leaf) or _is_secondary_mergeable(leaf)

    # A run is a contiguous span of any-mergeable children containing
    # at least 2 primary pieces. The 2+-primary anchor is what
    # distinguishes BARBS-style true acrostics from OWLET-style false
    # positives.
    runs = []
    i = 0
    while i < len(children):
        if not _is_any_mergeable(children[i]):
            i += 1
            continue
        j = i
        while j < len(children) and _is_any_mergeable(children[j]):
            j += 1
        primary_count = sum(1 for k in range(i, j)
                              if _is_primary_mergeable(children[k]))
        if primary_count >= 2:
            runs.append((i, j))
        i = j

    if not runs:
        return

    # Build consumed-token set from definition + every existing leaf's
    # source_word and indicator (recursively into any non-leaf nodes
    # already in `children`).
    consumed = set()
    for tok in _tokenize(def_phrase):
        consumed.add(tok.lower())
    for c in children:
        _collect_consumed_tokens(c, consumed)

    surface = _tokenize(clue_text)

    # Process runs in reverse so list slicing doesn't shift earlier
    # ranges.
    for start, end in reversed(runs):
        found = None
        for tok in surface:
            low = tok.lower()
            if low in consumed:
                continue
            types = db.get_indicator_types(low)
            type_set = {t[0] for t in types}
            # Require the strict `acrostic` type. Reject any indicator
            # also typed `alternating` (e.g. "alternately") — those
            # genuinely point to alternate-letters wordplay even
            # though the math may coincidentally work out for first
            # letters too. Loose `parts`-only indicators (e.g.
            # "regularly", "periodically") are not acrostic indicators
            # — they belong to other positional kinds.
            if "acrostic" in type_set and "alternating" not in type_set:
                found = tok
                break
        if not found:
            continue

        run_leaves = children[start:end]
        new_children = []
        joined = ""
        ok = True
        for lf in run_leaves:
            full = re.sub(r"[^A-Za-z]", "",
                           lf.source_word or "").upper()
            if not full:
                ok = False
                break
            new_children.append(lit(source_word=lf.source_word,
                                     value=full))
            joined += (lf.value or "").upper()
        if not ok:
            continue
        node = acrostic(*new_children, indicator=found, kind="first")
        node.value = joined
        children[start:end] = [node]
        consumed.add(found.lower())


# --- Helpers ------------------------------------------------------------

def _err(kind: str, detail: str) -> dict:
    return {"kind": kind, "detail": detail}


def _find_indicator(clue_text: str, def_phrase: str, leaves: list,
                     db, expected_types: set) -> Optional[str]:
    """Find an indicator phrase in the clue of the given expected
    types (e.g. {'anagram'} or {'hidden'} or {'deletion','parts'}).

    Searches contiguous spans of unconsumed clue tokens (longest
    first) for any phrase whose indicators-DB type matches
    `expected_types`. Multi-word indicators of any length are
    accepted — there is no arbitrary cap. Returns the matched phrase
    (in the clue's surface case) or None if no span qualifies.
    """
    surface = _tokenize(clue_text)
    consumed_words = set()
    for tok in _tokenize(def_phrase):
        consumed_words.add(tok.lower())
    for leaf in leaves:
        for tok in _tokenize(leaf.source_word or ""):
            consumed_words.add(tok.lower())

    n = len(surface)
    is_consumed = [surface[i].lower() in consumed_words for i in range(n)]

    # Try every contiguous span (start, end) where no token is
    # consumed. Longest-first across all valid spans.
    spans = []
    for start in range(n):
        if is_consumed[start]:
            continue
        for end in range(start + 1, n + 1):
            if is_consumed[end - 1]:
                break
            spans.append((start, end))
    spans.sort(key=lambda se: -(se[1] - se[0]))

    for start, end in spans:
        phrase = " ".join(surface[start:end])
        types = db.get_indicator_types(phrase.lower())
        if any(t[0] in expected_types for t in types):
            return phrase
    return None


def _first_wordplay_type(comp: dict) -> Optional[str]:
    wt = comp.get("wordplay_type")
    if isinstance(wt, str) and wt:
        return wt
    wts = comp.get("wordplay_types")
    if isinstance(wts, list) and wts:
        return wts[0]
    return None
