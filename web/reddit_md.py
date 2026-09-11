"""The WFW card, reformatted as a Reddit comment.

The user answers clue-writing threads on Reddit (2026-09-10) and those forums
take no screenshots, so the card has to travel as text.

THIS IS A REFORMATTER, NOT A SECOND RENDERER. Its input is the card HTML that
`core.wfw_card.stored_card` already produces — the very thing on the WFW page —
and it does nothing but change the markup. Every judgement was made upstream:

    <div class="wfw-clue">    the clue, with <span class="wfw-lit"> on the
                              lit letters          ->  **bold**
    <div class="wfw-tiles">   the answer tiles     ->  >!spoiler!<
    <div class="wfw-build">   the one-line summary ->  a quote line
    <div class="wfw-row">     wfw-role + wfw-content
                              (DEFINITION | Fruit) ->  - **Definition** — Fruit

An earlier attempt built this from `web.wfw_read.load_breakdown` instead — the
overlay's own re-derivation — and immediately had to re-derive the hidden-letter
lighting and rearrange the pills with regexes to get back to what the card
already said. Reformat the card; never rebuild it. (User, 2026-09-10: "all you
should be doing is reformatting the WFW page".)

The solver chips the public site strips — the PASS badge, the engine tag, the
provenance chip — are dropped here for the same reason web/serving.py strips
them: they are review furniture, and a Reddit reader has no use for them.
"""

import re
from html.parser import HTMLParser

# Reddit's spoiler syntax is >!text!< — the WHOLE comment goes inside one
# (user, 2026-09-10), not just the answer: the forums require a solution to be
# spoilered, and the wordplay gives the answer away as surely as the answer does.
# See _spoiler_block for what Reddit will and will not let a spoiler span.
#
# A markdown list would break the spoiler (each item is its own block), so the
# rows are prefixed with a literal bullet character instead, which is just text.
BULLET = "•"

# Review furniture: dropped, exactly as web/serving._INTERNAL_CHIPS drops it.
_SKIP_CLASSES = ("wfw-engine", "wfw-verdict", "wfw-prov")

# Markdown metacharacters that would otherwise change the rendered text. Narrow
# on purpose: a clue's own punctuation is part of the clue and survives verbatim.
_MD_SPECIAL = re.compile(r"([\\`*_\[\]~])")


def esc(text):
    return _MD_SPECIAL.sub(r"\\\1", text or "")


class _Card(HTMLParser):
    """Pulls the card's parts out of its HTML. One pass, no dependencies.

    Text is collected per section; `wfw-lit` toggles a bold marker so the hidden
    run the card lights survives into Markdown, which is the one thing a
    screenshot was really carrying.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.type_label = ""
        self.clue = []
        self.tiles = []
        self.build = []
        self.rows = []              # [[role, content], ...]
        self._sec = None            # which section we are inside
        self._depth = 0             # nesting depth of the current section
        self._skip = 0              # inside a chip we drop
        self._cell = None           # "role" | "content" while in a row

    # -- helpers ------------------------------------------------------------
    def _classes(self, attrs):
        return dict(attrs).get("class", "").split()

    def handle_starttag(self, tag, attrs):
        cls = self._classes(attrs)
        if self._skip:
            self._skip += 1
            return
        if any(c in _SKIP_CLASSES for c in cls):
            self._skip = 1
            return
        if "wfw-lit" in cls:
            self.clue.append("\x01")            # bold toggle, resolved later
            # Count its depth like any other tag: this span's </span> WILL
            # decrement, and skipping the increment closed the clue section at
            # the very first lit letter (the clue truncated to "P**i**").
            self._depth += 1
            return
        if "wfw-type" in cls:
            self._sec, self._depth = "type", 1
            return
        for name, sec in (("wfw-clue", "clue"), ("wfw-tiles", "tiles"),
                          ("wfw-build", "build")):
            if name in cls:
                self._sec, self._depth = sec, 1
                return
        if "wfw-row" in cls:
            self.rows.append(["", ""])
            self._sec, self._depth = "row", 1
            return
        if self._sec == "row":
            if "wfw-role" in cls:
                self._cell = "role"
            elif "wfw-content" in cls:
                self._cell = "content"
        if self._sec:
            self._depth += 1

    def handle_endtag(self, tag):
        if self._skip:
            self._skip -= 1
            return
        if self._sec:
            self._depth -= 1
            if self._depth <= 0:
                self._sec, self._cell = None, None

    def handle_data(self, data):
        if self._skip or not self._sec:
            return
        if self._sec == "type":
            self.type_label += data
        elif self._sec == "clue":
            self.clue.append(data)
        elif self._sec == "tiles":
            self.tiles.append(data)
        elif self._sec == "build":
            self.build.append(data)
        elif self._sec == "row" and self._cell:
            i = 0 if self._cell == "role" else 1
            self.rows[-1][i] += data


def _bold_runs(parts):
    """Join the clue's pieces, bolding the lit run.

    The card emits ONE <span class="wfw-lit"> per letter, and the space between
    two host words is not itself lit — so bolding each span separately would
    give doct**o****r** **a**nge, not doct**or ange**red. The card SHOWS one
    continuous highlight, so the markdown marks one continuous run: from the
    first lit letter to the last.
    """
    chunks, lit_next, flags = [], False, []
    for part in parts:
        if part == "":
            lit_next = True
            continue
        chunks.append(part)
        flags.append(lit_next)
        lit_next = False
    lit_idx = [i for i, f in enumerate(flags) if f]
    if not lit_idx:
        return "".join(chunks)
    first, last = lit_idx[0], lit_idx[-1]
    out = []
    for i, chunk in enumerate(chunks):
        if i == first:
            out.append("**")
        out.append(chunk)
        if i == last:
            out.append("**")
    return "".join(out)


def build_markdown(card_html):
    """The card HTML as a Reddit comment."""
    p = _Card()
    p.feed(card_html)

    clue = _bold_runs(p.clue).strip()
    # The enumeration rides along inside wfw-clue, so it is already in `clue`.
    # The card draws a hyphenated answer's break as an en-dash tile
    # (core/wfw_render's wfw-tile-sep). In running text a solver writes SO-SO,
    # so the tile glyph becomes the hyphen it stands for.
    answer = "".join(p.tiles).strip().replace("–", "-").replace("—", "-")
    summary = re.sub(r"\s+", " ", "".join(p.build)).strip()
    label = _sentence_case(re.sub(r"\s+", " ", p.type_label).strip())

    lines = []
    if clue:
        # The clue is NOT bolded whole — its lit run already carries bold, and
        # two overlapping bolds do not nest in Markdown.
        lines.append(_esc_keeping_bold(clue))
    if answer:
        lines.append("Answer: **%s**" % esc(answer))
    if summary:
        lines.append(esc(summary))
    for role, content in p.rows:
        role = re.sub(r"\s+", " ", role).strip()
        content = re.sub(r"\s+", " ", content).strip()
        if not (role or content):
            continue
        lines.append("%s **%s** — %s"
                     % (BULLET, esc(_sentence_case(role)), esc(content)))
    if label:
        lines.append("*%s*" % esc(label))
    return _spoiler_block(lines)


def _spoiler_block(lines):
    """The whole comment inside ONE Reddit spoiler (user, 2026-09-10: "we have
    to post it within a spoiler").

    TWO THINGS REDDIT FORCES, and they are why this is not simply ">!" around
    the earlier layout:

    1. A spoiler cannot cross a BLANK LINE — a blank line starts a new
       paragraph and the spoiler ends with the old one, leaving the rest of the
       comment in plain sight. So the block carries no blank lines at all.
    2. A spoiler cannot cross a markdown LIST — each item is its own block. So
       the "- " bullets become a literal bullet character, which is text and
       stays inside the spoiler.

    Lines are joined with a HARD line break (two trailing spaces): that keeps
    one paragraph, so the spoiler spans it, while still breaking the line on
    old Reddit, where a bare newline would otherwise run every line together.

    A literal "!<" in the content would close the spoiler early, so any is
    broken with a zero-width space. Nothing else about the text changes.
    """
    body = "  \n".join(line.replace("!<", "!​<") for line in lines if line)
    return ">!" + body + "!<\n" if body else ""


def _sentence_case(text):
    """ALL-CAPS chip text -> sentence case. The card shouts its pills because
    they are small coloured chips; a bullet list should not shout."""
    return text[:1].upper() + text[1:].lower() if text.isupper() else text


def _esc_keeping_bold(text):
    """Escape Markdown specials but leave the bold markers we inserted."""
    return "**".join(esc(part) for part in text.split("**"))


def for_clue(clue_id, db_path=None):
    """The Reddit comment for one clue, or None when it has no PASS card.

    `stored_card` is the gate: it returns None unless there is a stored PASS
    parse — so a clue must be Confirmed before it can be posted, which is the
    right order anyway.
    """
    from core.wfw_card import stored_card
    html = stored_card(clue_id, db_path=db_path)
    if html is None:
        return None
    return build_markdown(html)
