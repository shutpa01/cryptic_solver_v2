"""Atom-signature data model + extraction from an existing Parse.

An AtomSignature is the ABSTRACTED shape of a clue's atom-map (words and lengths
stripped): for each contributing piece, its role (synonym/abbreviation/...), the
permutation class read off the atom-map (identity/reversed/anagram/selection), and
whether its answer atoms are contiguous or split; plus the assembly, the validating
indicator types present, and the definition edge.

`signature_from_parse(parse, ctx)` reads a SOLVED Parse — which is already a
one-link-per-answer-letter atom-map — and returns (AtomSignature, instance) or
(None, reason). It is a pure read; it does not solve anything.
"""

from dataclasses import dataclass


# --------------------------------------------------------------- atom-id helpers

def _char_index(atom_id):
    """The integer character index encoded in a clue atom id ('clue_char_0004'->4)."""
    try:
        return int(str(atom_id).rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def _span(atom_ids):
    """(min,max) char index over a tuple of clue atom ids, or None if empty."""
    idx = [i for i in (_char_index(a) for a in atom_ids) if i is not None]
    return (min(idx), max(idx)) if idx else None


# --------------------------------------------------------------- the model

@dataclass(frozen=True)
class Piece:
    role: str          # synonym | abbreviation | raw | hidden | first_letter |
                       #   homophone | anagram_fodder  (the Source.mechanism)
    transform: str     # identity | reversed | anagram | selection
    placement: str     # contiguous | split   (charade-like vs container-outer)
    n_atoms: int       # answer atoms supplied (INSTANCE detail; not in the key)

    def key(self):
        return "%s:%s:%s" % (self.role, self.transform, self.placement)


@dataclass(frozen=True)
class AtomSignature:
    assembly: str          # single | charade | container
    pieces: tuple          # tuple[Piece], in answer order
    indicators: tuple      # tuple[str] indicator types present, sorted
    def_pos: str           # start | end | mixed
    operation: str         # the Parse's top-level operation label (reference only)

    def key(self):
        """Canonical, length-stripped key for frequency counting / catalogue id."""
        pk = " + ".join(p.key() for p in self.pieces)
        ind = ",".join(self.indicators)
        return "%s | %s | def:%s | ind:%s" % (self.assembly, pk, self.def_pos, ind)


# --------------------------------------------------------------- classification

def _transform_of(mechanism, transforms):
    """Permutation class for a piece from its Source.mechanism + the set of
    Link.transform values across its answer atoms."""
    t = " ".join(sorted(x for x in transforms if x))
    if "rev" in t:
        return "reversed"
    if "anag" in t:
        return "anagram"
    if mechanism == "anagram_fodder":
        return "anagram"
    if mechanism in ("hidden", "first_letter"):
        return "selection"
    return "identity"


def _placement_of(positions):
    """contiguous if the piece's answer positions form an unbroken run, else split."""
    ps = sorted(positions)
    return "contiguous" if ps == list(range(ps[0], ps[-1] + 1)) else "split"


def _indicator_type(note):
    """'anagram indicator' -> 'anagram'; 'deletion indicator' -> 'deletion'; etc.
    Falls back to the whole note (trimmed) if it does not end in ' indicator'."""
    n = (note or "").strip().lower()
    if n.endswith(" indicator"):
        return n[: -len(" indicator")].strip()
    return n


def _def_pos(def_src, wordplay_sources):
    """Definition edge: 'start' if the definition sits before all wordplay pieces,
    'end' if after, else 'mixed'."""
    d = _span(def_src.clue_atom_ids) if def_src else None
    if d is None:
        return "mixed"
    wp = [s for s in (_span(s.clue_atom_ids) for s in wordplay_sources) if s]
    if not wp:
        return "mixed"
    wmin = min(a for a, _ in wp)
    wmax = max(b for _, b in wp)
    if d[1] < wmin:
        return "start"
    if d[0] > wmax:
        return "end"
    return "mixed"


# --------------------------------------------------------------- extraction

def signature_from_parse(parse, ctx):
    """Convert a SOLVED Parse (status 'pass') into (AtomSignature, instance).
    Returns (None, reason) if the parse is not a clean, complete atom-map.

    `instance` is a dict of the concrete per-clue facts (piece values, clue spans)
    kept for seeding / display; the AtomSignature itself is length/word-stripped.
    """
    if parse is None:
        return None, "no parse"
    if parse.status != "pass":
        return None, "status:%s" % parse.status
    if not parse.is_complete():
        return None, "incomplete: answer atoms not fully linked"
    if parse.unexplained_words(ctx):
        return None, "unaccounted words: %s" % ", ".join(parse.unexplained_words(ctx))
    if not parse.sources:
        return None, "no wordplay sources"

    n_src = len(parse.sources)
    # Gather, per source, the answer positions it supplies and the transforms seen.
    positions = {i: [] for i in range(n_src)}
    transforms = {i: set() for i in range(n_src)}
    for link in parse.links:
        si = link.source_index
        if si is None or si < 0 or si >= n_src:
            return None, "link points outside sources"
        positions[si].append(link.answer_pos)
        transforms[si].add(link.transform)

    # Order sources by their first answer atom (answer order).
    order = sorted((i for i in range(n_src) if positions[i]),
                   key=lambda i: min(positions[i]))
    if not order:
        return None, "no source supplies any answer atom"

    # Classify each source first; anagram FODDER is one operation over several base
    # words, so a run of adjacent anagram sources is merged into ONE piece (its
    # letters legitimately scatter, which must NOT read as containment).
    classified = []
    for i in order:
        src = parse.sources[i]
        tf = _transform_of(src.mechanism, transforms[i])
        classified.append((i, src, tf))

    pieces, inst_pieces = [], []
    j = 0
    while j < len(classified):
        i, src, tf = classified[j]
        if tf == "anagram":
            grp = [(i, src)]
            k = j + 1
            while k < len(classified) and classified[k][2] == "anagram":
                grp.append((classified[k][0], classified[k][1]))
                k += 1
            pos = sorted(p for gi, _ in grp for p in positions[gi])
            place = _placement_of(pos)
            pieces.append(Piece(role="anagram_fodder", transform="anagram",
                                placement=place, n_atoms=len(pos)))
            inst_pieces.append({"text": " ".join(s.text for _, s in grp),
                                "value": "".join(s.value for _, s in grp),
                                "mechanism": "anagram_fodder", "transform": "anagram",
                                "placement": place, "answer_positions": pos})
            j = k
            continue
        place = _placement_of(positions[i])
        pieces.append(Piece(role=src.mechanism, transform=tf, placement=place,
                            n_atoms=len(positions[i])))
        inst_pieces.append({"text": src.text, "value": src.value,
                            "mechanism": src.mechanism, "transform": tf,
                            "placement": place,
                            "answer_positions": sorted(positions[i])})
        j += 1

    # Containment is meaningful only for an IDENTITY piece that is split around
    # another; an anagram/selection piece scattering its letters is not a container.
    any_split = any(p.placement == "split" and p.transform == "identity"
                    for p in pieces)
    if len(pieces) == 1:
        assembly = "single"
    elif any_split:
        assembly = "container"
    else:
        assembly = "charade"

    indicators = tuple(sorted({_indicator_type(a.note)
                               for a in parse.annotations if a.role == "indicator"}))
    def_pos = _def_pos(parse.definition, parse.sources)

    sig = AtomSignature(assembly=assembly, pieces=tuple(pieces),
                        indicators=indicators, def_pos=def_pos,
                        operation=parse.operation or "")
    instance = {"clue": parse.clue_text, "answer": parse.answer_text,
                "operation": parse.operation, "solved_by": parse.solved_by,
                "def": (parse.definition.text if parse.definition else None),
                "def_pos": def_pos, "pieces": inst_pieces,
                "indicators": [{"text": a.text, "type": _indicator_type(a.note)}
                               for a in parse.annotations if a.role == "indicator"]}
    return sig, instance
