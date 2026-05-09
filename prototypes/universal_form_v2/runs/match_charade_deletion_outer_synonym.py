"""Run signature charade(deletion[outer](literal), synonym) against a corpus.

For each clue:
  * tokenise clue text
  * try every (literal_word, indicator_word adjacent) pair
  * try every (synonym source span) on remaining words
  * for combinations whose deletion[outer]+synonym (in either order) equals
    the answer, find a definition span (must synonymise the answer)
  * any leftover wordplay words must all be on the LINK_WORDS allow-list
  * build the form, run clipboard_verifier
  * record verifier-PASSes

Read-only, no DB writes. Prints a markdown-style report to stdout.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from prototypes.universal_form_v2.schema import (
    Form, Definition, Node, lit, syn, charade, deletion,
)
from prototypes.universal_form_v2.clipboard_verifier import verify, LINK_WORDS


db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))


def tokenise(text: str) -> list:
    return re.findall(r"[A-Za-z]+", text or "")


def is_deletion_indicator(word: str) -> bool:
    types = db.get_indicator_types(word.lower())
    return any(t[0] in ("deletion", "parts") for t in types)


def synonyms_of(phrase: str):
    return [v.upper() for v in db.get_synonyms(phrase.lower())]


def try_match(clue_text: str, answer: str) -> list:
    """Returns list of (form, verdict) for verifier-PASS fits."""
    answer_u = re.sub(r"[^A-Za-z]", "", answer or "").upper()
    if not answer_u:
        return []
    words = tokenise(clue_text)
    n = len(words)
    if n < 3:
        return []
    passes = []

    for li in range(n):
        lw = words[li]
        if len(lw) < 3:
            continue
        full = lw.upper()
        extracted = full[1:-1]
        if not extracted:
            continue

        for ii in (li - 1, li + 1):
            if not (0 <= ii < n):
                continue
            iw = words[ii]
            if not is_deletion_indicator(iw):
                continue

            for sa in range(n):
                for sb in range(sa + 1, n + 1):
                    if any(k in (li, ii) for k in range(sa, sb)):
                        continue
                    syn_phrase = " ".join(words[sa:sb])
                    syn_vals = synonyms_of(syn_phrase)
                    if not syn_vals:
                        continue

                    for syn_val in syn_vals:
                        for first_val, second_val, first_kind in [
                            (extracted, syn_val, "delfirst"),
                            (syn_val, extracted, "synfirst"),
                        ]:
                            if first_val + second_val != answer_u:
                                continue
                            used_idx = {li, ii, *range(sa, sb)}
                            for da in range(n):
                                for db_ in range(da + 1, n + 1):
                                    if any(k in used_idx
                                           for k in range(da, db_)):
                                        continue
                                    def_phrase = " ".join(words[da:db_])
                                    if answer_u not in [
                                        v.upper()
                                        for v in db.get_synonyms(
                                            def_phrase.lower())
                                    ]:
                                        continue

                                    used2 = used_idx | set(range(da, db_))
                                    leftovers = [
                                        words[k].lower()
                                        for k in range(n)
                                        if k not in used2
                                    ]
                                    if any(w not in LINK_WORDS
                                           for w in leftovers):
                                        continue

                                    if first_kind == "delfirst":
                                        p_del = deletion(
                                            lit(source_word=lw,
                                                value=full),
                                            kind="outer",
                                            indicator=iw)
                                        p_syn = syn(
                                            source_word=syn_phrase,
                                            value=syn_val)
                                        tree = charade(p_del, p_syn)
                                    else:
                                        p_syn = syn(
                                            source_word=syn_phrase,
                                            value=syn_val)
                                        p_del = deletion(
                                            lit(source_word=lw,
                                                value=full),
                                            kind="outer",
                                            indicator=iw)
                                        tree = charade(p_syn, p_del)

                                    form = Form(
                                        tree=tree,
                                        definition=Definition(
                                            phrase=def_phrase,
                                            answer=answer_u),
                                        link_words=leftovers,
                                    )
                                    verdict = verify(
                                        form, clue_text, db)
                                    if verdict.verdict == "PASS":
                                        passes.append((form, verdict))
    return passes


def fmt_form(form: Form) -> str:
    """Compact stringification of the form tree."""
    def fmt(n: Node) -> str:
        op = n.operation
        if not n.sources:
            v = n.value or ""
            s = n.source_word or ""
            if op == "deletion":
                return f"deletion[{n.deletion_kind}]({v})"
            return f"{op}({v} <- {s!r})"
        ind = f" [{n.indicator}]" if n.indicator else ""
        chs = ", ".join(fmt(c) for c in n.sources)
        if op == "deletion":
            return f"deletion[{n.deletion_kind}]{ind}({chs})"
        return f"{op}{ind}({chs})"
    return fmt(form.tree)


def fetch_corpus(canary_id: int, n_per_source: int) -> list:
    """Return the canary clue plus n random clues from each major source."""
    master = sqlite3.connect(
        str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    canary_row = master.execute(
        "SELECT id, source, puzzle_number, clue_number, direction, "
        "clue_text, answer FROM clues WHERE id=?", (canary_id,)
    ).fetchone()
    if canary_row is None:
        raise SystemExit(f"canary id {canary_id} not in DB")
    rows = [dict(canary_row)]

    sources = ("times", "telegraph", "guardian", "independent",
                "telegraph-toughie", "cordelia")
    for src in sources:
        cur = master.execute(
            "SELECT id, source, puzzle_number, clue_number, direction, "
            "clue_text, answer FROM clues "
            "WHERE source=? AND answer IS NOT NULL AND answer != '' "
            "ORDER BY RANDOM() LIMIT ?", (src, n_per_source))
        for r in cur:
            d = dict(r)
            if d["id"] != canary_id:
                rows.append(d)
    master.close()
    return rows


def main():
    n_per_source = int(sys.argv[1]) if len(sys.argv) > 1 else 80
    canary_id = 10063362  # Times 29535 9a, ELITE
    corpus = fetch_corpus(canary_id, n_per_source)
    print(f"corpus size: {len(corpus)} (canary + {n_per_source} per source)")
    print(f"signature: charade(deletion[outer](literal), synonym)")
    print()

    matches = []
    canary_passed = False
    for i, c in enumerate(corpus, 1):
        passes = try_match(c["clue_text"], c["answer"])
        if c["id"] == canary_id:
            canary_passed = bool(passes)
        if passes:
            matches.append((c, passes))
        if i % 100 == 0:
            print(f"  ... {i}/{len(corpus)} processed, "
                  f"{len(matches)} matches so far",
                  file=sys.stderr)

    print(f"canary (ELITE) passed: {canary_passed}")
    print(f"clues with at least one PASS fit: "
          f"{len(matches)} / {len(corpus)}")
    print()

    by_source = Counter(c["source"] for c, _ in matches)
    print("matches by source:")
    for s, n in by_source.most_common():
        print(f"  {s:25s} {n}")
    print()

    print("=" * 70)
    print("PASSes:")
    print("=" * 70)
    for c, passes in matches:
        cn = c["clue_number"]
        d = c["direction"][0] if c["direction"] else "?"
        print(f"\n## {c['source']} {c['puzzle_number']} "
              f"{cn}{d}  {c['answer']}")
        print(f"   clue: {c['clue_text']}")
        for form, _ in passes[:3]:
            print(f"   form: {fmt_form(form)}")
            print(f"   def:  {form.definition.phrase!r}")
            if form.link_words:
                print(f"   link: {form.link_words}")


if __name__ == "__main__":
    main()
