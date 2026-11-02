"""Read-only probe: classify PASS clues into Crypticker difficulty buckets.

Reads clues_master.db, never writes. Groups each answer's letters by the
source word they came from (via wfw_link), and buckets each clue by whether
any source word got split around another.
"""
import sqlite3, json, collections, os

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clues_master.db")

# Assembly-style operations (spec 4.3 whitelist). Anagram counted separately.
ASSEMBLY_OPS = {"charade", "container", "container_charade"}


def letters(s):
    return "".join(ch for ch in s.upper() if ch.isalpha())


def runs_for(conn, clue_id):
    """Return list of (source_index, length) runs in answer order, or None."""
    rows = conn.execute(
        "select answer_pos, source_index from wfw_link "
        "where clue_id=? order by answer_pos", (clue_id,)
    ).fetchall()
    if not rows:
        return None
    seq = [r[1] for r in rows]
    runs = []
    for s in seq:
        if runs and runs[-1][0] == s:
            runs[-1][1] += 1
        else:
            runs.append([s, 1])
    return [(s, n) for s, n in runs]


def classify(runs):
    """charade | reorder | split, given the answer-order runs."""
    src_seq = [s for s, _ in runs]
    if len(src_seq) != len(set(src_seq)):
        return "split"                    # a source appears in >1 run
    if src_seq == sorted(src_seq):
        return "charade"                  # runs already in source (clue) order
    return "reorder"


def main():
    conn = sqlite3.connect(DB)
    buckets = collections.Counter()
    examples = collections.defaultdict(list)
    degenerate = 0
    no_map = 0

    rows = conn.execute(
        "select s.clue_id, s.clue_text, s.answer_text, s.operation "
        "from wfw_solve s where s.status='pass' and s.operation in "
        "('charade','container','container_charade')"
    ).fetchall()

    for cid, clue, ans, op in rows:
        runs = runs_for(conn, cid)
        if runs is None:
            no_map += 1
            continue
        L = len(letters(ans))
        ntiles = len(runs)
        # degenerate: fewer than 2 tiles, or every tile is 1-2 letters
        if ntiles < 2 or all(n <= 2 for _, n in runs):
            degenerate += 1
            examples["degenerate"].append((clue, ans, [n for _, n in runs]))
            continue
        b = classify(runs)
        buckets[b] += 1
        if len(examples[b]) < 4:
            examples[b].append((clue, ans, [n for _, n in runs]))

    # Anagram supply (spec: answer length 7-10)
    anag = conn.execute(
        "select count(*) from wfw_solve where status='pass' and operation='anagram'"
    ).fetchone()[0]
    anag_ok = 0
    for (ans,) in conn.execute(
        "select answer_text from wfw_solve where status='pass' and operation='anagram'"
    ):
        if 7 <= len(letters(ans)) <= 10:
            anag_ok += 1

    print("ASSEMBLY CLUES (charade / container / container_charade PASS)")
    print("  total with letter-map :", sum(buckets.values()) + degenerate)
    print("  no letter-map (skipped):", no_map)
    print()
    print("  BUCKET            COUNT   meaning")
    print("  charade           %5d   stick tiles together in clue order (jumble-safe)" % buckets["charade"])
    print("  reorder           %5d   whole tiles, shuffled out of clue order" % buckets["reorder"])
    print("  split             %5d   a word broken around another (hard)" % buckets["split"])
    print("  degenerate        %5d   too trivial (all tiny tiles / one tile)" % degenerate)
    print()
    print("ANAGRAM CLUES")
    print("  total pass        %5d" % anag)
    print("  answer len 7-10   %5d" % anag_ok)
    print()
    for b in ("charade", "reorder", "split", "degenerate"):
        print("--- %s examples ---" % b)
        for clue, ans, sizes in examples[b][:4]:
            print("   %-42s = %-12s tiles=%s" % (clue[:42], ans, sizes))
        print()


if __name__ == "__main__":
    main()
