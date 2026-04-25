"""Re-verify currently-HIGH clues from the last week with the updated verifier.

Read-only: reports clues that the new verifier would demote below HIGH.
"""
import os
import sys
import sqlite3
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sonnet_pipeline.verify_explanation import ExplanationVerifier

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "data", "clues_master.db")

DATE_START = "2026-04-15"
DATE_END = "2026-04-20"


def main():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT c.id, c.source, c.publication_date, c.puzzle_number, c.clue_number,
               c.direction, c.clue_text, c.answer, c.definition, c.wordplay_type,
               c.ai_explanation,
               se.confidence, se.model_version
        FROM clues c
        JOIN structured_explanations se ON se.clue_id = c.id
        WHERE se.confidence >= 0.70
          AND c.publication_date >= date(?)
          AND c.publication_date <= date(?)
          AND c.ai_explanation IS NOT NULL
          AND c.ai_explanation != ''
        ORDER BY c.publication_date, c.source, c.puzzle_number, c.direction,
                 CAST(c.clue_number AS INTEGER)
    """, (DATE_START, DATE_END)).fetchall()

    v = ExplanationVerifier()
    demoted = []
    skipped_manual = 0
    errors = 0
    processed = 0

    for r in rows:
        mv = r["model_version"] or ""
        if mv in ("manual_approve", "claude_review"):
            skipped_manual += 1
            continue
        try:
            res = v.verify(r["clue_text"], r["answer"], r["definition"],
                           r["wordplay_type"], r["ai_explanation"])
        except Exception:
            errors += 1
            continue
        processed += 1
        if res["score"] < 70:
            wrong = [c for c in res["checks"] if c["status"] == "wrong"]
            demoted.append((r, res["score"], res["verdict"], wrong))

    print(f"Processed: {processed}")
    print(f"Skipped manual: {skipped_manual}")
    print(f"Errors: {errors}")
    print(f"Demoted (new score < 70): {len(demoted)}")
    print()

    # By date / source
    by_src_date = Counter()
    for r, sc, vd, _ in demoted:
        by_src_date[(r["publication_date"], r["source"])] += 1
    print("By date and source:")
    for k in sorted(by_src_date):
        print(f"  {k[0]}  {k[1]:12s}  {by_src_date[k]}")
    print()

    # By failure signature
    sig_counter = Counter()
    for r, sc, vd, wrong in demoted:
        sigs = tuple(sorted({c["check"] for c in wrong}))
        sig_counter[sigs] += 1
    print("By wrong-check signature:")
    for sig, n in sig_counter.most_common():
        print(f"  {n:4d}  {sig}")
    print()

    # Sample a few examples
    print("Sample demotions (first 15):")
    for r, sc, vd, wrong in demoted[:15]:
        print(f"  {r['publication_date']} {r['source']} #{r['puzzle_number']} "
              f"{r['clue_number']}{r['direction'][0]}  {r['answer']:15s} "
              f"conf={r['confidence']:.2f} -> new={sc:3d} [{vd}]")
        print(f"    clue: {r['clue_text']}")
        print(f"    expl: {(r['ai_explanation'] or '')[:140]}")
        for c in wrong[:3]:
            print(f"    WRONG: {c['check']}: {c['detail']}")
        print()

    # Write full list
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "demoted_last_week.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Demoted HIGH clues for {DATE_START} to {DATE_END}\n")
        f.write(f"# Processed: {processed}, Demoted: {len(demoted)}\n\n")
        for r, sc, vd, wrong in demoted:
            f.write(f"[{r['publication_date']}] {r['source']} #{r['puzzle_number']} "
                    f"{r['clue_number']}{r['direction'][0]} {r['answer']} "
                    f"conf={r['confidence']:.2f} -> new={sc}\n")
            f.write(f"  clue: {r['clue_text']}\n")
            f.write(f"  def:  {r['definition']}\n")
            f.write(f"  type: {r['wordplay_type']}\n")
            f.write(f"  expl: {r['ai_explanation']}\n")
            f.write(f"  mv:   {r['model_version']}\n")
            for c in wrong:
                f.write(f"  WRONG {c['check']}: {c['detail']}\n")
            f.write("\n")
    print(f"Full list: {out}")


if __name__ == "__main__":
    main()
