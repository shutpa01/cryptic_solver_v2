"""Apply the v0 catalog to a held-out corpus — measure catalog utility.

For each clue (with known answer), no blog used:
  1. Run word_analyzer.analyze_phrases on the clue
  2. Try every template in the catalog
  3. First template whose strict-verified instantiation produces the
     known answer = HIT (record which template)
  4. No template fits = MISS

Outputs:
  - per-clue results
  - catalog hit rate
  - per-template hit count (which templates are doing the work)
  - MISS distribution

No production touched. No new templates extracted (catalog is fixed).
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from .schema import Form
from .strict_verifier import verify
from .template_match import match_template
from .strict_pipeline_runner import fetch_clues
from signature_solver.db import RefDB
from signature_solver.word_analyzer import analyze_phrases

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def run(catalog_path: str, source: str, puzzle_range: str,
        limit: int, out_path: str):
    print(f"Loading catalog from {catalog_path}")
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    print(f"  {len(catalog['entries'])} templates")

    print(f"Fetching clues: source={source} range={puzzle_range} "
          f"limit={limit}")
    clues = fetch_clues(source, puzzle_range, limit)
    # Filter to clues with answer
    clues = [c for c in clues if c.get("answer")]
    print(f"  {len(clues)} clues")

    db = RefDB()

    results = []
    template_hits = Counter()
    miss_clues = []
    t0 = time.time()
    for i, clue in enumerate(clues):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(clues)} processed "
                  f"({elapsed:.0f}s, {elapsed/(i+1):.1f}s/clue)")
        clue_words = _tokenise(clue["clue_text"])
        if not clue_words:
            results.append({**clue, "verdict": "BAD_TOKENISE"})
            continue
        answer = clue["answer"].upper().replace(" ", "").replace("-", "")
        try:
            single, phrases = analyze_phrases(clue_words, answer, db)
        except Exception as e:
            results.append({**clue, "verdict": "ANALYSE_ERROR",
                             "error": str(e)})
            continue

        hit_template = None
        hit_form = None
        for entry in catalog["entries"]:
            try:
                matches = match_template(
                    entry["structure"], clue_words, single, phrases,
                    answer, db, None)
            except Exception:
                continue
            if matches:
                hit_template = entry["id"]
                hit_form = matches[0][0]
                break
        if hit_template:
            template_hits[hit_template] += 1
            results.append({
                **clue,
                "verdict": "HIT",
                "template": hit_template,
                "form": hit_form.to_dict(),
            })
        else:
            miss_clues.append(clue)
            results.append({**clue, "verdict": "MISS"})

    summary = Counter(r["verdict"] for r in results)
    elapsed = time.time() - t0
    print(f"\nFinished in {elapsed:.0f}s ({elapsed/len(clues):.1f}s/clue)")
    print(f"Summary: {dict(summary)}")
    print(f"Catalog hit rate: "
          f"{summary['HIT']}/{len(clues)} = "
          f"{100*summary['HIT']/len(clues):.1f}%")
    print()
    print("Top template hits:")
    for tid, n in template_hits.most_common(15):
        print(f"  {n:3d}  {tid}")
    if miss_clues:
        print(f"\nFirst 10 MISSes:")
        for c in miss_clues[:10]:
            print(f"  {c['answer']:18}  {c['clue_text'][:60]}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "catalog_path": catalog_path,
            "n_clues": len(clues),
            "summary": dict(summary),
            "template_hits": dict(template_hits),
            "per_clue": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nWritten {out_path}")


def _tokenise(text: str) -> list:
    """Same tokenisation the production word_analyzer expects:
    split on whitespace, strip simple punctuation (apostrophes preserved).
    """
    if not text:
        return []
    # Strip punctuation but keep words
    import re
    return re.findall(r"[A-Za-z][A-Za-z'’]*", text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default=
        "prototypes/universal_form_v2/runs/catalog_v0.json")
    ap.add_argument("--source", default="times")
    ap.add_argument("--puzzle-range", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args.catalog, args.source, args.puzzle_range,
        args.limit, args.out)


if __name__ == "__main__":
    main()
