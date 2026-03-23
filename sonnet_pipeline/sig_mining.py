"""Mine the clue database using signature matching to discover synonyms,
abbreviations, indicators, and definitions.

For each clue that matches a known signature, we know exactly which word
performs which role. This lets us extract high-confidence DB entries
without any API calls.

Usage:
    python -m sonnet_pipeline.sig_mining
    python -m sonnet_pipeline.sig_mining --source times --limit 10000
    python -m sonnet_pipeline.sig_mining --resume
    python -m sonnet_pipeline.sig_mining --min-confidence 82
"""

import argparse
import json
import os
import sqlite3
import time
from collections import Counter

from signature_solver.solver import solve_clue
from signature_solver.db import RefDB
from signature_solver.tokens import (
    SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F,
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
    INDICATOR_TOKENS, LNK, DEF,
)

MASTER_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db"
OUTPUT_DIR = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data"
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "mining_checkpoint.txt")

# Map indicator tokens to DB wordplay_type strings
_INDICATOR_TOKEN_TO_DB_TYPE = {
    ANA_I: "anagram",
    REV_I: "reversal",
    CON_I: "container",
    DEL_I: "deletion",
    HID_I: "hidden",
    HOM_I: "homophone",
    POS_I_FIRST: "acrostic",
    POS_I_LAST: "parts",
    POS_I_OUTER: "parts",
    POS_I_MIDDLE: "parts",
    POS_I_ALTERNATE: "alternating",
    POS_I_TRIM_FIRST: "parts",
    POS_I_TRIM_LAST: "parts",
    POS_I_TRIM_MIDDLE: "parts",
    POS_I_TRIM_OUTER: "parts",
    POS_I_HALF: "parts",
}


def extract_discoveries(sr, answer):
    """Extract synonym/abbreviation/indicator/definition entries from a SolveResult.

    Returns list of (type, word, value) tuples.
    """
    if not sr.result:
        return []

    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    discoveries = []

    for word, tok, val in sr.result.word_roles:
        word_clean = word.lower().strip(".,;:!?\"'()-").strip()
        if not word_clean:
            continue

        if tok == SYN_F and val:
            val_upper = val.upper()
            # Skip circularity: synonym that equals the full answer
            if val_upper == answer_clean:
                continue
            # Skip unreasonably long synonyms
            if len(val_upper) > 15:
                continue
            discoveries.append(("synonym", word_clean, val_upper))

        elif tok == ABR_F and val:
            val_upper = val.upper()
            if val_upper == answer_clean:
                continue
            if len(val_upper) > 5:
                continue
            discoveries.append(("abbreviation", word_clean, val_upper))

        elif tok in INDICATOR_TOKENS:
            db_type = _INDICATOR_TOKEN_TO_DB_TYPE.get(tok)
            if db_type:
                discoveries.append(("indicator", word_clean, db_type))

    # Extract definition -> answer
    defn = getattr(sr, "definition", None)
    if defn and answer_clean:
        discoveries.append(("definition", defn.lower().strip(), answer_clean))

    return discoveries


def deduplicate(counter, ref_db):
    """Split counter into new (not in DB) and existing (already in DB).

    Returns (new_counter, existing_counter).
    """
    new = Counter()
    existing = Counter()

    for key, count in counter.items():
        typ, word, val = key

        in_db = False
        if typ == "synonym":
            syns = ref_db.get_synonyms(word, max_len=20)
            in_db = val in syns
        elif typ == "abbreviation":
            abbrs = ref_db.get_abbreviations(word)
            in_db = val in abbrs
        elif typ == "indicator":
            ind_types = ref_db.get_indicator_types(word)
            in_db = any(wt == val for wt, _, _ in ind_types)
        elif typ == "definition":
            in_db = ref_db.is_definition_of(word, val)

        if in_db:
            existing[key] = count
        else:
            new[key] = count

    return new, existing


def mine_clues(source=None, limit=None, min_confidence=80, resume=False):
    """Run signature matching over the clue database.

    Returns (all_discoveries Counter, stats dict).
    """
    print("Loading RefDB...")
    ref_db = RefDB()

    conn = sqlite3.connect(MASTER_DB, timeout=30)

    # Build query
    where = ["answer IS NOT NULL", "answer != ''"]
    params = []
    if source:
        where.append("source = ?")
        params.append(source)

    resume_id = 0
    if resume and os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            resume_id = int(f.read().strip())
        print("Resuming from id > %d" % resume_id)
        where.append("id > ?")
        params.append(resume_id)

    sql = "SELECT id, clue_text, answer FROM clues WHERE %s ORDER BY id" % (
        " AND ".join(where))
    if limit:
        sql += " LIMIT %d" % limit

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    total = len(rows)
    print("Mining %d clues (min_confidence=%d)..." % (total, min_confidence))

    all_discoveries = Counter()
    stats = {
        "total_clues": total,
        "matched": 0,
        "high_confidence": 0,
        "discoveries_raw": 0,
    }

    t0 = time.time()
    last_checkpoint = 0

    for i, (cid, clue_text, answer) in enumerate(rows):
        if not clue_text or not answer:
            continue

        # Strip enumeration from clue text
        import re
        clue_clean = re.sub(r'\s*\(\d+(?:[,-]\d+)*\)\s*$', '', clue_text).strip()
        if not clue_clean:
            continue

        try:
            sr = solve_clue(clue_clean, answer, ref_db, min_confidence=min_confidence)
        except Exception:
            continue

        if sr.solved and sr.confidence >= min_confidence:
            stats["matched"] += 1
            if sr.confidence >= 80:
                stats["high_confidence"] += 1

            entries = extract_discoveries(sr, answer)
            for entry in entries:
                all_discoveries[entry] += 1
                stats["discoveries_raw"] += 1

        # Progress + checkpoint
        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print("  %d/%d (%.0f/sec) — %d matched, %d discoveries" % (
                i + 1, total, rate, stats["matched"], stats["discoveries_raw"]))

            # Save checkpoint
            with open(CHECKPOINT_FILE, "w") as f:
                f.write(str(cid))
            last_checkpoint = cid

    elapsed = time.time() - t0
    stats["elapsed_sec"] = elapsed
    stats["unique_discoveries"] = len(all_discoveries)

    # Clean up checkpoint on completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    return all_discoveries, stats, ref_db


def generate_report(all_discoveries, new_discoveries, existing, stats):
    """Write human-readable report and machine-readable JSONL."""

    # --- Summary report ---
    lines = []
    lines.append("=" * 70)
    lines.append("SIGNATURE MINING REPORT")
    lines.append("Generated: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 70)
    lines.append("")
    lines.append("STATS")
    lines.append("-" * 40)
    lines.append("  Clues scanned:     %d" % stats["total_clues"])
    lines.append("  Matched:           %d (%.1f%%)" % (
        stats["matched"], 100 * stats["matched"] / max(stats["total_clues"], 1)))
    lines.append("  High confidence:   %d" % stats["high_confidence"])
    lines.append("  Raw discoveries:   %d" % stats["discoveries_raw"])
    lines.append("  Unique discoveries:%d" % stats["unique_discoveries"])
    lines.append("  New (not in DB):   %d" % len(new_discoveries))
    lines.append("  Already in DB:     %d" % len(existing))
    lines.append("  Elapsed:           %.1fs" % stats.get("elapsed_sec", 0))
    lines.append("")

    # Break down by type
    for typ in ("synonym", "abbreviation", "indicator", "definition"):
        new_of_type = {k: v for k, v in new_discoveries.items() if k[0] == typ}
        exist_of_type = {k: v for k, v in existing.items() if k[0] == typ}
        lines.append("  %s: %d new, %d existing" % (
            typ.upper(), len(new_of_type), len(exist_of_type)))
    lines.append("")

    # --- High-frequency new discoveries (count >= 3) ---
    lines.append("HIGH-FREQUENCY NEW DISCOVERIES (seen in 3+ clues)")
    lines.append("-" * 60)
    high_freq = sorted(
        [(k, v) for k, v in new_discoveries.items() if v >= 3],
        key=lambda x: -x[1]
    )
    if not high_freq:
        lines.append("  (none)")
    for (typ, word, val), count in high_freq[:200]:
        lines.append("  [%d] %s: %s -> %s" % (count, typ, word, val))
    lines.append("")

    # --- All new discoveries by type ---
    for typ in ("synonym", "abbreviation", "indicator", "definition"):
        typed = sorted(
            [(k, v) for k, v in new_discoveries.items() if k[0] == typ],
            key=lambda x: -x[1]
        )
        lines.append("ALL NEW %s (%d entries)" % (typ.upper(), len(typed)))
        lines.append("-" * 60)
        for (_, word, val), count in typed[:500]:
            lines.append("  [%d] %s -> %s" % (count, word, val))
        if len(typed) > 500:
            lines.append("  ... and %d more" % (len(typed) - 500))
        lines.append("")

    report_path = os.path.join(OUTPUT_DIR, "mining_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report written to %s" % report_path)

    # --- Machine-readable JSONL ---
    jsonl_path = os.path.join(OUTPUT_DIR, "mining_discoveries.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for (typ, word, val), count in sorted(
            new_discoveries.items(), key=lambda x: (-x[1], x[0])
        ):
            f.write(json.dumps({
                "type": typ, "word": word, "value": val,
                "count": count, "in_db": False
            }) + "\n")
    print("Discoveries written to %s (%d entries)" % (jsonl_path, len(new_discoveries)))

    return report_path, jsonl_path


def main():
    parser = argparse.ArgumentParser(description="Mine clue DB using signature matching")
    parser.add_argument("--source", help="Filter by source (times, telegraph, etc.)")
    parser.add_argument("--limit", type=int, help="Max clues to process")
    parser.add_argument("--min-confidence", type=int, default=80,
                        help="Minimum confidence to accept (default: 80)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    all_discoveries, stats, ref_db = mine_clues(
        source=args.source,
        limit=args.limit,
        min_confidence=args.min_confidence,
        resume=args.resume,
    )

    print("\nDeduplicating against existing DB...")
    new_discoveries, existing = deduplicate(all_discoveries, ref_db)

    print("\n%d unique discoveries, %d new, %d already in DB" % (
        len(all_discoveries), len(new_discoveries), len(existing)))

    generate_report(all_discoveries, new_discoveries, existing, stats)


if __name__ == "__main__":
    main()
