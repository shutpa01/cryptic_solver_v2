"""
Label unlabeled clues with wordplay_type based on keyword patterns in explanations.

Two tiers of confidence:
  - Tier 1: Explicit type keywords (e.g., "anagram", "double definition")
  - Tier 2: Structural patterns (e.g., "+" for charades, "around" for containers)

Only labels when exactly ONE type is detected (single-match rule).
Writes a log of all changes for reversibility.

Usage:
    python classifier/label_from_explanations.py              # Dry run (default)
    python classifier/label_from_explanations.py --apply      # Apply labels to DB
    python classifier/label_from_explanations.py --revert     # Undo all applied labels
"""

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DB_PATH = PROJECT_ROOT / "data" / "cryptic_new.db"
LOG_PATH = SCRIPT_DIR / "label_log.json"

# ---------------------------------------------------------------------------
# Tier 1: Explicit type keywords (high confidence)
# ---------------------------------------------------------------------------
TIER1_PATTERNS = {
    "anagram": re.compile(
        r"\banagram\w*\b|\brearrange\w*\b|\bscramble\w*\b|\bshuffle\w*\b"
        r"|\bjumble\w*\b|\bmixed up\b|\bmixing\b",
        re.IGNORECASE,
    ),
    "hidden": re.compile(
        r"\bhidden\b|\bhides?\b|\bconcealed\b|\blurking\b"
        r"|\bwithin\b|\bpart of\b",
        re.IGNORECASE,
    ),
    "double_definition": re.compile(
        r"\bdouble definition\b|\btwo definitions\b|\btwo meanings\b"
        r"|\bboth mean\b|\balso means\b|\btwo senses\b",
        re.IGNORECASE,
    ),
    "cryptic_definition": re.compile(
        r"\bcryptic definition\b|\bcryptic def\b"
        r"|\bwholly cryptic\b|\bpure cryptic\b|\ball-in-one\b",
        re.IGNORECASE,
    ),
    "homophone": re.compile(
        r"\bhomophone\b|\bsounds like\b|\bwe hear\b|\bas heard\b"
        r"|\baudibly\b|\bpronounce\w*\b|\bspoken\b|\bwhen spoken\b"
        r"|\bto the ear\b|\bto the listener\b|\bsaid aloud\b",
        re.IGNORECASE,
    ),
    "reversal": re.compile(
        r"\breversed\b|\brevers\w+\b|\bbackwards\b|\breflected\b"
        r"|\bgoing up\b|\bbrought up\b|\bturned up\b|\blifted\b"
        r"|\bread upwards\b|\bfrom bottom\b|\bread up\b",
        re.IGNORECASE,
    ),
    "acrostic": re.compile(
        r"\bfirst letters?\b|\binitial letters?\b|\bacrostic\b"
        r"|\bleading letters?\b|\binitials of\b|\bheads? of\b"
        r"|\bstarts? of\b|\bfirst letter of each\b|\bopening letters?\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# Tier 2: Structural patterns (medium-high confidence)
# These only fire if NO Tier 1 pattern matched, and are more specific
# ---------------------------------------------------------------------------
TIER2_PATTERNS = {
    "container": re.compile(
        r"\binside\b|\binsert\w*\b|\bwrapp\w*\b|\bsurrounding\b"
        r"|\bembrac\w+\b|\bcontain\w+\b|\bcontained\b|\bgoes into\b"
        r"|\bput in\b|\btaken in\b|\bgoing into\b|\bplaced in\b"
        r"|\bengulfing\b|\bswallowing\b",
        re.IGNORECASE,
    ),
    "deletion": re.compile(
        r"\bwithout\b|\bremov\w+\b|\bdropp\w+\b|\blosing\b|\bloses?\b"
        r"|\bmissing\b|\bless\b|\bdeleting\b|\bcut\w*\b"
        r"|\bshort(?:ened)?\b|\btruncated\b",
        re.IGNORECASE,
    ),
    "charade": re.compile(
        r"\bfollowed by\b|\bplus\b|\bnext to\b|\bpreceded by\b"
        r"|\bbefore\b|\bafter\b|\bthen\b",
        re.IGNORECASE,
    ),
}

# ---------------------------------------------------------------------------
# "+" notation detector for charades
# Matches explanations that use A + B or A + B + C notation
# ---------------------------------------------------------------------------
PLUS_PATTERN = re.compile(r"[A-Z]{2,}\s*\+\s*[A-Z]{2,}", re.IGNORECASE)
# Also match "word (meaning) + word (meaning)" style
PLUS_PATTERN2 = re.compile(r"\)\s*\+\s*\w", re.IGNORECASE)
# Also match the pattern "X + Y" with possible explanation in parens
PLUS_PATTERN3 = re.compile(r"\w\s*\+\s*\(", re.IGNORECASE)


def classify_explanation(explanation):
    """
    Classify an explanation into a wordplay type.
    Returns (type_name, tier, confidence) or (None, None, None) if no match.
    """
    exp_lower = explanation.lower()

    # Tier 1: Explicit keywords
    tier1_matches = []
    for wtype, pattern in TIER1_PATTERNS.items():
        if pattern.search(explanation):
            tier1_matches.append(wtype)

    if len(tier1_matches) == 1:
        return tier1_matches[0], 1, "high"

    # If multiple Tier 1 matches, skip (ambiguous)
    if len(tier1_matches) > 1:
        return None, None, None

    # Tier 2: Structural patterns (only if Tier 1 had no matches)
    tier2_matches = []
    for wtype, pattern in TIER2_PATTERNS.items():
        if pattern.search(explanation):
            tier2_matches.append(wtype)

    # Check "+" notation for charades
    has_plus = (
        PLUS_PATTERN.search(explanation)
        or PLUS_PATTERN2.search(explanation)
        or PLUS_PATTERN3.search(explanation)
    )

    if has_plus:
        # "+" notation is strong charade signal
        # But if container/deletion also matched, it's a compound clue
        # Only label as charade if it's the sole signal or only charade matched
        if "charade" not in tier2_matches:
            tier2_matches.append("charade")

    if len(tier2_matches) == 1:
        return tier2_matches[0], 2, "medium"

    # Special case: "+" notation alone (no other tier 2 matches)
    if has_plus and len(tier2_matches) == 1 and tier2_matches[0] == "charade":
        return "charade", 2, "medium"

    return None, None, None


def load_unlabeled(db_path):
    """Load clues that have explanations but no wordplay_type."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, clue_text, answer, explanation FROM clues "
        "WHERE explanation IS NOT NULL AND explanation != '' "
        "AND (wordplay_type IS NULL OR wordplay_type = '')"
    ).fetchall()
    conn.close()
    return rows


def dry_run(db_path):
    """Classify and report without modifying the DB."""
    rows = load_unlabeled(db_path)
    print(f"Unlabeled clues with explanations: {len(rows):,}\n")

    results = {"tier1": {}, "tier2": {}, "unmatched": 0, "ambiguous": 0}
    labeled = []

    for row_id, clue, answer, explanation in rows:
        wtype, tier, confidence = classify_explanation(explanation)
        if wtype:
            tier_key = f"tier{tier}"
            results[tier_key][wtype] = results[tier_key].get(wtype, 0) + 1
            labeled.append((row_id, wtype, tier, confidence))
        else:
            results["unmatched"] += 1

    total_labeled = len(labeled)
    print(f"Total labeled: {total_labeled:,} ({total_labeled/len(rows)*100:.1f}%)\n")

    print("Tier 1 (explicit keywords):")
    tier1_total = 0
    for wtype, count in sorted(results["tier1"].items(), key=lambda x: -x[1]):
        print(f"  {wtype:<25} {count:>6,}")
        tier1_total += count
    print(f"  {'TOTAL':<25} {tier1_total:>6,}\n")

    print("Tier 2 (structural patterns):")
    tier2_total = 0
    for wtype, count in sorted(results["tier2"].items(), key=lambda x: -x[1]):
        print(f"  {wtype:<25} {count:>6,}")
        tier2_total += count
    print(f"  {'TOTAL':<25} {tier2_total:>6,}\n")

    print(f"Unmatched: {results['unmatched']:,} ({results['unmatched']/len(rows)*100:.1f}%)")

    # Combined with existing labeled data
    conn = sqlite3.connect(db_path)
    existing = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE wordplay_type IS NOT NULL AND wordplay_type != ''"
    ).fetchone()[0]
    conn.close()

    print(f"\nExisting labeled: {existing:,}")
    print(f"New labels:       {total_labeled:,}")
    print(f"New total:        {existing + total_labeled:,} ({(existing + total_labeled)/existing*100 - 100:.0f}% increase)")

    return labeled


def apply_labels(db_path, log_path):
    """Apply labels to the DB and write a revert log."""
    rows = load_unlabeled(db_path)
    labeled = []

    for row_id, clue, answer, explanation in rows:
        wtype, tier, confidence = classify_explanation(explanation)
        if wtype:
            labeled.append({
                "id": row_id,
                "wordplay_type": wtype,
                "tier": tier,
                "confidence": confidence,
            })

    if not labeled:
        print("No labels to apply.")
        return

    # Write revert log FIRST (so we can always undo)
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "count": len(labeled),
        "labels": labeled,
    }
    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"Revert log written: {log_path} ({len(labeled):,} entries)")

    # Apply to DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for entry in labeled:
        cursor.execute(
            "UPDATE clues SET wordplay_type = ? WHERE id = ?",
            (entry["wordplay_type"], entry["id"]),
        )
    conn.commit()
    conn.close()

    print(f"Applied {len(labeled):,} labels to {db_path.name}")

    # Summary by type
    type_counts = {}
    for entry in labeled:
        t = entry["wordplay_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:<25} {c:>6,}")


def revert_labels(db_path, log_path):
    """Revert all labels applied by the last apply run."""
    if not log_path.exists():
        print(f"No revert log found at {log_path}")
        return

    log_data = json.loads(log_path.read_text())
    labels = log_data["labels"]
    print(f"Reverting {len(labels):,} labels from {log_data['timestamp']}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for entry in labels:
        cursor.execute(
            "UPDATE clues SET wordplay_type = NULL WHERE id = ?",
            (entry["id"],),
        )
    conn.commit()
    conn.close()

    print(f"Reverted {len(labels):,} labels. DB restored to pre-label state.")


def main():
    parser = argparse.ArgumentParser(description="Label clues from explanations")
    parser.add_argument("--apply", action="store_true", help="Apply labels to DB")
    parser.add_argument("--revert", action="store_true", help="Revert last applied labels")
    parser.add_argument("--db", type=str, default=str(DB_PATH))
    args = parser.parse_args()

    db_path = Path(args.db)

    if args.revert:
        revert_labels(db_path, LOG_PATH)
    elif args.apply:
        # Show dry run first, then apply
        labeled = dry_run(db_path)
        print("\n" + "=" * 50)
        print("APPLYING LABELS...")
        print("=" * 50)
        apply_labels(db_path, LOG_PATH)
    else:
        dry_run(db_path)
        print("\nDry run complete. Use --apply to write labels to DB.")


if __name__ == "__main__":
    main()
