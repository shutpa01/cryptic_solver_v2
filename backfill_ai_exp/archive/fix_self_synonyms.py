"""Post-processor to fix self-synonym annotations in backfill results.

Reads JSONL results, identifies pieces where clue_word == letters (self-synonym),
and reclassifies them:
  1. Literal: letters match a full clue word exactly
  2. Deletion: letters are a substring of a clue word (with letters removed)
  3. Real synonym: letters are a synonym/abbreviation of a clue word (fix the annotation)

Writes corrected JSONL to a new file.

Usage:
    python scripts/fix_self_synonyms.py data/parsed_explanations_v2.jsonl data/parsed_explanations_v2_fixed.jsonl
"""

import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.db import RefDB

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")


def norm(s):
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


def fix_pieces(pieces, clue_text, ref_db):
    """Fix self-synonym pieces in a clue's ai_pieces list.

    Returns (fixed_pieces, n_fixed).
    """
    clue_words_raw = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).split()
    clue_words_norm = [norm(w) for w in clue_words_raw]

    fixed = []
    n_fixed = 0

    for p in pieces:
        cw = p.get("clue_word", "")
        lt = p.get("letters", "")
        mech = p.get("mechanism", "")

        if mech != "synonym" or not lt or not cw or norm(lt) != norm(cw):
            fixed.append(p)
            continue

        lt_norm = norm(lt)
        new_piece = dict(p)

        # 1. Literal: letters match a full clue word
        if lt_norm in clue_words_norm:
            idx = clue_words_norm.index(lt_norm)
            new_piece["mechanism"] = "literal"
            new_piece["clue_word"] = clue_words_raw[idx]
            fixed.append(new_piece)
            n_fixed += 1
            continue

        # 2. Deletion: letters are a substring of a clue word
        found_deletion = False
        for i, cw_full in enumerate(clue_words_norm):
            if len(cw_full) > len(lt_norm) and lt_norm in cw_full:
                # Found the source word
                # Determine what was deleted
                if cw_full.startswith(lt_norm):
                    deleted = cw_full[len(lt_norm):]
                    new_piece["mechanism"] = "deletion"
                    new_piece["clue_word"] = clue_words_raw[i]
                    new_piece["source"] = cw_full
                    new_piece["deleted"] = deleted
                elif cw_full.endswith(lt_norm):
                    deleted = cw_full[:len(cw_full) - len(lt_norm)]
                    new_piece["mechanism"] = "deletion"
                    new_piece["clue_word"] = clue_words_raw[i]
                    new_piece["source"] = cw_full
                    new_piece["deleted"] = deleted
                else:
                    # Middle deletion
                    idx_in = cw_full.index(lt_norm)
                    deleted = cw_full[:idx_in] + cw_full[idx_in + len(lt_norm):]
                    new_piece["mechanism"] = "deletion"
                    new_piece["clue_word"] = clue_words_raw[i]
                    new_piece["source"] = cw_full
                    new_piece["deleted"] = deleted

                fixed.append(new_piece)
                n_fixed += 1
                found_deletion = True
                break

        if found_deletion:
            continue

        # 3. Real synonym: find which clue word this is a synonym/abbreviation of
        found_source = False
        for i, cw_raw in enumerate(clue_words_raw):
            cw_clean = norm(cw_raw)
            if not cw_clean or len(cw_clean) < 2:
                continue
            # Check if clue_word -> letters exists in DB
            syns = ref_db.get_synonyms(cw_clean, max_len=len(lt_norm))
            if lt_norm in [norm(s) for s in syns]:
                new_piece["clue_word"] = cw_raw
                fixed.append(new_piece)
                n_fixed += 1
                found_source = True
                break
            # Check abbreviations
            abbrs = ref_db.get_abbreviations(cw_clean)
            if lt_norm in [norm(a) for a in abbrs]:
                new_piece["mechanism"] = "abbreviation"
                new_piece["clue_word"] = cw_raw
                fixed.append(new_piece)
                n_fixed += 1
                found_source = True
                break

        if found_source:
            continue

        # 4. Try multi-word clue phrases (2-word combinations)
        found_multi = False
        for i in range(len(clue_words_raw) - 1):
            phrase = clue_words_raw[i] + " " + clue_words_raw[i + 1]
            syns = ref_db.get_synonyms(phrase, max_len=len(lt_norm))
            if lt_norm in [norm(s) for s in syns]:
                new_piece["clue_word"] = phrase
                fixed.append(new_piece)
                n_fixed += 1
                found_multi = True
                break

        if found_multi:
            continue

        # Couldn't fix — keep as-is but mark as unresolved
        new_piece["_self_synonym"] = True
        fixed.append(new_piece)

    return fixed, n_fixed


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/fix_self_synonyms.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print("Loading RefDB...")
    ref_db = RefDB()
    print("Loading clues DB...")
    conn = sqlite3.connect(CLUES_DB, timeout=30)

    with open(input_path, encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]

    print(f"Processing {len(lines):,} results...")
    t0 = time.time()
    total_fixed = 0
    total_self = 0

    for i, l in enumerate(lines):
        p = l["payload"]
        comps = p.get("components", {})
        pieces = comps.get("ai_pieces", [])

        # Check if any self-synonyms exist
        has_self = any(
            p2.get("mechanism") == "synonym" and norm(p2.get("letters", "")) == norm(p2.get("clue_word", ""))
            for p2 in pieces
        )
        if not has_self:
            continue

        total_self += 1
        cid = l["clue_id"]
        row = conn.execute("SELECT clue_text FROM clues WHERE id=?", (cid,)).fetchone()
        if not row:
            continue

        fixed_pieces, n_fixed = fix_pieces(pieces, row[0], ref_db)
        total_fixed += n_fixed

        # Update the result
        comps["ai_pieces"] = fixed_pieces
        p["components"] = comps

        if (i + 1) % 10000 == 0:
            print(f"  {i + 1:,}/{len(lines):,}...")

    elapsed = time.time() - t0

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(json.dumps(l, ensure_ascii=False) + "\n")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Results with self-synonyms: {total_self:,}")
    print(f"  Pieces fixed: {total_fixed:,}")
    print(f"  Output: {output_path}")

    conn.close()


if __name__ == "__main__":
    main()
