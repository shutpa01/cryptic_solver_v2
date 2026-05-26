"""Build a GT V2 research cohort from existing structured explanations.

This is an R&D extractor, not a solver stage.

The supervision source is the structured explanation itself:
clue_word -> answer letters -> mechanism, plus assembly.

Blog explanations are retained when present, but they are not required. This
lets Daily Mail and other structured-only sources participate in the core
science cohort.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "data" / "clues_master.db"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "structured_research_cohort_sample_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "structured_research_cohort_2026-05-17.md"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text or "")]


def clean_word(text: str) -> str:
    return (text or "").lower().strip(".,;:!?\"'()[]{}\u2018\u2019\u201c\u201d")


def norm_tokens(tokens: list[str]) -> list[str]:
    return [clean_word(t) for t in tokens]


def clean_piece_text(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"^[=:\-\s]+", "", text)
    text = re.sub(r"\bi\.e\..*$", "", text, flags=re.I)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\".*?\"", " ", text)
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contiguous_span(tokens: list[str], phrase: str, used: set[int] | None = None) -> dict[str, Any] | None:
    used = used or set()
    phrase_tokens = norm_tokens(tokenize(phrase))
    all_tokens = norm_tokens(tokens)
    if not phrase_tokens:
        return None
    n = len(phrase_tokens)
    for i in range(0, len(all_tokens) - n + 1):
        idxs = set(range(i, i + n))
        if idxs & used:
            continue
        if all_tokens[i : i + n] == phrase_tokens:
            return {"start": i, "span": n, "text": " ".join(tokens[i : i + n])}
    return None


def definition_indices(clue_tokens: list[str], definition: str | None) -> set[int]:
    """Find exact leading/trailing definition token spans."""
    if not definition:
        return set()
    def_tokens = norm_tokens(tokenize(definition))
    all_tokens = norm_tokens(clue_tokens)
    if not def_tokens or len(def_tokens) > len(all_tokens):
        return set()
    n = len(def_tokens)
    if all_tokens[:n] == def_tokens:
        return set(range(0, n))
    if all_tokens[-n:] == def_tokens:
        return set(range(len(all_tokens) - n, len(all_tokens)))
    return set()


def parse_json_maybe(text: str | None, fallback: Any) -> Any:
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        return fallback


def normalise_wordplay_types(raw: str | None) -> list[str]:
    value = parse_json_maybe(raw, None)
    if value is None:
        value = raw
    if isinstance(value, str):
        return [value] if value else ["blank"]
    if isinstance(value, list):
        return [str(v) for v in value] or ["blank"]
    return ["blank"]


def extract_pieces(components: dict[str, Any]) -> list[dict[str, Any]]:
    pieces = components.get("ai_pieces") or components.get("pieces") or []
    if not isinstance(pieces, list):
        return []
    out = []
    for piece in pieces:
        if not isinstance(piece, dict):
            continue
        clue_word = piece.get("clue_word") or piece.get("source") or piece.get("text") or ""
        letters = piece.get("letters") or piece.get("value") or ""
        mechanism = piece.get("mechanism") or piece.get("type") or ""
        out.append(
            {
                "clue_word": str(clue_word),
                "letters": str(letters),
                "mechanism": str(mechanism),
                "raw": piece,
            }
        )
    return out


def map_piece_to_clue(piece: dict[str, Any], clue_tokens: list[str], used: set[int]) -> dict[str, Any]:
    raw_text = piece["clue_word"]
    candidates = []
    cleaned = clean_piece_text(raw_text)
    if cleaned:
        candidates.append(cleaned)
    if raw_text and raw_text != cleaned:
        candidates.append(raw_text)

    seen = set()
    for candidate in candidates:
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        span = contiguous_span(clue_tokens, candidate, used)
        if span:
            return {"status": "mapped", "method": "exact_piece_text", **span}

    # Last resort for pieces like "style," or "'s".
    words = tokenize(cleaned)
    if len(words) == 1:
        target = clean_word(words[0])
        matches = [
            idx for idx, token in enumerate(clue_tokens)
            if idx not in used and clean_word(token) == target
        ]
        if len(matches) == 1:
            idx = matches[0]
            return {"status": "mapped", "method": "single_token", "start": idx, "span": 1, "text": clue_tokens[idx]}

    return {"status": "unmapped", "candidate_text": cleaned or raw_text}


def build_record(row: sqlite3.Row) -> dict[str, Any] | None:
    components = parse_json_maybe(row["components"], {})
    if not isinstance(components, dict):
        return None
    pieces = extract_pieces(components)
    assembly = components.get("assembly")
    if not pieces or not assembly:
        return None

    clue_tokens = tokenize(row["clue_text"])
    definition = row["definition_text"] or row["clue_definition"]
    def_indices = definition_indices(clue_tokens, definition)
    used: set[int] = set(def_indices)
    mapped_pieces = []
    for idx, piece in enumerate(pieces):
        mapping = map_piece_to_clue(piece, clue_tokens, used)
        if mapping["status"] == "mapped":
            used.update(range(mapping["start"], mapping["start"] + mapping["span"]))
        mapped_pieces.append({**piece, "piece_index": idx, "mapping": mapping})

    labels = []
    for idx, token in enumerate(clue_tokens):
        label = "DEF" if idx in def_indices else "RESIDUE"
        piece_index = None
        for piece in mapped_pieces:
            mapping = piece["mapping"]
            if mapping["status"] != "mapped":
                continue
            start = mapping["start"]
            end = start + mapping["span"]
            if start <= idx < end:
                label = "SOURCE"
                piece_index = piece["piece_index"]
                break
        labels.append({"index": idx, "text": token, "label": label, "piece_index": piece_index})

    residue = [x for x in labels if x["label"] == "RESIDUE"]
    mapped_count = sum(1 for p in mapped_pieces if p["mapping"]["status"] == "mapped")
    wordplay_types = normalise_wordplay_types(row["wordplay_types"])

    return {
        "id": row["id"],
        "source": row["source"],
        "puzzle_number": row["puzzle_number"],
        "clue_number": row["clue_number"],
        "clue": row["clue_text"],
        "answer": row["answer"],
        "definition": definition,
        "definition_span": {
            "start": row["definition_start"],
            "end": row["definition_end"],
        },
        "blog_explanation": row["blog_explanation"] or "",
        "ai_explanation": row["ai_explanation"] or "",
        "wordplay_types": wordplay_types,
        "model_version": row["model_version"],
        "confidence": row["confidence"],
        "pieces": mapped_pieces,
        "assembly": assembly,
        "token_labels": labels,
        "residue": residue,
        "quality": {
            "piece_count": len(mapped_pieces),
            "mapped_piece_count": mapped_count,
            "complete_piece_mapping": mapped_count == len(mapped_pieces),
            "has_blog": bool((row["blog_explanation"] or "").strip()),
            "has_definition_span": row["definition_start"] is not None and row["definition_end"] is not None,
            "has_exact_definition_tokens": bool(def_indices),
        },
    }


def load_records(db_path: Path, min_confidence: float) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.clue_text,
               c.answer, c.definition AS clue_definition,
               c.explanation AS blog_explanation, c.ai_explanation,
               se.definition_text, se.definition_start, se.definition_end,
               se.wordplay_types, se.components, se.model_version, se.confidence
        FROM structured_explanations se
        JOIN clues c ON c.id = se.clue_id
        WHERE se.confidence >= ?
          AND c.clue_text IS NOT NULL AND TRIM(c.clue_text) != ''
          AND c.answer IS NOT NULL AND TRIM(c.answer) != ''
          AND se.components IS NOT NULL AND TRIM(se.components) != ''
        """,
        (min_confidence,),
    ).fetchall()
    conn.close()

    records = []
    for row in rows:
        record = build_record(row)
        if record:
            records.append(record)
    return records


def balanced_sample(records: list[dict[str, Any]], per_source_type: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        primary_type = record["wordplay_types"][0] if record["wordplay_types"] else "blank"
        buckets[(record["source"], primary_type)].append(record)

    sample = []
    for key in sorted(buckets):
        complete = [r for r in buckets[key] if r["quality"]["complete_piece_mapping"]]
        partial = [r for r in buckets[key] if not r["quality"]["complete_piece_mapping"]]
        chosen = complete[:per_source_type]
        if len(chosen) < per_source_type:
            chosen.extend(partial[: per_source_type - len(chosen)])
        sample.extend(chosen)
    return sample


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(all_records: list[dict[str, Any]], sample: list[dict[str, Any]], path: Path) -> None:
    by_source = defaultdict(Counter)
    by_type = defaultdict(Counter)
    examples = defaultdict(list)

    for record in all_records:
        source = record["source"]
        q = record["quality"]
        by_source[source]["eligible"] += 1
        by_source[source]["has_blog"] += q["has_blog"]
        by_source[source]["has_definition_span"] += q["has_definition_span"]
        by_source[source]["has_exact_definition_tokens"] += q["has_exact_definition_tokens"]
        by_source[source]["complete_piece_mapping"] += q["complete_piece_mapping"]
        by_source[source]["partial_piece_mapping"] += not q["complete_piece_mapping"]
        by_source[source]["pieces"] += q["piece_count"]
        by_source[source]["mapped_pieces"] += q["mapped_piece_count"]
        primary_type = record["wordplay_types"][0] if record["wordplay_types"] else "blank"
        by_type[source][primary_type] += 1
        if len(examples[source]) < 2 and q["piece_count"] and record["assembly"]:
            examples[source].append(record)

    lines = [
        "# Structured Research Cohort",
        "",
        "Date: 2026-05-17",
        "",
        "This is the first GT V2 research cohort built from existing structured explanations.",
        "The supervision is the structured mapping from clue words to answer pieces, plus assembly.",
        "Blog explanations are retained when present, but they are not required.",
        "",
        f"Eligible high-confidence records: {len(all_records)}",
        f"Balanced sample records written: {len(sample)}",
        "",
        "## Source Inventory",
        "",
    ]

    for source in sorted(by_source):
        c = by_source[source]
        complete = c["complete_piece_mapping"]
        eligible = c["eligible"]
        mapped = c["mapped_pieces"]
        pieces = c["pieces"]
        lines.append(f"- `{source}`: {eligible} eligible; {complete} complete piece mappings; {c['has_blog']} with blog text; {c['has_exact_definition_tokens']} with exact definition tokens; {mapped}/{pieces} pieces mapped")

    lines.extend(["", "## Operation Mix By Source", ""])
    for source in sorted(by_type):
        top = ", ".join(f"{typ}={count}" for typ, count in by_type[source].most_common(8))
        lines.append(f"- `{source}`: {top}")

    lines.extend(["", "## Example Records", ""])
    for source in sorted(examples):
        lines.append(f"`{source}`")
        for record in examples[source]:
            pieces = []
            for piece in record["pieces"][:4]:
                mapping = piece["mapping"]
                mapped = mapping["text"] if mapping["status"] == "mapped" else "unmapped"
                pieces.append(f"{piece['clue_word']} -> {piece['letters']} ({mapped})")
            residue = " ".join(x["text"] for x in record["residue"])
            lines.append(f"- `{record['clue']}` ({record['answer']})")
            lines.append(f"  Definition: `{record['definition']}`")
            lines.append(f"  Pieces: {'; '.join(pieces)}")
            lines.append(f"  Assembly: `{record['assembly']}`")
            lines.append(f"  Residue: `{residue}`")
            if record["blog_explanation"]:
                lines.append(f"  Blog: `{record['blog_explanation'][:240]}`")

    lines.extend(
        [
            "",
            "## Research Use",
            "",
            "This cohort should be used before any further Times-only mining.",
            "The next science question is whether surface grammar can recover the same",
            "SOURCE/RESIDUE boundaries and operation attachment found in these structured records.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--per-source-type", type=int, default=80)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = load_records(args.db, args.min_confidence)
    sample = balanced_sample(records, args.per_source_type)
    write_jsonl(sample, args.jsonl_out)
    write_report(records, sample, args.report_out)
    print(f"eligible={len(records)}")
    print(f"sample={len(sample)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
