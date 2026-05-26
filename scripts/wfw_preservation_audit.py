"""Generate an initial WFW preservation audit sample from existing corpora.

This is not a truth generator. It drafts word-role coverage from current
structured artifacts so the gaps can be inspected systematically. The point is
to expose what the existing data shape cannot preserve before changing solver
logic.

Usage:
    python scripts/wfw_preservation_audit.py --limit 1000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCES = [
    ("data/gold_training_data.jsonl", 350),
    ("data/batch_mechanical_all.jsonl", 350),
    ("data/parsed_explanations_v2_HIGH.jsonl", 300),
]


def _norm_word(word: str) -> str:
    return re.sub(r"^[^a-z0-9']+|[^a-z0-9']+$", "", (word or "").lower())


def _tokens(text: str) -> list[dict]:
    text = re.sub(r"\([^)]*\)\s*$", "", text or "")
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
    return [
        {"index": i, "text": word, "norm": _norm_word(word)}
        for i, word in enumerate(words)
    ]


def _phrase_tokens(phrase: str) -> list[str]:
    return [tok["norm"] for tok in _tokens(phrase) if tok["norm"]]


def _find_phrase_span(clue_tokens: list[dict], phrase: str) -> dict | None:
    phrase_norms = _phrase_tokens(phrase)
    if not phrase_norms:
        return None
    for start in range(0, len(clue_tokens) - len(phrase_norms) + 1):
        window = clue_tokens[start:start + len(phrase_norms)]
        if [tok["norm"] for tok in window] == phrase_norms:
            return {
                "start": start,
                "end": start + len(phrase_norms) - 1,
                "words": [tok["text"] for tok in window],
            }
    return None


def _load_jsonl(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _classify_row(row: dict, source_file: str) -> dict:
    payload = row.get("payload") or row
    structured = payload.get("structured") or row.get("structured") or {}
    components = payload.get("components") or structured or {}

    clue = payload.get("clue") or row.get("clue") or payload.get("clue_text") or ""
    answer = payload.get("answer") or row.get("answer") or ""
    explanation = (
        payload.get("explanation")
        or row.get("explanation")
        or payload.get("ai_explanation")
        or ""
    )
    definition = payload.get("definition", structured.get("definition"))
    wordplay_types = (
        payload.get("wordplay_types")
        or structured.get("wordplay_types")
        or ([payload["wordplay_type"]] if payload.get("wordplay_type") else [])
    )
    pieces = (
        components.get("ai_pieces")
        or components.get("components")
        or payload.get("pieces")
        or []
    )

    clue_tokens = _tokens(clue)
    roles = [
        {
            "index": tok["index"],
            "word": tok["text"],
            "role": "unaccounted",
            "evidence": None,
            "letters": None,
            "piece_id": None,
        }
        for tok in clue_tokens
    ]

    piece_records = []
    for piece_id, piece in enumerate(pieces, start=1):
        phrase = piece.get("clue_word") or piece.get("fodder") or ""
        letters = piece.get("letters") or piece.get("yields") or ""
        mechanism = piece.get("mechanism") or piece.get("type") or "unknown"
        span = _find_phrase_span(clue_tokens, phrase)
        piece_records.append({
            "piece_id": piece_id,
            "mechanism": mechanism,
            "clue_phrase": phrase,
            "letters": letters,
            "span": span,
            "raw": piece,
        })
        if span:
            for index in range(span["start"], span["end"] + 1):
                roles[index]["role"] = f"{mechanism}_source"
                roles[index]["evidence"] = phrase
                roles[index]["letters"] = letters
                roles[index]["piece_id"] = piece_id

    if definition:
        def_span = _find_phrase_span(clue_tokens, definition)
        if def_span:
            for index in range(def_span["start"], def_span["end"] + 1):
                roles[index]["role"] = "definition"
                roles[index]["evidence"] = definition
                roles[index]["letters"] = answer

    unaccounted = [role["word"] for role in roles if role["role"] == "unaccounted"]
    return {
        "source_file": source_file,
        "clue_id": row.get("clue_id") or payload.get("clue_id"),
        "clue": clue,
        "answer": answer,
        "explanation": explanation,
        "definition": definition,
        "wordplay_types": wordplay_types,
        "pieces": piece_records,
        "word_roles": roles,
        "unaccounted": unaccounted,
        "coverage": {
            "tokens": len(roles),
            "accounted": len(roles) - len(unaccounted),
            "unaccounted": len(unaccounted),
        },
    }


def generate(limit: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    emitted = 0
    with output.open("w", encoding="utf-8") as handle:
        for rel_path, source_limit in DEFAULT_SOURCES:
            if emitted >= limit:
                break
            path = ROOT / rel_path
            if not path.exists():
                continue
            remaining = limit - emitted
            for row in _load_jsonl(path, min(source_limit, remaining)):
                handle.write(json.dumps(_classify_row(row, rel_path), ensure_ascii=False))
                handle.write("\n")
                emitted += 1
                if emitted >= limit:
                    break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "documents" / "wfw_preservation_corpus_initial_1000.jsonl",
    )
    args = parser.parse_args()
    generate(args.limit, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
