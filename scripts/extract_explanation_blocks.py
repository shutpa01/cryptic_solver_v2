"""Extract block-candidate records from human explanation text.

This is a mining/design tool, not a solver stage.

It starts from the Times explanation notation parser and preserves both the raw
human explanation evidence and the parser's current interpretation. The output is
intended for Grammar Triage / WFW design review.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cryptic_taxonomy.analysis.notation_parser import parse_explanation

DEFAULT_INPUT_DB = PROJECT_ROOT / "data" / "times_explanations.db"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "explanation_block_candidates_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "explanation_block_candidates_2026-05-17.md"


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")
PAREN_RE = re.compile(r"\(([^)]+)\)")


def clean_answer(answer: str) -> str:
    return re.sub(r"[\s\-]", "", (answer or "").upper())


def clean_phrase(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[{}[\]()*~+]", " ", text)
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower().strip("'") for m in WORD_RE.finditer(text or "")]


def strip_definition(clue: str, definition: str | None) -> str:
    """Remove an exact leading/trailing definition, if the corpus supplies one."""
    clue = (clue or "").strip()
    definition = (definition or "").strip()
    if not clue or not definition:
        return clue

    clue_tokens = tokenize(clue)
    def_tokens = tokenize(definition)
    if not def_tokens or len(def_tokens) > len(clue_tokens):
        return clue

    if clue_tokens[: len(def_tokens)] == def_tokens:
        return " ".join(clue.split()[len(def_tokens) :]).strip()
    if clue_tokens[-len(def_tokens) :] == def_tokens:
        return " ".join(clue.split()[: -len(def_tokens)]).strip()
    return clue


def contiguous_span(tokens: list[str], phrase: str) -> dict[str, Any] | None:
    phrase_tokens = tokenize(phrase)
    if not phrase_tokens or len(phrase_tokens) > len(tokens):
        return None
    n = len(phrase_tokens)
    for i in range(0, len(tokens) - n + 1):
        if tokens[i : i + n] == phrase_tokens:
            return {
                "start_token": i,
                "token_count": n,
                "text": " ".join(tokens[i : i + n]),
            }
    return None


def split_hint_text(text: str) -> str:
    text = re.split(r"[\u2014\u2013/\[]", text or "")[0]
    text = text.strip().rstrip(".,;:")
    return text


def is_compact_source(text: str) -> bool:
    """Allow raw source text only when it does not look like blog prose."""
    if not text:
        return False
    if len(text) > 45:
        return False
    if re.search(r"[.!?;]", text):
        return False
    if len(tokenize(text)) > 5:
        return False
    return True


def acceptable_candidate(text: str, source_type: str) -> bool:
    words = tokenize(text)
    if not words:
        return False
    if source_type in {"ANA", "HID"}:
        return len(words) <= 10
    return len(words) <= 5


def add_candidate(candidates: list[str], text: str, source_type: str) -> None:
    text = split_hint_text(text)
    if acceptable_candidate(text, source_type):
        candidates.append(text)


def candidate_phrases(source_word: str, gloss: str, source_type: str) -> list[str]:
    """Return phrase candidates without collapsing raw evidence.

    The raw source text is still stored on the block. This function is narrower:
    it extracts plausible clue-facing phrases from glosses, parentheses, and
    compact source expressions.
    """
    candidates: list[str] = []

    if gloss:
        add_candidate(candidates, gloss, source_type)

    for paren in PAREN_RE.findall(source_word or ""):
        add_candidate(candidates, paren, source_type)

    m = re.match(r"^[A-Z.]+\s*(?:=|is|for)\s+(.+)$", source_word or "")
    if m:
        add_candidate(candidates, m.group(1), source_type)

    if source_word and is_compact_source(source_word):
        if source_type in {"ANA", "HID", "HOM", "REV"}:
            add_candidate(candidates, source_word, source_type)
        elif not re.search(r"[(){}[\]~+=]", source_word) and not re.match(
            r"^[A-Z.]+\s+(?:is|for)\s+", source_word
        ):
            add_candidate(candidates, source_word, source_type)

    # Notation like RUNT (small issue of litter) containing ECO (green) often
    # reaches us as a single container piece. Preserve the internal hints.
    if source_type in {"CON", "CON+REV"}:
        for chunk in re.split(r"\b(?:in|inside|containing|contains|around)\b", source_word or "", flags=re.I):
            if is_compact_source(chunk) and not re.search(r"[(){}[\]~+]", chunk):
                add_candidate(candidates, chunk, source_type)

    cleaned: list[str] = []
    seen = set()
    for candidate in candidates:
        phrase = clean_phrase(candidate)
        if not phrase or phrase in seen:
            continue
        seen.add(phrase)
        cleaned.append(phrase)
    return cleaned


def operation_markers(explanation: str) -> list[str]:
    markers = [clean_phrase(m) for m in BRACKET_RE.findall(explanation or "")]
    keywords = [
        "contained",
        "containing",
        "contains",
        "inside",
        "around",
        "reversed",
        "backing",
        "rising",
        "without",
        "missing",
        "losing",
        "anagram",
        "sounds like",
        "homophone",
        "hidden",
    ]
    low = (explanation or "").lower()
    markers.extend(k for k in keywords if k in low)
    out = []
    seen = set()
    for marker in markers:
        if marker and marker not in seen:
            seen.add(marker)
            out.append(marker)
    return out


def phrase_kind(phrase: str) -> str:
    token_count = len(tokenize(phrase))
    if token_count >= 2:
        return "phrase"
    if token_count == 1:
        return "single"
    return "empty"


def block_objections(block: dict[str, Any]) -> list[str]:
    objections = []
    if not block["candidate_phrases"]:
        objections.append("no_source_phrase")
    if block["candidate_phrases"] and not block["clue_span_candidates"]:
        objections.append("no_exact_clue_span_match")
    if block["source_type"] in {"CON", "CON+REV"} and " in " in block["source_word"]:
        objections.append("compound_container_source_needs_internal_blocks")
    if any(c["kind"] == "phrase" for c in block["candidate_phrases"]):
        objections.append("preserve_as_possible_phrase_block")
    return objections


def build_record(row: sqlite3.Row) -> dict[str, Any]:
    clue = row["clue_text"] or ""
    answer = clean_answer(row["answer"] or "")
    definition = row["definition"] or ""
    explanation = row["explanation"] or ""
    wordplay = strip_definition(clue, definition)
    clue_tokens = tokenize(clue)
    wordplay_tokens = tokenize(wordplay)

    parsed = parse_explanation(explanation, answer)
    blocks = []
    for index, piece in enumerate(parsed.pieces):
        phrases = candidate_phrases(piece.source_word, piece.gloss, piece.source_type)
        phrase_records = [
            {
                "text": phrase,
                "kind": phrase_kind(phrase),
            }
            for phrase in phrases
        ]
        span_candidates = []
        for phrase in phrases:
            clue_span = contiguous_span(clue_tokens, phrase)
            wordplay_span = contiguous_span(wordplay_tokens, phrase)
            if clue_span:
                span_candidates.append({"space": "clue", **clue_span})
            if wordplay_span:
                span_candidates.append({"space": "wordplay", **wordplay_span})

        block = {
            "piece_index": index,
            "letters": piece.letters,
            "source_type": piece.source_type,
            "source_word": piece.source_word,
            "gloss": piece.gloss,
            "candidate_phrases": phrase_records,
            "clue_span_candidates": span_candidates,
        }
        block["objections"] = block_objections(block)
        blocks.append(block)

    return {
        "puzzle_number": row["puzzle_number"],
        "clue": clue,
        "answer": answer,
        "definition": definition,
        "wordplay_space": wordplay,
        "explanation": explanation,
        "parse": {
            "operation": parsed.operation,
            "verified": parsed.verified,
            "sub_operations": parsed.sub_operations,
            "operation_markers": operation_markers(explanation),
        },
        "blocks": blocks,
    }


def record_is_interesting(record: dict[str, Any]) -> bool:
    if record["parse"]["operation"] in {"unparsed", "double_definition", "cryptic_definition"}:
        return False
    for block in record["blocks"]:
        if any(p["kind"] == "phrase" for p in block["candidate_phrases"]):
            return True
        if block["source_type"] in {"CON", "CON+REV", "DEL", "HOM", "REV", "ANA", "HID"}:
            return True
        if block["objections"]:
            return True
    return False


def load_rows(db_path: Path, limit: int | None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT puzzle_number, clue_text, answer, definition, explanation "
        "FROM clues WHERE explanation IS NOT NULL AND answer IS NOT NULL "
        "AND clue_text IS NOT NULL AND TRIM(clue_text) != '' "
        "ORDER BY puzzle_number"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    conn.close()
    return rows


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(records: list[dict[str, Any]], path: Path, scanned: int) -> None:
    op_counts = Counter(r["parse"]["operation"] for r in records)
    source_counts = Counter(
        b["source_type"] for r in records for b in r["blocks"] if b["source_type"]
    )
    phrase_blocks = [
        (r, b)
        for r in records
        for b in r["blocks"]
        if any(p["kind"] == "phrase" for p in b["candidate_phrases"])
    ]
    objection_counts = Counter(
        objection for r in records for b in r["blocks"] for objection in b["objections"]
    )

    lines = [
        "# Explanation Block Candidate Extraction",
        "",
        "Date: 2026-05-17",
        "",
        "This is a generated mining report from `scripts/extract_explanation_blocks.py`.",
        "It is not a solver change.",
        "",
        f"Rows scanned: {scanned}",
        f"Interesting records written: {len(records)}",
        f"Extracted blocks: {sum(len(r['blocks']) for r in records)}",
        f"Phrase-shaped blocks: {len(phrase_blocks)}",
        "",
        "## Operation Mix",
        "",
    ]
    for op, count in op_counts.most_common(12):
        lines.append(f"- `{op}`: {count}")

    lines.extend(["", "## Source Type Mix", ""])
    for source_type, count in source_counts.most_common(12):
        lines.append(f"- `{source_type}`: {count}")

    lines.extend(["", "## Mapping Objections", ""])
    for objection, count in objection_counts.most_common(12):
        lines.append(f"- `{objection}`: {count}")

    def phrase_priority(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[int, int, str]:
        record, block = item
        has_span = bool(block["clue_span_candidates"])
        has_clue = bool(record["clue"])
        source_rank = 0 if block["source_type"] in {"SYN", "ABR", "DEL"} else 1
        return (0 if has_span else 1, 0 if has_clue else 1, source_rank, record["clue"])

    lines.extend(["", "## Phrase Examples", ""])
    for record, block in sorted(phrase_blocks, key=phrase_priority)[:24]:
        phrase_texts = [p["text"] for p in block["candidate_phrases"] if p["kind"] == "phrase"]
        phrase = phrase_texts[0] if phrase_texts else ""
        span_status = "span found" if block["clue_span_candidates"] else "no exact span"
        lines.append(
            f"- `{phrase}` -> `{block['letters']}` [{block['source_type']}] "
            f"({span_status}) in `{record['clue']}`"
        )

    lines.extend(["", "## Qualitative Review Seeds", ""])
    seed_terms = [
        "in charge",
        "at home",
        "on the way",
        "hospital department",
        "small issue of litter",
        "without delay",
    ]
    added = set()
    for term in seed_terms:
        for record, block in phrase_blocks:
            phrase_texts = [p["text"] for p in block["candidate_phrases"] if p["kind"] == "phrase"]
            if any(term in p for p in phrase_texts):
                key = (term, record["clue"], block["letters"])
                if key in added:
                    continue
                added.add(key)
                span_status = "span found" if block["clue_span_candidates"] else "no exact span"
                lines.append(
                    f"- `{term}` appears as `{block['letters']}` [{block['source_type']}] "
                    f"({span_status}) in `{record['clue']}`"
                )
                break

    lines.extend(
        [
            "",
            "## Design Reading",
            "",
            "The extractor deliberately keeps raw explanation text, parsed pieces,",
            "candidate phrases, exact clue-span candidates, and objections together.",
            "That is the preservation rule in executable form.",
            "",
            "The important next review is qualitative: inspect the JSONL examples where",
            "phrase blocks have no exact clue span, and where container sources need",
            "internal blocks. Those are likely to teach Grammar Triage the most.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-db", type=Path, default=DEFAULT_INPUT_DB)
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    rows = load_rows(args.input_db, args.limit)
    records = []
    for row in rows:
        record = build_record(row)
        if record_is_interesting(record):
            records.append(record)

    write_jsonl(records, args.jsonl_out)
    write_report(records, args.report_out, len(rows))
    print(f"rows_scanned={len(rows)}")
    print(f"interesting_records={len(records)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
