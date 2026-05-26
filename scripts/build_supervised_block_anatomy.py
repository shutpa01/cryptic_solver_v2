"""Build explanation-supervised clue block anatomy records.

This is an R&D corpus builder, not a solver stage.

The science question is whether surface grammar can predict cryptic block
boundaries. This script uses sanitised human explanations as supervision:

1. Parse the human explanation into answer-producing pieces.
2. Map those pieces back to clue spans where possible.
3. Treat unmapped clue tokens as residue.
4. Store the grammar signature over the same token spans.

The resulting JSONL is training/evaluation material for Grammar Triage V2.
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

from cryptic_taxonomy.analysis.improved_mapper import MappingDB
from cryptic_taxonomy.analysis.notation_parser import Piece, parse_explanation


DEFAULT_TIMES_DB = PROJECT_ROOT / "data" / "times_explanations.db"
DEFAULT_REF_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "supervised_block_anatomy_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "supervised_block_anatomy_2026-05-17.md"

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
PAREN_RE = re.compile(r"\(([^)]+)\)")
BRACKET_RE = re.compile(r"\[([^\]]+)\]")


try:
    import spacy

    _NLP = None

    def get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load("en_core_web_sm")
        return _NLP

except Exception:

    def get_nlp():
        return None


def clean_answer(answer: str) -> str:
    return re.sub(r"[\s\-]", "", (answer or "").upper())


def clean_word(text: str) -> str:
    return (text or "").lower().strip(".,;:!?\"'()[]{}\u2018\u2019\u201c\u201d")


def clean_phrase(text: str) -> str:
    text = (text or "").replace("\u00a0", " ")
    text = re.sub(r"[{}[\]()*~+]", " ", text)
    text = re.sub(r"[^A-Za-z0-9' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text or "")]


def norm_tokens(tokens: list[str]) -> list[str]:
    return [clean_word(t) for t in tokens]


def strip_definition_tokens(clue_tokens: list[str], definition: str) -> tuple[list[str], int, int | None]:
    def_tokens = tokenize(definition or "")
    clean_clue = norm_tokens(clue_tokens)
    clean_def = norm_tokens(def_tokens)
    if not clean_def or len(clean_def) > len(clean_clue):
        return clue_tokens, 0, None
    if clean_clue[: len(clean_def)] == clean_def:
        return clue_tokens[len(clean_def) :], len(clean_def), 0
    if clean_clue[-len(clean_def) :] == clean_def:
        return clue_tokens[: -len(clean_def)], 0, len(clean_clue) - len(clean_def)
    return clue_tokens, 0, None


def contiguous_span(tokens: list[str], phrase: str, used: set[int] | None = None) -> tuple[int, int] | None:
    used = used or set()
    phrase_tokens = norm_tokens(tokenize(phrase))
    token_norm = norm_tokens(tokens)
    if not phrase_tokens:
        return None
    span = len(phrase_tokens)
    for i in range(0, len(token_norm) - span + 1):
        indices = set(range(i, i + span))
        if indices & used:
            continue
        if token_norm[i : i + span] == phrase_tokens:
            return i, span
    return None


def piece_hints(piece: Piece) -> list[str]:
    hints = []
    if piece.gloss:
        hints.append(piece.gloss)
    for paren in PAREN_RE.findall(piece.source_word or ""):
        hints.append(paren)
    match = re.match(r"^[A-Z.]+\s*(?:=|is|for)\s+(.+)$", piece.source_word or "")
    if match:
        hints.append(match.group(1))
    if piece.source_type in {"ANA", "HID", "HOM", "REV"} and piece.source_word:
        hints.append(piece.source_word)

    out = []
    seen = set()
    for hint in hints:
        hint = clean_phrase(re.split(r"[\u2014\u2013/\[]", hint)[0])
        if hint and hint not in seen and len(tokenize(hint)) <= 6:
            seen.add(hint)
            out.append(hint)
    return out


def map_piece(piece: Piece, tokens: list[str], db: MappingDB, used: set[int]) -> dict[str, Any]:
    hints = piece_hints(piece)
    for hint in hints:
        found = contiguous_span(tokens, hint, used)
        if found:
            start, span = found
            return {
                "status": "mapped",
                "method": "hint_exact",
                "start": start,
                "span": span,
                "text": " ".join(tokens[start : start + span]),
                "hint": hint,
            }

    letters = piece.letters.upper()
    candidates: list[tuple[int, int, str]] = []
    for span in range(min(6, len(tokens)), 0, -1):
        for start in range(0, len(tokens) - span + 1):
            indices = set(range(start, start + span))
            if indices & used:
                continue
            words = tokens[start : start + span]
            if piece.source_type == "ABR" and db.phrase_produces_abbr(words, letters):
                candidates.append((start, span, "db_abbreviation"))
            elif piece.source_type in {"SYN", "DEL", "REV"}:
                lookup = letters[::-1] if piece.source_type == "REV" else letters
                if db.phrase_produces_syn(words, lookup):
                    candidates.append((start, span, "db_synonym"))

    if len(candidates) == 1:
        start, span, method = candidates[0]
        return {
            "status": "mapped",
            "method": method,
            "start": start,
            "span": span,
            "text": " ".join(tokens[start : start + span]),
            "hint": None,
        }

    if len(candidates) > 1:
        return {
            "status": "ambiguous",
            "method": "db_multiple",
            "candidates": [
                {"start": s, "span": sp, "text": " ".join(tokens[s : s + sp]), "method": m}
                for s, sp, m in candidates[:8]
            ],
            "hints": hints,
        }

    return {"status": "unmapped", "hints": hints}


def grammar_tokens(tokens: list[str]) -> list[dict[str, Any]]:
    nlp = get_nlp()
    if nlp is None:
        return [{"text": t, "pos": None, "tag": None, "dep": None, "head": None} for t in tokens]

    doc = nlp(" ".join(tokens))
    out = []
    for token in doc:
        if token.is_space:
            continue
        out.append(
            {
                "text": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "dep": token.dep_,
                "head": token.head.i,
            }
        )
    return out


def operation_markers(explanation: str) -> list[str]:
    markers = [clean_phrase(m) for m in BRACKET_RE.findall(explanation or "")]
    keywords = [
        "contained",
        "containing",
        "contains",
        "inside",
        "inserted",
        "insert",
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


def supervised_pieces(parsed_pieces: list[Piece], explanation: str) -> tuple[list[Piece], list[str]]:
    """Use clear human notation to recover source blocks hidden inside CON pieces."""
    notes = []
    pieces: list[Piece] = []
    consumed_container = False

    containing = re.search(
        r"([A-Z][A-Z~{}]*)\s*\(([^)]{1,80})\)\s+"
        r"(?:containing|contains)\s*(?:\([^)]{1,80}\)\s*)?"
        r"([A-Z][A-Z~{}]*)\s*\(([^)]{1,80})\)",
        explanation or "",
    )
    contained_by = re.search(
        r"([A-Z][A-Z~{}]*)\s*\(([^)]{1,80})\)\s+"
        r"contained\s+by\s*(?:\[[^\]]{1,80}\]\s*)?"
        r"([A-Z][A-Z~{}]*)\s*\(([^)]{1,80})\)",
        explanation or "",
    )

    expansion: list[Piece] = []
    if containing:
        outer_letters, outer_gloss, inner_letters, inner_gloss = containing.groups()
        expansion = [
            Piece(clean_answer(outer_letters), "SYN", f"{outer_letters} ({outer_gloss})", outer_gloss),
            Piece(clean_answer(inner_letters), "SYN", f"{inner_letters} ({inner_gloss})", inner_gloss),
        ]
        notes.append("expanded_container_containing")
    elif contained_by:
        inner_letters, inner_gloss, outer_letters, outer_gloss = contained_by.groups()
        expansion = [
            Piece(clean_answer(inner_letters), "SYN", f"{inner_letters} ({inner_gloss})", inner_gloss),
            Piece(clean_answer(outer_letters), "SYN", f"{outer_letters} ({outer_gloss})", outer_gloss),
        ]
        notes.append("expanded_container_contained_by")

    for piece in parsed_pieces:
        if expansion and piece.source_type in {"CON", "CON+REV"} and not consumed_container:
            pieces.extend(expansion)
            consumed_container = True
        else:
            pieces.append(piece)

    return pieces, notes


def build_record(row: sqlite3.Row, db: MappingDB) -> dict[str, Any] | None:
    clue = row["clue_text"] or ""
    answer = clean_answer(row["answer"] or "")
    definition = row["definition"] or ""
    explanation = row["explanation"] or ""
    clue_tokens = tokenize(clue)
    wordplay_tokens, left_def_len, right_def_start = strip_definition_tokens(clue_tokens, definition)
    if not clue_tokens or not wordplay_tokens or not answer or not explanation:
        return None

    parsed = parse_explanation(explanation, answer)
    if not parsed.pieces or parsed.operation in {"unparsed", "double_definition", "cryptic_definition"}:
        return None

    used: set[int] = set()
    pieces, supervision_notes = supervised_pieces(parsed.pieces, explanation)

    blocks = []
    for idx, piece in enumerate(pieces):
        mapped = map_piece(piece, wordplay_tokens, db, used)
        if mapped["status"] == "mapped":
            used.update(range(mapped["start"], mapped["start"] + mapped["span"]))
        blocks.append(
            {
                "piece_index": idx,
                "letters": piece.letters,
                "source_type": piece.source_type,
                "source_word": piece.source_word,
                "gloss": piece.gloss,
                "hints": piece_hints(piece),
                "mapping": mapped,
            }
        )

    residue = [
        {"index": idx, "text": token}
        for idx, token in enumerate(wordplay_tokens)
        if idx not in used
    ]

    token_labels = []
    for idx, token in enumerate(wordplay_tokens):
        label = "RESIDUE"
        block_index = None
        for block in blocks:
            mapping = block["mapping"]
            if mapping.get("status") == "mapped":
                start = mapping["start"]
                end = start + mapping["span"]
                if start <= idx < end:
                    label = "SOURCE"
                    block_index = block["piece_index"]
                    break
        token_labels.append({"index": idx, "text": token, "label": label, "block_index": block_index})

    mapped_count = sum(1 for block in blocks if block["mapping"].get("status") == "mapped")
    return {
        "puzzle_number": row["puzzle_number"],
        "clue": clue,
        "answer": answer,
        "definition": definition,
        "definition_position": {
            "left_token_count": left_def_len,
            "right_start": right_def_start,
        },
        "wordplay_tokens": wordplay_tokens,
        "explanation": explanation,
        "parse": {
            "operation": parsed.operation,
            "verified": parsed.verified,
            "sub_operations": parsed.sub_operations,
            "operation_markers": operation_markers(explanation),
            "supervision_notes": supervision_notes,
        },
        "blocks": blocks,
        "residue": residue,
        "token_labels": token_labels,
        "grammar": grammar_tokens(wordplay_tokens),
        "quality": {
            "mapped_piece_count": mapped_count,
            "piece_count": len(blocks),
            "complete_mapping": mapped_count == len(blocks),
            "residue_token_count": len(residue),
        },
    }


def load_rows(db_path: Path, limit: int | None) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT puzzle_number, clue_text, answer, definition, explanation "
        "FROM clues "
        "WHERE clue_text IS NOT NULL AND TRIM(clue_text) != '' "
        "AND answer IS NOT NULL AND TRIM(answer) != '' "
        "AND definition IS NOT NULL AND TRIM(definition) != '' "
        "AND explanation IS NOT NULL AND TRIM(explanation) != '' "
        "ORDER BY puzzle_number DESC"
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


def block_text(record: dict[str, Any], block: dict[str, Any]) -> str:
    mapping = block["mapping"]
    if mapping.get("status") == "mapped":
        return mapping["text"]
    if block["hints"]:
        return block["hints"][0]
    return block["source_word"]


def write_report(records: list[dict[str, Any]], path: Path, scanned: int) -> None:
    complete = [r for r in records if r["quality"]["complete_mapping"]]
    useful = [
        r
        for r in complete
        if 1 <= r["quality"]["residue_token_count"] <= 4
        and r["parse"]["operation"] in {"container", "container_charade", "reversal_container", "charade", "del"}
    ]
    op_counts = Counter(r["parse"]["operation"] for r in records)
    residue_counts = Counter(" ".join(t["text"].lower() for t in r["residue"]) for r in useful)
    grammar_available = any(
        token.get("tag") for record in records for token in record.get("grammar", [])
    )

    lines = [
        "# Supervised Block Anatomy Corpus",
        "",
        "Date: 2026-05-17",
        "",
        "This is an R&D artefact for GT V2. It uses sanitised explanations as",
        "supervision to mark answer-producing clue blocks, then records the",
        "unmapped residue and the surface grammar over the same wordplay span.",
        "",
        f"Rows scanned: {scanned}",
        f"Records written: {len(records)}",
        f"Complete block mappings: {len(complete)}",
        f"Complete mappings with compact residue: {len(useful)}",
        f"Grammar tags available: {'yes' if grammar_available else 'no'}",
        "",
        "## Operation Mix",
        "",
    ]
    for op, count in op_counts.most_common(12):
        lines.append(f"- `{op}`: {count}")

    lines.extend(["", "## Common Compact Residues", ""])
    for residue, count in residue_counts.most_common(18):
        if residue:
            lines.append(f"- `{residue}`: {count}")

    lines.extend(["", "## Anchor Example", ""])
    anchor = next(
        (
            r for r in records
            if r["clue"].lower() == "report on small issue of litter engulfing green"
        ),
        None,
    )
    if anchor:
        blocks = [f"`{block_text(anchor, b)}` -> `{b['letters']}`" for b in anchor["blocks"]]
        residue = " ".join(t["text"] for t in anchor["residue"])
        labels = " ".join(t["label"][0] for t in anchor["token_labels"])
        lines.append(f"`{anchor['clue']}` ({anchor['answer']})")
        lines.append("")
        lines.append(f"Definition: `{anchor['definition']}`")
        lines.append(f"Source blocks: {', '.join(blocks)}")
        lines.append(f"Residue: `{residue}`")
        lines.append(f"Boundary labels: `{labels}`")
        lines.append("")
        lines.append(
            "This is the target shape for the science project: explanation-supervised "
            "source spans, compact residue, and a boundary pattern that can later be "
            "tested against grammar."
        )

    lines.extend(["", "## Worked Review Seeds", ""])
    for record in useful[:20]:
        blocks = []
        for block in record["blocks"]:
            blocks.append(f"`{block_text(record, block)}` -> `{block['letters']}`")
        residue = " ".join(t["text"] for t in record["residue"])
        labels = " ".join(t["label"][0] for t in record["token_labels"])
        grammar = " ".join(g["tag"] or "?" for g in record["grammar"])
        lines.append(f"- `{record['clue']}` ({record['answer']})")
        lines.append(f"  Definition: `{record['definition']}`")
        lines.append(f"  Source blocks: {', '.join(blocks)}")
        lines.append(f"  Residue: `{residue}`")
        lines.append(f"  Labels: `{labels}`")
        lines.append(f"  Grammar tags: `{grammar}`")

    lines.extend(
        [
            "",
            "## Research Reading",
            "",
            "The useful records are not final parses. They are supervised examples of",
            "where answer-producing blocks appear in the clue, what residue remains,",
            "and what the ordinary grammar looked like before cryptic interpretation.",
            "",
            "The next research question is whether the grammar tags and dependency",
            "shape can predict the same SOURCE/RESIDUE boundaries without seeing the",
            "human explanation.",
            "",
        ]
    )
    if not grammar_available:
        lines.extend(
            [
                "This run does not include POS/dependency tags because spaCy is not",
                "installed in the active environment. The SOURCE/RESIDUE supervision is",
                "still useful; the grammar layer can be added once the language model is",
                "available.",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--times-db", type=Path, default=DEFAULT_TIMES_DB)
    parser.add_argument("--ref-db", type=Path, default=DEFAULT_REF_DB)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    db = MappingDB(str(args.ref_db))
    rows = load_rows(args.times_db, args.limit)
    records = []
    for row in rows:
        record = build_record(row, db)
        if record is not None:
            records.append(record)

    write_jsonl(records, args.jsonl_out)
    write_report(records, args.report_out, len(rows))
    complete = sum(1 for r in records if r["quality"]["complete_mapping"])
    print(f"rows_scanned={len(rows)}")
    print(f"records={len(records)}")
    print(f"complete_mappings={complete}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
