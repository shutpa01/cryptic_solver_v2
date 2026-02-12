# solver_engine/resources.py

import sqlite3
import re
from typing import Any, Dict, List, Optional, Tuple

import presets


# ------------------------- NORMALISATION -------------------------

def clean_key(x: str) -> str:
    if not x:
        return ""
    x = x.strip().lower()
    # Normalise apostrophes (Unicode + ASCII) for key matching
    x = x.replace("’", "'").replace("‘", "'")
    x = x.replace("'", "")
    x = re.sub(r"^[^a-z]+", "", x)
    x = re.sub(r"[^a-z]+$", "", x)
    return x


def clean_val(x: str) -> str:
    if not x:
        return ""
    x = x.strip()
    x = re.sub(r"^[^A-Za-z]+", "", x)
    x = re.sub(r"[^A-Za-z]+$", "", x)
    return x


def norm_letters(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


# ------------------------- ENUMERATION -------------------------

def parse_enum(en):
    return sum(map(int, re.findall(r"\d+", en or "")))


def matches_enumeration(word, enumeration):
    parts = list(map(int, re.findall(r"\d+", enumeration or "")))
    cleaned = re.sub(r"[^A-Za-z ]", "", word or "")
    chunks = cleaned.split()
    if len(chunks) != len(parts):
        return False
    return all(len(chunks[i]) == parts[i] for i in range(len(parts)))


# ------------------------- TOKENISATION -------------------------

def tokenize(s):
    return [
        re.sub(r"^[^\w]+|[^\w]+$", "", w)
        for w in (s or "").split()
        if w
    ]


# ---------------------- DB ----------------------

def connect_db() -> sqlite3.Connection:
    """Connect to reference DB (synonyms, wordplay, indicators etc)."""
    conn = sqlite3.connect(presets.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def connect_clues_db() -> sqlite3.Connection:
    """Connect to clues DB (clue selection only)."""
    conn = sqlite3.connect(presets.CLUES_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------- WORDLIST (always rebuild from DB) ----------------------

def build_wordlist() -> List[str]:
    # Clue answers from clues DB
    clues_conn = sqlite3.connect(presets.CLUES_DB_PATH)
    clues_cur = clues_conn.cursor()
    # Reference words from main DB
    conn = sqlite3.connect(presets.DB_PATH)
    cur = conn.cursor()

    words = set()

    clues_cur.execute("SELECT DISTINCT answer FROM clues")
    for (a,) in clues_cur.fetchall():
        v = clean_val(a)
        if v:
            words.add(v)

    cur.execute("SELECT DISTINCT answer FROM definition_answers")
    for (a,) in cur.fetchall():
        v = clean_val(a)
        if v:
            words.add(v)

    cur.execute("SELECT DISTINCT word FROM synonyms_pairs")
    for (w,) in cur.fetchall():
        v = clean_val(w)
        if v:
            words.add(v)

    cur.execute("SELECT DISTINCT synonym FROM synonyms_pairs")
    for (s,) in cur.fetchall():
        v = clean_val(s)
        if v:
            words.add(v)

    # ---- synonyms table (lexeme admission only; same basis as others) ----
    # headwords
    cur.execute("SELECT DISTINCT word FROM synonyms")
    for (w,) in cur.fetchall():
        v = clean_val(w)
        if v:
            words.add(v)

    # synonym items (JSON array stored as text)
    cur.execute("SELECT synonyms FROM synonyms")
    for (syn_json,) in cur.fetchall():
        for s in re.findall(r'"([^"]+)"', syn_json or ""):
            v = clean_val(s)
            if v:
                words.add(v)

    conn.close()
    return sorted(words)


# ---------------------- COHORT (criteria-driven; no filesystem) ----------------------

def _validate_criteria(criteria: Dict[str, Any]) -> None:
    if not isinstance(criteria, dict):
        raise TypeError("CURRENT_CRITERIA must be a dict.")

    allowed_keys = {"source", "where", "limit", "order"}
    unknown = set(criteria.keys()) - allowed_keys
    if unknown:
        raise ValueError(f"Unsupported criteria keys: {sorted(unknown)}")

    if criteria.get("source") != "clues":
        raise ValueError("Only source='clues' is supported by load_cohort().")

    where = criteria.get("where")
    if where is not None and not isinstance(where, dict):
        raise TypeError("criteria['where'] must be a dict of column -> value.")

    limit = criteria.get("limit")
    if limit is not None and (not isinstance(limit, int) or limit <= 0):
        raise ValueError("criteria['limit'] must be a positive int if provided.")

    order = criteria.get("order")
    if order is not None and order not in {"id", "random"}:
        raise ValueError("criteria['order'] must be 'id' or 'random' if provided.")


def _build_cohort_sql(criteria: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build a SQL query that returns rows with keys:
      - clue      (aliased from clues.clue_text)
      - enum      (aliased from clues.enumeration)
      - answer    (clues.answer)
      - plus any other columns from clues (kept as-is)
    """
    _validate_criteria(criteria)

    sql = """
        SELECT
            id,
            source,
            puzzle_number,
            publication_date,
            clue_number,
            direction,
            clue_text,
            enumeration,
            answer,
            definition,
            explanation,
            ai_explanation,
            wordplay_type,
            clue_text AS clue,
            enumeration AS enum
        FROM clues
    """.strip()

    params: List[Any] = []

    where = criteria.get("where") or {}
    if where:
        clauses = []
        for col, val in where.items():
            # Strict: only equality filtering; no guessing, no operators.
            if not isinstance(col, str) or not col:
                raise ValueError("criteria['where'] keys must be non-empty strings.")
            clauses.append(f"{col} = ?")
            params.append(val)
        sql += " WHERE " + " AND ".join(clauses)

    order = criteria.get("order")
    if order == "id":
        sql += " ORDER BY id"
    elif order == "random":
        sql += " ORDER BY RANDOM()"

    limit = criteria.get("limit")
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return sql, params


def load_cohort(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """
    Load cohort strictly via presets.CURRENT_CRITERIA.
    No file access. No fallbacks.
    """
    criteria = presets.CURRENT_CRITERIA
    sql, params = _build_cohort_sql(criteria)

    cur = conn.cursor()
    cur.execute(sql, params)
    return cur.fetchall()


# ---------------------- GRAPH ----------------------

def add_pair(graph, a, b):
    ak = clean_key(a)
    bv = clean_val(b)
    if ak and bv:
        graph.setdefault(ak, set()).add(bv)

    bk = clean_key(b)
    av = clean_val(a)
    if bk and av:
        graph.setdefault(bk, set()).add(av)


def load_graph(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    G = {}

    cur.execute("SELECT definition, answer FROM definition_answers_augmented")
    #cur.execute("SELECT definition, answer FROM definition_answers")
    for d, ans in cur.fetchall():
        add_pair(G, d, ans)

    cur.execute("SELECT word, synonym FROM synonyms_pairs")
    for w, s in cur.fetchall():
        add_pair(G, w, s)

    return {k: list(v) for k, v in G.items()}
