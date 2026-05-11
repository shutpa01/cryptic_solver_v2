"""Leftover authoring — prototype dashboard tab.

Top-level view is the enrichment queue: every verifier-suggested
candidate across the shadow_db FAIL queue, grouped by kind, with
Add / Reject buttons per row. Mirrors the existing live "DB
Enrichment" tab shape.

A secondary "FAIL detail" tab covers the per-clue deep dive
(production reading, blog, author a whole form) for cases the
queue can't address.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
CRYPTIC_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
SHADOW_DB = PROJECT_ROOT / "data" / "shadow_blog_v0.db"


# --- DB helpers ---------------------------------------------------------

def _ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rw_shadow() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SHADOW_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    _ensure_decisions_table(conn)
    return conn


def _rw_live() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CRYPTIC_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_decisions_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shadow_candidate_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_key TEXT NOT NULL UNIQUE,
            decision TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()


def _candidate_key(cand: dict) -> str:
    """A stable key for a candidate so we can track its decision."""
    payload = "|".join([
        str(cand.get("kind") or ""),
        str(cand.get("source_word") or "").lower(),
        str(cand.get("value") or "").upper(),
        str(cand.get("operation") or ""),
        str(cand.get("subtype") or ""),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


# --- Queue queries -----------------------------------------------------

def fetch_candidates() -> list:
    """Flatten every non-empty enrichments_json across seed_failures
    into one row per candidate. Excludes candidates that already
    have a decision (accepted or rejected) recorded."""
    shadow = _rw_shadow()
    decisions = {
        r["candidate_key"]: r["decision"]
        for r in shadow.execute(
            "SELECT candidate_key, decision FROM shadow_candidate_decisions"
        )
    }
    rows = shadow.execute(
        """
        SELECT clue_id, source, puzzle_number, clue_number, direction,
               clue_text, answer, enrichments_json
        FROM seed_failures
        WHERE enrichments_json IS NOT NULL
          AND enrichments_json != ''
          AND enrichments_json != '[]'
        ORDER BY source, puzzle_number,
                 CAST(clue_number AS INTEGER), direction
        """
    ).fetchall()
    out: list = []
    for r in rows:
        try:
            cands = json.loads(r["enrichments_json"])
        except Exception:
            continue
        if not isinstance(cands, list):
            continue
        for c in cands:
            if not isinstance(c, dict):
                continue
            key = _candidate_key(c)
            if key in decisions:
                continue
            out.append({
                "key": key,
                "kind": c.get("kind") or "",
                "word": c.get("source_word") or "",
                "value": c.get("value") or "",
                "operation": c.get("operation") or "",
                "subtype": c.get("subtype") or "",
                "detail": c.get("detail") or "",
                "clue_id": r["clue_id"],
                "source": r["source"],
                "puzzle_number": r["puzzle_number"],
                "clue_number": r["clue_number"],
                "direction": r["direction"],
                "clue_text": r["clue_text"],
                "answer": r["answer"],
            })
    return out


def record_decision(key: str, decision: str) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT OR REPLACE INTO shadow_candidate_decisions "
        "(candidate_key, decision) VALUES (?, ?)",
        (key, decision),
    )
    conn.commit()


# --- Writes -----------------------------------------------------------

def insert_shadow_synonym(word: str, value: str, clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym, source, clue_id) "
        "VALUES (?, ?, 'leftover_dashboard', ?)",
        (word.lower().strip(), value.upper().strip(), clue_id),
    )
    conn.commit()


def insert_shadow_abbreviation(word: str, value: str, clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO wordplay (indicator, substitution, category, "
        "confidence, clue_id) "
        "VALUES (?, ?, 'abbreviation', 'medium', ?)",
        (word.lower().strip(), value.upper().strip(), clue_id),
    )
    conn.commit()


def insert_shadow_indicator(word: str, op: str, subtype: str,
                              clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, subtype, "
        "confidence, source, clue_id) "
        "VALUES (?, ?, ?, 'medium', 'leftover_dashboard', ?)",
        (word.lower().strip(), op, subtype or None, clue_id),
    )
    conn.commit()


def insert_shadow_definition(definition: str, answer: str,
                               clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO definition_answers_augmented (definition, answer, "
        "source, clue_id) "
        "VALUES (?, ?, 'leftover_dashboard', ?)",
        (definition.lower().strip(), answer.upper().strip(), clue_id),
    )
    conn.commit()


def promote_synonym_live(word: str, value: str) -> None:
    conn = _rw_live()
    if conn.execute(
        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
        (word.lower().strip(), value.upper().strip()),
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym) VALUES (?, ?)",
        (word.lower().strip(), value.upper().strip()),
    )
    conn.commit()


def promote_abbreviation_live(word: str, value: str) -> None:
    conn = _rw_live()
    if conn.execute(
        "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
        "AND UPPER(substitution)=? AND category='abbreviation'",
        (word.lower().strip(), value.upper().strip()),
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO wordplay (indicator, substitution, category, "
        "confidence) VALUES (?, ?, 'abbreviation', 'high')",
        (word.lower().strip(), value.upper().strip()),
    )
    conn.commit()


def promote_indicator_live(word: str, op: str, subtype: str) -> None:
    conn = _rw_live()
    if conn.execute(
        "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=? "
        "AND COALESCE(subtype,'')=?",
        (word.lower().strip(), op, subtype or ""),
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, subtype, confidence) "
        "VALUES (?, ?, ?, 'high')",
        (word.lower().strip(), op, subtype or None),
    )
    conn.commit()


def promote_definition_live(definition: str, answer: str) -> None:
    conn = _rw_live()
    if conn.execute(
        "SELECT 1 FROM definition_answers_augmented "
        "WHERE LOWER(definition)=? AND UPPER(answer)=?",
        (definition.lower().strip(), answer.upper().strip()),
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO definition_answers_augmented (definition, answer) "
        "VALUES (?, ?)",
        (definition.lower().strip(), answer.upper().strip()),
    )
    conn.commit()


def accept_candidate(c: dict, also_live: bool) -> None:
    kind = c["kind"]
    if kind == "synonym":
        insert_shadow_synonym(c["word"], c["value"], c["clue_id"])
        if also_live:
            promote_synonym_live(c["word"], c["value"])
    elif kind == "abbreviation":
        insert_shadow_abbreviation(c["word"], c["value"], c["clue_id"])
        if also_live:
            promote_abbreviation_live(c["word"], c["value"])
    elif kind == "indicator":
        op = c["operation"]
        if not op:
            raise ValueError("indicator candidate missing operation")
        insert_shadow_indicator(
            c["word"], op, c.get("subtype") or "", c["clue_id"])
        if also_live:
            promote_indicator_live(c["word"], op, c.get("subtype") or "")
    elif kind == "definition":
        insert_shadow_definition(c["word"], c["value"], c["clue_id"])
        if also_live:
            promote_definition_live(c["word"], c["value"])
    else:
        raise ValueError(f"unknown candidate kind: {kind}")
    record_decision(c["key"], "accepted")


def reject_candidate(c: dict) -> None:
    record_decision(c["key"], "rejected")


# --- Rendering --------------------------------------------------------

def render():
    st.header("Prototype enrichment review")
    st.caption(
        "Verifier-suggested rows from the prototype's FAIL queue. "
        "Add writes to shadow; the optional 'Also live' checkbox at "
        "the top promotes accepted rows into `cryptic_new` too. "
        "Reject records the decision and removes the row from the "
        "queue."
    )

    tab1, tab2 = st.tabs(["Enrichment queue", "FAIL detail"])
    with tab1:
        _render_queue()
    with tab2:
        _render_detail()


def _render_queue():
    also_live = st.checkbox(
        "Also write accepted rows to live `cryptic_new`",
        value=False, key="lo_live")

    candidates = fetch_candidates()
    by_kind: dict = {"synonym": [], "abbreviation": [],
                       "indicator": [], "definition": []}
    for c in candidates:
        by_kind.setdefault(c["kind"], []).append(c)

    cols = st.columns(5)
    cols[0].metric("Total", len(candidates))
    cols[1].metric("Synonyms", len(by_kind.get("synonym", [])))
    cols[2].metric("Abbrevs", len(by_kind.get("abbreviation", [])))
    cols[3].metric("Indicators", len(by_kind.get("indicator", [])))
    cols[4].metric("Definitions", len(by_kind.get("definition", [])))

    if not candidates:
        st.info(
            "No pending candidates. Either the FAIL queue is empty, "
            "or all candidates have been decided. Decisions live in "
            "`shadow_db.shadow_candidate_decisions`."
        )
        return

    if by_kind.get("synonym"):
        _render_group("Synonyms", by_kind["synonym"], also_live)
    if by_kind.get("abbreviation"):
        _render_group("Abbreviations", by_kind["abbreviation"], also_live)
    if by_kind.get("indicator"):
        _render_group("Indicators", by_kind["indicator"], also_live,
                       indicator=True)
    if by_kind.get("definition"):
        _render_group("Definitions", by_kind["definition"], also_live,
                       definition=True)


def _render_group(title: str, items: list, also_live: bool,
                    indicator: bool = False, definition: bool = False):
    st.subheader(f"{title} ({len(items)})")
    for c in items:
        ctx = (
            f"{c['source']} #{c['puzzle_number']} "
            f"{c['clue_number']}{(c['direction'] or '')[:1]}  "
            f"= {c['answer']}  — {(c['clue_text'] or '')[:70]}"
        )
        if indicator:
            cols = st.columns([4, 3, 2, 4, 1.2, 1.2])
        else:
            cols = st.columns([4, 4, 5, 1.2, 1.2])

        with cols[0]:
            new_word = st.text_input(
                "word", value=c["word"], key=f"w_{c['key']}",
                label_visibility="collapsed",
            )
        with cols[1]:
            label = "definition" if definition else "value"
            new_value = st.text_input(
                label, value=c["value"], key=f"v_{c['key']}",
                label_visibility="collapsed",
            )
        if indicator:
            with cols[2]:
                op_default = c.get("operation") or "anagram"
                op_choices = [
                    "anagram", "hidden", "container", "insertion",
                    "deletion", "parts", "reversal", "homophone",
                    "acrostic", "alternating", "charade",
                ]
                if op_default not in op_choices:
                    op_choices.insert(0, op_default)
                new_op = st.selectbox(
                    "op", op_choices,
                    index=op_choices.index(op_default),
                    key=f"o_{c['key']}",
                    label_visibility="collapsed",
                )
            ctx_col = cols[3]
            add_col = cols[4]
            rej_col = cols[5]
        else:
            new_op = None
            ctx_col = cols[2]
            add_col = cols[3]
            rej_col = cols[4]

        with ctx_col:
            st.markdown(
                f"<span style='font-size:0.85em;color:#666'>{ctx}</span>",
                unsafe_allow_html=True,
            )
        with add_col:
            if st.button("Add", key=f"add_{c['key']}"):
                edited = dict(c)
                edited["word"] = new_word.strip()
                edited["value"] = new_value.strip()
                if new_op:
                    edited["operation"] = new_op
                try:
                    accept_candidate(edited, also_live)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Add failed: {e}")
        with rej_col:
            if st.button("Reject", key=f"rej_{c['key']}"):
                try:
                    reject_candidate(c)
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Reject failed: {e}")


# --- FAIL detail tab (per-clue deep dive) ---------------------------------

def fetch_fail_queue(source: str, puzzle: str, limit: int) -> list:
    shadow = _ro(SHADOW_DB)
    where = ["1=1"]
    params: list = []
    if source and source != "All":
        where.append("sf.source = ?")
        params.append(source)
    if puzzle:
        where.append("sf.puzzle_number = ?")
        params.append(puzzle)
    where_sql = " AND ".join(where)
    rows = shadow.execute(
        f"""
        SELECT sf.id AS fail_id, sf.clue_id, sf.source, sf.puzzle_number,
               sf.clue_number, sf.direction, sf.clue_text, sf.answer,
               sf.failure_kind, sf.failure_detail, sf.enrichments_json,
               sf.run_number, sf.created_at
        FROM seed_failures sf
        WHERE {where_sql}
        ORDER BY sf.source, sf.puzzle_number,
                 CAST(sf.clue_number AS INTEGER), sf.direction,
                 sf.run_number DESC
        LIMIT ?
        """,
        params + [limit],
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_clue_context(clue_id: int) -> dict:
    conn = _ro(CLUES_DB)
    row = conn.execute(
        """
        SELECT c.clue_text, c.answer, c.definition, c.wordplay_type,
               c.ai_explanation, c.explanation,
               se.components, se.confidence, se.model_version
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.id = ?
        """,
        (clue_id,),
    ).fetchone()
    return dict(row) if row else {}


def rerun_clue(source: str, puzzle: str) -> tuple:
    cmd = [
        sys.executable, "-u", "-X", "utf8", "-m",
        "prototypes.universal_form_v2.run_puzzle",
        source, str(puzzle), "--second-pass",
    ]
    proc = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT),
        capture_output=True, text=True, timeout=3600,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _render_detail():
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent",
             "dailymail"],
            key="lod_source",
        )
    with col_b:
        puzzle = st.text_input("Puzzle number (optional)", key="lod_puzzle")
    with col_c:
        limit = st.number_input(
            "Limit", min_value=10, max_value=500, value=50, step=10,
            key="lod_limit",
        )

    rows = fetch_fail_queue(source, puzzle.strip(), int(limit))
    if not rows:
        st.info("No FAIL rows match.")
        return

    options = [
        f"{r['source']} {r['puzzle_number']} "
        f"{r['clue_number']}{(r['direction'] or '')[:1]}  "
        f"— {(r['answer'] or '').strip()}  — {(r['clue_text'] or '')[:60]}"
        for r in rows
    ]
    idx = st.selectbox("Select a FAIL",
                        options=list(range(len(rows))),
                        format_func=lambda i: options[i],
                        key="lod_idx")
    fail = rows[idx]
    ctx = fetch_clue_context(fail["clue_id"])

    st.subheader(
        f"{fail['source']} {fail['puzzle_number']} "
        f"{fail['clue_number']}{(fail['direction'] or '')[:1]}  "
        f"— {fail['answer']}"
    )
    st.write(f"**Clue:** {fail['clue_text']}")
    st.write(f"**Answer:** {fail['answer']}")
    if ctx.get("definition"):
        st.write(f"**Recorded definition:** `{ctx['definition']}`")
    if ctx.get("ai_explanation"):
        st.write(f"**Production reading:** {ctx['ai_explanation']}")
    if ctx.get("explanation"):
        with st.expander("Blog explanation"):
            st.text(ctx["explanation"])
    if fail.get("failure_detail"):
        st.write(f"**Failure detail:** {fail['failure_detail']}")

    _render_author_form(fail, ctx)

    st.divider()
    if st.button("Re-run second pass on this puzzle",
                  key=f"rerun_{fail['clue_id']}"):
        with st.spinner("Running ..."):
            rc, out, err = rerun_clue(fail["source"], fail["puzzle_number"])
        st.write(f"Exit code: {rc}")
        if out:
            with st.expander("stdout"):
                st.text(out)
        if err:
            with st.expander("stderr"):
                st.text(err)


def _render_author_form(fail: dict, ctx: dict):
    st.subheader("Author form (catalog entry)")
    default_components = None
    if ctx.get("components"):
        try:
            default_components = json.loads(ctx["components"])
        except Exception:
            default_components = None
    if default_components is None:
        default_components = {
            "ai_pieces": [
                {"mechanism": "synonym", "clue_word": "<source>",
                 "letters": "<VALUE>"},
            ],
            "assembly": {"op": "charade", "order": ["<VALUE>"]},
            "wordplay_type": "charade",
        }
    default_def = ctx.get("definition") or ""

    def_phrase = st.text_input(
        "Definition phrase", value=default_def,
        key=f"author_def_{fail['clue_id']}",
    )
    components_text = st.text_area(
        "Components JSON",
        value=json.dumps(default_components, indent=2),
        height=240,
        key=f"author_comp_{fail['clue_id']}",
    )

    col_v, col_s = st.columns(2)
    with col_v:
        if st.button("Verify form",
                       key=f"author_verify_{fail['clue_id']}"):
            _verify(fail, def_phrase, components_text, save=False)
    with col_s:
        if st.button("Verify and save (on PASS)",
                       key=f"author_save_{fail['clue_id']}"):
            _verify(fail, def_phrase, components_text, save=True)


def _verify(fail: dict, def_phrase: str, components_text: str,
              save: bool) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from signature_solver.db import RefDB
    from prototypes.universal_form_v2.json_translator import (
        translate_components,
    )
    from prototypes.universal_form_v2.clipboard_verifier import verify
    from prototypes.universal_form_v2.extract_catalog import (
        signature as form_signature,
    )
    from prototypes.universal_form_v2.shadow_db import (
        ensure_shadow, write_solve,
    )

    try:
        components_obj = json.loads(components_text)
    except Exception as e:  # noqa: BLE001
        st.error(f"JSON parse failed: {e}")
        return
    row = {
        "clue_text": fail["clue_text"],
        "answer": fail["answer"],
        "components": json.dumps(components_obj),
        "definition_text": def_phrase,
    }
    db = RefDB(str(CRYPTIC_DB))
    shadow_conn = ensure_shadow()
    try:
        form, err = translate_components(row, db)
    except Exception as e:  # noqa: BLE001
        st.error(f"Translator exception: {e}")
        return
    if form is None:
        st.error(f"Translator rejected: {err}")
        return
    try:
        verdict = verify(form, fail["clue_text"], db, shadow_conn)
    except Exception as e:  # noqa: BLE001
        st.error(f"Verifier exception: {e}")
        return
    if verdict.verdict == "PASS":
        st.success("Verifier PASS")
        try:
            sig = form_signature(form.tree)
        except Exception:
            sig = "authored"
        st.write(f"Signature: `{sig}`")
        st.json(form.to_dict())
        if save:
            try:
                solve_id = write_solve(
                    shadow_conn,
                    clue_id=fail["clue_id"],
                    signature=sig,
                    verdict="PASS",
                    answer=fail["answer"],
                    form_dict=form.to_dict(),
                    run_number=3,
                )
                st.success(
                    f"Saved (solve_id={solve_id}, run_number=3)."
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Save failed: {e}")
    else:
        st.warning(f"Verifier {verdict.verdict}")
        for c in verdict.checks:
            if c.status != "pass":
                st.write(f" - {c.name}: {c.detail}")


render()
