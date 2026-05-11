"""Leftover authoring — prototype dashboard tab.

Surfaces the prototype's FAIL queue from `shadow_db.seed_failures`,
joined with the clue's metadata and the production explanation
where available. Lets a human reviewer:

  - Accept individual enrichment candidates (verifier-suggested
    missing DB rows) into the shadow vocabulary tables.
  - Add manual DB rows (synonym / abbreviation / indicator /
    definition) where the verifier didn't surface a candidate.
  - Re-run the cascade for the FAIL clue's puzzle (second pass)
    so newly-added rows feed back into a fresh attempt.
  - Promote approved shadow rows into the live `cryptic_new` DB
    (deliberate manual step; never automatic).

Per PARALLEL_SYSTEM_DESIGN.md §3.6 / §6.2, this is the human
review surface the design names. The clipboard verifier remains
the trust anchor: nothing is written to live `cryptic_new`
without a reviewer's explicit approval, and the re-run on second
pass goes through the verifier again.
"""
from __future__ import annotations

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


# --- DB helpers ----------------------------------------------------------

def _ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rw_shadow() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SHADOW_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _rw_live() -> sqlite3.Connection:
    conn = sqlite3.connect(str(CRYPTIC_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# --- Queries -------------------------------------------------------------

def fetch_fail_queue(source: str, puzzle: str, limit: int) -> list:
    shadow = _ro(SHADOW_DB)
    where = ["sf.failure_kind IN ('verifier_fail', 'translation_error')"]
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
    """Pull blog explanation + production reading from clues_master."""
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


def fetch_existing_solves(clue_id: int) -> list:
    shadow = _ro(SHADOW_DB)
    rows = shadow.execute(
        """
        SELECT signature, verdict, answer, run_number, created_at
        FROM solves WHERE clue_id = ?
        ORDER BY created_at DESC
        """,
        (clue_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def fetch_shadow_vocab_for_clue(clue_id: int) -> dict:
    """Return any shadow vocabulary rows whose provenance is this clue."""
    shadow = _ro(SHADOW_DB)
    out = {"synonyms": [], "wordplay": [], "indicators": [], "definitions": []}
    for r in shadow.execute(
        "SELECT word, synonym FROM synonyms_pairs WHERE clue_id = ?",
        (clue_id,),
    ):
        out["synonyms"].append((r["word"], r["synonym"]))
    for r in shadow.execute(
        "SELECT indicator, substitution FROM wordplay WHERE clue_id = ?",
        (clue_id,),
    ):
        out["wordplay"].append((r["indicator"], r["substitution"]))
    for r in shadow.execute(
        "SELECT word, wordplay_type, subtype FROM indicators WHERE clue_id = ?",
        (clue_id,),
    ):
        out["indicators"].append((r["word"], r["wordplay_type"], r["subtype"]))
    for r in shadow.execute(
        "SELECT definition, answer FROM definition_answers_augmented WHERE clue_id = ?",
        (clue_id,),
    ):
        out["definitions"].append((r["definition"], r["answer"]))
    return out


# --- Writes (shadow first, optional live promotion) ----------------------

def insert_shadow_synonym(word: str, synonym: str, clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym, source, clue_id) "
        "VALUES (?, ?, 'leftover_dashboard', ?)",
        (word.lower().strip(), synonym.upper().strip(), clue_id),
    )
    conn.commit()


def insert_shadow_abbreviation(indicator: str, substitution: str,
                                 clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO wordplay (indicator, substitution, category, "
        "confidence, clue_id) "
        "VALUES (?, ?, 'abbreviation', 'medium', ?)",
        (indicator.lower().strip(), substitution.upper().strip(), clue_id),
    )
    conn.commit()


def insert_shadow_indicator(word: str, wordplay_type: str,
                              subtype: str, clue_id: int) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, subtype, "
        "confidence, source, clue_id) "
        "VALUES (?, ?, ?, 'medium', 'leftover_dashboard', ?)",
        (word.lower().strip(), wordplay_type, subtype or None, clue_id),
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


def promote_to_live_synonym(word: str, synonym: str) -> None:
    conn = _rw_live()
    existing = conn.execute(
        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
        (word.lower().strip(), synonym.upper().strip()),
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym) VALUES (?, ?)",
        (word.lower().strip(), synonym.upper().strip()),
    )
    conn.commit()


def promote_to_live_abbreviation(indicator: str, substitution: str) -> None:
    conn = _rw_live()
    existing = conn.execute(
        "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
        "AND UPPER(substitution)=? AND category='abbreviation'",
        (indicator.lower().strip(), substitution.upper().strip()),
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO wordplay (indicator, substitution, category, "
        "confidence) VALUES (?, ?, 'abbreviation', 'high')",
        (indicator.lower().strip(), substitution.upper().strip()),
    )
    conn.commit()


def promote_to_live_indicator(word: str, wordplay_type: str,
                                subtype: str) -> None:
    conn = _rw_live()
    existing = conn.execute(
        "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=? "
        "AND COALESCE(subtype,'')=?",
        (word.lower().strip(), wordplay_type, subtype or ""),
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, subtype, confidence) "
        "VALUES (?, ?, ?, 'high')",
        (word.lower().strip(), wordplay_type, subtype or None),
    )
    conn.commit()


def promote_to_live_definition(definition: str, answer: str) -> None:
    conn = _rw_live()
    existing = conn.execute(
        "SELECT 1 FROM definition_answers_augmented "
        "WHERE LOWER(definition)=? AND UPPER(answer)=?",
        (definition.lower().strip(), answer.upper().strip()),
    ).fetchone()
    if existing:
        return
    conn.execute(
        "INSERT INTO definition_answers_augmented (definition, answer) "
        "VALUES (?, ?)",
        (definition.lower().strip(), answer.upper().strip()),
    )
    conn.commit()


# --- Cascade re-run -----------------------------------------------------

def rerun_clue(source: str, puzzle: str) -> tuple:
    """Invoke run_puzzle as a subprocess for the clue's puzzle with
    --second-pass. Returns (returncode, stdout, stderr)."""
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


# --- Rendering ----------------------------------------------------------

def render():
    st.header("Prototype leftover authoring")
    st.caption(
        "FAIL queue from `shadow_db.seed_failures`. Accept enrichment "
        "candidates into shadow vocabulary, then re-run the cascade. "
        "Promotion to the live `cryptic_new` DB is a deliberate manual "
        "step shown per row."
    )

    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent",
             "dailymail"],
            key="lo_source",
        )
    with col_b:
        puzzle = st.text_input("Puzzle number (optional)", key="lo_puzzle")
    with col_c:
        limit = st.number_input(
            "Limit", min_value=10, max_value=500, value=50, step=10,
            key="lo_limit",
        )

    rows = fetch_fail_queue(source, puzzle.strip(), int(limit))
    st.metric("FAILs in queue", len(rows))

    if not rows:
        st.info("No FAIL rows match the current filters.")
        return

    # Compact list of FAILs with a select. Show clue id + clue text.
    options = [
        f"{r['source']} {r['puzzle_number']} "
        f"{r['clue_number']}{(r['direction'] or '')[:1]}  "
        f"— {(r['answer'] or '').strip()}  — {(r['clue_text'] or '')[:60]}"
        for r in rows
    ]
    idx = st.selectbox("Select a FAIL to author",
                        options=list(range(len(rows))),
                        format_func=lambda i: options[i],
                        key="lo_idx")
    fail = rows[idx]
    _render_fail_detail(fail)


def _render_fail_detail(fail: dict) -> None:
    clue_id = fail["clue_id"]
    ctx = fetch_clue_context(clue_id)

    st.subheader(
        f"{fail['source']} {fail['puzzle_number']} "
        f"{fail['clue_number']}{(fail['direction'] or '')[:1]}  "
        f"— {fail['answer']}"
    )
    st.write(f"**Clue:** {fail['clue_text']}")
    st.write(f"**Answer:** {fail['answer']}")
    if ctx.get("definition"):
        st.write(f"**Recorded definition:** `{ctx['definition']}`")
    if ctx.get("wordplay_type"):
        st.write(f"**Wordplay type:** `{ctx['wordplay_type']}`")
    if ctx.get("ai_explanation"):
        st.write(f"**Production reading:** {ctx['ai_explanation']}")
    if ctx.get("explanation"):
        with st.expander("Blog explanation"):
            st.text(ctx["explanation"])

    st.write(f"**Failure kind:** `{fail['failure_kind']}`")
    if fail["failure_detail"]:
        st.write(f"**Verifier detail:** {fail['failure_detail']}")

    # Enrichment candidates from the verifier
    enrichments = []
    if fail.get("enrichments_json"):
        try:
            enrichments = json.loads(fail["enrichments_json"])
        except Exception:
            enrichments = []
    if enrichments:
        st.subheader("Enrichment candidates (verifier-suggested)")
        for i, cand in enumerate(enrichments):
            kind = cand.get("kind") or "?"
            source_word = cand.get("source_word") or "?"
            value = cand.get("value") or "?"
            operation = cand.get("operation")
            subtype = cand.get("subtype")
            label = f"{kind}: `{source_word}` → `{value}`"
            if kind == "indicator" and operation:
                label += f"   (op: `{operation}`"
                if subtype:
                    label += f", subtype: `{subtype}`"
                label += ")"
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.write(label)
            with col2:
                if st.button("Add to shadow", key=f"lo_add_{clue_id}_{i}"):
                    try:
                        _accept_candidate(cand, clue_id)
                        st.success("Added to shadow.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Insert failed: {e}")
            with col3:
                if st.button("Promote to live", key=f"lo_prom_{clue_id}_{i}"):
                    try:
                        _promote_candidate(cand)
                        st.success("Promoted to cryptic_new.")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Promote failed: {e}")
    else:
        st.info("No verifier-suggested enrichment candidates on this row.")

    # Manual row entry
    st.subheader("Manual row entry (shadow)")
    _render_manual_forms(clue_id, fail["answer"])

    # Existing shadow rows for this clue
    vocab = fetch_shadow_vocab_for_clue(clue_id)
    if any(vocab.values()):
        st.subheader("Shadow rows already added for this clue")
        for k, items in vocab.items():
            if items:
                st.write(f"**{k}**:")
                for it in items:
                    st.write(f"   - {it}")

    # Existing solves on this clue (any verdict)
    solves = fetch_existing_solves(clue_id)
    if solves:
        with st.expander("Solve history (shadow_db.solves)"):
            for s in solves:
                st.write(
                    f"- {s['verdict']}  run {s['run_number']}  "
                    f"`{s['signature']}`  {s['created_at']}"
                )

    st.divider()
    st.subheader("Re-run cascade for this puzzle")
    st.caption(
        "Runs run_puzzle.py with --second-pass on this puzzle, so any "
        "shadow rows just added feed the verifier on the next attempt."
    )
    if st.button("Re-run (second pass)", key=f"lo_rerun_{clue_id}"):
        with st.spinner("Running run_puzzle --second-pass..."):
            rc, out, err = rerun_clue(fail["source"], fail["puzzle_number"])
        st.write(f"Exit code: {rc}")
        if out:
            with st.expander("stdout"):
                st.text(out)
        if err:
            with st.expander("stderr"):
                st.text(err)


def _accept_candidate(cand: dict, clue_id: int) -> None:
    kind = cand.get("kind")
    word = cand.get("source_word")
    value = cand.get("value")
    if not (kind and word and value):
        raise ValueError("candidate missing kind/source_word/value")
    if kind == "synonym":
        insert_shadow_synonym(word, value, clue_id)
    elif kind == "abbreviation":
        insert_shadow_abbreviation(word, value, clue_id)
    elif kind == "indicator":
        op = cand.get("operation") or cand.get("wordplay_type")
        if not op:
            raise ValueError("indicator candidate missing operation")
        insert_shadow_indicator(word, op, cand.get("subtype") or "", clue_id)
    elif kind == "definition":
        insert_shadow_definition(word, value, clue_id)
    else:
        raise ValueError(f"unknown candidate kind: {kind}")


def _promote_candidate(cand: dict) -> None:
    kind = cand.get("kind")
    word = cand.get("source_word")
    value = cand.get("value")
    if not (kind and word and value):
        raise ValueError("candidate missing kind/source_word/value")
    if kind == "synonym":
        promote_to_live_synonym(word, value)
    elif kind == "abbreviation":
        promote_to_live_abbreviation(word, value)
    elif kind == "indicator":
        op = cand.get("operation") or cand.get("wordplay_type")
        if not op:
            raise ValueError("indicator candidate missing operation")
        promote_to_live_indicator(word, op, cand.get("subtype") or "")
    elif kind == "definition":
        promote_to_live_definition(word, value)
    else:
        raise ValueError(f"unknown candidate kind: {kind}")


def _render_manual_forms(clue_id: int, answer: str) -> None:
    tabs = st.tabs(["Synonym", "Abbreviation", "Indicator", "Definition"])

    with tabs[0]:
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            word = st.text_input("Word / phrase", key=f"man_syn_word_{clue_id}")
        with col2:
            syn = st.text_input("Synonym (value)",
                                 key=f"man_syn_value_{clue_id}")
        with col3:
            promote = st.checkbox("Promote to live",
                                    key=f"man_syn_promote_{clue_id}")
        if st.button("Add synonym", key=f"man_syn_btn_{clue_id}"):
            if word and syn:
                insert_shadow_synonym(word, syn, clue_id)
                if promote:
                    promote_to_live_synonym(word, syn)
                st.success("Added.")
            else:
                st.error("word and synonym required")

    with tabs[1]:
        col1, col2, col3 = st.columns([3, 3, 1])
        with col1:
            word = st.text_input(
                "Word / phrase", key=f"man_abr_word_{clue_id}")
        with col2:
            sub = st.text_input(
                "Abbreviation (value)", key=f"man_abr_value_{clue_id}")
        with col3:
            promote = st.checkbox(
                "Promote to live", key=f"man_abr_promote_{clue_id}")
        if st.button("Add abbreviation", key=f"man_abr_btn_{clue_id}"):
            if word and sub:
                insert_shadow_abbreviation(word, sub, clue_id)
                if promote:
                    promote_to_live_abbreviation(word, sub)
                st.success("Added.")
            else:
                st.error("word and abbreviation required")

    with tabs[2]:
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        with col1:
            word = st.text_input(
                "Word / phrase", key=f"man_ind_word_{clue_id}")
        with col2:
            op = st.selectbox(
                "Wordplay type",
                ["anagram", "hidden", "container", "insertion",
                 "deletion", "parts", "reversal", "homophone",
                 "acrostic", "alternating", "charade"],
                key=f"man_ind_op_{clue_id}",
            )
        with col3:
            subtype = st.text_input(
                "Subtype (optional)", key=f"man_ind_sub_{clue_id}")
        with col4:
            promote = st.checkbox(
                "Promote to live", key=f"man_ind_promote_{clue_id}")
        if st.button("Add indicator", key=f"man_ind_btn_{clue_id}"):
            if word and op:
                insert_shadow_indicator(word, op, subtype, clue_id)
                if promote:
                    promote_to_live_indicator(word, op, subtype)
                st.success("Added.")
            else:
                st.error("word and wordplay type required")

    with tabs[3]:
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            phrase = st.text_input(
                "Definition phrase", key=f"man_def_phrase_{clue_id}")
        with col2:
            ans = st.text_input(
                "Answer", value=answer, key=f"man_def_ans_{clue_id}")
        with col3:
            promote = st.checkbox(
                "Promote to live", key=f"man_def_promote_{clue_id}")
        if st.button("Add definition", key=f"man_def_btn_{clue_id}"):
            if phrase and ans:
                insert_shadow_definition(phrase, ans, clue_id)
                if promote:
                    promote_to_live_definition(phrase, ans)
                st.success("Added.")
            else:
                st.error("definition phrase and answer required")


render()
