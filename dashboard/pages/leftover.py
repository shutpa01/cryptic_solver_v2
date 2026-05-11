"""Leftover authoring — prototype enrichment review.

Right-shape pattern: every clue in the shadow FAIL queue that has a
recorded reading (production's ai_explanation / structured_explanations
components) is analysed to derive the exact DB rows the verifier
would need to PASS it. We then check which of those rows are
missing from (live ∪ shadow) and surface only the missing ones as
candidates.

This is *deterministic enrichment*, not guesswork: each candidate
is a row we know is needed for that clue. Accept writes it.

For FAILs with no recorded reading, the second tab is a form-
authoring surface — the human supplies a reading and the same
analyser fires.
"""
from __future__ import annotations

import hashlib
import json
import re
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


# --- Reading → needed rows ---------------------------------------------

# Maps assembly.op → required indicator wordplay_type(s).
OP_INDICATOR_TYPES = {
    "anagram":   {"anagram"},
    "reversal":  {"reversal"},
    "container": {"container", "insertion"},
    "deletion":  {"deletion", "parts"},
    "hidden":    {"hidden"},
    "homophone": {"homophone"},
    "acrostic":  {"acrostic", "parts"},
}


def derive_needed_rows(components: dict, definition_text: str,
                        answer: str, ai_explanation: str) -> list:
    """From a recorded reading, derive the DB rows the verifier
    needs to PASS this clue. Returns list of dicts with kind, word,
    value, plus optional op/subtype for indicators.

    Coverage today: synonym, abbreviation, homophone (leaf-level
    bridges from clue word → piece letters), definition (definition
    phrase → answer), and top-level indicators where the op needs
    one and the indicator word is named in the recorded explanation.
    """
    needs: list = []
    answer_u = (answer or "").upper().strip()

    # Definition
    def_text = (definition_text or "").strip()
    if def_text and answer_u:
        needs.append({
            "kind": "definition",
            "word": def_text,
            "value": answer_u,
        })

    # Leaf-level bridges from ai_pieces
    for p in components.get("ai_pieces") or []:
        if not isinstance(p, dict):
            continue
        mech = (p.get("mechanism") or "").lower().strip()
        cw = (p.get("clue_word") or "").strip()
        letters = (p.get("letters") or "").upper().strip()
        if not cw or not letters:
            continue
        if mech == "synonym":
            needs.append({"kind": "synonym", "word": cw, "value": letters})
        elif mech == "abbreviation":
            needs.append({"kind": "abbreviation", "word": cw, "value": letters})
        elif mech == "homophone":
            needs.append({"kind": "homophone", "word": cw, "value": letters})

    # Top-level indicator (when the op needs one). Emit the
    # canonical wordplay_type for the op (container for container
    # ops, deletion for deletion ops, etc.) — not the alternative
    # types the verifier also tolerates on lookup ("insertion",
    # "parts"). Adding rows under the wrong type pollutes the DB.
    assembly = components.get("assembly") or {}
    op = (assembly.get("op") or "").lower()
    if op in OP_INDICATOR_TYPES:
        # Try to read the indicator word from the ai_explanation,
        # which typically names it in square brackets:
        #    [anagram: "shifting"]   [container: "in"]
        ind_word = _extract_indicator_from_explanation(ai_explanation, op)
        if ind_word:
            needs.append({
                "kind": "indicator", "word": ind_word,
                "value": "", "op": op,
            })

    return needs


_IND_RE = re.compile(
    r'\[\s*(anagram|reversal|container|insertion|deletion|parts|'
    r'hidden|homophone|acrostic)\s*:\s*["“‘]([^"”’]+)'
    r'["”’]\s*\]',
    re.IGNORECASE,
)


def _extract_indicator_from_explanation(explanation: str, op: str) -> str:
    if not explanation:
        return ""
    for m in _IND_RE.finditer(explanation):
        tag = m.group(1).lower()
        if op == "container" and tag in ("container", "insertion"):
            return m.group(2).strip()
        if op == "deletion" and tag in ("deletion", "parts"):
            return m.group(2).strip()
        if op == "acrostic" and tag in ("acrostic", "parts"):
            return m.group(2).strip()
        if tag == op:
            return m.group(2).strip()
    return ""


# --- DB membership checks ----------------------------------------------

def has_synonym(word: str, value: str, live, shadow) -> bool:
    w, v = word.lower().strip(), value.upper().strip()
    for c in (live, shadow):
        if c.execute(
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, v),
        ).fetchone():
            return True
    return False


def has_abbreviation(word: str, value: str, live, shadow) -> bool:
    w, v = word.lower().strip(), value.upper().strip()
    for c in (live, shadow):
        if c.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
            "AND UPPER(substitution)=? AND category='abbreviation' LIMIT 1",
            (w, v),
        ).fetchone():
            return True
    return False


def has_definition(definition: str, answer: str, live, shadow) -> bool:
    d, a = definition.lower().strip(), answer.upper().strip()
    for c in (live, shadow):
        if c.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (d, a),
        ).fetchone():
            return True
    return False


def has_homophone(word: str, value: str, live, shadow) -> bool:
    w, v = word.lower().strip(), value.upper().strip()
    for c in (live, shadow):
        try:
            if c.execute(
                "SELECT 1 FROM homophones WHERE LOWER(word)=? "
                "AND UPPER(homophone)=? LIMIT 1",
                (w, v),
            ).fetchone():
                return True
        except sqlite3.OperationalError:
            pass
    return False


def has_indicator(word: str, op: str, live, shadow) -> bool:
    w = word.lower().strip()
    for c in (live, shadow):
        if c.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? "
            "AND wordplay_type=? LIMIT 1",
            (w, op),
        ).fetchone():
            return True
    return False


def need_is_missing(need: dict, live, shadow) -> bool:
    k = need["kind"]
    if k == "synonym":
        return not has_synonym(need["word"], need["value"], live, shadow)
    if k == "abbreviation":
        return not has_abbreviation(need["word"], need["value"], live, shadow)
    if k == "definition":
        return not has_definition(need["word"], need["value"], live, shadow)
    if k == "homophone":
        return not has_homophone(need["word"], need["value"], live, shadow)
    if k == "indicator":
        return not has_indicator(need["word"], need["op"], live, shadow)
    return False


# --- Inserts -----------------------------------------------------------

def _candidate_key(clue_id: int, need: dict) -> str:
    payload = "|".join([
        str(clue_id),
        str(need.get("kind") or ""),
        str(need.get("word") or "").lower(),
        str(need.get("value") or "").upper(),
        str(need.get("op") or ""),
    ])
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def record_decision(key: str, decision: str) -> None:
    conn = _rw_shadow()
    conn.execute(
        "INSERT OR REPLACE INTO shadow_candidate_decisions "
        "(candidate_key, decision) VALUES (?, ?)",
        (key, decision),
    )
    conn.commit()


def insert_shadow_synonym(word: str, value: str, clue_id: int) -> None:
    _rw_shadow().execute(
        "INSERT INTO synonyms_pairs (word, synonym, source, clue_id) "
        "VALUES (?, ?, 'leftover_dashboard', ?)",
        (word.lower().strip(), value.upper().strip(), clue_id),
    ).connection.commit()


def insert_shadow_abbreviation(word: str, value: str, clue_id: int) -> None:
    _rw_shadow().execute(
        "INSERT INTO wordplay (indicator, substitution, category, "
        "confidence, clue_id) "
        "VALUES (?, ?, 'abbreviation', 'medium', ?)",
        (word.lower().strip(), value.upper().strip(), clue_id),
    ).connection.commit()


def insert_shadow_indicator(word: str, op: str, clue_id: int) -> None:
    _rw_shadow().execute(
        "INSERT INTO indicators (word, wordplay_type, confidence, "
        "source, clue_id) "
        "VALUES (?, ?, 'medium', 'leftover_dashboard', ?)",
        (word.lower().strip(), op, clue_id),
    ).connection.commit()


def insert_shadow_definition(definition: str, answer: str,
                               clue_id: int) -> None:
    _rw_shadow().execute(
        "INSERT INTO definition_answers_augmented (definition, answer, "
        "source, clue_id) "
        "VALUES (?, ?, 'leftover_dashboard', ?)",
        (definition.lower().strip(), answer.upper().strip(), clue_id),
    ).connection.commit()


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


def promote_indicator_live(word: str, op: str) -> None:
    conn = _rw_live()
    if conn.execute(
        "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=?",
        (word.lower().strip(), op),
    ).fetchone():
        return
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, confidence) "
        "VALUES (?, ?, 'high')",
        (word.lower().strip(), op),
    )
    conn.commit()


def accept_need(need: dict, clue_id: int, also_live: bool) -> None:
    k = need["kind"]
    if k == "synonym":
        insert_shadow_synonym(need["word"], need["value"], clue_id)
        if also_live:
            promote_synonym_live(need["word"], need["value"])
    elif k == "abbreviation":
        insert_shadow_abbreviation(need["word"], need["value"], clue_id)
        if also_live:
            promote_abbreviation_live(need["word"], need["value"])
    elif k == "definition":
        insert_shadow_definition(need["word"], need["value"], clue_id)
        if also_live:
            promote_definition_live(need["word"], need["value"])
    elif k == "indicator":
        insert_shadow_indicator(need["word"], need["op"], clue_id)
        if also_live:
            promote_indicator_live(need["word"], need["op"])


# --- Queries -----------------------------------------------------------

def fetch_unsolved_clues_with_reading(source_filter: str,
                                        puzzle_filter: str) -> list:
    """Pull the clues that:
    - have a row in shadow_db.seed_failures (= FAILed)
    - have NO PASS row in shadow_db.solves (still unsolved)
    - have EITHER (a) a structured_explanations row with components,
      OR (b) a seed_failures row carrying diagnostics_json (so the
      cascade run captured candidate hypotheses at FAIL time).

    Returns dicts with clue_id + reading payload + clue context.
    LEFT JOINs structured_explanations so cold clues come through
    too (they'll have NULL components / definition_text).
    """
    shadow = _ro(SHADOW_DB)
    where_sf = ["1=1"]
    params_sf: list = []
    if source_filter and source_filter != "All":
        where_sf.append("source = ?")
        params_sf.append(source_filter)
    if puzzle_filter:
        where_sf.append("puzzle_number = ?")
        params_sf.append(puzzle_filter)

    fail_clue_ids = set(
        r[0] for r in shadow.execute(
            f"SELECT DISTINCT clue_id FROM seed_failures "
            f"WHERE {' AND '.join(where_sf)}",
            params_sf,
        )
    )
    diag_clue_ids = set(
        r[0] for r in shadow.execute(
            f"SELECT DISTINCT clue_id FROM seed_failures "
            f"WHERE diagnostics_json IS NOT NULL "
            f"  AND ({' AND '.join(where_sf)})",
            params_sf,
        )
    )
    pass_clue_ids = set(
        r[0] for r in shadow.execute(
            "SELECT DISTINCT clue_id FROM solves WHERE verdict='PASS'"
        )
    )
    unsolved_ids = fail_clue_ids - pass_clue_ids
    if not unsolved_ids:
        return []

    clues = _ro(CLUES_DB)
    out: list = []
    # Chunk the IN list to avoid SQL limits.
    ids = list(unsolved_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = clues.execute(
            f"""
            SELECT c.id AS clue_id, c.source, c.puzzle_number,
                   c.clue_number, c.direction, c.clue_text, c.answer,
                   c.definition, c.wordplay_type, c.ai_explanation,
                   c.explanation,
                   se.components, se.definition_text, se.confidence,
                   se.model_version
            FROM clues c
            LEFT JOIN structured_explanations se ON se.clue_id = c.id
            WHERE c.id IN ({placeholders})
            """,
            chunk,
        ).fetchall()
        for r in rows:
            rec = dict(r)
            # Keep clues that EITHER have a recorded reading OR
            # have diagnostics captured. Otherwise nothing to
            # surface — they belong in the no-reading authoring tab.
            has_reading = bool((rec.get("components") or "").strip())
            has_diag = rec["clue_id"] in diag_clue_ids
            if not (has_reading or has_diag):
                continue
            rec["has_reading"] = has_reading
            rec["has_diagnostics"] = has_diag
            out.append(rec)
    out.sort(key=lambda x: (
        x.get("source") or "", x.get("puzzle_number") or "",
        int(str(x.get("clue_number") or 0).split("-")[0]) if
            str(x.get("clue_number") or "").replace("-", "").isdigit() else 0,
        x.get("direction") or "",
    ))
    return out


def compute_missing_for_clue(clue: dict, live, shadow) -> list:
    """Parse the clue's recorded reading, derive needed rows, filter
    to missing ones, attach a stable key, and exclude any candidate
    that's been previously rejected via shadow_candidate_decisions."""
    try:
        components = json.loads(clue.get("components") or "{}")
    except Exception:
        return []
    if not isinstance(components, dict):
        return []
    def_text = (clue.get("definition_text") or clue.get("definition")
                or "").strip()
    needs = derive_needed_rows(
        components, def_text, clue.get("answer") or "",
        clue.get("ai_explanation") or "",
    )

    return _filter_missing(clue["clue_id"], needs, live, shadow)


def compute_diagnostic_candidates(clue: dict, live, shadow) -> list:
    """Mine seed_failures.diagnostics_json (captured during the
    cascade run) for candidate enrichments. Three sources:

      - haiku_definition: the (phrase, answer) Haiku suggested when
        the DB had no def candidate. → definition row.
      - haiku_dbe: per-word Haiku DBE category-mate suggestions
        when the clue carries a DBE marker. → synonym rows.
      - grammar_triage word_roles: for each SYN_F / ABR_F entry
        across all attempted readings, the (clue_word, letters)
        pair the system hypothesised. → synonym / abbreviation row.

    These are hypotheses, not deterministic needs. The reviewer
    sees them with the same Add / Reject UI as the
    `compute_missing_for_clue` flow; filter to missing-in-DB and
    not-already-decided. Returned dicts carry the same shape so
    the renderer can treat them uniformly.
    """
    sh_ro = _ro(SHADOW_DB)
    rows = sh_ro.execute(
        """
        SELECT diagnostics_json FROM seed_failures
        WHERE clue_id = ? AND diagnostics_json IS NOT NULL
        ORDER BY run_number DESC, created_at DESC
        LIMIT 1
        """,
        (clue["clue_id"],),
    ).fetchall()
    if not rows:
        return []
    try:
        diag = json.loads(rows[0]["diagnostics_json"])
    except Exception:
        return []
    if not isinstance(diag, dict):
        return []

    needs: list = []
    answer_u = (clue.get("answer") or "").upper().strip()

    # haiku_definition → definition_answers_augmented row
    hd = diag.get("haiku_definition") or {}
    if isinstance(hd, dict) and hd.get("phrase"):
        needs.append({
            "kind": "definition",
            "word": hd["phrase"],
            "value": answer_u,
        })

    # haiku_dbe → synonym rows for each DBE-marked word
    hdbe = diag.get("haiku_dbe") or {}
    if isinstance(hdbe, dict):
        for word, cands in hdbe.items():
            if not isinstance(cands, list):
                continue
            for cand in cands:
                needs.append({
                    "kind": "synonym",
                    "word": word,
                    "value": str(cand).upper(),
                })

    # grammar_triage + production_solve roles → synonym / abbreviation
    # hypotheses. production_solve produces a SINGLE word_roles list
    # while grammar_triage produces a list-of-readings; unify both
    # under the same dedup so a high-confidence production role
    # doesn't get duplicated by a lower-confidence triage role for
    # the same (word, value).
    role_readings: list = []
    gt_list = diag.get("grammar_triage") or []
    if isinstance(gt_list, list):
        for gt in gt_list:
            if isinstance(gt, dict) and gt.get("word_roles"):
                role_readings.append(gt["word_roles"])
    ps = diag.get("production_solve") or {}
    if isinstance(ps, dict) and ps.get("word_roles"):
        role_readings.append(ps["word_roles"])

    seen_roles: set = set()
    for word_roles in role_readings:
        for role in word_roles:
            if not isinstance(role, (list, tuple)) or len(role) < 3:
                continue
            word, tok, val = role[0], role[1], role[2]
            if not word or not val:
                continue
            kind = None
            if tok == "SYN_F":
                kind = "synonym"
            elif tok == "ABR_F":
                kind = "abbreviation"
            if not kind:
                continue
            key = (kind, str(word).lower(), str(val).upper())
            if key in seen_roles:
                continue
            seen_roles.add(key)
            needs.append({
                "kind": kind,
                "word": str(word),
                "value": str(val).upper(),
            })

    return _filter_missing(clue["clue_id"], needs, live, shadow)


def _filter_missing(clue_id: int, needs: list, live, shadow) -> list:
    """Common tail for the two candidate sources: keep only the
    needs that aren't already in the DB and haven't been decided."""
    sh_w = _rw_shadow()
    decisions = {
        r["candidate_key"]: r["decision"]
        for r in sh_w.execute(
            "SELECT candidate_key, decision "
            "FROM shadow_candidate_decisions"
        )
    }
    out: list = []
    seen_keys: set = set()
    for n in needs:
        if not need_is_missing(n, live, shadow):
            continue
        key = _candidate_key(clue_id, n)
        if key in seen_keys or key in decisions:
            continue
        seen_keys.add(key)
        out.append({**n, "key": key, "clue_id": clue_id})
    return out


# --- Rendering --------------------------------------------------------

def render():
    st.header("Prototype enrichment review")
    st.caption(
        "Per clue: parse the recorded reading, work out exactly which "
        "DB rows the verifier needs to PASS it, show the missing ones. "
        "Add writes to shadow; the optional 'Also live' checkbox at "
        "the top promotes accepted rows into `cryptic_new` too. "
        "Reject records the decision and removes the row."
    )

    tab1, tab2, tab3 = st.tabs([
        "Unsolved (with reading)",
        "Unsolved (no reading)",
        "Puzzle overview",
    ])
    with tab1:
        _render_with_reading()
    with tab2:
        _render_no_reading()
    with tab3:
        _render_overview()


def _render_with_reading():
    col_a, col_b, col_c = st.columns([2, 2, 1])
    with col_a:
        source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent",
             "dailymail"],
            key="lo_src",
        )
    with col_b:
        puzzle = st.text_input("Puzzle (optional)", key="lo_puz")
    with col_c:
        also_live = st.checkbox(
            "Also live", value=False, key="lo_live",
            help="Promote Add'd rows to cryptic_new as well as shadow.",
        )

    if source == "All" and not puzzle.strip():
        st.info(
            "Pick a source (or enter a puzzle number) before loading. "
            "The unfiltered queue is many thousands of clues and "
            "would make the page hang."
        )
        return

    clues = fetch_unsolved_clues_with_reading(source, puzzle.strip())
    st.metric("Unsolved clues with a recorded reading", len(clues))

    if not clues:
        st.info(
            "No clues match. The combination might be: this filter "
            "matches no FAILs, all matching clues are already PASS, "
            "or none of them have a `structured_explanations` row."
        )
        return

    live = _rw_live()
    shadow = _rw_shadow()

    rendered_anything = False
    for c in clues:
        # Deterministic needs derived from a recorded reading.
        deterministic = compute_missing_for_clue(c, live, shadow)
        for n in deterministic:
            n["source"] = "reading"
        # Hypotheses captured during the cascade FAIL run.
        diagnostic = compute_diagnostic_candidates(c, live, shadow)
        for n in diagnostic:
            n["source"] = "diagnostic"
        # Dedup: a deterministic candidate trumps a diagnostic with
        # the same (kind, word, value); both have the same key so a
        # set of seen keys suffices.
        seen: set = set()
        combined: list = []
        for n in deterministic + diagnostic:
            if n["key"] in seen:
                continue
            seen.add(n["key"])
            combined.append(n)
        if not combined:
            continue
        rendered_anything = True
        _render_clue_with_missing(c, combined, also_live)

    if not rendered_anything:
        st.info(
            "Every clue with a recorded reading has all its rows "
            "already in (live ∪ shadow). Either the readings are "
            "fully covered, or the cascade is failing on these "
            "clues for some other reason (translator gap, schema "
            "mismatch). Look at the 'Unsolved (no reading)' tab "
            "for the rest of the FAIL queue."
        )


def _render_clue_with_missing(clue: dict, missing: list,
                                also_live: bool) -> None:
    badges = []
    if clue.get("has_reading"):
        badges.append(
            "<span style='background:#1a4480;color:white;padding:1px "
            "6px;border-radius:3px;font-size:0.75em'>reading</span>"
        )
    if clue.get("has_diagnostics"):
        badges.append(
            "<span style='background:#b58a00;color:white;padding:1px "
            "6px;border-radius:3px;font-size:0.75em'>diagnostics</span>"
        )
    badge_html = "&nbsp;".join(badges)
    st.markdown(
        f"**{clue.get('source','?')} {clue.get('puzzle_number','?')}  "
        f"{clue.get('clue_number','?')}{(clue.get('direction') or '')[:1]}  "
        f"— {clue.get('answer','?')}** &nbsp;{badge_html}",
        unsafe_allow_html=True,
    )
    st.write(clue.get("clue_text", ""))
    if clue.get("ai_explanation"):
        st.markdown(
            f"<span style='font-size:0.85em;color:#666'>"
            f"reading: {clue['ai_explanation']}</span>",
            unsafe_allow_html=True,
        )

    for n in missing:
        kind = n["kind"]
        if kind == "indicator":
            label = (f"indicator: `{n['word']}` "
                       f"(wordplay_type=`{n['op']}`)")
        elif kind == "definition":
            label = f"definition: `{n['word']}` → `{n['value']}`"
        else:
            label = f"{kind}: `{n['word']}` → `{n['value']}`"
        # Tag deterministic-from-reading vs hypothesis-from-diag.
        src = n.get("source", "reading")
        if src == "diagnostic":
            tag = (
                "<span style='background:#b58a00;color:white;"
                "padding:1px 6px;border-radius:3px;font-size:0.75em;"
                "margin-right:6px'>hypothesis</span>"
            )
        else:
            tag = (
                "<span style='background:#1a4480;color:white;"
                "padding:1px 6px;border-radius:3px;font-size:0.75em;"
                "margin-right:6px'>from reading</span>"
            )
        cols = st.columns([6, 1, 1])
        with cols[0]:
            st.markdown(tag + label, unsafe_allow_html=True)
        with cols[1]:
            if st.button("Add", key=f"add_{n['key']}"):
                try:
                    accept_need(n, clue["clue_id"], also_live)
                    record_decision(n["key"], "accepted")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Add failed: {e}")
        with cols[2]:
            if st.button("Reject", key=f"rej_{n['key']}"):
                try:
                    record_decision(n["key"], "rejected")
                    st.rerun()
                except Exception as e:  # noqa: BLE001
                    st.error(f"Reject failed: {e}")
    st.divider()


# --- No-reading tab (form authoring) ---------------------------------

def fetch_unsolved_clues_no_reading(source_filter: str,
                                      puzzle_filter: str) -> list:
    shadow = _ro(SHADOW_DB)
    where_sf = ["1=1"]
    params_sf: list = []
    if source_filter and source_filter != "All":
        where_sf.append("source = ?")
        params_sf.append(source_filter)
    if puzzle_filter:
        where_sf.append("puzzle_number = ?")
        params_sf.append(puzzle_filter)
    fail_clue_ids = set(
        r[0] for r in shadow.execute(
            f"SELECT DISTINCT clue_id FROM seed_failures "
            f"WHERE {' AND '.join(where_sf)}",
            params_sf,
        )
    )
    pass_clue_ids = set(
        r[0] for r in shadow.execute(
            "SELECT DISTINCT clue_id FROM solves WHERE verdict='PASS'"
        )
    )
    unsolved_ids = fail_clue_ids - pass_clue_ids
    if not unsolved_ids:
        return []

    clues = _ro(CLUES_DB)
    out: list = []
    ids = list(unsolved_ids)
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        placeholders = ",".join("?" * len(chunk))
        rows = clues.execute(
            f"""
            SELECT c.id AS clue_id, c.source, c.puzzle_number,
                   c.clue_number, c.direction, c.clue_text, c.answer,
                   c.definition, c.ai_explanation, c.explanation
            FROM clues c
            LEFT JOIN structured_explanations se ON se.clue_id = c.id
            WHERE c.id IN ({placeholders})
              AND (se.components IS NULL OR se.components = '')
            """,
            chunk,
        ).fetchall()
        for r in rows:
            out.append(dict(r))
    out.sort(key=lambda x: (
        x.get("source") or "", x.get("puzzle_number") or "",
        int(str(x.get("clue_number") or 0).split("-")[0]) if
            str(x.get("clue_number") or "").replace("-", "").isdigit() else 0,
        x.get("direction") or "",
    ))
    return out


def _render_no_reading():
    col_a, col_b = st.columns([2, 2])
    with col_a:
        source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent",
             "dailymail"],
            key="lo_nr_src",
        )
    with col_b:
        puzzle = st.text_input("Puzzle (optional)", key="lo_nr_puz")

    if source == "All" and not puzzle.strip():
        st.info(
            "Pick a source (or enter a puzzle number) before loading."
        )
        return

    clues = fetch_unsolved_clues_no_reading(source, puzzle.strip())
    st.metric("Unsolved clues with NO recorded reading", len(clues))

    if not clues:
        st.info("Nothing to author here.")
        return

    options = [
        f"{c.get('source','?')} {c.get('puzzle_number','?')} "
        f"{c.get('clue_number','?')}"
        f"{(c.get('direction') or '')[:1]}  "
        f"— {(c.get('answer') or '').strip()}  "
        f"— {(c.get('clue_text') or '')[:60]}"
        for c in clues
    ]
    idx = st.selectbox("Pick a clue to author",
                        options=list(range(len(clues))),
                        format_func=lambda i: options[i],
                        key="lo_nr_idx")
    clue = clues[idx]

    st.markdown(f"**Clue:** {clue.get('clue_text','')}")
    st.markdown(f"**Answer:** {clue.get('answer','')}")
    if clue.get("definition"):
        st.markdown(f"**Recorded definition:** `{clue['definition']}`")
    if clue.get("ai_explanation"):
        st.markdown(
            f"<span style='font-size:0.85em;color:#666'>"
            f"production reading: {clue['ai_explanation']}</span>",
            unsafe_allow_html=True,
        )
    if clue.get("explanation"):
        with st.expander("Blog explanation"):
            st.text(clue["explanation"])

    st.caption(
        "Author a reading by supplying a `components` JSON in the "
        "production format. The translator builds a Form, clipboard "
        "verifier checks it. On PASS we save to shadow_db.solves AND "
        "compute the rows the reading needs; missing rows appear in "
        "the 'with reading' tab for accept/reject."
    )

    default_template = {
        "ai_pieces": [
            {"mechanism": "synonym", "clue_word": "<source>",
             "letters": "<VALUE>"},
        ],
        "assembly": {"op": "charade", "order": ["<VALUE>"]},
        "wordplay_type": "charade",
    }
    def_phrase = st.text_input(
        "Definition phrase",
        value=clue.get("definition") or "",
        key=f"auth_def_{clue['clue_id']}",
    )
    components_text = st.text_area(
        "Components JSON",
        value=json.dumps(default_template, indent=2),
        height=220,
        key=f"auth_comp_{clue['clue_id']}",
    )
    if st.button("Verify and save (on PASS)",
                  key=f"auth_save_{clue['clue_id']}"):
        _verify_and_save(clue, def_phrase, components_text)


def _verify_and_save(clue: dict, def_phrase: str,
                      components_text: str) -> None:
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
        "clue_text": clue["clue_text"],
        "answer": clue["answer"],
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
        verdict = verify(form, clue["clue_text"], db, shadow_conn)
    except Exception as e:  # noqa: BLE001
        st.error(f"Verifier exception: {e}")
        return
    if verdict.verdict != "PASS":
        st.warning(f"Verifier {verdict.verdict}")
        for c in verdict.checks:
            if c.status != "pass":
                st.write(f"   - {c.name}: {c.detail}")
        return
    try:
        sig = form_signature(form.tree)
    except Exception:
        sig = "authored"
    write_solve(
        shadow_conn,
        clue_id=clue["clue_id"],
        signature=sig,
        verdict="PASS",
        answer=clue["answer"],
        form_dict=form.to_dict(),
        run_number=3,
    )
    st.success(f"Saved. Signature: `{sig}`")


# --- Puzzle overview tab ---------------------------------------------

def fetch_puzzle_state(source: str, puzzle: str) -> list:
    """For every clue in the puzzle, return its current state in our
    shadow_db plus any recorded production reading. Verdict priority:
    PASS > PENDING > FAIL > not-run.
    """
    clues = _ro(CLUES_DB)
    rows = clues.execute(
        """
        SELECT c.id AS clue_id, c.clue_number, c.direction, c.clue_text,
               c.answer, c.definition, c.ai_explanation,
               se.confidence AS prod_conf, se.model_version
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
        ORDER BY CAST(c.clue_number AS INTEGER), c.direction
        """,
        (source, puzzle),
    ).fetchall()
    out: list = []

    shadow = _ro(SHADOW_DB)
    for r in rows:
        rec = dict(r)
        clue_id = rec["clue_id"]
        verdict = None
        signature = None
        form_json = None
        run_number = None
        for s in shadow.execute(
            "SELECT verdict, signature, form_json, run_number, created_at "
            "FROM solves WHERE clue_id = ? "
            "ORDER BY CASE verdict WHEN 'PASS' THEN 0 "
            "  WHEN 'PENDING' THEN 1 ELSE 2 END, created_at DESC",
            (clue_id,),
        ):
            verdict = s["verdict"]
            signature = s["signature"]
            form_json = s["form_json"]
            run_number = s["run_number"]
            break
        if verdict is None:
            fail = shadow.execute(
                "SELECT 1 FROM seed_failures WHERE clue_id = ? LIMIT 1",
                (clue_id,),
            ).fetchone()
            if fail:
                verdict = "FAIL"
        rec["verdict"] = verdict or "not-run"
        rec["signature"] = signature
        rec["form_json"] = form_json
        rec["run_number"] = run_number
        out.append(rec)
    return out


def _render_overview():
    col_a, col_b = st.columns([2, 2])
    with col_a:
        source = st.selectbox(
            "Source",
            ["telegraph", "times", "guardian", "independent", "dailymail"],
            key="lo_ov_src",
        )
    with col_b:
        puzzle = st.text_input("Puzzle number", key="lo_ov_puz")
    if not puzzle.strip():
        st.info("Enter a puzzle number above.")
        return

    rows = fetch_puzzle_state(source, puzzle.strip())
    if not rows:
        st.info("No clues found for that source / puzzle.")
        return

    counts = {"PASS": 0, "PENDING": 0, "FAIL": 0, "not-run": 0}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    cs = st.columns(5)
    cs[0].metric("Total", len(rows))
    cs[1].metric("PASS", counts.get("PASS", 0))
    cs[2].metric("PENDING", counts.get("PENDING", 0))
    cs[3].metric("FAIL", counts.get("FAIL", 0))
    cs[4].metric("Not run", counts.get("not-run", 0))

    verdict_filter = st.selectbox(
        "Filter",
        ["All", "PASS", "PENDING", "FAIL", "not-run"],
        key="lo_ov_filter",
    )
    filtered = (rows if verdict_filter == "All"
                else [r for r in rows if r["verdict"] == verdict_filter])

    for r in filtered:
        _render_overview_card(r)
        st.divider()


def _render_overview_card(r: dict) -> None:
    """One card per clue: header, clue text with per-word attribution
    (multi-word spans grouped under a single label), and a fall-back
    summary line when no form is available.
    """
    verdict = r["verdict"]
    badge_colour = {
        "PASS": "#1a9e3e", "PENDING": "#b58a00",
        "FAIL": "#b03030", "not-run": "#888",
    }.get(verdict, "#888")
    cnum = f"{r['clue_number']}{(r['direction'] or '')[:1]}"
    header = (
        f"<span style='font-weight:bold;font-size:1.05em'>"
        f"{cnum} — {r['answer'] or ''}</span> &nbsp;"
        f"<span style='background:{badge_colour};color:white;padding:2px "
        f"8px;border-radius:8px;font-size:0.85em'>{verdict}</span>"
    )
    st.markdown(header, unsafe_allow_html=True)
    st.markdown(
        f"<div style='font-size:1.05em;margin:6px 0'>"
        f"{r['clue_text'] or ''}</div>",
        unsafe_allow_html=True,
    )

    form = None
    if r.get("form_json"):
        try:
            form = json.loads(r["form_json"])
        except Exception:
            form = None

    if form:
        attrs = _attribute_clue(r["clue_text"] or "", form)
        for span_text, label, colour in attrs:
            st.markdown(
                f"<div style='margin:2px 0'>"
                f"<span style='background:{colour};padding:2px 8px;"
                f"border-radius:4px;color:#fff;font-size:0.85em;"
                f"margin-right:8px;display:inline-block;min-width:140px'>"
                f"{label}</span>"
                f"<span style='font-size:1em'>{span_text}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if r.get("signature"):
            st.caption(f"signature: `{r['signature']}` "
                         f"(run {r.get('run_number','?')})")
    else:
        if r.get("ai_explanation"):
            st.markdown(
                f"<div style='font-size:0.9em;color:#666'>"
                f"production reading: {r['ai_explanation']}</div>",
                unsafe_allow_html=True,
            )
        elif r.get("definition"):
            st.markdown(
                f"<div style='font-size:0.9em;color:#666'>"
                f"recorded definition: <code>{r['definition']}</code></div>",
                unsafe_allow_html=True,
            )


def _attribute_clue(clue_text: str, form: dict) -> list:
    """Walk the form and turn the clue text into a list of
    (span_text, label, colour) tuples — one entry per attribution
    group. Multi-word spans (definition, multi-word source word,
    multi-word indicator) are grouped together.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from prototypes.universal_form_v2.surface import tokenize

    tokens = tokenize(clue_text)
    n = len(tokens)
    # For each surface position, we'll attach (label, colour).
    role: list = [None] * n

    def _label_positions(phrase: str, label: str, colour: str) -> None:
        if not phrase:
            return
        phrase_tokens = [t.lower() for t in tokenize(phrase)]
        if not phrase_tokens:
            return
        # Find phrase as contiguous lowercase token match in clue.
        lower = [t.lower() for t in tokens]
        for i in range(n - len(phrase_tokens) + 1):
            if lower[i:i + len(phrase_tokens)] == phrase_tokens:
                for j in range(i, i + len(phrase_tokens)):
                    if role[j] is None:
                        role[j] = (label, colour)
                break

    # Definition first (multi-word).
    def_phrase = (form.get("definition") or {}).get("phrase", "")
    if def_phrase:
        _label_positions(def_phrase, f"definition", "#1a4480")

    # Walk tree: collect leaves and op-indicators.
    def _walk(node):
        if not isinstance(node, dict):
            return
        op = node.get("operation")
        sources = node.get("sources") or []
        indicator = node.get("indicator")
        if sources:
            if indicator:
                colour = _OP_COLOURS.get(op, "#666")
                _label_positions(indicator, f"{op} indicator", colour)
            for c in sources:
                _walk(c)
        else:
            # Leaf
            cw = node.get("source_word") or ""
            value = node.get("value") or ""
            mech = op or "leaf"
            if node.get("positional_kind"):
                mech = f"{node['positional_kind']}-letter"
            label = f"{mech} → {value}"
            _label_positions(cw, label, _MECH_COLOURS.get(op, "#5a5a5a"))
            # If positional leaf carried its own indicator, label that too
            if indicator:
                _label_positions(indicator, f"{op} indicator",
                                  _OP_COLOURS.get(op, "#666"))

    _walk(form.get("tree") or {})

    # link_words
    for lw in form.get("link_words") or []:
        for i, t in enumerate(tokens):
            if t.lower() == lw.lower() and role[i] is None:
                role[i] = ("link", "#808080")

    # Anything still None is unattributed.
    for i in range(n):
        if role[i] is None:
            role[i] = ("(unaccounted)", "#a00")

    # Group consecutive tokens that share the same (label, colour).
    groups: list = []
    cur_label = None
    cur_colour = None
    cur_words: list = []
    for tok, (label, colour) in zip(tokens, role):
        if (label, colour) == (cur_label, cur_colour):
            cur_words.append(tok)
        else:
            if cur_words:
                groups.append((" ".join(cur_words), cur_label, cur_colour))
            cur_label, cur_colour = label, colour
            cur_words = [tok]
    if cur_words:
        groups.append((" ".join(cur_words), cur_label, cur_colour))
    return groups


_MECH_COLOURS = {
    "synonym":     "#3a6e9e",
    "abbreviation": "#7a4a9e",
    "literal":     "#4a7a4a",
    "raw":         "#4a7a4a",
    "homophone":   "#9e7a3a",
    "positional":  "#5a5a5a",
}

_OP_COLOURS = {
    "charade":   "#3a6e9e",
    "anagram":   "#9e3a3a",
    "container": "#3a9e6e",
    "deletion":  "#9e6e3a",
    "reversal":  "#6e3a9e",
    "hidden":    "#3a9e9e",
    "homophone": "#9e7a3a",
    "acrostic":  "#9e3a6e",
    "double_definition": "#1a4480",
}


def _render_overview_detail(r: dict) -> None:
    st.markdown(
        f"**{r['clue_number']}{(r['direction'] or '')[:1]}  "
        f"— {r['answer']}  ({r['verdict']})**"
    )
    st.write(r["clue_text"])
    if r.get("definition"):
        st.write(f"**Recorded definition:** `{r['definition']}`")
    if r.get("ai_explanation"):
        st.write(f"**Production reading:** {r['ai_explanation']}")
        if r.get("prod_conf"):
            st.caption(
                f"(production confidence {r['prod_conf']:.2f}, "
                f"source {r.get('model_version', '?')})"
            )

    if r.get("form_json"):
        try:
            form = json.loads(r["form_json"])
        except Exception:
            form = None
        if form:
            st.write("**Universal-form reading (our system):**")
            _render_form_pretty(form)
            with st.expander("Full form JSON"):
                st.json(form)

    if r["verdict"] == "FAIL":
        st.info(
            "Use the 'Unsolved (with reading)' tab to accept missing "
            "DB rows, or the 'Unsolved (no reading)' tab to author a "
            "form."
        )


def _render_form_pretty(form: dict) -> None:
    tree = form.get("tree") or {}
    def_phrase = (form.get("definition") or {}).get("phrase", "")
    link_words = form.get("link_words") or []
    is_and_lit = form.get("is_and_lit") or False

    if def_phrase:
        st.write(f"**Definition:** `{def_phrase}`")
    if link_words:
        st.write(f"**Link words:** {', '.join(link_words)}")
    if is_and_lit:
        st.write("**&lit** (whole clue is both definition and wordplay)")

    pieces = _walk_form(tree)
    if pieces:
        for line in pieces:
            st.write(f"- {line}")


def _walk_form(node: dict, depth: int = 0) -> list:
    out: list = []
    if not isinstance(node, dict):
        return out
    op = node.get("operation")
    indicator = node.get("indicator")
    value = node.get("value")
    source_word = node.get("source_word")
    sources = node.get("sources") or []
    prefix = "  " * depth

    if sources:
        ind = f' [{indicator}]' if indicator else ''
        kind_tag = ""
        if node.get("deletion_kind"):
            kind_tag = f"[{node['deletion_kind']}]"
        elif node.get("acrostic_kind"):
            kind_tag = f"[{node['acrostic_kind']}]"
        elif node.get("positional_kind"):
            kind_tag = f"[{node['positional_kind']}]"
        out.append(f"{prefix}{op}{kind_tag}{ind}")
        for c in sources:
            out.extend(_walk_form(c, depth + 1))
    else:
        ind = f' [{indicator}]' if indicator else ''
        kind_tag = ""
        if node.get("positional_kind"):
            kind_tag = f"[{node['positional_kind']}]"
        out.append(
            f"{prefix}{op}{kind_tag}: `{source_word}` → `{value}`{ind}"
        )
    return out


render()
