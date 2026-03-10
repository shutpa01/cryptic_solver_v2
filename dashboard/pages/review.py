"""Review Queue — approve/reject/edit explanations, enrich reference DB, re-run clues."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
CRYPTIC_DB = PROJECT_ROOT / "data" / "cryptic_new.db"


def _get_conn(db_path=CLUES_DB, readonly=True):
    if readonly:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=300)
def _load_ref_sets():
    """Preload reference DB into sets for fast enrichment lookups. Cached 5 min."""
    import sqlite3
    ref_conn = sqlite3.connect(str(CRYPTIC_DB))
    syn_set = set()
    for r in ref_conn.execute("SELECT LOWER(word), LOWER(synonym) FROM synonyms_pairs"):
        syn_set.add((r[0], r[1]))
    abbr_set = set()
    for r in ref_conn.execute("SELECT LOWER(indicator), LOWER(substitution) FROM wordplay"):
        abbr_set.add((r[0], r[1]))
    def_set = set()
    for r in ref_conn.execute("SELECT LOWER(definition), LOWER(answer) FROM definition_answers_augmented"):
        def_set.add((r[0], r[1]))
    ref_conn.close()
    return syn_set, abbr_set, def_set


def render():
    st.header("Review Queue")

    tab1, tab2, tab3 = st.tabs(["Review Queue", "DB Enrichment", "Unprocessed Clues"])

    with tab1:
        _render_review_queue()

    with tab2:
        _render_enrichment_queue()

    with tab3:
        _render_unprocessed()


# =====================================================================
# Unified Review Queue (uses clues.reviewed column)
# =====================================================================

def _render_review_queue():
    """Show all clues with explanations that need review."""
    conn = _get_conn()

    # Count unreviewed (only clues with solutions, not failures)
    unreviewed_count = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE reviewed = 0 AND has_solution IN (1, 2)"
    ).fetchone()[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        rq_status = st.selectbox(
            "Status",
            ["Unreviewed", "Approved", "Rejected", "All"],
            key="rq_status",
        )
    with col2:
        rq_source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent"],
            key="rq_source",
        )
    with col3:
        rq_limit = st.number_input(
            "Limit", min_value=10, max_value=500, value=50, key="rq_limit"
        )

    st.metric("Unreviewed", f"{unreviewed_count:,}")

    conditions = ["c.reviewed IS NOT NULL", "c.has_solution IN (1, 2)"]
    params = []

    status_map = {"Unreviewed": 0, "Approved": 1, "Rejected": 2}
    if rq_status != "All":
        conditions.append("c.reviewed = ?")
        params.append(status_map[rq_status])

    if rq_source != "All":
        conditions.append("c.source = ?")
        params.append(rq_source)

    where = " AND ".join(conditions)
    rows = conn.execute(f"""
        SELECT c.id, c.source, c.puzzle_number, c.publication_date,
               c.clue_text, c.enumeration, c.answer,
               c.definition, c.wordplay_type, c.ai_explanation,
               c.has_solution, c.reviewed,
               se.confidence as score
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE {where}
        ORDER BY c.publication_date DESC, c.puzzle_number DESC
        LIMIT ?
    """, params + [rq_limit]).fetchall()
    conn.close()

    if not rows:
        st.info(f"No {rq_status.lower()} clues to review.")
        return

    st.caption(f"{len(rows)} results (most recent first)")

    for row in rows:
        _render_review_card(row)


def _render_review_card(row):
    """Render a single review card with approve/reject/edit actions."""
    clue_id = row["id"]
    reviewed = row["reviewed"]
    badge = {0: ":orange[Unreviewed]", 1: ":green[Approved]", 2: ":red[Rejected]"}.get(
        reviewed, "Unknown"
    )
    solution_badge = ""
    if row["has_solution"] == 1:
        solution_badge = " :green[Full]"
    elif row["has_solution"] == 2:
        solution_badge = " :orange[Partial]"
    elif row["has_solution"] == 0:
        solution_badge = " :red[Failed]"

    score_val = row["score"]
    score_pct = int(score_val * 100) if score_val is not None else None
    score_str = f"  {score_pct}/100" if score_pct is not None else ""

    with st.expander(
        f"**{row['clue_text']}** ({row['enumeration'] or '?'}) = {row['answer'] or '?'}  "
        f"— {badge}{solution_badge}{score_str}  |  {row['source']} #{row['puzzle_number']}",
        expanded=(reviewed == 0),
    ):
        st.text(f"Definition: {row['definition'] or '—'}")
        st.text(f"Type: {row['wordplay_type'] or '—'}")
        st.text(f"Explanation: {row['ai_explanation'] or '—'}")
        st.text(f"Date: {row['publication_date'] or '—'}")

        # Action buttons
        col_a, col_b, col_c, col_d = st.columns(4)
        if reviewed != 1:
            with col_a:
                if st.button("Approve", key=f"approve_{clue_id}"):
                    _set_clue_reviewed(clue_id, 1)
                    st.rerun()
        if reviewed != 2:
            with col_b:
                if st.button("Reject", key=f"reject_{clue_id}"):
                    _set_clue_reviewed(clue_id, 2)
                    st.rerun()
        with col_c:
            if st.button("Edit", key=f"edit_toggle_{clue_id}"):
                st.session_state[f"editing_{clue_id}"] = True
        with col_d:
            if st.button("Re-run", key=f"rerun_{clue_id}"):
                _rerun_clue(clue_id)

        # Inline editor
        if st.session_state.get(f"editing_{clue_id}"):
            _render_editor(clue_id, row)

        # Manual enrichment (collapsed — bulk enrichment is in the DB Enrichment tab)
        with st.expander("Manual DB entry", expanded=False):
            _render_manual_enrichment(row)


def _set_clue_reviewed(clue_id, status):
    conn = _get_conn(CLUES_DB, readonly=False)
    conn.execute("UPDATE clues SET reviewed = ? WHERE id = ?", (status, clue_id))
    conn.commit()
    conn.close()


def _render_editor(clue_id, row):
    def_key = f"def_{clue_id}"
    type_key = f"type_{clue_id}"
    expl_key = f"expl_{clue_id}"
    if def_key not in st.session_state:
        st.session_state[def_key] = row["definition"] or ""
    if type_key not in st.session_state:
        st.session_state[type_key] = row["wordplay_type"] or ""
    if expl_key not in st.session_state:
        st.session_state[expl_key] = row["ai_explanation"] or ""

    with st.form(key=f"edit_form_{clue_id}"):
        new_def = st.text_input("Definition", key=def_key)
        new_type = st.text_input("Wordplay type", key=type_key)
        new_expl = st.text_area("Explanation", key=expl_key)
        if st.form_submit_button("Save & Approve"):
            conn = _get_conn(CLUES_DB, readonly=False)
            conn.execute(
                "UPDATE clues SET definition = ?, wordplay_type = ?, ai_explanation = ?, reviewed = 1 WHERE id = ?",
                (new_def or None, new_type or None, new_expl or None, clue_id),
            )
            conn.commit()
            conn.close()
            for k in (def_key, type_key, expl_key, f"editing_{clue_id}"):
                st.session_state.pop(k, None)
            st.rerun()


def _rerun_clue(clue_id):
    """Re-run a single clue through the web explainer (full pipeline)."""
    conn = _get_conn(CLUES_DB, readonly=False)
    conn.execute(
        "UPDATE clues SET definition = NULL, wordplay_type = NULL, ai_explanation = NULL, "
        "reviewed = NULL WHERE id = ?",
        (clue_id,),
    )
    conn.commit()
    conn.close()

    with st.spinner("Running pipeline..."):
        try:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from web.explainer import generate_explanation
            success, message, result = generate_explanation(clue_id)

            if success:
                st.success(
                    f"Done! Score: {result.get('score', '?')} | "
                    f"Op: {result.get('assembly_op', '?')} | "
                    f"Time: {result.get('total_ms', '?')}ms"
                )
                st.text(f"Definition: {result.get('definition', '—')}")
                st.text(f"Type: {result.get('wordplay_type', '—')}")
                st.text(f"Explanation: {result.get('ai_explanation', '—')}")
            else:
                st.error(f"Re-run failed: {message}")
        except Exception as e:
            st.error(f"Error: {e}")


# =====================================================================
# Enrichment tools
# =====================================================================

def _get_pending_enrichments(row):
    """Check AI pieces for this clue against the reference DB.

    Returns a list of dicts: {type, word, letters, status} where status is
    'missing' (not in DB) or 'exists' (already in DB).
    """
    clue_id = row["id"]
    conn = _get_conn()
    se = conn.execute(
        "SELECT components FROM structured_explanations WHERE clue_id = ?",
        (clue_id,)
    ).fetchone()
    conn.close()
    if not se or not se["components"]:
        return []

    import json
    try:
        comps = json.loads(se["components"])
    except (json.JSONDecodeError, TypeError):
        return []

    pieces = comps.get("ai_pieces", [])
    if not pieces:
        return []

    answer_clean = (row["answer"] or "").upper().replace(" ", "").replace("-", "")
    ref_conn = _get_conn(CRYPTIC_DB)
    suggestions = []

    for p in pieces:
        clue_word = (p.get("clue_word") or p.get("fodder") or "").strip()
        letters = (p.get("letters") or p.get("yields") or "").strip().upper()
        mech = (p.get("mechanism") or p.get("type") or "").lower()
        import re
        letters_clean = re.sub(r"[^A-Z]", "", letters)

        if not clue_word or not letters_clean:
            continue
        # Skip pieces that are the full answer, or non-enrichable mechanisms
        if letters_clean == answer_clean:
            continue
        if mech in ("literal", "anagram_fodder", "first_letter", "last_letter",
                     "hidden", "sound_of", "alternate_letters", "core_letters"):
            continue

        word_lower = clue_word.lower().strip(".,;:!?\"'()-")

        if mech == "synonym":
            exists = ref_conn.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND LOWER(synonym)=?",
                (word_lower, letters_clean.lower())
            ).fetchone()
            if not exists:
                suggestions.append({
                    "type": "synonym", "word": word_lower,
                    "letters": letters_clean, "status": "missing"
                })

        elif mech == "abbreviation":
            exists = ref_conn.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND LOWER(substitution)=?",
                (word_lower, letters_clean.lower())
            ).fetchone()
            if not exists:
                suggestions.append({
                    "type": "abbreviation", "word": word_lower,
                    "letters": letters_clean, "status": "missing"
                })

    # Check definition
    ai_def = row["definition"]
    if ai_def and answer_clean:
        exists = ref_conn.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND LOWER(answer)=?",
            (ai_def.lower(), answer_clean.lower())
        ).fetchone()
        if not exists:
            suggestions.append({
                "type": "definition", "word": ai_def.lower(),
                "letters": answer_clean, "status": "missing"
            })

    ref_conn.close()
    return suggestions


def _render_manual_enrichment(row):
    """Render manual DB enrichment forms (blank inputs for ad-hoc additions)."""
    answer = row["answer"] or ""
    uid = row["id"]

    tab_syn, tab_ind, tab_abbr, tab_homo, tab_def = st.tabs(
        ["Add Synonym", "Add Indicator", "Add Abbreviation", "Add Homophone", "Add Definition"]
    )

    with tab_syn:
        with st.form(key=f"syn_form_{uid}"):
            c1, c2 = st.columns(2)
            with c1:
                syn_word = st.text_input("Word (from clue)", key=f"syn_word_{uid}")
            with c2:
                syn_value = st.text_input("Synonym (letters)", key=f"syn_val_{uid}").upper()
            if st.form_submit_button("Add synonym pair"):
                if syn_word and syn_value:
                    if _add_synonym(syn_word.lower().strip(), syn_value.strip()):
                        st.success(f"Added: {syn_word} → {syn_value}")
                    else:
                        st.info(f"Already exists: {syn_word} → {syn_value}")
                else:
                    st.warning("Both fields required.")

    with tab_ind:
        with st.form(key=f"ind_form_{uid}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                ind_word = st.text_input("Indicator word", key=f"ind_word_{uid}")
            with c2:
                ind_type = st.selectbox("Wordplay type", [
                    "anagram", "container", "deletion", "hidden", "reversal",
                    "homophone", "acrostic", "alternation", "spoonerism",
                ], key=f"ind_type_{uid}")
            with c3:
                ind_conf = st.selectbox("Confidence", ["high", "medium", "low"],
                                        key=f"ind_conf_{uid}")
            if st.form_submit_button("Add indicator"):
                if ind_word:
                    if _add_indicator(ind_word.lower().strip(), ind_type, ind_conf):
                        st.success(f"Added indicator: {ind_word} → {ind_type} ({ind_conf})")
                    else:
                        st.info(f"Already exists: {ind_word} → {ind_type}")
                else:
                    st.warning("Word required.")

    with tab_abbr:
        with st.form(key=f"abbr_form_{uid}"):
            c1, c2 = st.columns(2)
            with c1:
                abbr_word = st.text_input("Word (from clue)", key=f"abbr_word_{uid}")
            with c2:
                abbr_sub = st.text_input("Substitution (letters)", key=f"abbr_sub_{uid}").upper()
            if st.form_submit_button("Add abbreviation"):
                if abbr_word and abbr_sub:
                    if _add_abbreviation(abbr_word.lower().strip(), abbr_sub.strip()):
                        st.success(f"Added: {abbr_word} → {abbr_sub}")
                    else:
                        st.info(f"Already exists: {abbr_word} → {abbr_sub}")
                else:
                    st.warning("Both fields required.")

    with tab_homo:
        with st.form(key=f"homo_form_{uid}"):
            c1, c2 = st.columns(2)
            with c1:
                homo_word = st.text_input("Word", key=f"homo_word_{uid}")
            with c2:
                homo_sounds = st.text_input("Sounds like", key=f"homo_sounds_{uid}")
            if st.form_submit_button("Add homophone"):
                if homo_word and homo_sounds:
                    if _add_homophone(homo_word.lower().strip(), homo_sounds.lower().strip()):
                        st.success(f"Added: {homo_word} sounds like {homo_sounds}")
                    else:
                        st.info(f"Already exists: {homo_word} / {homo_sounds}")
                else:
                    st.warning("Both fields required.")

    with tab_def:
        with st.form(key=f"def_form_{uid}"):
            c1, c2 = st.columns(2)
            with c1:
                def_phrase = st.text_input("Definition phrase", key=f"def_phrase_{uid}")
            with c2:
                def_answer = st.text_input("Answer", value=answer, key=f"def_answer_{uid}").upper()
            if st.form_submit_button("Add definition"):
                if def_phrase and def_answer:
                    if _add_definition(def_phrase.lower().strip(), def_answer.strip()):
                        st.success(f"Added: \"{def_phrase}\" → {def_answer}")
                    else:
                        st.info(f"Already exists: \"{def_phrase}\" → {def_answer}")
                else:
                    st.warning("Both fields required.")


# =====================================================================
# DB write helpers (all write to cryptic_new.db)
# =====================================================================

def _add_synonym(word, synonym):
    conn = _get_conn(CRYPTIC_DB, readonly=False)
    existing = conn.execute(
        "SELECT id FROM synonyms_pairs WHERE word = ? AND synonym = ?",
        (word, synonym),
    ).fetchone()
    if existing:
        conn.close()
        return False
    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, 'dashboard')",
        (word, synonym),
    )
    conn.commit()
    conn.close()
    return True


def _add_indicator(word, wordplay_type, confidence):
    conn = _get_conn(CRYPTIC_DB, readonly=False)
    existing = conn.execute(
        "SELECT id FROM indicators WHERE word = ? AND wordplay_type = ?",
        (word, wordplay_type),
    ).fetchone()
    if existing:
        conn.close()
        return False
    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, confidence, source) "
        "VALUES (?, ?, ?, 'dashboard')",
        (word, wordplay_type, confidence),
    )
    conn.commit()
    conn.close()
    return True


def _add_abbreviation(indicator, substitution):
    conn = _get_conn(CRYPTIC_DB, readonly=False)
    existing = conn.execute(
        "SELECT id FROM wordplay WHERE indicator = ? AND substitution = ?",
        (indicator, substitution),
    ).fetchone()
    if existing:
        conn.close()
        return False
    conn.execute(
        "INSERT INTO wordplay (indicator, substitution, category, confidence, notes) "
        "VALUES (?, ?, 'dashboard', 'high', '')",
        (indicator, substitution),
    )
    conn.commit()
    conn.close()
    return True


def _add_homophone(word, homophone):
    conn = _get_conn(CRYPTIC_DB, readonly=False)
    existing = conn.execute(
        "SELECT id FROM homophones WHERE word = ? AND homophone = ?",
        (word, homophone),
    ).fetchone()
    if existing:
        conn.close()
        return False
    max_group = conn.execute("SELECT MAX(group_id) FROM homophones").fetchone()[0] or 0
    conn.execute(
        "INSERT INTO homophones (word, homophone, group_id) VALUES (?, ?, ?)",
        (word, homophone, max_group + 1),
    )
    conn.execute(
        "INSERT INTO homophones (word, homophone, group_id) VALUES (?, ?, ?)",
        (homophone, word, max_group + 1),
    )
    conn.commit()
    conn.close()
    return True


def _add_definition(definition, answer):
    conn = _get_conn(CRYPTIC_DB, readonly=False)
    existing = conn.execute(
        "SELECT rowid FROM definition_answers_augmented WHERE definition = ? AND answer = ?",
        (definition, answer),
    ).fetchone()
    if existing:
        conn.close()
        return False
    conn.execute(
        "INSERT INTO definition_answers_augmented (definition, answer, source) "
        "VALUES (?, ?, 'dashboard')",
        (definition, answer),
    )
    conn.commit()
    conn.close()
    return True


# =====================================================================
# DB Enrichment tab
# =====================================================================

def _render_enrichment_queue():
    """Show all pending DB enrichments from recent pipeline runs.

    Scans structured_explanations for AI pieces not in the reference DB,
    grouped by type (synonyms, abbreviations, definitions).
    """
    import json
    import re

    col1, col2 = st.columns(2)
    with col1:
        eq_source = st.selectbox(
            "Source",
            ["All", "telegraph", "times", "guardian", "independent"],
            key="eq_source",
        )
    with col2:
        eq_limit = st.number_input(
            "Max clues to scan", min_value=50, max_value=2000, value=200,
            key="eq_limit",
        )

    conn = _get_conn()
    syn_set, abbr_set, def_set = _load_ref_sets()

    # Get recent clues with structured_explanations
    conditions = ["se.components IS NOT NULL"]
    params = []
    if eq_source != "All":
        conditions.append("c.source = ?")
        params.append(eq_source)

    where = " AND ".join(conditions)
    rows = conn.execute(f"""
        SELECT c.id, c.source, c.puzzle_number, c.clue_text, c.answer,
               c.definition, se.components
        FROM structured_explanations se
        JOIN clues c ON se.clue_id = c.id
        WHERE {where}
        ORDER BY c.publication_date DESC
        LIMIT ?
    """, params + [eq_limit]).fetchall()
    conn.close()

    # Collect all missing enrichments
    syn_missing = []
    abbr_missing = []
    def_missing = []

    for row in rows:
        answer = row["answer"] or ""
        answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
        clue_text = row["clue_text"] or ""

        try:
            comps = json.loads(row["components"])
        except (json.JSONDecodeError, TypeError):
            continue

        pieces = comps.get("ai_pieces", [])
        for p in pieces:
            clue_word = (p.get("clue_word") or p.get("fodder") or "").strip()
            letters = (p.get("letters") or p.get("yields") or "").strip().upper()
            mech = (p.get("mechanism") or p.get("type") or "").lower()
            letters_clean = re.sub(r"[^A-Z]", "", letters)

            if not clue_word or not letters_clean:
                continue
            if letters_clean == answer_clean:
                continue
            if mech in ("literal", "anagram_fodder", "first_letter", "last_letter",
                        "hidden", "sound_of", "alternate_letters", "core_letters"):
                continue

            word_lower = clue_word.lower().strip(".,;:!?\"'()-")

            if mech == "synonym":
                if (word_lower, letters_clean.lower()) not in syn_set:
                    syn_missing.append({
                        "word": word_lower, "letters": letters_clean,
                        "clue": clue_text[:50], "answer": answer,
                        "source": row["source"], "puzzle": row["puzzle_number"],
                    })

            elif mech == "abbreviation":
                if (word_lower, letters_clean.lower()) not in abbr_set:
                    abbr_missing.append({
                        "word": word_lower, "letters": letters_clean,
                        "clue": clue_text[:50], "answer": answer,
                        "source": row["source"], "puzzle": row["puzzle_number"],
                    })

        # Check definition
        ai_def = row["definition"]
        if ai_def and answer_clean:
            if (ai_def.lower(), answer_clean.lower()) not in def_set:
                def_missing.append({
                    "word": ai_def.lower(), "letters": answer_clean,
                    "clue": clue_text[:50], "answer": answer,
                    "source": row["source"], "puzzle": row["puzzle_number"],
                })

    # Deduplicate (same word→letters pair can appear from multiple clues)
    def _dedup(items):
        seen = set()
        result = []
        for item in items:
            key = (item["word"], item["letters"])
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result

    syn_missing = _dedup(syn_missing)
    abbr_missing = _dedup(abbr_missing)
    def_missing = _dedup(def_missing)

    total = len(syn_missing) + len(abbr_missing) + len(def_missing)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total", total)
    c2.metric("Synonyms", len(syn_missing))
    c3.metric("Abbreviations", len(abbr_missing))
    c4.metric("Definitions", len(def_missing))

    if total == 0:
        st.info("No pending enrichments found. Run the pipeline on more puzzles to generate suggestions.")
        return

    # --- Synonyms ---
    if syn_missing:
        st.subheader(f"Synonyms ({len(syn_missing)})")
        for i, s in enumerate(syn_missing):
            col1, col2, col3 = st.columns([4, 4, 1])
            with col1:
                st.text(f"{s['word']}  ->  {s['letters']}")
            with col2:
                st.caption(f"{s['source']} #{s['puzzle']} | {s['clue']}... = {s['answer']}")
            with col3:
                if st.button("Add", key=f"eq_syn_{i}"):
                    _add_synonym(s["word"], s["letters"])
                    st.rerun()

    # --- Abbreviations ---
    if abbr_missing:
        st.subheader(f"Abbreviations ({len(abbr_missing)})")
        for i, s in enumerate(abbr_missing):
            col1, col2, col3 = st.columns([4, 4, 1])
            with col1:
                st.text(f"{s['word']}  ->  {s['letters']}")
            with col2:
                st.caption(f"{s['source']} #{s['puzzle']} | {s['clue']}... = {s['answer']}")
            with col3:
                if st.button("Add", key=f"eq_abbr_{i}"):
                    _add_abbreviation(s["word"], s["letters"])
                    st.rerun()

    # --- Definitions ---
    if def_missing:
        st.subheader(f"Definitions ({len(def_missing)})")
        for i, s in enumerate(def_missing):
            col1, col2, col3 = st.columns([4, 4, 1])
            with col1:
                st.text(f"\"{s['word']}\"  ->  {s['letters']}")
            with col2:
                st.caption(f"{s['source']} #{s['puzzle']} | {s['clue']}... = {s['answer']}")
            with col3:
                if st.button("Add", key=f"eq_def_{i}"):
                    _add_definition(s["word"], s["letters"])
                    st.rerun()


# =====================================================================
# Unprocessed Clues tab
# =====================================================================

def _render_unprocessed():
    """Show clues that have answers but incomplete hints."""
    conn = _get_conn()

    col1, col2, col3 = st.columns(3)
    with col1:
        source_filter = st.selectbox(
            "Source", ["All", "telegraph", "times", "guardian", "independent"],
            key="unproc_source",
        )
    with col2:
        tier_filter = st.selectbox("Tier", ["NONE", "LOW", "MEDIUM", "All non-HIGH"], key="unproc_tier")
    with col3:
        limit = st.number_input("Limit", min_value=10, max_value=500, value=50, key="unproc_limit")

    conditions = [
        "c.answer IS NOT NULL", "c.answer != ''",
        "c.source IN ('telegraph', 'times', 'guardian', 'independent')",
    ]
    params = []

    if source_filter != "All":
        conditions.append("c.source = ?")
        params.append(source_filter)

    if tier_filter == "NONE":
        conditions.append("c.definition IS NULL AND c.wordplay_type IS NULL")
    elif tier_filter == "LOW":
        conditions.append("c.definition IS NOT NULL AND c.wordplay_type IS NULL")
    elif tier_filter == "MEDIUM":
        conditions.append("c.definition IS NOT NULL AND c.wordplay_type IS NOT NULL")
        conditions.append("c.ai_explanation IS NULL")
    else:
        conditions.append("""NOT (
            c.definition IS NOT NULL AND c.wordplay_type IS NOT NULL
            AND c.ai_explanation IS NOT NULL
        )""")

    where = " AND ".join(conditions)
    rows = conn.execute(f"""
        SELECT c.id, c.source, c.puzzle_number, c.publication_date,
               c.clue_number, c.direction, c.clue_text, c.enumeration,
               c.answer, c.definition, c.wordplay_type, c.ai_explanation
        FROM clues c
        WHERE {where}
        ORDER BY c.publication_date DESC
        LIMIT ?
    """, params + [limit]).fetchall()

    # Summary counts
    counts = conn.execute("""
        SELECT
            SUM(CASE WHEN c.definition IS NULL AND c.wordplay_type IS NULL THEN 1 ELSE 0 END) as none_tier,
            SUM(CASE WHEN c.definition IS NOT NULL AND c.wordplay_type IS NULL THEN 1 ELSE 0 END) as low_tier,
            SUM(CASE WHEN c.definition IS NOT NULL AND c.wordplay_type IS NOT NULL
                      AND c.ai_explanation IS NULL THEN 1 ELSE 0 END) as medium_tier,
            COUNT(*) as total_non_high
        FROM clues c
        WHERE c.answer IS NOT NULL AND c.answer != ''
          AND c.source IN ('telegraph', 'times', 'guardian', 'independent')
          AND NOT (c.definition IS NOT NULL AND c.wordplay_type IS NOT NULL
                   AND c.ai_explanation IS NOT NULL)
    """).fetchone()
    conn.close()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NONE", f"{counts['none_tier'] or 0:,}")
    c2.metric("LOW", f"{counts['low_tier'] or 0:,}")
    c3.metric("MEDIUM", f"{counts['medium_tier'] or 0:,}")
    c4.metric("Total non-HIGH", f"{counts['total_non_high'] or 0:,}")

    if not rows:
        st.info("No matching clues.")
        return

    data = []
    for r in rows:
        tier = "NONE"
        if r["definition"] and r["wordplay_type"] and r["ai_explanation"]:
            tier = "HIGH"
        elif r["definition"] and r["wordplay_type"]:
            tier = "MEDIUM"
        elif r["definition"]:
            tier = "LOW"

        data.append({
            "ID": r["id"],
            "Source": r["source"],
            "Puzzle": r["puzzle_number"],
            "Date": r["publication_date"],
            "Clue": r["clue_text"][:60] if r["clue_text"] else "",
            "Answer": r["answer"],
            "Tier": tier,
            "Def": "Y" if r["definition"] else "",
            "Type": "Y" if r["wordplay_type"] else "",
            "Expl": "Y" if r["ai_explanation"] else "",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
