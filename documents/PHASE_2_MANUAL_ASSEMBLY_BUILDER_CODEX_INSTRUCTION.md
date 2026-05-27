# Phase 2 Manual Assembly Builder — Codex Instruction (rev 4)

## Task

Wire manual WFW word roles into an authoritative Stage Three assembly so that
admin-assigned roles change proof["blocks"], proof["atomic_links"], and the
assembly checks — not just word_purposes. Also add piece_key support to the
admin UI so phrase sources like "That man -> HE" can be created from the browser.

This is Phase 1 of the manual-roles-as-assembly plan: charade, pure anagram,
and mixed anagram (one fixed prefix or suffix plus anagrammed fodder) only.
Container assembly (ENDEARED) is Phase 2 and is out of scope.

Four files change:
  web/routes/admin.py
  web/routes/clue.py
  web/templates/clue.html
  signature_solver/stage_three_proof.py

Do not change stage_two_casefile.py, wfw_display_adapter.py, wfw_proof_store.py,
word_roles_store.py, or any test file.


---

## Background

clue_word_roles holds per-word manual role assignments made via the admin WFW UI.
_write_manual_role_stage_three_for_clue already attaches these to the casefile
and passes them to build_stage_three_proof. They currently reach _word_purposes
(word classification) and _definition_check (definition acceptance) only.

They have zero effect on the assembly, blocks, or atomic_links because nothing
ever calls a manual assembly builder. This instruction adds that builder and wires
it in.

piece_key groups consecutive words into one proof piece (e.g. "That man" -> HE).
write_manual_role already accepts piece_key and saves it to the database, but the
admin route does not read it from the form and the templates have no input for it.
Changes 0a-0c fix this.

The design document is:
  documents/MANUAL_ROLES_AUTHORITATIVE_PROOF_PROPOSAL_2026-05-26.md (v4)

Read it before implementing if any design question arises.


---

## Change 0: piece_key UI support

### Change 0a: admin.py — read piece_key in set_word_role

Location: admin.py, set_word_role function (line 301).

Current function body reads role, word_text, and letters_raw from the form.
It calls write_manual_role without piece_key. Add piece_key reading between the
letters_raw line and the role validation:

Current (lines 309-318):

    role = (request.form.get("role") or "").strip()
    word_text = (request.form.get("word_text") or "").strip()
    letters_raw = (request.form.get("letters") or "").strip().upper()
    letters = letters_raw if letters_raw else None
    if role not in WORD_ROLE_CHOICES:
        abort(400)
    if not word_text:
        abort(400)
    from sonnet_pipeline.word_roles_store import write_manual_role
    write_manual_role(clue_id, word_index, word_text, role, letters=letters)

Replacement:

    role = (request.form.get("role") or "").strip()
    word_text = (request.form.get("word_text") or "").strip()
    letters_raw = (request.form.get("letters") or "").strip().upper()
    letters = letters_raw if letters_raw else None
    piece_key_raw = (request.form.get("piece_key") or "").strip()
    try:
        piece_key = int(piece_key_raw) if piece_key_raw else None
    except (ValueError, TypeError):
        piece_key = None
    if role not in WORD_ROLE_CHOICES:
        abort(400)
    if not word_text:
        abort(400)
    from sonnet_pipeline.word_roles_store import write_manual_role
    write_manual_role(
        clue_id, word_index, word_text, role,
        letters=letters, piece_key=piece_key)

No other change to set_word_role.


### Change 0b: clue.py — add piece_key to wfw_role_rows

Location: clue.py, the wfw_role_rows builder (line 603).

Current append block (lines 603-610):

    wfw_role_rows.append({
        "word_index": word_index,
        "token_index": token_index,
        "word_text": token.get("text") or "",
        "role": saved.get("role") or "unaccounted",
        "source": saved.get("source") or "wfw",
        "letters": saved.get("letters"),
    })

Replacement — add piece_key from the saved dict:

    wfw_role_rows.append({
        "word_index": word_index,
        "token_index": token_index,
        "word_text": token.get("text") or "",
        "role": saved.get("role") or "unaccounted",
        "source": saved.get("source") or "wfw",
        "letters": saved.get("letters"),
        "piece_key": saved.get("piece_key"),
    })

No other change to this builder. Note: word_role_rows (the non-WFW admin panel)
already includes piece_key (line 517) — no change needed there.


### Change 0c: clue.html — add piece_key input to both admin forms

There are two admin forms that call the set_word_role endpoint. Both need a
piece_key input so the field is submitted with hx-include="closest form".

#### Form 1: "Admin: word-role overrides" (line 258, iterates clue.word_roles)

After the existing letters input (ending at line 285), add inside the same
<form> element:

    <input type="text" name="piece_key"
           value="{{ w.piece_key if w.piece_key is not none else '' }}"
           placeholder="#"
           hx-post="/admin/word-role/{{ clue.id }}/{{ w.word_index }}"
           hx-target="#role-saved-{{ clue.id }}-{{ w.word_index }}"
           hx-swap="innerHTML"
           hx-trigger="change, blur"
           hx-include="closest form"
           class="text-xs border border-slate-300 rounded px-1 py-0.5 w-8
                  text-center bg-white text-gray-700"
           title="Piece group number (same integer = same phrase piece)">

Place it as the third field inside the form's flex container, immediately after
the letters input. The form currently ends at </form> on line 285. The new input
goes before that closing tag.

#### Form 2: "Admin: WFW word roles" (line 324, iterates clue.wfw_role_rows)

The current form is a two-column grid (grid-cols-[1fr_4.5rem]). Change it to a
three-column grid and add the piece_key input as the third column.

Current opening tag (line 324):

    <form class="grid grid-cols-[1fr_4.5rem] gap-1">

Replacement:

    <form class="grid grid-cols-[1fr_4.5rem_2rem] gap-1">

After the existing letters input (ending at line 345), add before </form>:

    <input type="text" name="piece_key"
           value="{{ w.piece_key if w.piece_key is not none else '' }}"
           placeholder="#"
           hx-post="/admin/word-role/{{ clue.id }}/{{ w.word_index }}"
           hx-target="#wfw-role-saved-{{ clue.id }}-{{ w.word_index }}"
           hx-swap="innerHTML"
           hx-trigger="change"
           hx-include="closest form"
           class="text-xs border border-slate-300 rounded px-1 py-0.5
                  text-center bg-white text-gray-700"
           title="Piece group number (same integer = same phrase piece)">


---

## Change 1: _manual_roles_for_clue — add source and piece_key

Location: admin.py line 1871.

Current query and return:

    def _manual_roles_for_clue(db, clue_id):
        rows = db.execute(
            "SELECT word_index, word_text, role, letters "
            "FROM clue_word_roles WHERE clue_id = ? ORDER BY word_index",
            (clue_id,),
        ).fetchall()
        return [
            {
                "index": row["word_index"],
                "text": row["word_text"],
                "role": row["role"],
                "letters": row["letters"],
            }
            for row in rows
        ]

Replacement — add source and piece_key to the SELECT and the returned dict:

    def _manual_roles_for_clue(db, clue_id):
        rows = db.execute(
            "SELECT word_index, word_text, role, letters, source, piece_key "
            "FROM clue_word_roles WHERE clue_id = ? ORDER BY word_index",
            (clue_id,),
        ).fetchall()
        return [
            {
                "index": row["word_index"],
                "text": row["word_text"],
                "role": row["role"],
                "letters": row["letters"],
                "source": row["source"],
                "piece_key": row["piece_key"],
            }
            for row in rows
        ]

No other change to this function.


---

## Change 2: WORD_ROLE_CHOICES — add new container role values

Location: admin.py line 163. WORD_ROLE_CHOICES is a tuple. Add the two new
container role values. "anagram_fodder" already exists in the tuple — do not
add it again.

Add these two strings to WORD_ROLE_CHOICES (place them after "anagram_fodder"):

    "container_frame",
    "container_content_source",


---

## Change 3: New function _build_manual_assembly

Add this function to admin.py, placed immediately before
_write_manual_role_stage_three_for_clue (line 1909).

The function signature:

    def _build_manual_assembly(manual_roles, answer, clue_words):

### Step 3a — imports and filter

    import re as _re

    def _clean(s):
        return _re.sub(r'[^A-Za-z]', '', s or '').upper()

    manual_only = [r for r in manual_roles if r.get("source") == "manual"]

### Step 3b — piece_key grouping

Group manual_only rows into proof pieces. Rows with piece_key=None are each
their own piece. Rows with the same non-null piece_key are merged into one piece.

    from collections import defaultdict

    singles = [r for r in manual_only if r.get("piece_key") is None]
    grouped = defaultdict(list)
    for r in manual_only:
        if r.get("piece_key") is not None:
            grouped[r["piece_key"]].append(r)

    pieces = []
    # singles: one piece per row
    for r in singles:
        pieces.append({
            "index": r["index"],
            "span": [r["index"], r["index"] + 1],
            "text": r["text"],
            "role": r["role"],
            "letters": r.get("letters") or "",
        })
    # grouped: merge rows with same piece_key
    for key, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["index"])
        # Validate: all rows in the group must have the same role
        role_set = {r.get("role") for r in rows}
        if len(role_set) > 1:
            print(f"[MANUAL_ASSEMBLY] piece_key={key} has inconsistent roles: {role_set}")
            return None
        # Validate: indices must be contiguous
        indices = [r["index"] for r in rows]
        if indices != list(range(indices[0], indices[-1] + 1)):
            print(f"[MANUAL_ASSEMBLY] piece_key={key} has non-contiguous indices: {indices}")
            return None
        # Validate: letters must be consistent (same value in every row)
        letters_set = {r.get("letters") or "" for r in rows}
        if len(letters_set) > 1:
            print(f"[MANUAL_ASSEMBLY] piece_key={key} has inconsistent letters: {letters_set}")
            return None
        pieces.append({
            "index": rows[0]["index"],
            "span": [rows[0]["index"], rows[-1]["index"] + 1],
            "text": " ".join(r["text"] for r in rows),
            "role": rows[0]["role"],
            "letters": rows[0].get("letters") or "",
        })
    # Sort pieces by span start
    pieces.sort(key=lambda p: p["index"])

### Step 3c — completeness check

Every word index in range(len(clue_words)) must be covered by a piece.
Structural roles are exempt from the letters requirement. All source-role
pieces must have non-empty letters.

    STRUCTURAL_ROLES = {
        "definition", "link", "surface", "charade_joiner",
        "anagram_indicator", "container_indicator", "reversal_indicator",
        "deletion_indicator", "hidden_indicator", "homophone_indicator",
        "first_letter_indicator", "last_letter_indicator",
        "letter_position_indicator", "alternating_indicator",
        "parts_indicator", "positional_indicator", "spoonerism_indicator",
        "indicator",
    }

    piece_by_index = {}
    for p in pieces:
        for i in range(p["span"][0], p["span"][1]):
            piece_by_index[i] = p

    for i in range(len(clue_words)):
        if i not in piece_by_index:
            print(f"[MANUAL_ASSEMBLY] index {i} ({clue_words[i]}) has no manual row — incomplete")
            return None
        p = piece_by_index[i]
        if p["role"] not in STRUCTURAL_ROLES and not p["letters"]:
            print(f"[MANUAL_ASSEMBLY] piece '{p['text']}' role={p['role']} has no letters — incomplete")
            return None

### Step 3d — separate piece types

    SOURCE_ROLES = {
        "synonym", "synonym_source", "abbreviation", "abbreviation_source",
        "single_letter", "double_letter", "roman_numeral", "nato_phonetic",
        "container_frame", "container_content_source",
        "literal_source", "letter_source", "positional_source",
        "reversal_source", "deletion_source", "hidden_source",
        "homophone_source", "first_letter", "pronoun", "name",
        "cricket", "chemistry", "musical", "shape", "example",
        "british_slang", "slang", "suffix", "reference",
    }

    # Deduplicate pieces by span start (piece_key groups already merged above)
    seen_starts = set()
    unique_pieces = []
    for p in pieces:
        if p["index"] not in seen_starts:
            seen_starts.add(p["index"])
            unique_pieces.append(p)

    # Only anagram_indicator triggers the anagram builder.
    # Generic "indicator" is a legacy role and must NOT be treated as an
    # anagram trigger.
    indicator_pieces = [p for p in unique_pieces if p["role"] == "anagram_indicator"]
    fodder_pieces    = [p for p in unique_pieces if p["role"] == "anagram_fodder"]
    container_pieces = [p for p in unique_pieces if p["role"] in {
        "container_frame", "container_content_source", "container_indicator"}]
    fixed_pieces = [p for p in unique_pieces
                    if p["role"] in SOURCE_ROLES
                    and p["role"] not in {"anagram_fodder",
                                          "container_frame",
                                          "container_content_source"}]
    definition_pieces = [p for p in unique_pieces if p["role"] == "definition"]

### Step 3d.5 — role-to-token helper

Define a small helper inside _build_manual_assembly (above Step 3e):

    def _role_to_token(role):
        if role in ("synonym", "synonym_source", "pronoun", "name",
                    "literal_source", "example", "reference"):
            return "SYN_F"
        if role in ("abbreviation", "abbreviation_source", "single_letter",
                    "double_letter", "roman_numeral", "nato_phonetic",
                    "cricket", "chemistry", "musical", "shape",
                    "british_slang", "slang", "suffix"):
            return "ABR_F"
        if role == "anagram_fodder":
            return "ANA_F"
        if role in ("positional_source", "letter_source", "first_letter"):
            return "POS_F"
        return "SYN_F"  # fallback for unrecognised source roles

### Step 3e — charade builder

Condition: no anagram_indicator, no anagram_fodder, and no container roles.
Return None immediately if any container role is present (container is Phase 2).
Verification: fixed piece letters concatenated in span order == answer.

    # Explicitly reject container roles in Phase 1
    if container_pieces:
        print(f"[MANUAL_ASSEMBLY] container roles present — Phase 2 only")
        return None

    if not indicator_pieces and not fodder_pieces:
        charade_letters = _clean("".join(p["letters"] for p in fixed_pieces))
        if charade_letters == _clean(answer):
            parts = []
            for p in fixed_pieces:
                parts.append({
                    "text": p["text"],
                    "span": p["span"],
                    "value": _clean(p["letters"]),
                    "token": _role_to_token(p["role"]),
                    "evidence_status": "manual",
                })
            return {
                "kind": "charade",
                "status": "manual_fit",
                "output": _clean(answer),
                "parts": parts,
            }
        print(f"[MANUAL_ASSEMBLY] charade verification failed: "
              f"{''.join(p['letters'] for p in fixed_pieces)!r} != {answer!r}")
        return None

### Step 3f — anagram builder

Condition: at least one anagram_indicator or anagram_fodder piece is present.
Container pieces are already rejected above.

Guard: anagram_fodder is required. If anagram_indicator is present but there is
no anagram_fodder, return None — an indicator alone cannot produce an anagram
assembly.

Scope restriction: Phase 1 supports only one contiguous fixed prefix or one
contiguous fixed suffix. If fixed_pieces is non-empty and their combined letters
do not appear as a clean prefix or suffix of the answer, return None.

The returned kind is:
  "anagram"       — if fixed_pieces is empty (pure anagram)
  "mixed_anagram" — if fixed_pieces is non-empty (fixed prefix/suffix + fodder)

    if indicator_pieces or fodder_pieces:
        # Anagram fodder is required; indicator alone is not enough.
        if not fodder_pieces:
            print(f"[MANUAL_ASSEMBLY] anagram_indicator present but no "
                  f"anagram_fodder — cannot build anagram assembly")
            return None

        ans = _clean(answer)
        fodder_letters = _clean("".join(p["letters"] for p in fodder_pieces))
        fixed_letters  = _clean("".join(p["letters"] for p in fixed_pieces))

        if not fixed_letters:
            # Pure anagram: all fodder, no fixed pieces
            if sorted(fodder_letters) == sorted(ans):
                parts = []
                for p in fodder_pieces:
                    parts.append({
                        "text": p["text"],
                        "span": p["span"],
                        "value": _clean(p["letters"]),
                        "token": "ANA_F",
                        "evidence_status": "manual",
                    })
                return {
                    "kind": "anagram",
                    "status": "manual_fit",
                    "output": ans,
                    "parts": parts,
                }
            print(f"[MANUAL_ASSEMBLY] pure anagram failed: "
                  f"sorted({fodder_letters!r}) != sorted({ans!r})")
            return None

        # Mixed anagram: fixed must be a prefix or suffix of the answer
        flen = len(fixed_letters)
        if ans[:flen] == fixed_letters:
            remaining = ans[flen:]
        elif ans[-flen:] == fixed_letters:
            remaining = ans[:-flen]
        else:
            print(f"[MANUAL_ASSEMBLY] mixed anagram: fixed {fixed_letters!r} "
                  f"is neither prefix nor suffix of {ans!r}")
            return None

        if sorted(fodder_letters) != sorted(remaining):
            print(f"[MANUAL_ASSEMBLY] mixed anagram: sorted fodder "
                  f"{sorted(fodder_letters)!r} != sorted remaining "
                  f"{sorted(remaining)!r}")
            return None

        parts = []
        for p in fixed_pieces:
            parts.append({
                "text": p["text"],
                "span": p["span"],
                "value": _clean(p["letters"]),
                "token": _role_to_token(p["role"]),
                "evidence_status": "manual",
            })
        for p in fodder_pieces:
            parts.append({
                "text": p["text"],
                "span": p["span"],
                "value": _clean(p["letters"]),
                "token": "ANA_F",
                "evidence_status": "manual",
            })
        return {
            "kind": "mixed_anagram",
            "status": "manual_fit",
            "output": ans,
            "parts": parts,
        }

    # No recognised structure
    return None

### Full function outline

The complete _build_manual_assembly function runs steps in order:
  Step 3a  (filter to manual_only)
  Step 3b  (piece_key grouping with three validations)
  Step 3c  (completeness check)
  Step 3d  (classify piece types)
  Step 3d.5 (_role_to_token helper definition)
  Step 3e  (container rejection guard, then charade branch — returns early)
  Step 3f  (anagram/mixed-anagram branch with fodder guard — returns early)
  return None  (no recognised structure)


---

## Change 4: _write_manual_role_stage_three_for_clue

Location: admin.py line 1909. Current function body (lines 1909–1957).

### Change 4a: filter to source="manual" before _build_manual_definition_candidates

_manual_roles_for_clue returns all rows including auto rows. Auto definition rows
must not become manual definition candidates. Replace the three-line block:

Current (lines 1931-1936):

    casefile.manual_roles = _manual_roles_for_clue(db, clue_id)
    ref_db = current_app.get_shared_ref_db()
    casefile.manual_definition_candidates = (
        _build_manual_definition_candidates(
            casefile.answer, casefile.manual_roles, ref_db)
    )

Replacement:

    casefile.manual_roles = _manual_roles_for_clue(db, clue_id)
    _manual_only = [r for r in casefile.manual_roles if r.get("source") == "manual"]
    ref_db = current_app.get_shared_ref_db()
    casefile.manual_definition_candidates = (
        _build_manual_definition_candidates(
            casefile.answer, _manual_only, ref_db)
    )

No other change to these lines.

### Change 4b: attach manual_assembly before build_stage_three_proof

After the _build_manual_definition_candidates block and before the call to
build_stage_three_proof (line 1938), add:

    from signature_solver.stage_three_proof import _clue_words as _st3_clue_words
    _clue_words_list = list(_st3_clue_words(casefile.clue_text))
    _manual_assembly = _build_manual_assembly(
        casefile.manual_roles, casefile.answer, _clue_words_list)
    if _manual_assembly is not None:
        casefile.manual_assembly = _manual_assembly

### Change 4c: set reviewed=1 on PASS with manual assembly

After the block that writes stage_three_json (lines 1947–1952) and before the
return statement, add:

    if _manual_assembly is not None and stage_three_proof.get("status") == "PASS":
        db.execute(
            "UPDATE clues SET reviewed = 1 WHERE id = ?",
            (clue_id,),
        )
        db.commit()

Note: _clue_words is a module-level private function in stage_three_proof.py
(line 1416). Import it inside the function body as a local import.


---

## Change 5: stage_three_proof.py — five targeted changes

### Change 5a: _purpose_for_manual_role — add new roles

Location: stage_three_proof.py line 1128.

Current:
    if role in ("synonym", "synonym_source", "abbreviation",
                "abbreviation_source", "nato_phonetic",
                "literal_source", "letter_source", "roman_numeral",
                "single_letter", "positional_source", "reversal_source",
                "deletion_source", "hidden_source", "homophone_source"):
        return "answer_source"

Replacement:
    if role in ("synonym", "synonym_source", "abbreviation",
                "abbreviation_source", "nato_phonetic",
                "literal_source", "letter_source", "roman_numeral",
                "single_letter", "positional_source", "reversal_source",
                "deletion_source", "hidden_source", "homophone_source",
                "container_frame", "container_content_source",
                "anagram_fodder"):
        return "answer_source"

No other change to _purpose_for_manual_role.


### Change 5b: build_stage_three_proof — prefer manual_assembly

Location: stage_three_proof.py line 72. Current line:

    assembly = _best_answer_fit_assembly(casefile.assemblies, answer)

Replacement:

    manual_assembly = getattr(casefile, "manual_assembly", None)
    assembly = manual_assembly or _best_answer_fit_assembly(casefile.assemblies, answer)

No other change to build_stage_three_proof.


### Change 5c: _assembly_check — trust output for manual_fit

Location: stage_three_proof.py line 216.

Add a new branch at the very start of the function body, before the existing
"if assembly:" block:

    def _assembly_check(answer, assembly, conditional_assemblies):
        if assembly and assembly.get("status") == "manual_fit":
            if _clean_answer(assembly.get("output", "")) == answer:
                return StageThreeCheck(
                    "answer_assembly",
                    PASS,
                    "%s (manual) = %s" % (assembly.get("output", ""), answer),
                    assembly,
                )
            return StageThreeCheck(
                "answer_assembly",
                REVIEW,
                "manual assembly output %r does not match answer %r" % (
                    assembly.get("output", ""), answer),
            )
        if assembly:
            ... (rest of existing body unchanged)


### Change 5d: _source_check — pass manual_fit immediately

Location: stage_three_proof.py line 254.

Add a new branch after "if not assembly:" and before the "missing = [...]" check:

    def _source_check(casefile, assembly):
        if not assembly:
            _candidates = tuple(
                getattr(casefile, "source_candidates", ()) or ())
            if _candidates:
                ... (unchanged)
            return StageThreeCheck(
                "source_evidence",
                REVIEW,
                "no complete assembly source list to verify",
            )
        if assembly.get("status") == "manual_fit":
            return StageThreeCheck(
                "source_evidence",
                PASS,
                "manual assembly mechanically verified by assembly builder",
                list(assembly.get("parts", ())),
            )
        missing = [
            ... (rest of existing body unchanged)


### Change 5e: _atomic_links — mixed_anagram branch

Location: stage_three_proof.py, _atomic_links() function (line 1043).

The current function has two branches:
  - kind != "anagram": positional links (each letter in sequence)
  - kind == "anagram": pool match (all letters pooled)

Insert a new branch for kind == "mixed_anagram" at the top, before the existing
kind != "anagram" branch. The new branch uses enumerate throughout — never
list.index() — to avoid wrong results when two dicts compare equal.

    def _atomic_links(answer, assembly):
        if not assembly:
            return ()

        if assembly.get("kind") == "mixed_anagram":
            # Fixed parts (token != "ANA_F") contribute positionally at the
            # prefix or suffix of the answer. Fodder parts (token == "ANA_F")
            # are pool-matched over the remaining portion.
            _parts = list(assembly.get("parts", ()))

            # Partition using enumerate to preserve correct overall indices.
            _fixed_indexed = [
                (pi, p) for pi, p in enumerate(_parts)
                if p.get("token") != "ANA_F"
            ]
            _fodder_indexed = [
                (pi, p) for pi, p in enumerate(_parts)
                if p.get("token") == "ANA_F"
            ]

            _fixed_str = _clean_answer(
                "".join(p.get("value", "") for _, p in _fixed_indexed))

            # Build the fodder pool using overall part indices as source_role.
            _pool = []
            for _pi_overall, _part in _fodder_indexed:
                _val = _clean_answer(_part.get("value", ""))
                for _si, _letter in enumerate(_val):
                    _pool.append({
                        "letter": _letter,
                        "source_text": _part.get("text", ""),
                        "source_span": _part.get("span"),
                        "source_role": "piece_%d" % _pi_overall,
                        "source_value": _val,
                        "source_value_index": _si,
                        "used": False,
                    })

            # Determine whether fixed is prefix, suffix, or absent.
            flen = len(_fixed_str)
            if not _fixed_str:
                _fixed_answer_start  = 0
                _fodder_answer_start = 0
                _fodder_portion      = answer
                _fixed_indexed_to_emit = []
            elif answer[:flen] == _fixed_str:
                _fixed_answer_start  = 0
                _fodder_answer_start = flen
                _fodder_portion      = answer[flen:]
                _fixed_indexed_to_emit = _fixed_indexed
            elif answer[-flen:] == _fixed_str:
                _fixed_answer_start  = len(answer) - flen
                _fodder_answer_start = 0
                _fodder_portion      = answer[:-flen]
                _fixed_indexed_to_emit = _fixed_indexed
            else:
                return ()

            _links = []

            # Positional links for fixed parts.
            _ai = _fixed_answer_start
            for _pi_overall, _part in _fixed_indexed_to_emit:
                _val = _clean_answer(_part.get("value", ""))
                for _si, _letter in enumerate(_val):
                    _links.append({
                        "answer_index": _ai,
                        "letter": _letter,
                        "source_text": _part.get("text", ""),
                        "source_span": _part.get("span"),
                        "source_role": "piece_%d" % _pi_overall,
                        "source_value": _val,
                        "source_value_index": _si,
                    })
                    _ai += 1

            # Pool-matched links for the fodder portion of the answer.
            for _offset, _aletter in enumerate(_fodder_portion):
                _matched = None
                for _entry in _pool:
                    if not _entry["used"] and _entry["letter"] == _aletter:
                        _entry["used"] = True
                        _matched = _entry
                        break
                if _matched is None:
                    return ()
                _links.append({
                    "answer_index": _fodder_answer_start + _offset,
                    "letter": _aletter,
                    "source_text": _matched["source_text"],
                    "source_span": _matched["source_span"],
                    "source_role": _matched["source_role"],
                    "source_value": _matched["source_value"],
                    "source_value_index": _matched["source_value_index"],
                })

            _links.sort(key=lambda l: l["answer_index"])
            if len(_links) != len(answer):
                return ()
            return tuple(_links)

        if assembly.get("kind") != "anagram":
            # Existing positional branch — unchanged
            ...

        # Existing anagram pool-match branch — unchanged
        ...


---

## Change 6: _blocks — four additions

### Ordering within _blocks after all changes

1. Manual DEF_BLOCKs (new — Change 6b, emitted first)
2. Stage Two DEF_BLOCKs (suppressed entirely when any manual def exists)
3. SOURCE_BLOCKs from assembly parts (with container_role — Change 6c)
4. SOURCE_BLOCKs + OP_BLOCKs from working_pairs (unchanged)
5. OP_BLOCKs from operation_candidates (unchanged)
6. CONDITIONAL_ASSEMBLY_BLOCKs (unchanged)
7. Manual OP_BLOCKs (new — Change 6a, before REVIEW_BLOCK loop)
8. Build _covered_indices from source_spans, operation_spans, _manual_def_spans
9. REVIEW_BLOCKs with index-containment suppression (revised — Change 6a)


### Change 6a: manual OP_BLOCKs and REVIEW_BLOCK suppression

Location: stage_three_proof.py, _blocks() function.

Step 1 — Add a manual OP_BLOCK pass immediately BEFORE the REVIEW_BLOCK loop
(line 1033), after the CONDITIONAL_ASSEMBLY_BLOCK loop:

    for _mr in (getattr(casefile, "manual_roles", None) or ()):
        if _mr.get("source") != "manual":
            continue
        _mr_role = _mr.get("role", "")
        if _purpose_for_manual_role(_mr_role) != "operation_indicator":
            continue
        _mr_span = (_mr["index"], _mr["index"] + 1)
        if _mr_span in operation_spans:
            continue
        yield {
            "kind": "OP_BLOCK",
            "role": _mr_role,
            "text": _mr.get("text", ""),
            "span": list(_mr_span),
            "token": None,
            "source": "manual_role",
            "status": "verified",
        }
        operation_spans.add(_mr_span)

Step 2 — After the manual OP_BLOCK pass, build a set of all word indices covered
by any source block, operation block, or manual definition block. Phrase-spanning
source blocks (span [0,2]) must suppress REVIEW_BLOCKs for words at indices 0
and 1, not just an exact span match:

    _covered_indices = set()
    for _sp in source_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)
    for _sp in operation_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)
    for _sp in _manual_def_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)

Step 3 — Replace the existing REVIEW_BLOCK loop with a version that skips any
unresolved word whose word index is in _covered_indices:

    for item in _unresolved:
        if item.get("index") in _covered_indices:
            continue
        yield {
            "kind": "REVIEW_BLOCK",
            "role": "unresolved",
            "text": item.get("text", ""),
            "span": (item.get("index"), item.get("index", 0) + 1),
            "status": "review",
        }

Note: _manual_def_spans is defined at the start of _blocks() by Change 6b.


### Change 6b: manual DEF_BLOCKs emitted first; Stage Two suppressed when any exists

Location: stage_three_proof.py, _blocks() function. The current first loop
(line 927) emits DEF_BLOCKs from definition_candidates.

Replace the current first loop with the following two-part block:

Part 1 — Emit manual DEF_BLOCKs first and track their spans:

    _manual_def_spans = set()
    for _mc in (getattr(casefile, "manual_definition_candidates", None) or ()):
        if _mc.get("boundary_status") != "manual_definition":
            continue
        _mc_span = _mc.get("span")
        if not _mc_span:
            continue
        yield {
            "kind": "DEF_BLOCK",
            "text": _mc.get("text", ""),
            "span": _mc_span,
            "value": casefile.answer,
            "status": "manual",
        }
        _manual_def_spans.add(tuple(_mc_span))

Part 2 — Emit Stage Two DEF_BLOCKs only when no accepted manual definition
exists. If any manual definition was accepted, Stage Two definitions are
suppressed entirely:

    if not _manual_def_spans:
        for definition in _defs:
            yield {
                "kind": "DEF_BLOCK",
                "text": definition["text"],
                "span": definition["span"],
                "value": casefile.answer,
                "status": "verified",
            }

Rationale: if the Stage Two definition has the wrong span, both blocks would
appear under span-by-span suppression, leaving wrong colouring. Suppressing all
Stage Two DEF_BLOCKs when any manual definition is accepted ensures the manual
definition is the sole authoritative one.


### Change 6c: pass container_role through SOURCE_BLOCKs

Location: stage_three_proof.py, _blocks(), the assembly parts loop (line 936).

Current SOURCE_BLOCK yield — add container_role to the dict:

    yield {
        "kind": "SOURCE_BLOCK",
        "role": "piece_%d" % idx,
        "text": part.get("text", ""),
        "span": part.get("span"),
        "value": part.get("value", ""),
        "token": part.get("token"),
        "derivation_kind": part.get("derivation_kind"),
        "evidence_status": part.get("evidence_status"),
        "evidence_reason": part.get("evidence_reason"),
        "container_role": part.get("container_role"),
        "status": "verified",
    }


---

## What not to change

Do not change _word_purposes, _purpose_requests, _definition_check, or
_accepted_definition_candidate (all already correctly implemented).
Do not change build_stage_two_casefile or any Stage Two logic.
Do not change the anagram pool-match branch or the positional branch of
_atomic_links — only the new mixed_anagram branch is added.
Do not change wfw_display_adapter.py.
Do not change any test file.
Do not change the schema string "stage_three_proof:v1".
Do not add any new imports to stage_three_proof.py at the module top level.
Any additional imports go inside function bodies as local imports.
In admin.py, the import of _clue_words from stage_three_proof goes inside
_write_manual_role_stage_three_for_clue as a local import.


---

## Verification

### Check 1 — syntax

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile web\routes\admin.py

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile signature_solver\stage_three_proof.py

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        -m py_compile web\routes\clue.py

Expected: no output, exit 0 for all three.


### Check 2 — regression suite

    C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe
        signature_solver\test_stage_three_proof.py

Expected: "Stage Three proof contract passed"


### Check 3 — manual charade smoke test

Write and run a standalone script (do not save to repo):

    import sys
    sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
    from types import SimpleNamespace
    from signature_solver.stage_three_proof import build_stage_three_proof, _clue_words
    import web.routes.admin as adm

    manual_roles = [
        {"index": 0, "text": "te",   "role": "synonym_source", "letters": "TE",
         "source": "manual", "piece_key": None},
        {"index": 1, "text": "st",   "role": "synonym_source", "letters": "ST",
         "source": "manual", "piece_key": None},
        {"index": 2, "text": "test", "role": "definition",     "letters": "",
         "source": "manual", "piece_key": None},
    ]
    answer = "TEST"
    clue_text = "te st test"

    clue_words = list(_clue_words(clue_text))
    assembly = adm._build_manual_assembly(manual_roles, answer, clue_words)
    assert assembly is not None, "builder returned None"
    assert assembly["status"] == "manual_fit"
    assert assembly["kind"] == "charade"
    assert assembly["output"] == "TEST"

    casefile = SimpleNamespace(
        clue_text=clue_text, answer=answer,
        definition_candidates=[], grammar_phrases=[],
        source_candidates=[], operation_candidates=[],
        working_pairs=[], assemblies=[], enrichment_candidates=[],
        unresolved_words=[], status="unknown",
        manual_roles=manual_roles, manual_definition_candidates=[],
        manual_assembly=assembly,
    )
    proof = build_stage_three_proof(casefile).as_dict()
    for check in proof["checks"]:
        print(check["name"], check["status"])
    asm_check = next(c for c in proof["checks"] if c["name"] == "answer_assembly")
    src_check = next(c for c in proof["checks"] if c["name"] == "source_evidence")
    assert asm_check["status"] == "PASS", f"answer_assembly={asm_check['status']}"
    assert src_check["status"] == "PASS", f"source_evidence={src_check['status']}"
    # Overall proof status is not expected to be PASS — no definition candidate,
    # so definition_evidence will be REVIEW.
    print("charade smoke test PASSED")


### Check 4 — mixed anagram smoke test (HEATED)

    import sys
    sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
    from types import SimpleNamespace
    from signature_solver.stage_three_proof import (
        build_stage_three_proof, _clue_words, _atomic_links, _clean_answer)
    import web.routes.admin as adm

    manual_roles = [
        {"index": 0, "text": "That",   "role": "synonym_source",   "letters": "HE",
         "source": "manual", "piece_key": 1},
        {"index": 1, "text": "man",    "role": "synonym_source",   "letters": "HE",
         "source": "manual", "piece_key": 1},
        {"index": 2, "text": "date",   "role": "anagram_fodder",   "letters": "DATE",
         "source": "manual", "piece_key": None},
        {"index": 3, "text": "mixed",  "role": "anagram_indicator","letters": "",
         "source": "manual", "piece_key": None},
        {"index": 4, "text": "heated", "role": "definition",       "letters": "",
         "source": "manual", "piece_key": None},
    ]
    answer = "HEATED"
    clue_text = "That man date mixed heated"

    clue_words = list(_clue_words(clue_text))
    assembly = adm._build_manual_assembly(manual_roles, answer, clue_words)
    assert assembly is not None, "builder returned None"
    assert assembly["kind"] == "mixed_anagram"
    assert assembly["output"] == "HEATED"
    he_part   = next((p for p in assembly["parts"] if p["value"] == "HE"),   None)
    date_part = next((p for p in assembly["parts"] if p["value"] == "DATE"), None)
    assert he_part   is not None, "no HE part"
    assert date_part is not None, "no DATE part"
    assert he_part["token"]   == "SYN_F"
    assert date_part["token"] == "ANA_F"

    answer_clean = _clean_answer(answer)
    links = _atomic_links(answer_clean, assembly)
    assert len(links) == 6, f"expected 6 links, got {len(links)}"
    h_link = next(l for l in links if l["answer_index"] == 0)
    e_link = next(l for l in links if l["answer_index"] == 1)
    assert h_link["source_text"] == "That man"
    assert e_link["source_text"] == "That man"
    for ai in (2, 3, 4, 5):
        lnk = next(l for l in links if l["answer_index"] == ai)
        assert lnk["source_text"] == "date", f"pos {ai} from {lnk['source_text']!r}"

    print("mixed anagram smoke test PASSED")


### Check 5 — piece_key persistence through the admin save path

This check proves that piece_key travels from the set_word_role form-parsing
logic through write_manual_role into the database, and that _manual_roles_for_clue
(Change 1) reads it back correctly. It uses an isolated in-memory connection so
it does not touch the live database.

Write and run a standalone script (do not save to repo):

    import sys, sqlite3, os, tempfile
    sys.path.insert(0, r"C:\Users\shute\PycharmProjects\cryptic_solver_V2")
    from sonnet_pipeline.word_roles_store import write_manual_role, ensure_table
    import web.routes.admin as adm

    # Part A — prove the form-parsing logic converts "1" to integer 1.
    # This replicates the three lines added to set_word_role in Change 0a.
    for raw, expected in [("1", 1), ("2", 2), ("", None), ("abc", None)]:
        try:
            result = int(raw) if raw.strip() else None
        except (ValueError, TypeError):
            result = None
        assert result == expected, f"parse {raw!r}: got {result}, want {expected}"
    print("Part A (form parse) PASSED")

    # Part B — prove write_manual_role saves piece_key and _manual_roles_for_clue
    # reads it back. Use a temporary file DB so the live DB is untouched.
    tmp = tempfile.mktemp(suffix=".db")
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        ensure_table(conn)

        # Simulate two consecutive form submissions for "That" and "man"
        # both with piece_key=1, as the admin UI would send after Change 0a/0c.
        write_manual_role(
            clue_id=99999, word_index=0, word_text="That",
            role="synonym_source", letters="HE", piece_key=1, conn=conn)
        write_manual_role(
            clue_id=99999, word_index=1, word_text="man",
            role="synonym_source", letters="HE", piece_key=1, conn=conn)
        write_manual_role(
            clue_id=99999, word_index=2, word_text="date",
            role="anagram_fodder", letters="DATE", piece_key=None, conn=conn)
        conn.commit()

        # Read back using the updated _manual_roles_for_clue (Change 1).
        # It must return source and piece_key.
        roles = adm._manual_roles_for_clue(conn, 99999)
        assert len(roles) == 3, f"expected 3 rows, got {len(roles)}"

        that_row = next(r for r in roles if r["text"] == "That")
        man_row  = next(r for r in roles if r["text"] == "man")
        date_row = next(r for r in roles if r["text"] == "date")

        assert that_row["source"]    == "manual",        f"That source={that_row['source']}"
        assert that_row["role"]      == "synonym_source", f"That role={that_row['role']}"
        assert that_row["letters"]   == "HE",            f"That letters={that_row['letters']}"
        assert that_row["piece_key"] == 1,               f"That piece_key={that_row['piece_key']}"

        assert man_row["source"]    == "manual",         f"man source={man_row['source']}"
        assert man_row["piece_key"] == 1,                f"man piece_key={man_row['piece_key']}"

        assert date_row["piece_key"] is None,            f"date piece_key={date_row['piece_key']}"

        print(f"That: source={that_row['source']}, piece_key={that_row['piece_key']}")
        print(f"man:  source={man_row['source']},  piece_key={man_row['piece_key']}")
        print(f"date: source={date_row['source']}, piece_key={date_row['piece_key']}")

        # Part C — the saved rows must feed into the builder correctly.
        # piece_key=1 should group "That" and "man" into one piece with value HE.
        from signature_solver.stage_three_proof import _clue_words
        clue_words = list(_clue_words("That man date mixed heated"))
        # Add the remaining words as manual rows so the completeness check passes
        write_manual_role(
            clue_id=99999, word_index=3, word_text="mixed",
            role="anagram_indicator", letters=None, piece_key=None, conn=conn)
        write_manual_role(
            clue_id=99999, word_index=4, word_text="heated",
            role="definition", letters=None, piece_key=None, conn=conn)
        conn.commit()
        roles_full = adm._manual_roles_for_clue(conn, 99999)
        assembly = adm._build_manual_assembly(roles_full, "HEATED", clue_words)
        assert assembly is not None, "builder returned None with DB-sourced roles"
        assert assembly["kind"] == "mixed_anagram", f"kind={assembly['kind']}"
        he_part = next((p for p in assembly["parts"] if p["value"] == "HE"), None)
        assert he_part is not None, "no HE part"
        assert he_part["text"] == "That man", f"phrase text={he_part['text']!r}"
        print(f"Part C: assembly kind={assembly['kind']}, HE piece text={he_part['text']!r}")
        print("Part B+C (DB persistence and builder) PASSED")

        conn.close()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    print("piece_key persistence check PASSED")

Expected: all three parts print their PASSED message with no assertion errors.


---

## After writing

Paste:
  1. The full body of _build_manual_assembly.
  2. The changed set_word_role (Change 0a).
  3. The changed wfw_role_rows append block in clue.py (Change 0b).
  4. The changed _manual_roles_for_clue (Change 1).
  5. The full revised _write_manual_role_stage_three_for_clue body
     showing all three sub-changes (4a, 4b, 4c).
  6. The changed opening of _assembly_check (manual_fit branch only).
  7. The changed opening of _source_check (manual_fit branch only).
  8. The full new _atomic_links mixed_anagram branch.
  9. The revised DEF_BLOCK section of _blocks (both parts of Change 6b).
  10. The manual OP_BLOCK loop, the _covered_indices build, and the revised
      REVIEW_BLOCK loop (Change 6a).

Then run all five verification checks and paste the full output of each.
