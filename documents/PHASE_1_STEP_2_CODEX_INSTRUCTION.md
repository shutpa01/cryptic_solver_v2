# Phase 1 Step 2 — Codex Instruction (revised)

Task: Add build_stage_two_from_solve_result to stage_two_casefile.py,
and add a high-confidence branch in _attach_gt2_evidence in solver.py
that calls it instead of build_stage_two_casefile.

Two files change in this step. No other file is touched.

---

## Context

The existing _attach_gt2_evidence calls build_stage_two_casefile
unconditionally. That function builds Stage Two from grammar annotations
alone. After Step 2, when the legacy solver produced a high-confidence
result, Stage Two is built from the solver's own word_roles. The
grammar-only path remains as the fallback for unsolved clues.

---

## Change 1 — New function in stage_two_casefile.py

Add build_stage_two_from_solve_result at the end of the file, after
the existing build_stage_two_casefile function.

Function signature:

    def build_stage_two_from_solve_result(
        clue_text, answer, db, solve_result,
        *, stage_one_context=None):


### Imports

All tokens already imported at the top of the file are available.
To handle all positional indicator tokens without enumerating them,
check token.startswith("POS_I") for indicator classification.


### Guard — fall back to grammar-only path

If solve_result is None, solve_result.result is None, or
solve_result.result.word_roles is empty, fall back immediately:

    if (solve_result is None
            or solve_result.result is None
            or not solve_result.result.word_roles):
        return build_stage_two_casefile(
            clue_text, answer, db,
            stage_one_context=stage_one_context)


### Build stage_one and context

    answer_clean = _clean_answer(answer)
    stage_one = stage_one_context or build_clue_context(
        clue_text, answer_clean, db, annotate=False)
    context = with_wordplay_annotations(stage_one, db)

IMPORTANT: stage_one is built with annotate=False, so
stage_one.annotations is empty or minimal. All wordplay annotations
are on context.annotations. Every annotation lookup in this function
must use context.annotations, not stage_one.annotations.


### Span mapping

Build token texts once from stage_one.tokens:

    token_texts = [t.text.lower() for t in stage_one.tokens]
    n_tokens = len(token_texts)

Determine the wordplay window from definition_candidates:

    if stage_one.definition_candidates:
        first_candidate = stage_one.definition_candidates[0]
        wp_start = first_candidate.wordplay_span.start
        wp_end = first_candidate.wordplay_span.end
    else:
        wp_start = 0
        wp_end = n_tokens

Track used annotation indices and used token index ranges so that
repeated words and phrases get distinct spans (occurrence-aware).

    used_annotation_indices = set()
    used_token_spans = set()

For each (word, token, value) in word_roles, find the clue token span
using this priority:

Priority 1 — annotation match

    Search context.annotations for the first unused annotation where:
      - annotation.text.lower() == word.lower()
      - annotation.token == token (the role token from word_roles)
      - the produced value is represented in annotation.values:
        check _clean_answer(value) in
        [_clean_answer(v) for v in annotation.values
         if isinstance(v, str)]
        OR annotation.values == (True,)
        (SpanAnnotation.values is a tuple. The single-element tuple (True,)
        means the DB confirmed existence of the word as that role, without
        a specific produced value. Do NOT compare to [True] — that is a
        list and will never equal a tuple.)
    Mark the matched annotation index as used.
    Return annotation.span as the token span.

Priority 2 — token sequence match within wordplay window

    Normalise the role word to lower-case. Split on spaces to get a
    list of word_parts (length L = len(word_parts)). Search for the
    first i in [wp_start, wp_end - L] where:
      - (i, i + L) is not already in used_token_spans
      - " ".join(token_texts[i:i+L]) == " ".join(word_parts)
    Span is (i, i + L).
    Add the span tuple (i, i + L) to used_token_spans before returning.

Priority 3 — ambiguous

    If neither priority matched, span = None, span_status = "ambiguous".
    Do not fabricate a span. Record the gap honestly.

Helper function signature (internal):

    def _map_word_to_span(word, token, value,
                          context_annotations, token_texts,
                          wp_start, wp_end,
                          used_annotation_indices, used_token_spans):
        -> (span_tuple_or_None, span_status_string)


### Translate word_roles

For each (word, token, value) in solve_result.result.word_roles:

Call the span mapper to get (span, span_status).

Classify:

Token is SYN_F, ABR_F, or any other value-bearing source token that
is NOT an indicator and NOT LNK:

    Value-bearing source tokens include at minimum:
      SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, DEL_F, POS_F
    The safest test: token not in INDICATOR_TOKENS and
    not token.startswith("POS_I") and token != LNK
    and value is not None.

    Add to source_candidates:
      {
        "text": word,
        "span": span,
        "span_status": span_status,
        "token": token,
        "value": _clean_answer(value),
        "source": "word_roles",
        "relation_to_answer": (
          "contained_in_answer"
          if _clean_answer(value) in answer_clean
          else "unknown"
        ),
      }

Token is LNK:

    Do not add to source_candidates or operation_candidates.
    Record the span in a separate link_spans list so coverage
    tracking can account for these words.

Token is ANA_I, REV_I, CON_I, DEL_I, HOM_I, or startswith("POS_I"):

    Add to operation_candidates:
      {
        "text": word,
        "span": span,
        "span_status": span_status,
        "token": token,
        "source": "word_roles",
        "role": "operation",
      }

Any other token: skip silently.

Also collect indicator_spans and all source spans for coverage tracking.


### Definition candidates

    definitions = []
    if getattr(solve_result, "definition", None):
        definitions = [{
            "text": solve_result.definition,
            "span": None,
            "span_status": "from_solve_result",
            "wordplay_span": None,
            "wordplay_text": None,
            "boundary_status": "legacy_solver",
            "objections": [],
        }]
    else:
        definitions = _definition_candidates(stage_one)

If the legacy solver set sr.definition, use it. If not, fall back to
grammar-based extraction so this field is never empty without cause.

Note on Stage Three definition check: Stage Three's
_accepted_definition_candidate only accepts boundary_status values of
"complete_edge_phrase", "edge_db_hit_no_larger_pos_phrase", or
"non_edge_db_hit". The boundary_status "legacy_solver" used here is not
in that set, so Stage Three's definition_evidence check will always be
REVIEW for definitions supplied by the legacy solver. This is expected
and correct per the plan — Phase 1 does not aim for Stage Three PASS
on definitions from the legacy path. Do not add "legacy_solver" to
_accepted_definition_candidate to work around this; that would inflate
proof status dishonestly.


### Grammar phrases

    grammar = _grammar_phrases(stage_one)

Unchanged from the grammar-only path.


### Unresolved words

Track coverage by token index spans, not by text.

Build a set of covered token indices:

    covered_indices = set()

    # Cover source spans
    for item in source_candidates:
        if item["span"]:
            for i in range(item["span"][0], item["span"][1]):
                covered_indices.add(i)

    # Cover operation (indicator) spans
    for item in operation_candidates:
        if item["span"]:
            for i in range(item["span"][0], item["span"][1]):
                covered_indices.add(i)

    # Cover link word spans
    for span in link_spans:
        if span:
            for i in range(span[0], span[1]):
                covered_indices.add(i)

    # Cover definition span if known
    if definitions and definitions[0].get("span"):
        def_span = definitions[0]["span"]
        for i in range(def_span[0], def_span[1]):
            covered_indices.add(i)
    elif stage_one.definition_candidates:
        def_span = stage_one.definition_candidates[0].definition_span
        for i in range(def_span.start, def_span.end):
            covered_indices.add(i)

Then find uncovered tokens:

    unresolved = []
    for i, tok in enumerate(stage_one.tokens):
        if i not in covered_indices:
            unresolved.append({
                "text": tok.text,
                "span": (i, i + 1),
                "reason": "not covered by any mapped span",
            })

Spans are token index tuples throughout. Do not use character offsets.


### Assembly — Stage Three compatible shape

_best_answer_fit_assembly in stage_three_proof.py selects the first
assembly where status == "answer_fit" AND output == answer AND parts
is non-empty. _assembly_check then joins part["value"] from each part
and checks it equals answer. The assembly MUST have this shape or
Stage Three will never produce PASS.

Build parts from all source_candidates that have a non-empty value:

    parts = [
        {
            "text": item["text"],
            "value": item["value"],
            "span": item["span"],
        }
        for item in source_candidates
        if item.get("value")
    ]

Infer assembly kind from op_tokens:

    op_tokens = {tok for (_, tok, _) in solve_result.result.word_roles
                 if tok != LNK
                 and (tok in INDICATOR_TOKENS or tok.startswith("POS_I"))}

    Note: parentheses around the or-clause are mandatory. Python's
    operator precedence binds "and" before "or", so without parentheses
    the expression would be (tok != LNK and tok in INDICATOR_TOKENS)
    or tok.startswith("POS_I"), which is not the intended logic.
    if ANA_I in op_tokens:
        assembly_kind = "anagram"
    elif CON_I in op_tokens:
        assembly_kind = "container"
    elif REV_I in op_tokens:
        assembly_kind = "reversal"
    elif DEL_I in op_tokens:
        assembly_kind = "deletion"
    elif HOM_I in op_tokens:
        assembly_kind = "homophone"
    elif any(t.startswith("POS_I") for t in op_tokens):
        assembly_kind = "positional"
    elif not op_tokens:
        assembly_kind = "charade"
    else:
        assembly_kind = "compound"

Verify the assembly: join all part values and check against answer:

    joined = "".join(_clean_answer(p["value"]) for p in parts)
    assembly_status = "answer_fit" if joined == answer_clean else "evidence_only"

Build the assembly dict:

    assembly_dict = {
        "kind": assembly_kind,
        "status": assembly_status,
        "output": answer_clean,
        "parts": parts,
    }
    assemblies = (assembly_dict,)

If assembly_status is "evidence_only", Stage Three will produce REVIEW
(not PASS). That is correct and honest — we do not inflate confidence.


### Status

    status = "answer_fit" if solve_result.high_confidence else "evidence_only"


### Return StageTwoCaseFile

    return StageTwoCaseFile(
        clue_text=clue_text,
        answer=answer_clean,
        stage_one_context=stage_one,
        annotated_context=context,
        definition_candidates=tuple(definitions),
        grammar_phrases=tuple(grammar),
        source_candidates=tuple(source_candidates),
        operation_candidates=tuple(operation_candidates),
        working_pairs=(),
        assemblies=assemblies,
        enrichment_candidates=(),
        unresolved_words=tuple(unresolved),
        status=status,
    )

working_pairs is left empty. The legacy solver's word_roles are the
authoritative account of what happened. enrichment_candidates is empty
for now; Phase 4 will add grammar-span enrichments for this path.

---

## Change 2 — Branch in _attach_gt2_evidence in solver.py

Find this block (currently lines 107-118):

    if stage_one_context is not None:
        sr.stage_one_context = stage_one_context
        try:
            from .stage_two_casefile import build_stage_two_casefile
            from .stage_three_proof import build_stage_three_proof
            stage_two = build_stage_two_casefile(
                clue_text, answer_clean, db,
                stage_one_context=stage_one_context)
            sr.stage_two_casefile = stage_two
            sr.stage_three_proof = build_stage_three_proof(stage_two)
        except Exception:
            pass

Replace it with:

    if stage_one_context is not None:
        sr.stage_one_context = stage_one_context
        try:
            from .stage_two_casefile import (
                build_stage_two_casefile,
                build_stage_two_from_solve_result,
            )
            from .stage_three_proof import build_stage_three_proof
            if sr.high_confidence and sr.result is not None:
                stage_two = build_stage_two_from_solve_result(
                    clue_text, answer_clean, db, sr,
                    stage_one_context=stage_one_context)
            else:
                stage_two = build_stage_two_casefile(
                    clue_text, answer_clean, db,
                    stage_one_context=stage_one_context)
            sr.stage_two_casefile = stage_two
            sr.stage_three_proof = build_stage_three_proof(stage_two)
        except Exception:
            pass

The surrounding code is unchanged.

---

## Constraints

- Do not change build_stage_two_casefile.
- Do not change build_stage_three_proof.
- Do not change the solve_clue function.
- Do not import from sonnet_pipeline or sig_adapter in either file.
- If word_roles is non-empty, the function must produce a StageTwoCaseFile
  from those roles. The grammar-only fallback is only for the empty case.

---

## After Making the Changes

Paste the following for audit before running anything:

1. The complete build_stage_two_from_solve_result function from its def
   line to its final return statement.

2. The modified block in _attach_gt2_evidence from the line
   "if stage_one_context is not None:" to the closing "pass" of that
   try/except block.

Do not run the pipeline or any tests until Claude has audited those two
sections and confirmed the changes are correct.
