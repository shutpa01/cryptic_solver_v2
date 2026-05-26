# Phase 2 Anagram Stage Three Fix — Codex Instruction

## Task

Fix two functions in signature_solver/stage_three_proof.py so that
anagram assemblies produced by Stage Two are correctly verified and
linked by Stage Three.

Only this one file changes. No other file may be touched.

The stage_two_casefile.py changes (sorted-letter assembly fit test and
_normalize_words punctuation fix) and the _blocks OP_BLOCK change are
already implemented. This instruction covers only what remains.


---

## What to verify before writing

Read signature_solver/stage_three_proof.py in full before making any
change. Confirm:

1. _assembly_check (around line 148): currently reads
       values = "".join(part.get("value", "") for part in assembly.get("parts", ()))
       if _clean_answer(values) == answer:
   There is no sorted() comparison and no assembly["kind"] check.

2. _atomic_links (around line 894): currently iterates parts in source
   order, assigns letters to consecutive answer positions, and returns
   () if total letter count != len(answer). There is no pool-based
   algorithm and no assembly["kind"] check.

3. _blocks (around line 811): already emits OP_BLOCKs from
   operation_candidates (lines 860-874) when not already covered by
   working_pairs. Do not touch this function.


---

## Change 1 — _assembly_check

### Current code

    def _assembly_check(answer, assembly, conditional_assemblies):
        if assembly:
            values = "".join(part.get("value", "") for part in assembly.get("parts", ()))
            if _clean_answer(values) == answer:
                return StageThreeCheck(
                    "answer_assembly",
                    PASS,
                    "%s = %s" % (
                        " + ".join(part.get("value", "") for part in assembly["parts"]),
                        answer,
                    ),
                    assembly,
                )

### Replace with

    def _assembly_check(answer, assembly, conditional_assemblies):
        if assembly:
            values = "".join(part.get("value", "") for part in assembly.get("parts", ()))
            clean_values = _clean_answer(values)
            if assembly.get("kind") == "anagram":
                fits = sorted(clean_values) == sorted(answer)
            else:
                fits = clean_values == answer
            if fits:
                if assembly.get("kind") == "anagram":
                    detail = "%s (anagram) = %s" % (clean_values, answer)
                else:
                    detail = "%s = %s" % (
                        " + ".join(
                            part.get("value", "")
                            for part in assembly["parts"]),
                        answer,
                    )
                return StageThreeCheck(
                    "answer_assembly",
                    PASS,
                    detail,
                    assembly,
                )

The rest of _assembly_check (the conditional_assemblies branch and the
final REVIEW return) is unchanged.

Key points:
- assembly.get("kind") == "anagram" is the only guard. All other kinds
  keep the linear equality check.
- The PASS detail message distinguishes anagram assemblies clearly.
- No change to any other branch.


---

## Change 2 — _atomic_links

### Current code

    def _atomic_links(answer, assembly):
        if not assembly:
            return ()
        links = []
        answer_index = 0
        for piece_index, part in enumerate(assembly.get("parts", ())):
            value = _clean_answer(part.get("value", ""))
            for source_index, letter in enumerate(value):
                links.append({
                    "answer_index": answer_index,
                    "letter": letter,
                    "source_text": part.get("text", ""),
                    "source_span": part.get("span"),
                    "source_role": "piece_%d" % piece_index,
                    "source_value": value,
                    "source_value_index": source_index,
                })
                answer_index += 1
        if answer_index != len(answer):
            return ()
        return tuple(links)

### Replace with

    def _atomic_links(answer, assembly):
        if not assembly:
            return ()
        if assembly.get("kind") != "anagram":
            links = []
            answer_index = 0
            for piece_index, part in enumerate(assembly.get("parts", ())):
                value = _clean_answer(part.get("value", ""))
                for source_index, letter in enumerate(value):
                    links.append({
                        "answer_index": answer_index,
                        "letter": letter,
                        "source_text": part.get("text", ""),
                        "source_span": part.get("span"),
                        "source_role": "piece_%d" % piece_index,
                        "source_value": value,
                        "source_value_index": source_index,
                    })
                    answer_index += 1
            if answer_index != len(answer):
                return ()
            return tuple(links)

        # Anagram: match answer letters to source letters by letter value,
        # consuming from a pool so each source letter is used at most once.
        pool = []
        for piece_index, part in enumerate(assembly.get("parts", ())):
            value = _clean_answer(part.get("value", ""))
            for source_index, letter in enumerate(value):
                pool.append({
                    "letter": letter,
                    "source_text": part.get("text", ""),
                    "source_span": part.get("span"),
                    "source_role": "piece_%d" % piece_index,
                    "source_value": value,
                    "source_value_index": source_index,
                    "used": False,
                })
        links = []
        for answer_index, answer_letter in enumerate(answer):
            matched = None
            for entry in pool:
                if not entry["used"] and entry["letter"] == answer_letter:
                    entry["used"] = True
                    matched = entry
                    break
            if matched is None:
                return ()
            links.append({
                "answer_index": answer_index,
                "letter": answer_letter,
                "source_text": matched["source_text"],
                "source_span": matched["source_span"],
                "source_role": matched["source_role"],
                "source_value": matched["source_value"],
                "source_value_index": matched["source_value_index"],
            })
        return tuple(links)

Key points:
- The non-anagram path is the original code unchanged.
- The anagram path only fires when assembly.get("kind") == "anagram".
- Pool entries carry a "used" flag to handle repeated letters correctly.
- If any answer letter cannot be matched, return () (same as original).
- The "used" flag is internal to this call and is not returned.


---

## What not to do

Do not modify _blocks.
Do not modify _assembly_check branches other than the fits check.
Do not modify any function outside _assembly_check and _atomic_links.
Do not modify stage_two_casefile.py.
Do not modify any other file.


---

## After writing

Paste _assembly_check and _atomic_links in full so Claude can audit
before anything is run.
