# Phase 2 Manual Parse Editor UI Collapse Proposal for Claude

## Problem

The manual parse editor has become visually overwhelming. When
`Admin: manual parse editor` is opened, all parser forms are visible at once:

- Charade parse editor
- Reversal / anagram parse editor
- Container parse editor
- Saved structured parse
- Legacy graph editor

This creates too much vertical material and makes it hard to work accurately.

There is a second usability problem: once the user is working inside one of the
manual parser forms, the original clue text is often no longer visible. The user
has to scroll back up or remember the clue while filling exact clue words. That
increases transcription errors.

## Goal

Make the manual parse editor usable without changing parser behaviour, storage,
validation, scoring, or database logic.

This is a UI-only slice.

## Required Behaviour

When `Admin: manual parse editor` is opened:

1. Each parser form must be collapsed by default.
2. The saved structured parse panel may remain visible, because it is the
   current state summary.
3. The legacy graph editor remains collapsed, as it is now.
4. Each parser section must repeat the clue text near the top of the section.
5. The repeated clue text must include the answer/enumeration if that is already
   available in the template context.

## Recommended UI Shape

Inside `web/templates/clue.html`, keep the existing outer:

```html
<details class="mt-3">
    <summary>Admin: manual parse editor</summary>
    ...
</details>
```

Then wrap each parser form in its own inner `<details>`:

```html
<details class="rounded border border-sky-200 bg-white/80">
    <summary class="...">Charade parse editor</summary>
    <div class="...">
        <p class="...">{{ clue.clue_text }}</p>
        ...
        <form hx-post="/admin/structured-parse/{{ clue.id }}/source">
            ...
        </form>
    </div>
</details>
```

Do the same for:

- Charade parse editor
- Reversal / anagram parse editor
- Container parse editor

The inner parser `<details>` elements should not have the `open` attribute.
They should all be collapsed when the manual editor first opens.

## Clue Repeat

Each parser panel should include a compact clue reference at the top.

Suggested text:

```html
<div class="rounded border border-slate-200 bg-slate-50 px-3 py-2">
    <p class="text-xs font-semibold uppercase text-slate-500">Clue</p>
    <p class="text-sm font-semibold text-slate-900">
        {{ clue.clue_text }}
        {% if clue.enumeration %}
        <span class="font-normal text-slate-500">({{ clue.enumeration }})</span>
        {% endif %}
    </p>
    {% if clue.answer %}
    <p class="mt-1 text-xs text-slate-500">
        Answer: <span class="font-mono font-semibold text-slate-800">{{ clue.answer }}</span>
    </p>
    {% endif %}
</div>
```

This should be plain display text, not clickable word helper markup. The parser
needs a stable reference, not another interactive surface.

## Important Constraints

Do not change:

- Any route.
- Any `hx-post` target.
- Any form field name.
- Any validation logic.
- Any scoring logic.
- Any manual structured parse storage.
- Any DB/enrichment behaviour.
- Any Stage Three, WFW, solver, proof storage, or display adapter code.

This proposal is strictly about reducing visual overload and keeping the clue
visible while editing.

## Files

Expected file touched:

- `web/templates/clue.html`

No Python files should be touched for this UI slice.

## Verification

After implementing:

1. Start/restart Flask.
2. Open a clue page as admin.
3. Expand `Admin: manual parse editor`.
4. Confirm:
   - Charade parser is collapsed.
   - Reversal / anagram parser is collapsed.
   - Container parser is collapsed.
   - Saved structured parse summary is still visible if one exists.
   - Legacy graph editor is still collapsed.
5. Open each parser and confirm:
   - The clue text is visible inside the parser.
   - The answer is visible if available.
   - Existing fields and submit buttons still work.
6. Run the existing template/load checks used in prior slices.

## Anti-patterns To Avoid

- Do not hide the saved structured parse summary behind another collapsed
  section unless the user explicitly asks for that later.
- Do not add JavaScript for accordion behaviour in this slice.
- Do not make only one parser open at a time.
- Do not move forms to separate pages.
- Do not rename routes or field names.
- Do not change the manual parser data model.

