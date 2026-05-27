# Manual Evidence Graph Specification for Claude

Date: 2026-05-26
Status: specification and design request, not an implementation instruction

## 1. Objective

Build a persistent manual evidence system for cryptic clue parsing.

The success criterion is not that the automatic solver improves. The success
criterion is that a human can manually construct and preserve a correct parse
for any clue, and the final proof/display respects it.

The purpose is to let an admin manually record exact cryptic evidence when the
automatic solver fails or only partially succeeds. The system must then display,
preserve, and merge that evidence into the final clue proof.

This is not merely a manual word-role picker. It must let the admin model a
parse chain such as:

    CAT -> TOM
    BACKS reverses TOM -> MOT
    THE SPANISH -> EL
    MOT + EL = MOTEL
    hotel defines MOTEL

The key user need is: "I can always make the clue parseable by recording the
evidence myself, and that evidence will not be lost on rerun."

## 2. Problem With Current System

The current system has several pieces that look related but do not solve the
real problem:

1. `clue_word_roles` records per-word roles and optional letters.
2. Stage Three can display some blocks and atomic links.
3. Rerun/reverify can regenerate proof attempts.
4. Manual role rows may survive auto role writes.

However, this is insufficient because:

1. A role row does not express a full evidence relationship.
2. It cannot say which answer tiles the letters occupy.
3. It cannot say which operation applies to which source.
4. It cannot represent transformed output, such as TOM reversed to MOT.
5. It cannot represent containers, deletions, hidden selections, homophones, or
   anagrams as persistent user-authored evidence.
6. Rerun can rebuild failed automatic evidence, and the final display may still
   show that failed evidence unless manual evidence is explicitly merged with
   priority.

Previous work that tried to turn manual roles into a whole-answer assembly was
too narrow. Manual evidence must be useful even when it covers only part of a
clue. It must not require the whole clue to be manually completed before visible
evidence appears.

## 3. Core Principle

There are two evidence classes:

### 3.1 Automatic Evidence

Automatic evidence is produced by the solver.

It is useful but disposable:

- it can be wrong
- it can be incomplete
- it can be replaced by rerun
- it must not overwrite protected manual evidence

### 3.2 Manual Evidence

Manual evidence is produced by the admin.

It is protected and authoritative:

- it is stored separately from automatic proof attempts
- it survives rerun/reverify
- it is loaded after automatic solving
- it wins where it overlaps with automatic evidence
- it is merged into the final proof and display

The final displayed proof must be:

    protected manual evidence + latest automatic evidence for uncovered gaps

not:

    latest automatic evidence only

## 4. Evidence Graph Model

Manual evidence should be modeled as a graph. The graph has nodes and edges.

This is necessary because cryptic parsing is not always a direct clue-span to
answer-span mapping. Often a clue source is transformed by an operation before it
lands in the answer.

### 4.1 Node Types

The system should support at least these node types.

#### Source Node

A source node says a clue span produces raw letters.

Example:

    CAT -> TOM

Fields:

- node id
- clue id
- clue span, as word indices and display text
- role, such as synonym, abbreviation, literal, fodder, foreign, initial-letter
- raw letters, such as TOM
- source = manual
- group/colour id
- timestamps

#### Operator Node

An operator node says a clue span performs an operation.

Examples:

    BACKS = reversal
    shortly = deletion-final
    around = container
    ground = anagram
    reportedly = homophone
    hidden in = hidden/selection

Fields:

- node id
- clue id
- clue span and text
- operation type
- optional operation subtype, such as delete-first, delete-last, odd letters
- source = manual
- group/colour id or operation colour id
- timestamps

#### Transform Node

A transform node records the result of applying an operator to one or more source
nodes.

Examples:

    reverse(TOM) -> MOT
    delete_last(CASE) -> CAS
    homophone(DOCK) -> DOC
    hidden(local society) -> ALSO
    anagram(DATE) -> ATED

Fields:

- node id
- clue id
- operation node id
- input source node ids
- input letters
- output letters
- source = manual
- timestamps

#### Assembly Node

An assembly node records how final output pieces occupy the answer.

Examples:

    MOT -> answer positions [0, 1, 2]
    EL -> answer positions [3, 4]

Fields:

- node id
- clue id
- source or transform node id
- output letters
- answer positions, explicitly ordered
- group/colour id
- source = manual
- timestamps

#### Definition Node

A definition node records which clue span defines the answer.

Example:

    hotel defines MOTEL

Fields:

- node id
- clue id
- clue span and text
- answer
- source = manual
- group/colour id or definition colour id
- timestamps

#### Structural Node

A structural node records clue words that are intentionally not evidence-bearing
but should not remain unresolved.

Examples:

    with = link
    for = link
    the = surface

Fields:

- node id
- clue id
- clue span and text
- structural role, such as link, separator, surface
- source = manual
- timestamps

### 4.2 Edge Types

The graph also needs relationships.

#### Operator Applies To Source

Example:

    BACKS applies_to CAT -> TOM

Fields:

- edge id
- clue id
- from operator node id
- to source node id or source group id

#### Transform Uses Source

Example:

    reverse(TOM) uses CAT -> TOM

Fields:

- edge id
- clue id
- transform node id
- source node id

#### Assembly Uses Source Or Transform

Example:

    answer positions [0,1,2] use reverse(TOM)->MOT

Fields:

- edge id
- clue id
- assembly node id
- source or transform node id

## 5. Required Mechanisms

The manual evidence graph must be able to represent at least the following.

### 5.1 Direct Source / Charade

Example:

    THE SPANISH -> EL
    work -> IC
    attache's -> DIPLOMAT

The user can save the clue span, produced letters, and answer positions.

### 5.2 Reversal

Example:

    CAT -> TOM
    BACKS reverses TOM -> MOT
    MOT maps to answer positions [0,1,2]

The user must be able to attach the reversal operator to the source and save the
transformed output.

### 5.3 Deletion

Example from sample:

    Instance -> CASE
    shortly deletes final E
    CAS maps to answer positions [0,1,2]

Deletion needs at least:

- delete first
- delete last
- delete indicated letters
- delete contained substring

The first implementation can support only first/last deletion if that is the
safe starting point, but the data model must not prevent fuller deletion later.

### 5.4 Container / Insertion

Example from sample:

    Aim -> END
    expensive -> DEAR
    energy -> E
    limit applies container
    DEAR + E goes inside END
    output ENDEARED maps to all answer positions

Another example:

    Footballer -> WINGER
    hotel -> H
    around inserts H into WINGER
    output WHINGER

The graph must support:

- frame source
- content source or sources
- operator
- output letters
- answer placement

### 5.5 Anagram

Example from sample:

    date -> DATE
    affected anagrams DATE -> ATED
    That man -> HE stays fixed
    HE + ATED = HEATED

Example:

    parking -> P
    here -> HERE
    at -> AT
    old -> OLD
    ground anagrams all fodder -> PETROLHEAD

The graph must support:

- one or more fodder source nodes
- an anagram operator
- transformed output letters
- answer positions for the transformed output
- fixed pieces that are not part of the anagram pool

### 5.6 Hidden / Selection

Examples from sample:

    local society contains ALSO
    on clay at odd points -> OCA

The graph must support:

- source span with raw text or letters
- operator, such as hidden, odd, even, first letters, last letters
- selected output letters
- answer positions

### 5.7 Homophone

Example from sample:

    place to board ship -> DOCK
    we're told / reportedly indicates homophone
    DOCK sounds as DOC
    DOC maps to answer

The graph must support:

- source node
- homophone operator node
- transform node with output spelling
- answer positions

### 5.8 Definition

The user must be able to mark a clue span as definition even if the definition
database does not yet contain that definition-answer pair.

The definition should be shown as manual evidence and should suppress wrong
automatic definition blocks where they overlap or conflict.

### 5.9 Structural / Surface Words

The user must be able to mark words as link/surface/ignore so that they stop
appearing as unresolved when the parse is otherwise accounted for.

Examples:

    with
    for
    to
    is
    of

This matters because many current review statuses are caused by word-purpose
coverage rather than missing answer assembly.

## 6. Persistence Requirements

Manual evidence must not live only inside `wfw_proof_attempts.proof_json`.

It needs a durable storage layer that is outside the rerun blast radius.

The exact schema is for Claude to propose, but it must satisfy:

1. Store multiple manual evidence nodes per clue.
2. Store edges between nodes.
3. Store answer tile positions.
4. Store clue word spans.
5. Store raw and transformed letters.
6. Store group/colour ids.
7. Store operation type and subtype.
8. Store source = manual.
9. Store created_at and updated_at.
10. Allow edit/delete of manual evidence.
11. Allow rerun/reverify to read the manual evidence back.

Potential schema shape:

    manual_evidence_nodes
    manual_evidence_edges

or one JSON graph table:

    manual_evidence_graphs

Claude should evaluate which is safer for this codebase.

Important: if JSON graph storage is chosen, it must still be queryable enough for
the clue page to load and display evidence efficiently.

## 7. Merge Requirements

Every final proof/display build must merge evidence in this order:

1. Load latest automatic proof/evidence.
2. Load protected manual evidence graph.
3. Convert manual graph into proof blocks, atomic links, transformations, and
   word purposes.
4. Suppress automatic blocks that conflict with covered manual spans or answer
   positions.
5. Keep automatic evidence only for uncovered gaps.
6. Emit a final merged proof.

Manual evidence wins over automatic evidence in conflicts.

Conflict examples:

- Manual says `CAT -> TOM`, auto says `CAT -> TAB`: manual wins.
- Manual says answer positions `[0,1,2]`, auto maps another source to those
  positions: manual wins.
- Manual marks `hotel` as definition, auto marks a different phrase as
  definition: manual definition should be authoritative in the display.

Rerun must never delete manual evidence.

## 8. Display Requirements

The clue page must show manual evidence clearly.

Required display behaviours:

1. A clue-side block for each source, transform, operator, definition, and
   structural item where useful.
2. The answer tiles selected by an assembly node must share the same colour as
   the source/transform that produced them.
3. Operators should visibly attach to the source they operate on.
4. For operation chains, the UI should show the chain:

       CAT -> TOM; BACKS reverses -> MOT; MOT placed at 0-2

5. The display should not require the automatic solver to accept the parse.
6. The display should not show manually covered clue words as unresolved.
7. Failed automatic evidence should not visually override manual evidence.

## 9. UI Workflow Requirements

Claude should propose a UI that is practical in the existing admin clue page.

Minimum viable workflow:

1. Select clue word or phrase.
2. Choose evidence type:
   - source
   - operator
   - definition
   - structural
3. For source:
   - enter produced raw letters
   - choose role
   - optionally select answer tiles directly
4. For operator:
   - choose operation type
   - attach it to one or more source nodes
   - enter or confirm transformed output
   - select answer tiles for the transformed output
5. For definition:
   - mark clue span as definition
6. Save evidence.
7. Edit/delete existing evidence.

Important usability point:

The admin must be able to build evidence incrementally. A single saved source
mapping should immediately show useful colour, even if the rest of the clue is
not parsed yet.

## 10. Sample-Driven Requirements

The following real failed/review clue patterns were observed in the live DB.
These should guide the design.

### 10.1 ENDEARED

Clue:

    Aim to limit expensive energy is made more attractive

Answer:

    ENDEARED

Needed manual graph:

    Aim -> END
    expensive -> DEAR
    energy -> E
    limit = container operator
    limit applies to frame END and content DEAR+E
    output ENDEARED maps to answer positions 0-7
    made more attractive = definition
    to/is may be structural

### 10.2 HEATED

Clue:

    That man before date affected with passion?

Answer:

    HEATED

Needed manual graph:

    That man -> HE
    date -> DATE
    affected = anagram operator
    affected applies only to DATE
    DATE -> ATED
    HE maps to answer positions 0-1
    ATED maps to answer positions 2-5
    with passion? = definition
    before is structural/order indicator

This case proves the need for resource ownership. `HE` must not be swallowed
into the anagram pool once it is a fixed source.

### 10.3 CASSOCKS

Clue:

    Instance shortly with footwear and garments for clerics

Answer:

    CASSOCKS

Needed manual graph:

    Instance -> CASE
    shortly = delete final letter
    CASE -> CAS
    footwear -> SOCKS
    CAS maps to answer positions 0-2
    SOCKS maps to answer positions 3-7
    garments for clerics = definition
    with/and may be structural

### 10.4 CANVAS

Clue:

    Reportedly, examine material for sails

Answer:

    CANVAS

Needed manual graph:

    examine -> CANVASS
    Reportedly = homophone operator
    CANVASS sounds as CANVAS
    CANVAS maps to answer positions 0-5
    material for sails = definition

### 10.5 OCA

Clue:

    South American plant on clay at odd points

Answer:

    OCA

Needed manual graph:

    on clay at = source phrase
    odd points = selection operator
    selected letters -> OCA
    OCA maps to answer positions 0-2
    South American plant = definition

### 10.6 PETROLHEAD

Clue:

    Driving fan parking here at old ground

Answer:

    PETROLHEAD

Needed manual graph:

    parking -> P
    here -> HERE
    at -> AT
    old -> OLD
    ground = anagram operator
    fodder P+HERE+AT+OLD anagrams to PETROLHEAD
    output maps to answer positions 0-9
    Driving fan = definition

### 10.7 WHINGER

Clue:

    Footballer around hotel is one often complaining

Answer:

    WHINGER

Likely needed manual graph:

    Footballer -> WINGER
    hotel -> H
    around = insertion/container operator
    H inserted into WINGER -> WHINGER
    one often complaining = definition

### 10.8 Double Definition / Definition-Only

Example:

    House in Kent area for trainer, say
    answer SHOE

Needed manual graph may be:

    trainer, say -> SHOE
    House in Kent area = definition

Some clues may be cryptic definitions or double definitions where a complete
letter-by-letter assembly is not the right goal. The system must allow this
without forcing bogus source pieces.

## 11. Proof Output Requirements

Manual evidence graph must convert into Stage Three-style proof output:

### Blocks

Manual graph should emit:

- `SOURCE_BLOCK` for source nodes
- `OP_BLOCK` for operator nodes
- `TRANSFORM_BLOCK` or equivalent for transformed outputs
- `DEF_BLOCK` for definition nodes
- `STRUCTURAL_BLOCK` or purpose coverage for structural nodes

If the current display cannot render `TRANSFORM_BLOCK`, Claude should propose
either:

1. adding a new block kind, or
2. representing transforms in existing block fields without losing meaning.

### Atomic Links

Manual assembly placements must produce atomic links from source/transform to
answer positions.

For direct source:

    CAT -> TOM -> answer [0,1,2]

For transformed source:

    CAT -> TOM; reverse -> MOT -> answer [0,1,2]

Atomic links should point to the final output that lands in the answer, while
the display should still preserve the source-to-transform chain.

### Word Purposes

Manual evidence should mark covered clue words as accounted for.

The word-purpose layer should distinguish:

- manual source
- manual operation
- manual definition
- manual structural/surface

## 12. Acceptance Tests

No implementation should be considered complete without tests equivalent to
these.

### Test A: Direct Source Placement

Given a clue containing `CAT` and answer `TOM`:

1. Save manual evidence `CAT -> TOM`.
2. Select answer positions `[0,1,2]`.
3. Rebuild proof.
4. Confirm:
   - manual evidence persists in DB
   - clue block `CAT -> TOM` appears
   - answer tiles 0-2 are linked to that block
   - `CAT` is not unresolved
   - automatic failure does not remove it

### Test B: Reversal Chain

Given made-up clue:

    CAT BACKS THE SPANISH hotel

Answer:

    MOTEL

Save:

    CAT -> TOM
    BACKS = reversal, applies to CAT
    TOM -> MOT
    THE SPANISH -> EL
    MOT maps to [0,1,2]
    EL maps to [3,4]
    hotel = definition

Confirm:

- clue blocks show source and operator
- answer tiles 0-2 share colour with the MOT chain
- answer tiles 3-4 share colour with THE SPANISH -> EL
- final proof contains source, operation, transform, assembly, and definition
- rerun preserves all manual evidence

### Test C: Container

Use ENDEARED:

    Aim -> END
    expensive -> DEAR
    energy -> E
    limit container
    output ENDEARED
    definition made more attractive

Confirm final manual evidence can coexist with or override failed automatic
evidence.

### Test D: Deletion

Use CASSOCKS:

    Instance -> CASE
    shortly deletes final letter -> CAS
    footwear -> SOCKS

Confirm deletion chain and answer placement survive rerun.

### Test E: Anagram With Fixed Piece

Use HEATED:

    That man -> HE
    date -> DATE
    affected anagrams DATE -> ATED

Confirm HE is fixed and not consumed by the anagram operation.

### Test F: Hidden / Selection

Use OCA or ALSO:

    source phrase contains/selects answer letters

Confirm selected letters map to answer positions and the hidden/selection
operator is represented.

### Test G: Failed Auto Merge Protection

For any clue with saved manual evidence:

1. Force or simulate an automatic proof that fails or proposes conflicting
   evidence.
2. Rerun/reverify.
3. Confirm manual evidence remains in storage.
4. Confirm final proof/display still prefers manual evidence.

## 13. Non-Goals

This design request is not asking Claude to:

1. Rewrite the automatic solver.
2. Make the parser automatically understand all mechanisms.
3. Replace `clue_word_roles` immediately if it can be reused safely.
4. Build a polished UI in one pass.
5. Delete existing proof systems.

The goal is to add a durable manual evidence layer that can be merged into the
existing proof/display pipeline.

## 14. Questions For Claude To Answer

Please review this specification and propose an implementation plan.

Do not write code yet. The requested output is a careful design review and
implementation plan that can be reviewed before coding starts.

The response should answer:

1. What storage schema should be used?
2. Should the graph be stored as normalized node/edge rows or as JSON per clue?
3. How should manual evidence be converted into Stage Three proof blocks?
4. How should manual evidence merge with automatic evidence?
5. Where in the current code should the merge happen?
6. How should answer tile colouring be represented and rendered?
7. What is the smallest safe first implementation that still proves the whole
   architecture?
8. What should be explicitly deferred?
9. What migration/backfill is needed, if any?
10. What tests should be added before implementation is trusted?

## 15. Recommended First Implementation Slice

Claude should critique this, but a plausible first slice is:

1. Persistent storage for manual evidence graph.
2. UI/API to create direct source placements:

       clue span -> letters -> answer positions

3. UI/API to create definition and structural nodes.
4. Merge direct source placements into final proof blocks and atomic links.
5. Prove rerun preserves manual evidence.

Only after that first slice works should operation chains be added:

1. reversal
2. deletion
3. anagram
4. container
5. hidden/selection
6. homophone

The first slice must still be architecturally compatible with operation chains.
It must not be a dead-end patch.
