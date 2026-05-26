# Block Type Inspection

Date: 2026-05-17

This report samples graph nodes by block type for manual anatomy inspection.
The examples are weakly generated and should be treated as design evidence, not truth.

## Counts

- `DEF_BLOCK`: 865 nodes, 19 ambiguous
- `SCOPE_BLOCK`: 820 nodes, 707 ambiguous
- `OP_BLOCK`: 290 nodes
- `CONNECTOR_BLOCK`: 199 nodes
- `RELATION_BLOCK`: 103 nodes
- `POSITION_BLOCK`: 32 nodes
- `LOCATOR_BLOCK`: 22 nodes
- `DEF_MODIFIER_BLOCK`: 5 nodes

## DEF_BLOCK

Available nodes: `865`

Ambiguous definition gaps:

- `Supermarket among restaurants stricken in flood, maybe` (NATURALDISASTER)
  Node: `<definition gap>` / `ambiguous`
  Operation: `charade`
  Span: `None`
  Residue evidence: `none`
  Edges: `none`
- `It carries passengers in suburb, usually` (BUS)
  Node: `<definition gap>` / `ambiguous`
  Operation: `hidden`
  Span: `None`
  Residue evidence: `none`
  Edges: `none`
- `Some bloke choosing feature in a chamber?` (ECHO)
  Node: `<definition gap>` / `ambiguous`
  Operation: `hidden`
  Span: `None`
  Residue evidence: `none`
  Edges: `none`
- `Fifty plus acres, perhaps?` (LAND)
  Node: `<definition gap>` / `ambiguous`
  Operation: `charade`
  Span: `None`
  Residue evidence: `none`
  Edges: `none`
- `Son and family, following attack, hide from nanny perhaps` (GOATSKIN)
  Node: `<definition gap>` / `ambiguous`
  Operation: `charade`
  Span: `None`
  Residue evidence: `none`
  Edges: `none`

## DEF_MODIFIER_BLOCK

Available nodes: `5`

- `Slug, say, with time in gaps door created` (GASTROPOD)
  Node: `say` / `inferred`
  Operation: `charade`
  Span: `[0, 1]`
  Residue evidence: `say`
  Split evidence: `split from residue run say with`
  Edges: `MODIFIES_DEFINITION -> def_0:Slug [definition_modifier_candidate]`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Node: `say` / `inferred`
  Operation: `anagram`
  Span: `[0, 1]`
  Residue evidence: `say`
  Split evidence: `split from residue run say broadcast of`
  Edges: `MODIFIES_DEFINITION -> def_0:Hammer [definition_modifier_candidate]`
- `Port, perhaps, now in electric fences` (WINE)
  Node: `perhaps` / `inferred`
  Operation: `hidden`
  Span: `[0, 1]`
  Residue evidence: `perhaps`
  Edges: `MODIFIES_DEFINITION -> def_0:Port [definition_modifier_candidate]`

## RELATION_BLOCK

Available nodes: `103`

- `Slug, say, with time in gaps door created` (GASTROPOD)
  Node: `with` / `inferred`
  Operation: `charade`
  Span: `[1, 2]`
  Residue evidence: `with`
  Split evidence: `split from residue run say with`
  Edges: `CONTAINS -> src_0:time [weakly_scoped_from_operation]`
- `Malign falsehood about bishop on left` (LIBEL)
  Node: `about` / `inferred`
  Operation: `container`
  Span: `[1, 2]`
  Residue evidence: `about`
  Edges: `CONTAINS -> src_0:falsehood [weakly_scoped_from_operation]; CONTAINS -> src_1:bishop [weakly_scoped_from_operation]`

## CONNECTOR_BLOCK

Available nodes: `199`

- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Node: `in` / `inferred`
  Operation: `anagram`
  Span: `[0, 1]`
  Residue evidence: `in`
  Edges: `SURFACE_CONNECTS -> src_0:a bolder violet [surface_only_until_grammar_check]`
- `Bad smell began to spread around marsh plant` (BOGBEAN)
  Node: `to` / `inferred`
  Operation: `charade`
  Span: `[3, 4]`
  Residue evidence: `to`
  Split evidence: `split from residue run to spread around`
  Edges: `SURFACE_CONNECTS -> answer:BOGBEAN [surface_only_until_grammar_check]`
- `With sun gone, sail by unstable African country` (LIBYA)
  Node: `With` / `inferred`
  Operation: `deletion+anagram`
  Span: `[0, 1]`
  Residue evidence: `with`
  Edges: `SURFACE_CONNECTS -> src_0:sun gone sail by unstable [surface_only_until_grammar_check]`
- `Smear limited publicity for a book` (BLUR)
  Node: `for a` / `inferred`
  Operation: `deletion`
  Span: `[2, 4]`
  Residue evidence: `a, for`
  Edges: `SURFACE_CONNECTS -> src_0:publicity [surface_only_until_grammar_check]; SURFACE_CONNECTS -> src_1:book [surface_only_until_grammar_check]`
- `Part of a ground revealed is attracting attention?` (STANDOUT)
  Node: `is` / `inferred`
  Operation: `charade`
  Span: `[5, 6]`
  Residue evidence: `is`
  Edges: `SURFACE_CONNECTS -> src_1:revealed [surface_only_until_grammar_check]`

## LOCATOR_BLOCK

Available nodes: `22`

- `Renown at intervals for a very long period` (EON)
  Node: `intervals` / `inferred`
  Operation: `anagram`
  Span: `[2, 3]`
  Residue evidence: `intervals`
  Split evidence: `split from residue run Renown at intervals`
  Edges: `LOCATES_WITHIN -> src_0:for [weakly_scoped_from_operation]`
- `Strong soldiers turning up in front of sculpture` (ROBUST)
  Node: `front` / `inferred`
  Operation: `charade`
  Span: `[4, 5]`
  Residue evidence: `front`
  Split evidence: `split from residue run turning up in front of`
  Edges: `LOCATES_WITHIN -> src_0:soldiers [weakly_scoped_from_operation]; LOCATES_WITHIN -> src_1:sculpture [weakly_scoped_from_operation]`
- `Close friend losing top papers` (HUMID)
  Node: `top` / `inferred`
  Operation: `deletion`
  Span: `[2, 3]`
  Residue evidence: `top`
  Split evidence: `split from residue run losing top`
  Edges: `LOCATES_WITHIN -> src_0:friend [weakly_scoped_from_operation]; LOCATES_WITHIN -> src_1:papers [weakly_scoped_from_operation]`
- `Part of church in steeple's painted towards the top` (APSE)
  Node: `the top` / `inferred`
  Operation: `hidden_reversed`
  Span: `[4, 6]`
  Residue evidence: `the, top`
  Edges: `LOCATES_WITHIN -> src_0:steeple's painted towards [weakly_scoped_from_operation]`
- `Fish in net's tail thrashing` (TROUT)
  Node: `tail` / `inferred`
  Operation: `charade`
  Span: `[2, 3]`
  Residue evidence: `tail`
  Edges: `LOCATES_WITHIN -> src_0:net's [weakly_scoped_from_operation]; LOCATES_WITHIN -> src_1:thrashing [weakly_scoped_from_operation]`

## POSITION_BLOCK

Available nodes: `32`

- `Outline amount of money on the increase` (SUMUP)
  Node: `on` / `inferred`
  Operation: `charade`
  Span: `[3, 4]`
  Residue evidence: `on`
  Split evidence: `split from residue run of money on the`
  Edges: `ORDERS -> src_0:amount [answer_aware_scope_needed]; ORDERS -> src_1:increase [answer_aware_scope_needed]`

## OP_BLOCK

Available nodes: `290`

- `Follow lad after repairing emergency barrier` (FLOODWALL)
  Node: `repairing` / `inferred`
  Operation: `anagram`
  Span: `[3, 4]`
  Residue evidence: `repairing`
  Split evidence: `split from residue run after repairing`
  Edges: `OPERATES_ON -> src_0:Follow lad [weakly_scoped_from_operation]`
- `Reg is excited to catch large salmon` (GRILSE)
  Node: `excited` / `inferred`
  Operation: `deletion+anagram`
  Span: `[2, 3]`
  Residue evidence: `excited`
  Split evidence: `split from residue run excited to catch`
  Edges: `OPERATES_ON -> src_0:is [weakly_scoped_from_operation]; OPERATES_ON -> src_1:large [weakly_scoped_from_operation]`
- `Vegetable kept by Anneka least` (KALE)
  Node: `kept by` / `inferred`
  Operation: `hidden`
  Span: `[0, 2]`
  Residue evidence: `by, kept`
  Edges: `OPERATES_ON -> src_0:Anneka least [weakly_scoped_from_operation]`
- `Problem in the US simply rejected` (ISSUE)
  Node: `in` / `inferred`
  Operation: `hidden_reversed`
  Span: `[0, 1]`
  Residue evidence: `in`
  Edges: `OPERATES_ON -> src_0:the US simply [weakly_scoped_from_operation]`
- `Try a bit of latte MP tasted` (ATTEMPT)
  Node: `a bit of` / `inferred`
  Operation: `hidden`
  Span: `[0, 3]`
  Residue evidence: `a, bit, of`
  Edges: `OPERATES_ON -> src_0:latte MP tasted [weakly_scoped_from_operation]`

## SCOPE_BLOCK

Available nodes: `820`

- `Follow lad after repairing emergency barrier` (FLOODWALL)
  Node: `after` / `ambiguous`
  Operation: `anagram`
  Span: `[2, 3]`
  Residue evidence: `after`
  Split evidence: `split from residue run after repairing`
  Edges: `UNRESOLVED -> answer:FLOODWALL [unresolved]`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Node: `created` / `ambiguous`
  Operation: `charade`
  Span: `[6, 7]`
  Residue evidence: `created`
  Edges: `UNRESOLVED -> src_1:gaps door [unresolved]`
- `Reg is excited to catch large salmon` (GRILSE)
  Node: `Reg` / `ambiguous`
  Operation: `deletion+anagram`
  Span: `[0, 1]`
  Residue evidence: `reg`
  Edges: `UNRESOLVED -> src_0:is [unresolved]`
- `Smear limited publicity for a book` (BLUR)
  Node: `limited` / `ambiguous`
  Operation: `deletion`
  Span: `[0, 1]`
  Residue evidence: `limited`
  Edges: `UNRESOLVED -> src_0:publicity [unresolved]`
- `Part of a ground revealed is attracting attention?` (STANDOUT)
  Node: `Part` / `ambiguous`
  Operation: `charade`
  Span: `[0, 1]`
  Residue evidence: `part`
  Edges: `UNRESOLVED -> src_0:of a [unresolved]`

## Inspection Prompts

- Should this node kind exist as shown, or should it split/merge with a neighbour?
- Does the edge point to the right target, or merely to the nearest available source?
- Is the node really wordplay, definition modifier, source-internal phrase material, or surface grammar?
- Is scope known now, or should the graph preserve `AWAITING_SCOPE` until answer verification?
