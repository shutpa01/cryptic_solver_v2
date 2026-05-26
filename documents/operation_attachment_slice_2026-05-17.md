# Operation Attachment Slice

Date: 2026-05-17

This is a weakly-labelled R&D slice for residue attachment.
Labels are provisional and operation-derived; they are not production truth.

Residue runs: `1144`

## Label Counts

- `UNCLASSIFIED_RESIDUE`: 480
- `OPERATOR_SCOPE`: 281
- `CONNECTOR_OR_SURFACE`: 118
- `DIRECTION_OR_ORIENTATION`: 112
- `CONTAINER_RELATION`: 100
- `ORDER_OR_POSITION`: 27
- `LOCATOR_SCOPE`: 22
- `DEF_MODIFIER`: 4

## Labels By Operation

- `anagram`: `UNCLASSIFIED_RESIDUE`=160, `OPERATOR_SCOPE`=59, `CONNECTOR_OR_SURFACE`=27, `LOCATOR_SCOPE`=4
- `charade`: `UNCLASSIFIED_RESIDUE`=135, `CONTAINER_RELATION`=71, `CONNECTOR_OR_SURFACE`=47, `ORDER_OR_POSITION`=27, `LOCATOR_SCOPE`=15, `DEF_MODIFIER`=3
- `container`: `CONTAINER_RELATION`=29
- `deletion`: `CONNECTOR_OR_SURFACE`=3, `UNCLASSIFIED_RESIDUE`=2, `LOCATOR_SCOPE`=2
- `deletion+anagram`: `UNCLASSIFIED_RESIDUE`=7, `OPERATOR_SCOPE`=3, `CONNECTOR_OR_SURFACE`=1
- `hidden`: `OPERATOR_SCOPE`=106, `UNCLASSIFIED_RESIDUE`=92, `CONNECTOR_OR_SURFACE`=16, `DEF_MODIFIER`=1
- `hidden_reversed`: `DIRECTION_OR_ORIENTATION`=55, `OPERATOR_SCOPE`=40, `UNCLASSIFIED_RESIDUE`=23, `CONNECTOR_OR_SURFACE`=4, `LOCATOR_SCOPE`=1
- `homophone`: `OPERATOR_SCOPE`=73, `UNCLASSIFIED_RESIDUE`=12, `CONNECTOR_OR_SURFACE`=10
- `reversal`: `DIRECTION_OR_ORIENTATION`=57, `UNCLASSIFIED_RESIDUE`=49, `CONNECTOR_OR_SURFACE`=10

## Examples

- `Follow lad after repairing emergency barrier` (FLOODWALL)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `after repairing`
  Needs split: `after->UNCLASSIFIED_RESIDUE, repairing->OPERATOR_SCOPE`
  Sources: `Follow lad`
  Definitions: `emergency barrier`
  Parser source heads: `after->lad:prep`
- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `in`
  Sources: `a bolder violet`
  Definitions: `Sister say`
- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `when dressed`
  Needs split: `when->UNCLASSIFIED_RESIDUE, dressed->OPERATOR_SCOPE`
  Sources: `a bolder violet`
  Definitions: `Sister say`
- `Man is unaccompanied when cycling` (ELON)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `is`
  Sources: `unaccompanied`
  Definitions: `Man`
  Parser source heads: `is->unaccompanied:auxpass`
- `Man is unaccompanied when cycling` (ELON)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `when cycling`
  Needs split: `when->UNCLASSIFIED_RESIDUE, cycling->OPERATOR_SCOPE`
  Sources: `unaccompanied`
  Definitions: `Man`
  Parser source heads: `cycling->unaccompanied:advcl`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; label: `DEF_MODIFIER`; relationship: `MODIFIES_DEFINITION`
  Scope status: `definition_modifier_candidate`
  Residue: `say with`
  Needs split: `say->DEF_MODIFIER, with->CONTAINER_RELATION`
  Sources: `time`; `gaps door`
  Definitions: `Slug`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; label: `CONTAINER_RELATION`; relationship: `CONTAINS`
  Scope status: `weakly_scoped_from_operation`
  Residue: `in`
  Sources: `time`; `gaps door`
  Definitions: `Slug`
  Parser source heads: `in->time:prep`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `created`
  Sources: `time`; `gaps door`
  Definitions: `Slug`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `in event of emergency`
  Sources: `across a deep`
  Definitions: `Routes`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `rupture`
  Sources: `across a deep`
  Definitions: `Routes`
  Parser source heads: `rupture->across:pobj`
- `Bet Rod's upset those owing money` (DEBTORS)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `upset those`
  Needs split: `upset->OPERATOR_SCOPE, those->UNCLASSIFIED_RESIDUE`
  Sources: `Bet Rod's`
  Definitions: `owing money`
- `Nautical ode when reviewed having instructive value?` (EDUCATIONAL)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `when reviewed having`
  Needs split: `when->UNCLASSIFIED_RESIDUE, reviewed->OPERATOR_SCOPE, having->UNCLASSIFIED_RESIDUE`
  Sources: `Nautical ode`
  Definitions: `instructive value`
  Parser source heads: `reviewed->ode:relcl`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `say broadcast of`
  Needs split: `say->DEF_MODIFIER, broadcast->OPERATOR_SCOPE, of->CONNECTOR_OR_SURFACE`
  Sources: `stunt`; `nimble turn`
  Definitions: `Hammer`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `with`
  Sources: `stunt`; `nimble turn`
  Definitions: `Hammer`
- `Have a second job looming somehow by empty hut` (MOONLIGHT)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `by`
  Sources: `looming somehow`; `empty hut`
  Definitions: `Have a second job`
  Parser source heads: `by->looming:prep`
- `Bow to clerk bewildered by tall building` (TOWERBLOCK)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `bewildered by`
  Sources: `Bow to clerk`
  Definitions: `tall building`
  Parser source heads: `bewildered->clerk:acl`
- `Home Ron devised for bird` (MOORHEN)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `devised for`
  Sources: `Home Ron`
  Definitions: `bird`
- `Volatile sort heard in good or hard bargaining` (HORSETRADING)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `Volatile`
  Sources: `sort heard in good`
  Definitions: `hard bargaining`
  Parser source heads: `Volatile->sort:amod`
- `Volatile sort heard in good or hard bargaining` (HORSETRADING)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `or`
  Sources: `sort heard in good`
  Definitions: `hard bargaining`
  Parser source heads: `or->good:cc`
- `Bad smell began to spread around marsh plant` (BOGBEAN)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `Bad`
  Sources: `smell began`
  Definitions: `marsh plant`
  Parser source heads: `Bad->smell:amod`
- `Bad smell began to spread around marsh plant` (BOGBEAN)
  Operation: `charade`; label: `CONTAINER_RELATION`; relationship: `CONTAINS`
  Scope status: `weakly_scoped_from_operation`
  Residue: `to spread around`
  Needs split: `to->CONNECTOR_OR_SURFACE, spread->UNCLASSIFIED_RESIDUE, around->CONTAINER_RELATION`
  Sources: `smell began`
  Definitions: `marsh plant`
  Parser source heads: `spread->began:xcomp`
- `A term's e.g. formulated for one wagering money` (GAMESTER)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `formulated for one wagering`
  Needs split: `formulated->OPERATOR_SCOPE, for->CONNECTOR_OR_SURFACE, one->UNCLASSIFIED_RESIDUE, wagering->UNCLASSIFIED_RESIDUE`
  Sources: `A term's e g`
  Definitions: `money`
- `Thing to ride wrongly takes B-road` (SKATEBOARD)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `to ride wrongly`
  Needs split: `to->CONNECTOR_OR_SURFACE, ride->UNCLASSIFIED_RESIDUE, wrongly->OPERATOR_SCOPE`
  Sources: `takes B-road`
  Definitions: `Thing`
  Parser source heads: `wrongly->takes:advmod`
- `See glum eccentric getting plants grown as a crop` (LEGUMES)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `eccentric getting plants grown as a`
  Needs split: `eccentric->OPERATOR_SCOPE, getting->UNCLASSIFIED_RESIDUE, plants->UNCLASSIFIED_RESIDUE, grown->UNCLASSIFIED_RESIDUE, as->CONNECTOR_OR_SURFACE, a->CONNECTOR_OR_SURFACE`
  Sources: `See glum`
  Definitions: `crop`
  Parser source heads: `getting->See:dobj`
- `Gruel I suspect is less attractive` (UGLIER)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `suspect is`
  Sources: `Gruel I`
  Definitions: `less attractive`
- `Engineer dares to skirt eastern body of water` (REDSEA)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `Engineer`
  Sources: `dares`; `eastern`
  Definitions: `body of water`
  Parser source heads: `Engineer->dares:nsubj`
- `Engineer dares to skirt eastern body of water` (REDSEA)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `to skirt`
  Sources: `dares`; `eastern`
  Definitions: `body of water`
  Parser source heads: `skirt->dares:xcomp`
- `Take a risk having disrupted nice chat` (CHANCEIT)
  Operation: `anagram`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `having disrupted`
  Sources: `nice chat`
  Definitions: `Take a risk`
- `Bond hated other criminal in a state of anxiety?` (HOTANDBOTHERED)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `criminal in a state of`
  Needs split: `criminal->OPERATOR_SCOPE, in->CONNECTOR_OR_SURFACE, a->CONNECTOR_OR_SURFACE, state->UNCLASSIFIED_RESIDUE, of->CONNECTOR_OR_SURFACE`
  Sources: `Bond hated other`
  Definitions: `anxiety`
  Parser source heads: `criminal->hated:dobj, in->hated:prep`
- `Wobbling cedars making one afraid` (SCARED)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `Wobbling`
  Sources: `cedars`
  Definitions: `afraid`
- `Wobbling cedars making one afraid` (SCARED)
  Operation: `anagram`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `making one`
  Sources: `cedars`
  Definitions: `afraid`
  Parser source heads: `making->cedars:acl`
- `Underhand edict veep devised` (DECEPTIVE)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `devised`
  Sources: `edict veep`
  Definitions: `Underhand`
  Parser source heads: `devised->edict:dep`
- `Renown at intervals for a very long period` (EON)
  Operation: `anagram`; label: `LOCATOR_SCOPE`; relationship: `LOCATES_WITHIN`
  Scope status: `weakly_scoped_from_operation`
  Residue: `Renown at intervals`
  Needs split: `Renown->UNCLASSIFIED_RESIDUE, at->CONNECTOR_OR_SURFACE, intervals->LOCATOR_SCOPE`
  Sources: `for`
  Definitions: `a very long period`
- `Types about to run around scenic site` (BEAUTYSPOT)
  Operation: `anagram`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `to run around scenic`
  Sources: `Types about`
  Definitions: `site`
  Parser source heads: `run->about:xcomp`
- `Star set to change samples` (TASTERS)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `to change`
  Sources: `Star set`
  Definitions: `samples`
  Parser source heads: `change->set:xcomp`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `excited about`
  Needs split: `excited->OPERATOR_SCOPE, about->UNCLASSIFIED_RESIDUE`
  Sources: `Painter got`; `shift`
  Definitions: `financial projection`
  Parser source heads: `excited->got:acomp`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `in`
  Sources: `Painter got`; `shift`
  Definitions: `financial projection`
  Parser source heads: `in->shift:prep`
- `Rubbish flung in sides of enormous swamps` (ENGULFS)
  Operation: `anagram`; label: `CONNECTOR_OR_SURFACE`; relationship: `SURFACE_CONNECTS`
  Scope status: `surface_only_until_grammar_check`
  Residue: `of`
  Sources: `Rubbish flung in sides`; `enormous`
  Definitions: `swamps`
  Parser source heads: `of->sides:prep`
- `Programmes chess duel differently` (SCHEDULES)
  Operation: `anagram`; label: `OPERATOR_SCOPE`; relationship: `OPERATES_ON`
  Scope status: `weakly_scoped_from_operation`
  Residue: `differently`
  Sources: `chess duel`
  Definitions: `Programmes`
  Parser source heads: `differently->duel:advmod`
- `Mike had tub replaced for therapeutic treatment` (MUDBATH)
  Operation: `charade`; label: `UNCLASSIFIED_RESIDUE`; relationship: `UNRESOLVED`
  Scope status: `unresolved`
  Residue: `replaced for`
  Sources: `Mike had tub`
  Definitions: `therapeutic treatment`

## Reading

This slice is intended for manual inspection and later classifier experiments.
The useful next move is to inspect one weak label at a time and decide whether the relationship is anatomically right.
The `block_relationship` and `scope_status` fields are deliberately separate from the label so that operation, locator, and scope can be preserved without flattening.
