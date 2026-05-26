# Grammar Boundary Signal

Date: 2026-05-17

This report tests whether parser features give signal for GT V2 block anatomy.
It uses the supervised scaffold and does not alter solver behaviour.

## Coverage

- Records analysed: `931`
- Records with grammar features: `931`
- SOURCE tokens: `2235`
- RESIDUE tokens: `1913`
- SOURCE tokens whose parser head is another SOURCE token in the same source span: `883/1151` (`77%`)

## POS By Supervised Label

- `SOURCE`: `N`=935, `J`=288, `NP`=231, `Vi`=215, `P`=149, `R`=129, `Vb`=88, `D`=88, `PR`=54, `C`=19, `CD`=16, `RP`=13
- `RESIDUE`: `P`=559, `Vi`=431, `N`=299, `R`=132, `D`=126, `J`=125, `Vb`=77, `RP`=67, `C`=29, `NP`=25, `PR`=18, `X`=12
- `DEF`: `N`=591, `J`=233, `Vb`=103, `NP`=103, `Vi`=95, `P`=91, `R`=65, `D`=38, `RP`=13, `PR`=11, `CD`=8, `MD`=4

## Dependency Labels By Supervised Label

- `SOURCE`: `pobj`=359, `ROOT`=352, `amod`=266, `nsubj`=258, `compound`=198, `dobj`=150, `prep`=128, `advmod`=121, `det`=82, `poss`=39, `conj`=32, `npadvmod`=30
- `RESIDUE`: `prep`=477, `ROOT`=299, `pobj`=138, `advmod`=137, `amod`=127, `det`=107, `acl`=78, `prt`=67, `aux`=65, `dobj`=56, `nsubj`=54, `xcomp`=50
- `DEF`: `ROOT`=273, `pobj`=193, `amod`=182, `dobj`=139, `nsubj`=119, `compound`=111, `prep`=82, `advmod`=60, `det`=32, `acomp`=28, `acl`=15, `prt`=14

## Parser Head Crossings

`SOURCE->SOURCE`=1151, `RESIDUE->RESIDUE`=1003, `SOURCE->RESIDUE`=726, `DEF->DEF`=661, `RESIDUE->SOURCE`=583, `DEF->RESIDUE`=354, `DEF->SOURCE`=307, `SOURCE->DEF`=300, `RESIDUE->DEF`=290, `SOURCE-><none>`=58, `RESIDUE-><none>`=37, `DEF-><none>`=35

## SOURCE Head Pattern

`same-span`=883, `outside->RESIDUE`=726, `outside->DEF`=300, `other-source-span`=268, `outside-><none>`=58

## RESIDUE To Head Label

`of->RESIDUE`=53, `in->RESIDUE`=52, `to->RESIDUE`=49, `in->DEF`=43, `up->RESIDUE`=38, `in->SOURCE`=36, `some->SOURCE`=34, `the->RESIDUE`=27, `for->SOURCE`=26, `by->RESIDUE`=23, `for->RESIDUE`=23, `with->SOURCE`=23, `on->SOURCE`=18, `is->RESIDUE`=16, `from->DEF`=16, `of->DEF`=16, `and->SOURCE`=15, `to->SOURCE`=15, `back->RESIDUE`=14, `on->RESIDUE`=13, `picked->RESIDUE`=13, `with->RESIDUE`=12, `about->SOURCE`=12, `when->RESIDUE`=10, `a->RESIDUE`=10, `with->DEF`=10, `part->RESIDUE`=10, `around->RESIDUE`=9, `after->SOURCE`=8, `broadcast->RESIDUE`=8

## Preposition Attachment

- `of:RESIDUE->RESIDUE` = 53
- `in:RESIDUE->RESIDUE` = 50
- `to:RESIDUE->RESIDUE` = 49
- `in:RESIDUE->DEF` = 43
- `in:RESIDUE->SOURCE` = 36
- `in:SOURCE->SOURCE` = 29
- `for:RESIDUE->SOURCE` = 26
- `by:RESIDUE->RESIDUE` = 23
- `for:RESIDUE->RESIDUE` = 23
- `with:RESIDUE->SOURCE` = 23
- `of:DEF->DEF` = 22
- `on:RESIDUE->SOURCE` = 18
- `from:RESIDUE->DEF` = 16
- `of:RESIDUE->DEF` = 16
- `of:SOURCE->SOURCE` = 15
- `to:RESIDUE->SOURCE` = 15
- `to:SOURCE->SOURCE` = 13
- `on:RESIDUE->RESIDUE` = 13
- `with:RESIDUE->RESIDUE` = 12
- `about:RESIDUE->SOURCE` = 12
- `in:SOURCE->RESIDUE` = 12
- `in:DEF->DEF` = 10
- `with:RESIDUE->DEF` = 10
- `after:RESIDUE->SOURCE` = 8
- `by:RESIDUE->SOURCE` = 8
- `to:DEF->DEF` = 8
- `from:RESIDUE->RESIDUE` = 8
- `for:DEF->DEF` = 7
- `of:RESIDUE->SOURCE` = 6
- `on:SOURCE->SOURCE` = 6

## Glue Word Attachment

- `of:RESIDUE->RESIDUE` = 53
- `in:RESIDUE->RESIDUE` = 52
- `to:RESIDUE->RESIDUE` = 49
- `in:RESIDUE->DEF` = 43
- `a:SOURCE->SOURCE` = 36
- `in:RESIDUE->SOURCE` = 36
- `in:SOURCE->SOURCE` = 32
- `the:RESIDUE->RESIDUE` = 27
- `for:RESIDUE->SOURCE` = 26
- `by:RESIDUE->RESIDUE` = 23
- `for:RESIDUE->RESIDUE` = 23
- `with:RESIDUE->SOURCE` = 23
- `of:DEF->DEF` = 22
- `on:RESIDUE->SOURCE` = 18
- `a:DEF->DEF` = 16
- `is:RESIDUE->RESIDUE` = 16
- `from:RESIDUE->DEF` = 16
- `of:RESIDUE->DEF` = 16
- `of:SOURCE->SOURCE` = 15
- `and:RESIDUE->SOURCE` = 15
- `to:RESIDUE->SOURCE` = 15
- `to:SOURCE->SOURCE` = 13
- `on:RESIDUE->RESIDUE` = 13
- `with:RESIDUE->RESIDUE` = 12
- `is:SOURCE->SOURCE` = 12
- `in:SOURCE->RESIDUE` = 12
- `a:SOURCE->DEF` = 11
- `a:RESIDUE->RESIDUE` = 10
- `in:DEF->DEF` = 10
- `with:RESIDUE->DEF` = 10

## Glue Word Dependency Context

- `in:RESIDUE:prep` = 128
- `of:RESIDUE:prep` = 75
- `a:SOURCE:det` = 52
- `for:RESIDUE:prep` = 52
- `to:RESIDUE:aux` = 50
- `with:RESIDUE:prep` = 44
- `in:SOURCE:prep` = 44
- `the:RESIDUE:det` = 36
- `on:RESIDUE:prep` = 34
- `from:RESIDUE:prep` = 29
- `by:RESIDUE:agent` = 26
- `and:RESIDUE:cc` = 26
- `of:DEF:prep` = 22
- `a:RESIDUE:det` = 21
- `a:DEF:det` = 18
- `in:DEF:prep` = 18
- `of:SOURCE:prep` = 17
- `to:RESIDUE:prep` = 17
- `is:RESIDUE:ROOT` = 13
- `with:SOURCE:prep` = 10
- `at:SOURCE:prep` = 10
- `from:DEF:prep` = 10
- `over:RESIDUE:prep` = 10
- `to:SOURCE:aux` = 9
- `and:SOURCE:cc` = 9
- `for:DEF:prep` = 9
- `is:SOURCE:ROOT` = 8
- `an:RESIDUE:det` = 8
- `on:SOURCE:prep` = 7
- `the:SOURCE:det` = 7

## Operationish Word Context

- `up:RESIDUE:prt->RESIDUE` = 35
- `some:RESIDUE:det->SOURCE` = 31
- `picked:RESIDUE:ROOT->RESIDUE` = 13
- `about:RESIDUE:prep->SOURCE` = 12
- `back:RESIDUE:advmod->RESIDUE` = 11
- `radio:RESIDUE:pobj->RESIDUE` = 8
- `up:RESIDUE:prt->SOURCE` = 7
- `reportedly:RESIDUE:advmod->SOURCE` = 5
- `broadcast:RESIDUE:ROOT->RESIDUE` = 4
- `up:DEF:prt->DEF` = 4
- `broadcast:RESIDUE:pobj->RESIDUE` = 4
- `some:RESIDUE:det->RESIDUE` = 4
- `reportedly:RESIDUE:advmod->DEF` = 4
- `some:SOURCE:det->SOURCE` = 3
- `up:SOURCE:prt->SOURCE` = 3
- `east:RESIDUE:pobj->RESIDUE` = 3
- `partly:RESIDUE:advmod->SOURCE` = 3
- `some:RESIDUE:nsubj->SOURCE` = 3
- `back:RESIDUE:prt->RESIDUE` = 3
- `rising:RESIDUE:amod->SOURCE` = 3
- `heard:RESIDUE:ROOT->RESIDUE` = 3
- `excited:RESIDUE:acomp->SOURCE` = 2
- `excited:RESIDUE:ROOT->RESIDUE` = 2
- `up:RESIDUE:prt-><none>` = 2
- `partly:RESIDUE:advmod->RESIDUE` = 2
- `back:RESIDUE:advmod->DEF` = 2
- `some:RESIDUE:dobj->DEF` = 2
- `reported:RESIDUE:ROOT->RESIDUE` = 2
- `reverse:RESIDUE:pobj->RESIDUE` = 2
- `up:RESIDUE:advmod->RESIDUE` = 2

## Preposition Attachment By Operation

- `anagram`: `to:RESIDUE->RESIDUE`=14, `of:RESIDUE->RESIDUE`=10, `for:RESIDUE->SOURCE`=9, `for:RESIDUE->RESIDUE`=7, `in:RESIDUE->RESIDUE`=7, `of:RESIDUE->DEF`=7, `with:RESIDUE->SOURCE`=6, `in:RESIDUE->DEF`=5, `with:RESIDUE->RESIDUE`=5, `in:RESIDUE->SOURCE`=5
- `charade`: `to:RESIDUE->RESIDUE`=18, `with:RESIDUE->SOURCE`=16, `in:RESIDUE->SOURCE`=15, `of:RESIDUE->RESIDUE`=13, `of:SOURCE->SOURCE`=11, `in:RESIDUE->DEF`=11, `of:DEF->DEF`=10, `on:RESIDUE->SOURCE`=9, `in:RESIDUE->RESIDUE`=8, `with:RESIDUE->RESIDUE`=6
- `container`: `in:RESIDUE->SOURCE`=4, `about:RESIDUE->SOURCE`=2, `of:RESIDUE->DEF`=2, `of:RESIDUE->RESIDUE`=2, `on:RESIDUE->SOURCE`=1, `through:RESIDUE->DEF`=1, `beyond:DEF->DEF`=1, `in:RESIDUE->RESIDUE`=1, `with:RESIDUE->DEF`=1, `without:SOURCE->DEF`=1
- `deletion`: `for:RESIDUE->SOURCE`=1, `for:RESIDUE->RESIDUE`=1, `in:RESIDUE->RESIDUE`=1, `on:RESIDUE->DEF`=1
- `deletion+anagram`: `with:RESIDUE->SOURCE`=1, `by:SOURCE->SOURCE`=1, `to:RESIDUE->RESIDUE`=1, `in:SOURCE->SOURCE`=1, `to:SOURCE->RESIDUE`=1, `for:RESIDUE->RESIDUE`=1, `in:RESIDUE->RESIDUE`=1, `around:RESIDUE->RESIDUE`=1, `towards:RESIDUE->RESIDUE`=1
- `hidden`: `in:SOURCE->SOURCE`=19, `of:RESIDUE->RESIDUE`=15, `by:RESIDUE->RESIDUE`=13, `in:RESIDUE->RESIDUE`=12, `in:RESIDUE->DEF`=10, `to:RESIDUE->SOURCE`=6, `for:RESIDUE->SOURCE`=6, `to:RESIDUE->RESIDUE`=6, `in:RESIDUE->SOURCE`=5, `from:RESIDUE->DEF`=5
- `hidden_reversed`: `in:RESIDUE->RESIDUE`=12, `in:RESIDUE->DEF`=10, `of:RESIDUE->RESIDUE`=5, `from:RESIDUE->DEF`=3, `in:SOURCE->RESIDUE`=3, `to:RESIDUE->SOURCE`=2, `after:RESIDUE->SOURCE`=2, `of:RESIDUE->DEF`=2, `from:RESIDUE->RESIDUE`=2, `by:RESIDUE->RESIDUE`=2
- `homophone`: `for:RESIDUE->SOURCE`=5, `on:RESIDUE->SOURCE`=5, `in:RESIDUE->DEF`=4, `of:RESIDUE->RESIDUE`=4, `in:RESIDUE->SOURCE`=3, `on:RESIDUE->RESIDUE`=3, `like:RESIDUE->RESIDUE`=3, `to:DEF->DEF`=3, `of:RESIDUE->DEF`=2, `to:RESIDUE->RESIDUE`=2
- `reversal`: `in:RESIDUE->RESIDUE`=6, `to:RESIDUE->RESIDUE`=6, `in:RESIDUE->SOURCE`=4, `of:RESIDUE->RESIDUE`=4, `from:RESIDUE->SOURCE`=3, `on:RESIDUE->SOURCE`=3, `on:RESIDUE->DEF`=3, `in:RESIDUE->DEF`=3, `on:RESIDUE->RESIDUE`=2, `for:RESIDUE->RESIDUE`=2

## Boundary Crossing Examples

- `Sister, say, in a bolder violet when dressed` (BLOODRELATIVE)
  Operation: `anagram`; SOURCE `violet` depends on RESIDUE `in` as `pobj`
  Residue: `in when dressed`
  Labels: `D D R S S S R R`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; SOURCE `time` depends on RESIDUE `with` as `pobj`
  Residue: `say with in created`
  Labels: `D R R S R S S R`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; SOURCE `gaps` depends on RESIDUE `in` as `pobj`
  Residue: `say with in created`
  Labels: `D R R S R S S R`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; SOURCE `door` depends on RESIDUE `created` as `nsubj`
  Residue: `say with in created`
  Labels: `D R R S R S S R`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`; SOURCE `a` depends on RESIDUE `rupture` as `det`
  Residue: `in event of emergency rupture`
  Labels: `D R R R R S S S R`
- `Routes in event of emergency across a deep rupture` (ESCAPEROADS)
  Operation: `anagram`; SOURCE `deep` depends on RESIDUE `rupture` as `amod`
  Residue: `in event of emergency rupture`
  Labels: `D R R R R S S S R`
- `Bet Rod's upset those owing money` (DEBTORS)
  Operation: `anagram`; SOURCE `Rod's` depends on RESIDUE `those` as `nsubj`
  Residue: `upset those`
  Labels: `S S R R D D`
- `Hammer, say, broadcast of stunt with nimble turn` (BLUNTINSTRUMENT)
  Operation: `anagram`; SOURCE `stunt` depends on RESIDUE `of` as `pobj`
  Residue: `say broadcast of with`
  Labels: `D R R R S R S S`

## Residue Glue Attached To SOURCE Examples

- `Man is unaccompanied when cycling` (ELON)
  Operation: `anagram`; `is` heads to SOURCE `unaccompanied` as `auxpass`
  Residue: `is when cycling`
  Labels: `D R S R R`
- `Slug, say, with time in gaps door created` (GASTROPOD)
  Operation: `charade`; `in` heads to SOURCE `time` as `prep`
  Residue: `say with in created`
  Labels: `D R R S R S S R`
- `Have a second job looming somehow by empty hut` (MOONLIGHT)
  Operation: `anagram`; `by` heads to SOURCE `looming` as `prep`
  Residue: `by`
  Labels: `D D D D S S R S S`
- `Bond hated other criminal in a state of anxiety?` (HOTANDBOTHERED)
  Operation: `anagram`; `in` heads to SOURCE `hated` as `prep`
  Residue: `criminal in a state of`
  Labels: `S S S R R R R R D`
- `Painter got excited about shift in financial projection` (OPERATINGBUDGET)
  Operation: `anagram`; `in` heads to SOURCE `shift` as `prep`
  Residue: `excited about in`
  Labels: `S S R R S R D D`
- `Rubbish flung in sides of enormous swamps` (ENGULFS)
  Operation: `anagram`; `of` heads to SOURCE `sides` as `prep`
  Residue: `of`
  Labels: `S S S S R S D`
- `Unusual logo on bread and citrus fruit` (BLOODORANGE)
  Operation: `anagram`; `and` heads to SOURCE `bread` as `cc`
  Residue: `unusual and citrus`
  Labels: `R S S S R R D`
- `Stop awkwardly by a line relating to deliveries?` (POSTAL)
  Operation: `charade`; `by` heads to SOURCE `Stop` as `agent`
  Residue: `awkwardly by`
  Labels: `S R R S S D D D`

## Direction Word Examples

- `Awkward ride around back of park making one annoyed` (IRKED)
  Operation: `anagram`; direction `back` labelled `SOURCE` heads to `around` / `SOURCE`
  Residue: `<empty>`
  Labels: `S S S S S S D D D`
- `Strong soldiers turning up in front of sculpture` (ROBUST)
  Operation: `charade`; direction `up` labelled `RESIDUE` heads to `turning` / `RESIDUE`
  Residue: `turning up in front of`
  Labels: `D S R R R R R S`
- `Roll up at work given good measure on a course?` (FURLONG)
  Operation: `charade`; direction `up` labelled `SOURCE` heads to `Roll` / `SOURCE`
  Residue: `at given`
  Labels: `S S R S R S D D D D`
- `Hands from the East in trade` (SWAP)
  Operation: `reversal`; direction `East` labelled `RESIDUE` heads to `from` / `RESIDUE`
  Residue: `from the east in`
  Labels: `S R R R R D`
- `Cheer up figure behind judge` (HEARTEN)
  Operation: `charade`; direction `up` labelled `DEF` heads to `Cheer` / `DEF`
  Residue: `behind`
  Labels: `D D S R S`
- `Some road engineers in Middle East port` (ADEN)
  Operation: `hidden`; direction `East` labelled `SOURCE` heads to `port` / `DEF`
  Residue: `some`
  Labels: `R S S S S S D`
- `Smashing atoms up is creating element` (POTASSIUM)
  Operation: `anagram`; direction `up` labelled `SOURCE` heads to `atoms` / `SOURCE`
  Residue: `smashing creating`
  Labels: `R S S S R D`
- `Picked up perfumes for a little cash` (CENTS)
  Operation: `anagram`; direction `up` labelled `RESIDUE` heads to `Picked` / `RESIDUE`
  Residue: `picked up for`
  Labels: `R R S R D D D`

## Reading

The parser signal should be treated as weak supervision, not authority.
Useful signal appears where residue glue attaches syntactically to SOURCE material, because that marks the exact attachment question the residue baseline cannot answer.
The next useful test is a small classifier for token label or boundary transitions using residue features plus POS/dependency context, measured against the residue-only baseline buckets.
