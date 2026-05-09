# v0 wrapper - per-clue review

Test bed: DT 31132 / 31138 / 31150 (92 clues, unprocessed or lightly mechanical-only).

## Headline

- PASS: 10
- FAIL: 36
- NO_FORM: 46

## PASS - cases the wrapper got right (10)

| Puzzle | Clue # | Answer | Op | Clue text |
|---|---|---|---|---|
| 31132 | 1a | PARADISE | anagram | A padre is prepared for heaven |
| 31132 | 1d | PRECLUDE | container | Rule out opening boxing clubs |
| 31132 | 8d | COTTON | container | Against eating too much fibre |
| 31132 | 17d | HEIGHTEN | container | Reinforce layer protecting crew |
| 31132 | 20d | HEINOUS | anagram | Monstrous bats in house |
| 31138 | 11a | APPRAISED | anagram | Spare iPad somehow given a value |
| 31138 | 18a | MATADOR | charade | Ring performer Mike visiting a party with Republican |
| 31150 | 11a | Adrenalin | anagram | Daniel ran badly, which increases the heart rate |
| 31150 | 22a | TRIER | synonym | Judge city in Germany |
| 31150 | 2d | ADMIRER | anagram | Model married fan |

## FAIL - wrapper built a form but verifier rejected it (36)

Each entry: clue text -> answer, the tree, and the failed checks.

### 31132 5across - `PICNIC`

> Photograph largely enjoyable meal out

**Form:**

```
deletion ['largely'] kind=tail
  synonym('PIC', src='Photograph')
```

**Verifier:**
  X assembly               tree does not produce 'PICNIC'
  + bridge.leaves          all 1 leaves verified
  X bridge.indicators      indicator 'largely' not in DB for op 'deletion' (expected types ['deletion'])
  X residue                unaccounted: ['enjoyable']

**Flags:** op_inferred_from_token_mix_not_provided, op_name_not_preserved_in_solveresult

### 31132 9across - `ENDEAVOUR`

> Dexter's lead character in an oeuvre beaten up and shot

**Form:**

```
container ['in']
  abbreviation('U', src='up')
  synonym('ENDEAVOR', src='shot')
```

**Verifier:**
  + assembly               tree produces ENDEAVOUR
  X bridge.leaves          abbreviation 'U' not in DB for 'up'
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['beaten', 'character', "dexter's", 'lead', 'oeuvre']

**Flags:** container_outer_inner_not_preserved

### 31132 11across - `TEMPT`

> Lure agency worker with Tango

**Form:**

```
charade
  synonym('TEMP', src='agency')
  synonym('TEMP', src='worker')
  synonym('T', src='Tango')
```

**Verifier:**
  X assembly               tree does not produce 'TEMPT'
  X bridge.leaves          synonym 'TEMP' not in DB for 'agency'
  + bridge.indicators      
  + residue                

### 31132 18across - `service charge`

> Maintain storm creates additional damage for diner

**Form:**

```
charade
  synonym('SERVICE', src='Maintain')
  synonym('CHARGE', src='storm')
```

**Verifier:**
  + assembly               tree produces SERVICECHARGE
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['diner', 'for']; non-link words declared as link_words: ['creates']

### 31132 23across - `AIRING`

> Giving voice to group of chatbots?

**Form:**

```
synonym('AIRING', src='voice')
```

**Verifier:**
  + assembly               tree produces AIRING
  + bridge.leaves          all 1 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['chatbots']; non-link words declared as link_words: ['chatbots?', 'group']

### 31132 26across - `OWNER`

> Single run involving women's title holder

**Form:**

```
container ['involving']
  synonym('ONE', src='Single')
  charade
    synonym('R', src='run')
    synonym('W', src="women's")
```

**Verifier:**
  X assembly               tree does not produce 'OWNER'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['title']

**Flags:** container_charade_structure_not_preserved

### 31132 28across - `TIERED`

> Neckwear frequently sported by Trump, Donald, in banks

**Form:**

```
charade
  positional[last]('ED', src='sported')
  synonym('TIER', src='banks')
```

**Verifier:**
  X assembly               tree does not produce 'TIERED'
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['donald', 'frequently', 'trump']; non-link words declared as link_words: ['donald,', 'neckwear', 'trump,']

**Flags:** definition_phrase_missing

### 31132 2down - `RADON`

> Element of song and dance in Royal Navy

**Form:**

```
container ['in']
  synonym('ADO', src='and')
  synonym('RN', src='Navy')
```

**Verifier:**
  + assembly               tree produces RADON
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['dance', 'royal', 'song']

**Flags:** container_outer_inner_not_preserved

### 31132 4down - `SHOD`

> Nothing interrupting quiet day in flats?

**Form:**

```
charade
  synonym('O', src='Nothing')
  synonym('SH', src='quiet')
  synonym('D', src='day')
```

**Verifier:**
  X assembly               tree does not produce 'SHOD'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['interrupting']

### 31132 16down - `RESIDENCE`

> Sincere about taking retiring editor's place

**Form:**

```
anagram ['about']
  literal('SINCERE', src='Sincere')
  literal('EDITORS', src="editor's")
```

**Verifier:**
  X assembly               tree does not produce 'RESIDENCE'
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['retiring', 'taking']

### 31132 25down - `NOSH`

> Why deli doesn't make delish food?

**Form:**

```
charade
  synonym('NO', src='Why')
  positional[last]('SH', src='delish')
```

**Verifier:**
  + assembly               tree produces NOSH
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['make']; non-link words declared as link_words: ['deli', "doesn't"]

### 31138 1across - `MAHOGANY`

> Turn meat over with some wood

**Form:**

```
charade
  synonym('GO', src='Turn')
  synonym('HAM', src='meat')
  synonym('ANY', src='some')
```

**Verifier:**
  X assembly               tree does not produce 'MAHOGANY'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['over']

### 31138 5across - `ISOBAR`

> Curve on map is essentially ignored by local

**Form:**

```
charade
  positional[first]('I', src='is')
  synonym('SO', src='by')
  synonym('BAR', src='local')
```

**Verifier:**
  + assembly               tree produces ISOBAR
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['ignored']; non-link words declared as link_words: ['essentially']

### 31138 10across - `SLATE`

> Had food on back of empty steel pan

**Form:**

```
charade
  synonym('ATE', src='Had')
  abbreviation('S', src='on')
  positional[last]('L', src='steel')
```

**Verifier:**
  X assembly               tree does not produce 'SLATE'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['back']; non-link words declared as link_words: ['empty', 'food']

### 31138 20across - `SESAME`

> Advises American screening seed supplier

**Form:**

```
hidden
  literal('ADVISESAMERICANSCREENING', src='Advises American screening')
```

**Verifier:**
  + assembly               tree produces SESAME
  + bridge.leaves          all 1 leaves verified
  X bridge.indicators      hidden node has no indicator
  + residue                

**Flags:** hidden_indicator_not_tagged_by_solver, op_inferred_from_token_mix_not_provided, op_name_not_preserved_in_solveresult

### 31138 28across - `news desk`

> Modern offices secured finally by river for media centre

**Form:**

```
charade
  synonym('NEW', src='Modern')
  positional[last]('S', src='offices')
  positional[last]('D', src='secured')
  synonym('ESK', src='river')
```

**Verifier:**
  + assembly               tree produces NEWSDESK
  + bridge.leaves          all 4 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['finally']

### 31138 1down - `MUSLIN`

> Large amount rolled inside of fine fabric

**Form:**

```
charade
  synonym('L', src='Large')
  synonym('SUM', src='amount')
  synonym('IN', src='inside')
```

**Verifier:**
  X assembly               tree does not produce 'MUSLIN'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['rolled']; non-link words declared as link_words: ['fine']

### 31138 3down - `green woodpecker`

> Young wife precooked wild bird

**Form:**

```
charade
  synonym('GREEN', src='Young')
  synonym('WOODPECKER', src='bird')
```

**Verifier:**
  + assembly               tree produces GREENWOODPECKER
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['precooked', 'wife', 'wild']

### 31138 8down - `RADIATES`

> Fans out of area D sit all over the place

**Form:**

```
anagram ['over']
  literal('AREA', src='area')
  literal('D')
  literal('SIT', src='sit')
  literal('THE', src='the')
```

**Verifier:**
  X assembly               tree does not produce 'RADIATES'
  + bridge.leaves          all 4 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['place']

### 31138 16down - `Cambridge`

> Doctors caught scaling bank in city

**Form:**

```
charade
  synonym('MB', src='Doctors')
  positional[first]('CA', src='caught')
  synonym('RIDGE', src='bank')
```

**Verifier:**
  X assembly               tree does not produce 'CAMBRIDGE'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['scaling']

### 31138 17down - `AMICABLE`

> Friendly novelist abridged message

**Form:**

```
synonym('AMICABLE', src='Friendly')
```

**Verifier:**
  + assembly               tree produces AMICABLE
  + bridge.leaves          all 1 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['abridged', 'message', 'novelist']

### 31138 19down - `RHINOS`

> Queen greeting Poles housing old animals

**Form:**

```
charade
  abbreviation('S', src='Poles')
  synonym('RHINO', src='animals')
```

**Verifier:**
  X assembly               tree does not produce 'RHINOS'
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['greeting', 'housing', 'old', 'queen']

### 31138 20down - `so there`

> Triumphal expression of drunk with present

**Form:**

```
charade
  synonym('SOT', src='drunk')
  synonym('HERE', src='present')
```

**Verifier:**
  + assembly               tree produces SOTHERE
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['triumphal']

### 31138 21down - `STREAK`

> Romeo takes off for cheeky run across pitch?

**Form:**

```
anagram ['off']
  literal('ROMEO', src='Romeo')
  literal('TAKES', src='takes')
```

**Verifier:**
  X assembly               tree does not produce 'STREAK'
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['across', 'pitch']; non-link words declared as link_words: ['cheeky']

### 31138 23down - `CLASP`

> Colonel eviscerated reptile in hold

**Form:**

```
charade
  positional[outer]('CL', src='Colonel')
  synonym('ASP', src='reptile')
```

**Verifier:**
  + assembly               tree produces CLASP
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['eviscerated']

### 31150 1across - `what for`

> I'm sorry fine soldiers could get severe reprimand

**Form:**

```
charade
  synonym('WHAT', src='sorry')
  synonym('F', src='fine')
  synonym('OR', src='soldiers')
```

**Verifier:**
  + assembly               tree produces WHATFOR
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['get', "i'm", 'severe']

### 31150 12across - `SPEND`

> Exhaust pipe's initially carried by post

**Form:**

```
container
  abbreviation('P', src="pipe's")
  synonym('SEND', src='post')
```

**Verifier:**
  + assembly               tree produces SPEND
  X bridge.leaves          abbreviation 'P' not in DB for "pipe's"
  X bridge.indicators      container node has no indicator
  X residue                non-link words declared as link_words: ['carried', 'initially']

**Flags:** container_outer_inner_not_preserved

### 31150 17across - `last-ditch`

> Eleventh-hour note by daughter beset by sharp side pain

**Form:**

```
container ['beset']
  synonym('LA', src='note')
  charade
    synonym('D', src='daughter')
    synonym('STITCH', src='pain')
```

**Verifier:**
  X assembly               tree does not produce 'LASTDITCH'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['sharp', 'side']

**Flags:** container_charade_structure_not_preserved

### 31150 19across - `TABOO`

> Banned cheers, I don't like that

**Form:**

```
charade
  synonym('TA', src='cheers,')
  synonym('BOO', src='I')
  synonym('BOO', src="don't")
  synonym('BOO', src='like')
```

**Verifier:**
  X assembly               tree does not produce 'TABOO'
  X bridge.leaves          synonym 'BOO' not in DB for 'i'; synonym 'BOO' not in DB for "don't"; synonym 'BOO' not in DB for 'like'
  + bridge.indicators      
  + residue                

### 31150 23across - `SIDEBOARD`

> Cupboard love accepted by sad bride in a mess

**Form:**

```
anagram ['mess']
  literal('LOVE', src='love')
  literal('SAD', src='sad')
  literal('BRIDE', src='bride')
```

**Verifier:**
  X assembly               tree does not produce 'SIDEBOARD'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['accepted']

### 31150 1down - `WASTAGE`

> Wife with a leg that's showing degeneration

**Form:**

```
charade
  synonym('W', src='Wife')
  raw('A', src='a')
  synonym('STAGE', src='leg')
```

**Verifier:**
  + assembly               tree produces WASTAGE
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['showing', "that's"]

### 31150 3down - `FEIGN`

> Fake news primarily follows Spain plugging fruit

**Form:**

```
container ['plugging']
  synonym('N', src='news')
  charade
    abbreviation('E', src='Spain')
    synonym('FIG', src='fruit')
```

**Verifier:**
  X assembly               tree does not produce 'FEIGN'
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['follows', 'primarily']

**Flags:** container_charade_structure_not_preserved

### 31150 8down - `HANGDOG`

> Guilty-looking, hover over man's best friend

**Form:**

```
charade
  positional[last]('G', src='Guilty-looking,')
  synonym('HANG', src='hover')
  synonym('O', src='over')
  positional[last]('D', src='friend')
```

**Verifier:**
  X assembly               tree does not produce 'HANGDOG'
  + bridge.leaves          all 4 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['best']; non-link words declared as link_words: ["man's"]

### 31150 17down - `LATERAL`

> Not yet advanced student's kind of thinking

**Form:**

```
charade
  synonym('LATER', src='yet')
  synonym('A', src='advanced')
  abbreviation('L', src="student's")
```

**Verifier:**
  + assembly               tree produces LATERAL
  + bridge.leaves          all 3 leaves verified
  + bridge.indicators      
  X residue                non-link words declared as link_words: ['not']

### 31150 21down - `OLDSTER`

> Veteran undersold sterling pounds

**Form:**

```
hidden
  literal('UNDERSOLDSTERLINGPOUNDS', src='undersold sterling pounds')
```

**Verifier:**
  + assembly               tree produces OLDSTER
  + bridge.leaves          all 1 leaves verified
  X bridge.indicators      hidden node has no indicator
  + residue                

**Flags:** hidden_indicator_not_tagged_by_solver, op_inferred_from_token_mix_not_provided, op_name_not_preserved_in_solveresult

### 31150 24down - `BRACE`

> Cleared out bar propped up by one or two

**Form:**

```
charade
  positional[outer]('BR', src='bar')
  synonym('ACE', src='one')
```

**Verifier:**
  + assembly               tree produces BRACE
  + bridge.leaves          all 2 leaves verified
  + bridge.indicators      
  X residue                unaccounted: ['propped']; non-link words declared as link_words: ['cleared']

## NO_FORM - solve_clue returned nothing (46)

These are clues the production solver itself can't crack - grammar_triage + catalog matchers + Haiku-enrichment all failed. Listed for awareness; widening solver coverage is the next plan.

| Puzzle | Clue # | Answer | Clue text |
|---|---|---|---|
| 31132 | 12a | LENGTH | Stretch hats gingerly, every so often twisting |
| 31132 | 13a | SCENARIO | Dramatic scheme reduced smell near a South American port |
| 31132 | 15a | direct current | One emerges from cell to achieve what Canute couldn't? |
| 31132 | 22a | PEDESTAL | Parisian is trapped by cycle stand |
| 31132 | 27a | ECONOMIST | Delving into energy price, fancy I'm no financial expert |
| 31132 | 29a | CHESSMEN | Duchess mentions cuddling more than one bishop? |
| 31132 | 3d | DRASTIC | Severe Republican given fizzy drink in Washington |
| 31132 | 6d | INTENSE | How green bottles come close to blue, deep? |
| 31132 | 7d | NUMERATOR | Man with router breaking one in half |
| 31132 | 10d | RECORDER | One gets blown about in both directions, gripping rope |
| 31132 | 14d | ACTIVATE | Switch on aircon and television around one, brewing tea |
| 31132 | 19d | RESERVE | Book suggests what to do following a fault? |
| 31132 | 21d | SPROUT | French author promoting small green vegetable |
| 31132 | 24d | IDIOM | Fool wasting time over minute piece of cake, say |
| 31138 | 12a | INSINCERE | Lying at home after Sunday school class? |
| 31138 | 13a | SALSA | Charlie recalled hosting naked gala dance |
| 31138 | 14a | CAVORT | Victor interrupting frisky actor's frolic |
| 31138 | 15a | PINOCLE | Card game oddly calmer after a lot of wine |
| 31138 | 22a | CACHE | Adult and child inspired by church store |
| 31138 | 24a | INTERPRET | Read Pinter after cycling on front of tandem |
| 31138 | 25a | black hole | Liberal blocking second delivery of total budget shortfall? |
| 31138 | 26a | PADRE | Father quietly covering chapter in trained group |
| 31138 | 27a | EMPIRE | Politician welcomed by island conglomerate |
| 31138 | 2d | head start | Early advantage of promotion above street in centre |
| 31138 | 4d | NEAREST | Attention held by shelter easiest to reach? |
| 31138 | 6d | Stars and Stripes | Flag pests around beach rubbish on top of shingle |
| 31138 | 7d | Basel | Loyal voters left European city |
| 31138 | 9d | UPKEEP | Running ahead before spy returned |
| 31150 | 5a | FOOLISH | Trawl around gents, perhaps on reflection it's unwise |
| 31150 | 9a | SUMMITS | Tops Mike repeatedly included in outfits |
| 31150 | 10a | OVERRUN | Blackburn, for one, without wingers manage attack |
| 31150 | 13a | EERIE | Belgium's out of ale, that is frightening |
| 31150 | 15a | easy-going | Tolerant of one wearing medal after each case of superiority |
| 31150 | 25a | RETINUE | Entourage regret taking European money |
| 31150 | 26a | LEAFLET | Flyer in open grassy area left abandoned |
| 31150 | 27a | LENIENT | Clement Attlee's term embraced by old red back in parliament |
| 31150 | 28a | SLENDER | Slight slur making American butt of joke |
| 31150 | 4d | RESILIENT | Hardy novel in Leicester church banned |
| 31150 | 5d | FROWN | Refs regularly put up with individual's dirty look |
| 31150 | 6d | OVERSIGHT | Exhale deeply in public showing boob |
| 31150 | 7d | Israeli | National statesman of old, but no leading character |
| 31150 | 14d | ENDURANCE | Runner's strength running nude dash around North |
| 31150 | 16d | SCHEDULES | Diaries and school uniform in Leeds damaged |
| 31150 | 18d | SMITTEN | Fond of warm clothing getting recycled |
| 31150 | 20d | BRAWLED | Scrapped plot to pinch green lobbyist's crown |
| 31150 | 23d | SCENT | Broadcast picked up in Cologne, say |
