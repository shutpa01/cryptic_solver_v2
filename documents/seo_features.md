# SEO Features — Cryptic Crossword Hints

## Strategy

Individual clue pages as the primary SEO asset — 500k+ clues, each with a unique URL, targeting long-tail search queries like "companions shredded corset cryptic clue". Nobody else has individual clue pages WITH quality wordplay explanations.

---

## 1. Individual Clue Pages

**URL format**: `/clue/{slug}` where slug = `clue-text-words-ANSWER`
- Example: `/clue/companions-shredded-corset-ESCORT`
- Uppercase answer segment for visual separation
- `?id=N` parameter for disambiguation when multiple clues share a slug

**Content per page**:
- Clue text with enumeration
- Progressive hint buttons (Definition → Wordplay Type → Explanation → Answer)
- Source info: publication, puzzle number, date
- Clickable word helper (indicators, synonyms, abbreviations)
- "Also seen in" section — other puzzle appearances of the same clue
- Breadcrumb navigation

---

## 2. Structured Data (JSON-LD)

### FAQPage Schema
Drives FAQ rich results in Google. Three questions generated per clue:
1. "What does the cryptic crossword clue '[text]' mean?"
2. "What is the answer to '[text]'?"
3. "What type of wordplay is used in '[text]'?"

Answers include definition, wordplay type (human-friendly label), step-by-step guidance, and the answer itself.

### BreadcrumbList Schema
Drives breadcrumb display in SERPs:
`Home > Times Cryptic > #27492 > Clue text...`

---

## 3. Meta Tags

### Clue Pages
- **Title**: `"Companions shredded corset (6) — Cryptic Crossword Clue"`
- **Meta description** (dynamic, 150-160 chars): includes clue text, enumeration, source, puzzle number, available hints
- **Open Graph**: `og:title`, `og:description`, `og:type: article`
- **Canonical URL**: self-referencing

### Puzzle Pages
- **Title**: `"Times Cryptic #27492"`
- Generic meta description from base template

### Base Template
- `charset`, `viewport`, generic description
- Canonical URL on every page

---

## 4. Sitemaps

### Sitemap Index (`/sitemap.xml`)
References multiple sub-sitemaps:

### Clue Sitemaps (`/sitemap-clues-N.xml`)
- Paginated at 50,000 URLs per file
- All four sources: Guardian (252k), Telegraph (156k), Times (67k), Independent (32k) — all 99-100% answer coverage
- Includes `<lastmod>` (publication date), `<changefreq>` (monthly)
- Deduplicated by slug

### Puzzle Sitemaps (`/sitemap-puzzles.xml`)
- All puzzle pages: `/{source}/{type_slug}/{puzzle_number}`
- Supports "DT 31180" style searches

---

## 5. Robots.txt

```
User-agent: *
Allow: /
Disallow: /admin/
Disallow: /reveal
Disallow: /explain
Sitemap: /sitemap.xml
```

---

## 6. Crawlability

- **No orphan pages**: every clue reachable via sitemap, puzzle page links, breadcrumbs, and "also seen in"
- **Server-side rendered**: all content in HTML, no JS-only rendering
- **HTMX progressive enhancement**: hints load via HTMX but pages work without JS
- **Pagination**: puzzle lists and sitemaps paginated for scalability
- **Direct URL access**: all routes work as direct links, no single-page-app routing

---

## 7. Solver Tools (User Engagement)

These tools increase session time and repeat visits (positive ranking signals):

1. **Clickable clue words** — click any word to see indicators, synonyms, abbreviations
2. **Pattern finder** — enter known letters, find matching words
3. **Anagram solver** — click words to build letter chips, find anagrams
4. **Similar clue search** — click clue number to find similarly-worded clues from 500k database
5. **Interactive solve mode** — solve puzzles with crossing letters, grid progress, auto pattern matching

---

## 8. Content Quality Signals

- **508k clues with answers** — comprehensive coverage
- **Verified explanations** — mechanical verifier scores explanations, only HIGH and MEDIUM shown to users
- **Multiple sources**: Times, Telegraph, Guardian, Independent
- **Progressive hints**: users choose how much help they want
- **"Also seen in"**: cross-references between puzzles show depth of content

---

## 9. Honeypot Measurement Site

Separate site (clairesclues.xyz) to measure long-tail search traffic before investing further:
- 504k clue pages (answer + definition only)
- Google Search Console verified, 11 sitemaps submitted
- Measures: impressions, clicks, which clues get searched, old vs recent traffic

---

## 10. Not Yet Implemented

- `og:image` (social sharing image)
- Twitter Card tags
- Analytics (Google Analytics or similar)
- AMP pages
- Per-page keyword meta tags
- Schema for solver tools
- `og:url` explicit tag
