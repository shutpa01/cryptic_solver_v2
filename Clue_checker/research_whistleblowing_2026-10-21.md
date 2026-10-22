# Whistleblowing: research (2026-10-21)

**The aim (user, 2026-10-21):** design our own process from scratch, for one specific purpose: whistleblowing. The product is ultra-secure, ultra-anonymous two-way reporting. This replaces the Incogneato comparison (`research_incogneato_2026-10-21.md`) as the basis for the design.

**How this was produced:**
- Three research agents ran in parallel using web search and fetch: the law, the competitors, and buyers plus security. I merged their findings here.
- **[U]** means unverified: it came from a secondary source or a search summary.
- **[3P]** means a third-party figure, from a review site, an aggregator or an outside researcher.
- [Inference] means my own reasoning, not a sourced fact.
- None of this is legal advice.
- Several primary sources didn't load: EUR-Lex, Légifrance, the CNIL pages, the NAVEX PDFs, and the UK consultation PDF.

---

## 1. Summary

1. **The law creates the demand.**
   - In the EU, every private firm with 50 or more workers must run an internal reporting channel. The channel must acknowledge a report within 7 days and give feedback within 3 months.
   - Firms may outsource the channel to a third party.
   - The UK has no general duty yet. A government consultation running from 12 Oct 2026 to 8 Jan 2027 proposes one for employers with 50+ staff [U].
   - Japan requires a channel at 300+ workers.
   - US listed companies need an anonymous route to the audit committee (SOX §301).
2. **Volume is tiny.**
   - The median is 1.57 reports per 100 employees a year; Europe is 0.67.
   - So a 50–250 person firm gets about 0.3–4 reports a year.
   - In Europe 65% of reports are anonymous.
   - The product is low-volume and high-stakes. Running costs scale with the number of customers, not the number of reports. In a small firm, the details in a report are what give the reporter away.
3. **The SME end of the market is crowded and cheap.**
   - Trusty charges a flat €49–99 a month.
   - Whistlelink charges €79–299 a month by headcount.
   - Formalize charges €99–349 a month.
   - NAVEX starts at $1,980 a year for up to 250 staff.
   - Safecall charges about £1,540 a year.
   - No commercial product has a permanent free tier.
4. **Whistleblowers get caught through their work devices and networks, the details in their report, and management hunting for them. Broken encryption is rarely the cause.**
   - Barclays' chief executive, 2016: fined £642,430 for trying to identify a whistleblower.
   - Reality Winner: identified as one of six people who printed a document, and the only one who had emailed from a work machine.
   - A New Jersey hotline case: the employer could have worked out who reported from the report's details.
5. **No competitor fills five gaps** (§4). These are our opening.
6. **Biggest risk: trust, not technology.**
   - Buyers expect ISO 27001: about $10–60k and 3–6 months in year one [3P].
   - They also expect a data processing agreement, EU hosting and a list of sub-processors.
   - Italy's data regulator has fined whistleblowing software vendors directly, not only their customers.
   - The cheapest credible trust signal is open code plus a published independent audit.

---

## 2. The law and what it requires of the product

### EU Directive 2019/1937 (text from legislation.gov.uk/eudr/2019/1937)

**Who needs a channel (Art. 8):**
- Private firms with 50 or more workers.
- Firms in financial services, anti-money-laundering, transport safety and environment, whatever their size.
- All public bodies. Countries may exempt municipalities under 10,000 people and public bodies under 50 workers.

**Outsourcing and sharing:**
- The channel may be "provided externally by a third party" (Art. 8(5)).
- Firms with 50–249 workers may share resources (Art. 8(6)).
- The Commission's 2021 letters say a group-wide channel alone doesn't satisfy the law: each entity with 50+ workers needs its own [U].

**How the procedure must work (Art. 9):**
- The channel must be run "in a secure manner" that protects identity.
- Acknowledge receipt within 7 days.
- An impartial person or department is designated to handle reports.
- Diligent follow-up, including of anonymous reports where national law provides for them.
- Feedback within 3 months.
- Clear information on how to report externally to regulators.
- Reports in writing and/or orally (phone or voice message), plus a physical meeting on request.

**Anonymous reports (Art. 6(2)):** each country decides whether they must be accepted.
- Spain: must be accepted and processed (Ley 2/2023 art. 7.3).
- Germany: no duty, though the channel "should" handle them (HinSchG §16).
- Ireland: no duty, but employers must tell staff whether they accept them.
- France: accepted [U].
- Portugal and Belgium (250+ staff): must be accepted [U].
- Netherlands, Italy, Denmark, Sweden: not required, or not addressed [U].

**Confidentiality (Art. 16):**
- It covers the reporter's identity *and anything from which it could be deduced*.
- It can be overridden by a "necessary and proportionate" legal obligation, for example in court proceedings. So it isn't absolute.

**Records (Art. 18):**
- Keep a record of every report.
- Calls or voice messages may be recorded or transcribed only with consent. The reporter can check, correct and sign the transcript.

**Retention:**
- Germany: delete 3 years after the case closes.
- Spain: delete intake data after 3 months if no investigation starts; 10 years at most.

**Penalties for having no channel:**
- Germany: up to €20,000.
- Spain: €600,001 to €1,000,000 for a company.

**Enforcement and review:**
- On 6 Mar 2025 the EU Court of Justice fined five countries for implementing the Directive late, Germany reportedly €34m [U].
- The Commission's evaluation of the Directive is due in 2026.

### UK

- **Current law:** PIDA 1998 protects whistleblowers but doesn't require a channel.
- **Sexual harassment:** from 6 Apr 2026 it counts as a protected disclosure (Employment Rights Act 2025 s.23).
- **Consultation 12 Oct 2026 – 8 Jan 2027** (law-firm summaries, [U]):
  - a possible legal duty to have "appropriate and effective procedures", with civil penalties;
  - scope: private employers with 50+ staff, and public bodies;
  - possibly phased, 250+ staff first;
  - a proposed reward scheme run by the Serious Fraud Office.
  - [Inference] If this becomes law, UK demand opens up in the same way EU demand did in 2021–23.
- **Where a channel is already mandatory:**
  - Financial firms under the FCA's rules (SYSC 18): anonymous reports must be handled, a third party may run the channel, and the board gets an annual report.
  - The NHS, under the national Freedom to Speak Up policy.
- **ECCTA 2023, failure to prevent fraud** (in force 1 Sep 2025):
  - Applies to firms meeting two of: more than 250 staff, £36m turnover, £18m in assets.
  - The Home Office guidance §3.5.2 says these firms "should have appropriate whistleblowing arrangements" with independent channels.
  - A channel is evidence of reasonable procedures, not a guaranteed defence.

### US

- **SOX §301:** audit committees of listed companies must provide for "confidential, anonymous submission" of concerns.
- **SEC Rule 21F-17:** nothing may discourage reporting to the SEC. Seven companies settled in Sept 2024. So our wording must never deter external reporting.
- **DOJ compliance guidance (Sept 2024)** asks: "Does the company have an anonymous reporting mechanism…?" [U, secondary].
- **DOJ whistleblower awards pilot:** a company that self-reports within 120 days of an internal report can still get a declination. Fast handling of internal reports therefore has real value.

### Other markets

- **Australia:** public and large private companies must have a whistleblower policy; not having one is a strict-liability offence.
- **Japan:**
  - Employers with more than 300 workers need a system and a designated handler.
  - A 2025 amendment adds criminal penalties and bans attempts to identify whistleblowers. It takes effect about Dec 2026 [U].
- **Brazil:** companies that must have a CIPA committee must run an anonymous harassment reporting channel.
- **Switzerland:** no mandate.

### GDPR and data protection

- **Lawful basis:** legal obligation where a channel is mandatory, otherwise legitimate interest [inference].
- **DPIA:** France's regulator (CNIL) lists whistleblowing systems as requiring a DPIA. Its 2023 guidance binds outside providers to no other use of the data, to retention limits, and to returning or destroying the data at the end of the contract.
- **Person reported on:** telling them (GDPR Art. 14) can be delayed if it would seriously impair the investigation.
- **🚨 Italian regulator enforcement (Garante):**
  - An airport operator was fined €40k and its *software vendor €20k separately*. Firewall logs recorded employees' access to the app, there was no HTTPS, and sub-processors weren't under contract.
  - A hospital and its hosting provider were fined €40k in total, because hosting logs made users identifiable.
  - [secondary; Garante docweb 9768363 / 9768387]
  - **Lesson: the vendor's own logs are a regulatory liability.**
- **No regulator guidance found** from the UK ICO on whistleblowing tools, or on cross-border transfers specific to whistleblowing.

### ISO 37002:2021

- It is a guidelines standard ("should"), built on trust, impartiality and protection.
- It covers the buyer's whole process, so a vendor can only claim "alignment".
- The clause-level text is paywalled and wasn't seen.

### Vendor liability

- **As a GDPR processor:** liable where we breach a processor's own duties or act outside the customer's instructions (Art. 82(2)). Regulators can fine us directly (the Garante cases above).
- **Court orders:** none found against whistleblowing vendors.
- **A vendor that holds no keys and no identifying logs:** can only hand over what it has. That's a practical defence, not a legal privilege [inference — needs legal advice].

---

## 3. Competitors

| Product | Price | How the reporter follows up | Can the vendor read reports? | Notes |
|---|---|---|---|---|
| **GlobaLeaks** (open source) | Free to self-host. Hosted versions: WhistleblowingIT is free for Italian public bodies; private-sector price not published | 16-digit receipt (the reporter's key is derived from it) | Close to no. Keys are per user and password-protected, but generated on the server, and admins can switch on escrow | Tor onion service, about 5% file padding, 90-day default retention. A 2014 audit found the receipt guessable and a timing leak; fix status unknown |
| **SecureDrop** (journalism) | Free, needs dedicated hardware | 7-word codename, cannot be reset | No. The key sits on an air-gapped machine | The design reference. Tor only; assumes the server can be seized; repeated public audits (2024, 2026) |
| **Hush Line** | Free (nonprofit) | Optional anonymous email | No (PGP to the recipient's key) | Text only. Strips IP and country from logs. Audit funded by the Open Technology Fund |
| **NAVEX EthicsPoint / WhistleB** | From $1,980/yr (≤250 staff, US) | Report key + password; if lost, file a new report | Not documented | Market leader; owns WhistleB and Expolink (Expolink closed 2021) |
| **EQS Integrity Line** | From €129/mo [3P] | Not checked | Claims it "can at no time access" reports (PGP); key custody not stated | ISO 27001, EU hosting. AI transcribes voice reports and anonymises them. Reviews: "complicated to master" |
| **FaceUp** | Platform fee by headcount; $49–169/mo [3P] | 16-digit case key | Optional E2E tied to members' passwords | No IP logging, metadata stripped, voice altered. **States its form makes "no calls to external sites"**. 113 languages, an app |
| **Whistlelink** | €79–299/mo by headcount | Not checked | Yes (encrypted at rest only) | No IP logging, metadata removed, EU, ISO 27001. 30-day trial |
| **Formalize** (Whistleblower Software) | Core €99–349/mo; Advanced €149–529/mo | Not checked | Only if the customer exports the key; lose it and the data is gone | **[3P, June 2026]:** a researcher alleges attachments are handled unencrypted, and a US analytics service receives reporters' IP addresses. No vendor reply found |
| **Trusty Report** | Flat €49 / €69 / €99 a month, any headcount | Not checked | No encryption claim | The price floor. 7-day trial |
| **Hintbox** (Germany) | Basic from €99, Premium from €149 | Not checked | Claims E2E | ISO 27001 |
| **Whispli** | About $379–396/mo, 3-year contracts [3P] | Random username | AES-256 (at rest) | IP "deleted every single time" |
| **Safecall** (UK) | £1,540 + VAT a year, ≤250 staff | Not checked | Not checked | Phone hotline |
| **SpeakUp** | From about €3,000 [3P] | Not checked | Not checked | Voice: transcribed word for word, audio never passed to the client |
| **Vault / OneTrust / AllVoices / Got Ethics / Syntrio** | Quote only, or $5k+ [3P] | Not checked | Not checked | Enterprise end |

**Price ranges per year (Whistlelink, Formalize and Trusty from vendor pages; NAVEX and Safecall as their own lines):**

| Company size | Range |
|---|---|
| 50 staff | €588–1,788 |
| 250 staff | €828–4,548; NAVEX $1,980; Safecall about £1,850 |
| 1,000 staff | Mostly by quote; probably $5k–15k+ [inference] |

**Free:** no commercial product has a permanent free tier. The free options are open source (GlobaLeaks, SecureDrop, Hush Line), and each needs self-hosting or is narrow in what it does.

---

## 4. Gaps nobody fills: our opening

1. **Encryption the vendor can't read, plus a humane way to recover a lost key.**
   - Today you get one of three things: the vendor can read reports (most products); a lost key means the data is gone (Formalize export, SecureDrop); or admins can read everything (GlobaLeaks escrow).
   - **Nobody offers threshold recovery,** for example any 2 of 3 named officers together.
2. **Reporter safety at the moment of reporting.**
   - No commercial product warns against using a work laptop, the office network or a managed phone. Even SecureDrop says it cannot protect someone on a work computer.
   - No commercial SaaS offers Tor or onion access.
3. **Discreet access.**
   - Every product uses a branded reporting URL.
   - None offers a quick-exit button or protection against traces in browser history.
4. **Cleaning attachments on the reporter's own device.**
   - "Metadata stripping" usually means the server handles the unencrypted file.
   - Doing it in the browser *before* encryption is unclaimed. Offer an opt-out, because the original file can be evidence.
5. **A reporting page with no third-party code, and a way to verify that.**
   - Only FaceUp states it.
   - Browser encryption is only trustworthy if the served code can be checked: published, reproducible builds and an audit. Only the open-source tools publish audits.
6. **Warnings about identifying details while the reporter writes** (my addition).
   - In a 50-person firm, a detail like "the meeting on Tuesday" can name the reporter.
   - Nobody coaches reporters while they write. This is where an AI check could help without seeing anything *after* submission. It would have to run without sending the report anywhere; how is an open question.
7. **Telling the reporter safely that a reply is waiting.**
   - Everyone relies on the reporter checking back.
   - About 33% of anonymous reporters follow up [U, NAVEX 2022].
   - No documented design alerts them without a contact channel.

---

## 5. Tensions we must resolve in the design

| Tension | Detail |
|---|---|
| **Device-held ticket (Ask Us) vs work devices** | A ticket stored in the browser is evidence on a work machine. A memorised case key leaves no trace but is easily lost. Possible answer: memorised key by default, plus the threshold recovery described in §4. |
| **Ask Us runs on Cloudflare (a US company)** | Customers expect an EU-controlled operator; US law (the CLOUD Act) can compel US companies. Cloudflare's edge sees the reporter's IP even if we log nothing. End-to-end encryption protects the content but not that metadata. **Hosting choice must be re-decided** [inference]. |
| **Anonymity vs records and voice** | The law wants records and allows oral reports. Voice distortion doesn't remove vocabulary or speaking style. Transcribing and deleting the audio (as SpeakUp does) is the safer pattern. |
| **Stripping metadata vs keeping evidence** | GlobaLeaks deliberately doesn't strip, because metadata is evidence. Strip by default and let the reporter opt out. |
| **"We hold nothing" vs the legal duties** | Customers must keep records and meet retention rules. We can store encrypted blobs that only the customer can open, so "we can't read it" holds even though we store it. |
| **AI vs ultra-secure** | Sending reports to an outside AI is what whistleblowers fear. Drop the after-submission AI summary and sentiment from §5 of the Incogneato notes. Any AI help must run before encryption, on the reporter's side, and be optional. |
| **A free tier vs the cost of trust** | Paying customers expect ISO 27001 ($10–60k in year one) plus audits. The free tier itself is cheap to run because volume is tiny. The fixed cost is proving we're trustworthy. |
| **Our pricing frame** | The Incogneato plan (free = their $49 tier, top tier = $99) doesn't map onto this market, where SME channels cost €49–300 a month. **Pricing needs re-deciding for whistleblowing.** |

---

## 6. Requirements: the brief for our own design

### What the law requires

1. Acknowledge within 7 days and give feedback within 3 months. Timers alert the handler. (Ireland: further feedback every 3 months on written request.)
2. Anonymous two-way messaging. The law expects reporters to receive the acknowledgement and feedback.
3. Written and oral intake (voice message), plus a way to request a meeting. Recordings only with consent; the reporter can check and sign the transcript; delete the audio once transcribed.
4. Designated handler role(s). Access limited to them.
5. Anonymous intake as a per-customer switch: compulsory for Spain, and Portugal [U]. Tell staff the customer's policy.
6. No IP, device or firewall logging that could identify a reporter. HTTPS everywhere.
7. Retention rules per customer, with automatic deletion: Germany 3 years after closure; Spain a 3-month intake purge and a 10-year cap.
8. A record of every report, and an audit trail of handler actions that contains no reporter identity.
9. A separate channel for each legal entity, with an optional shared mode for 50–249 worker firms.
10. Information on external reporting routes, and no wording that discourages contacting regulators (SEC 21F-17).
11. A processor package for customers: data processing agreement, sub-processor list, DPIA template, return or destroy at contract end.
12. An annual report to the board with no identities (FCA rules, ECCTA guidance). For SOX customers, routing to the audit committee.
13. Tools to delay telling the person reported on, where the law allows.

### Security and anonymity (ranked)

1. Encrypt in the reporter's browser, with a key per report wrapped for each handler. No vendor decryption path and no escrow. Threshold recovery held by the customer's own officers.
2. Served code that can be verified: reproducible builds, published hashes, a Code-Verify-style check, open-source client and crypto.
3. Steer reporters away from work devices and networks. Offer an onion address and discreet access options.
4. No identifying logs at any layer, including the hosting and CDN provider.
5. A high-entropy case key with rate limiting, and honest wording about what happens if it's lost.
6. Strip attachment metadata in the browser before encryption, with an opt-out. Warn that printer tracking dots survive.
7. Coach the reporter about identifying details while they write.
8. EU hosting with an EU-controlled operator. Publish the sub-processor list.
9. A published independent audit (the Open Technology Fund has funded these). ISO 27001 for procurement. A warrant canary.
10. Accessibility: plain language and multiple languages. Voice reports transcribed, with the audio deleted.

### The host's dashboard

- Case inbox with deadline timers, states (received → acknowledged → investigating → feedback given → closed), handler assignment and notes.
- Two-way messages with the reporter.
- Retention countdown.
- Board report.
- Report volume is tiny, so the dashboard is about getting each case right and on time, not handling volume.

---

## 7. Open questions for the user

1. **Market first:** EU SMEs, where there's a mandate now? The UK, where a duty may come in 2027? Or both?
2. **Hosting:** stay on Cloudflare Workers, as Ask Us does, or move to an EU-controlled host?
3. **Pricing:** rethink it for this market (§5).
4. **Trust route:** open-source and a published audit first, then ISO 27001? Budget and timing?
5. **Legal advice:** needed on vendor liability and court orders before launch.

---

## Main sources

**Law**
- EU Directive: https://www.legislation.gov.uk/eudr/2019/1937/article/8 (and articles 6, 9, 16, 17, 18, 23, 26)
- Spain: https://www.boe.es/buscar/act.php?id=BOE-A-2023-4513
- Germany: https://www.gesetze-im-internet.de/hinschg/__16.html
- UK consultation: https://www.bclplaw.com/en-US/events-insights-news/tuning-the-whistle-uk-government-consults-on-reforming-whistleblowing.html
- FCA SYSC 18: https://www.handbook.fca.org.uk/handbook/SYSC/18/3.html
- ECCTA guidance: https://www.gov.uk/government/publications/offence-of-failure-to-prevent-fraud-introduced-by-eccta/economic-crime-and-corporate-transparency-act-2023-guidance-to-organisations-on-the-offence-of-failure-to-prevent-fraud-accessible-version
- SOX: https://www.law.cornell.edu/uscode/text/15/78j-1
- SEC 21F-17: https://corpgov.law.harvard.edu/2024/10/02/settled-actions-reiterate-secs-broad-interpretation-of-rule-21f-17a-whistleblower-impediments/
- DOJ pilot: https://www.justice.gov/criminal/criminal-division-corporate-whistleblower-awards-pilot-program
- CNIL: https://cnil.fr/fr/la-cnil-met-jour-son-referentiel-alertes-professionnelles
- Garante: https://lentepubblica.it/pa-digitale/whistleblowing-garante-sanziona-datore-lavoro-fornitore-applicativo/

**Volume and buyers**
- NAVEX benchmark: https://www.navex.com/en-us/northstar/whistleblowing-statistics-benchmark-report-2025/
- NAVEX regional: https://www.navex.com/en-us/company/press-room/regional-whistleblowing-benchmark-report-highlights-rise-in-anonymous-reports/
- Elker buyer's guide: https://elker.com/articles/whistleblowing-software
- ISO 27001 cost: https://consilien.com/news/iso-27001-certification-cost-timeline

**Unmasking cases**
- Barclays: https://www.bankofengland.co.uk/news/2018/may/fca-and-pra-jointly-fine-mr-james-staley-and-announce-special-requirements-at-barclays
- Reality Winner: https://www.cbsnews.com/amp/news/nsa-leaker-reality-winner-secret-printer
- New Jersey case: https://www.njemploymentlawfirmblog.com/court-reinstates-employees-retaliation-lawsuit-finding-employer-could-have-identified-her-as-anonymous-whistleblower/

**Security**
- GlobaLeaks: https://docs.globaleaks.org/en/devel/technical/security/encryption-protocol.html and https://docs.globaleaks.org/en/stable/technical/security/threat-model.html
- SecureDrop: https://docs.securedrop.org/en/stable/what_is_securedrop.html
- Hush Line audit: https://www.opentech.fund/security-safety-audits/hush-line-security-audit/
- Browser E2E weakness: https://cispa.de/en/research/publications/213253-trust-on-reload-securing-browser-based-end-to-end-encryption
- Code Verify: https://engineering.fb.com/2022/03/10/security/code-verify/

**Competitors**
- Whistlelink: https://whistlelink.com/pricing/
- Formalize: https://whistleblowersoftware.com/en/prices
- Trusty: https://trusty.report/plans/
- FaceUp: https://www.faceup.com/en/security
- NAVEX: https://www.navex.com/en-us/campaigns/epim-pricing-smb-dg/
- Safecall: https://safecall.co.uk/safecall-price-calculator
- Formalize allegation: https://seclists.org/fulldisclosure/2026/Jul/16
