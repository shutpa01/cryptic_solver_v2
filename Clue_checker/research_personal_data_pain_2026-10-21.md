# Where holding personal data hurts small organisations (research, 2026-10-21)

**The question:** which small organisations want to talk with their clients but are put off by the complications of holding personal data? Ask Us (no name, no email; the client's device holds a ticket and the reply waits there) is the answer we are testing.

**How this was done:**
- Three web research strands: the law, kinds of organisation, and what they use now.
- Many points come from search summaries rather than full pages. Labels:
  - **[S]**: secondary source only.
  - **[I]**: inference.
  - No label: a source is linked in the Sources section.
- Whistleblowing has been dropped (`research_whistleblowing_2026-10-21.md` is kept for reference only).

## 1. Headlines

1. **"We hold no personal data" is not true for the host.**
   - The host can usually work out who a client is from what they write, so the messages count as personal data, and often sensitive data such as health.
   - A ticket the host can link to a person counts as a pseudonym, not as anonymity.
   - **[I] What we can honestly sell:** *less to hold, less to lose*. We take away the ways things usually go wrong. We can't take away the law.
2. **The most common breach is human error, not hacking.** Most often it is data emailed to the wrong person or a CC instead of BCC.
   - The data protection regulator (ICO) fined the charities HIV Scotland (£10k) and Mermaids (£25k) for exactly this.
   - Ask Us has no email addresses, so this mistake cannot happen.
3. **Small organisations drift onto WhatsApp and volunteers' personal phones.** Sector guidance warns against both for vulnerable people.
   - "Safer than WhatsApp" is a message they will understand.
4. **The position is unclaimed.** No product found says "talk to your customers without holding their personal data".
   - The device-held ticket exists only in whistleblowing and HR tools (Incogneato, FaceUp, WhistleblowerSoftware) and in tools for schools (Whisper, Tootoot).
   - Nothing like it serves small organisations whose clients are adults.
5. **Proof the model works:** NHS ChatHealth lets 11–19 year olds "text a School Nurse anonymously", with replies within one working day. That is almost exactly our model, as a public service.

## 2. What holding personal data costs a small UK organisation

- **ICO fee** (from 17 Feb 2025) [S]:
  - £52 for micro organisations (≤10 staff or under £632k turnover), and for all charities.
  - £78 for organisations up to 250 staff.
  - Some are exempt. The exemptions list was not verified for 2026.
- **Standing duties under UK GDPR:**
  - A privacy notice.
  - A lawful basis for holding the data.
  - Records of what is held. [I] Small organisations are exempt only if the processing is occasional, low-risk and involves no sensitive data, so client messaging probably doesn't qualify.
  - A retention policy.
  - A risk assessment (DPIA) for high-risk processing.
  - Reporting a breach to the ICO within 72 hours.
- **Sensitive data** (health, sex life, religion and similar) needs an extra legal condition. It usually also needs a written "appropriate policy document" [S].
- **Children's data:** the ICO treats children as vulnerable and recommends a DPIA [S].
- **Subject access requests** (people asking for a copy of their data) must be answered within 1 month.
- **Data (Use and Access) Act 2025:**
  - Only a "reasonable and proportionate" search is needed [S].
  - The rules on subject access requests and time limits changed on 5 Feb 2026.
  - **Every organisation must have a complaints procedure** from 19 Jun 2026.
- **[I] What Ask Us genuinely reduces:**
  - Breach impact.
  - Retention exposure, through automatic deletion.
  - The search needed to answer a subject access request. The ticket also proves who is asking.
  - No cookies or third-party scripts, so no cookie banner is needed for our page.
- **What remains:** the lawful basis, a privacy notice, security, breach reporting and answering subject access requests.

### Our own position as supplier
- **The cautious view:** we are a **processor** (a supplier handling data for the host) and need a data processing agreement with each host.
  - **IP addresses (checked in the Ask Us code, `trainer-proof/support/worker.js`):**
    - Our code reads the IP only to limit each IP to 5 questions a minute (`worker.js:60`). Cloudflare's rate limiter holds that count briefly; the IP is never written to storage, logs or email.
    - Cloudflare passes the IP through to deliver each request, which no server can avoid. That counts as brief processing, not holding.
    - So "we never record your IP address" is accurate.
  - **Messages:** the client's question is emailed to the host's inbox (`worker.js:87`). It is held by the host's email provider, not by us. On our side we keep only the host's answers, for 60 days.
- **Holding only encrypted data the host alone can read:** the ICO describes this set-up but doesn't say whether it puts the supplier outside the law.
  - That guidance is "under review" because of the 2025 Act.
  - In the EU, the EDPS v SRB judgment (CJEU, Sep 2025) supports "data you can't decrypt isn't personal data for you".
  - The same judgment says opinions "relate to" their author, so the host is still bound.
  - **[I]** This is untested in the UK.

## 3. Evidence of pain

- **Breach reports by sector** [S]: health 17% and education 14% (2023). Charity reports doubled in two years to 148 (2017/18, dated). No recent charity figure was found.
- **Fines and reprimands of small organisations:**
  - DPP Law: £60k (2025). No two-factor login, client data published on the dark web, reported late.
  - Birthlink, a charity: £18k (2025). About 4,800 records destroyed.
  - Staines Health Group, a GP surgery: reprimand. A dying patient's records sent to an insurer; no written procedure.
  - A primary school: reprimand (2023).
  - Mermaids: £25k (2021).
- **Breaches and attacks** (Cyber Security Breaches Survey 2025/26) [S]: 28% of charities, 42% of micro businesses, 46% of small businesses.
- **Cost:** 7 hours a month, about £1,075 a year (FSB survey, 2018, dated) [S]. No survey since 2024 of the burden on small charities was found.

## 4. Who feels it most (ranked: data pain × wish to talk to clients × client wish to stay anonymous × ability to pay)

| # | Kind of organisation | Why | Weakness |
|---|---|---|---|
| 1 | **Small charities whose clients are vulnerable** (sexual health and HIV, LGBTQ+, addiction, domestic abuse) | Strongest evidence: ICO fines for the email failures we remove; privacy is the top barrier to addiction treatment; 46% fear judgement; volunteers use personal phones | Little money (grant-funded) [I]; may need a route through funders or local support bodies such as CVAs and NAVCA |
| 2 | **Private sexual health, independent pharmacies, online clinics:** "ask before you book" | About 1 in 4 delay seeing a GP out of embarrassment (Bupa 2019, Essity 2021); ChatHealth proves the model; about 10.4k pharmacies in England | Data pain is inferred, not shown; NHS buyers are slow and ChatHealth already serves them |
| 3 | **Private therapists and counsellors**, for first enquiries | UKCP and BACP guidance worries about texts (a partner might see them) and WhatsApp; 18k+ in BACP's directory; they already pay for practice software | Ongoing clients are known by name, so this fits **first contact** and privacy from family only |
| 4 | Staff feedback in small firms | 37% say speaking up "isn't worth the personal risk" (Culture Shift 2026) | In a team of 8, writing style identifies people; the market is crowded |
| — | Schools and youth clubs | Biggest share of breaches (15%, 2022) | Already served by Whisper (£350–450/yr) and Tootoot (about £400–500/yr) |
| — | Shops and trades | GDPR feels "overkill" | Customers don't want anonymity; shops want email addresses for marketing |

**[I] Design notes from this:**
- **Domestic abuse:** a ticket on a phone the abuser checks is a risk. We would need a quick-exit button and a disguised name for the ticket.
- **Therapists:** the job is the anonymous **enquiry before booking**. Practice software (WriteUpp, Cliniko, Zanda, at £16–50/mo) covers known clients but not this.

## 5. What they use now, and price anchors

| Group | Examples | Price | Data |
|---|---|---|---|
| Free chat | Tawk.to | Free | Profiles visitors (IP, location, pages); needs a cookie decision |
| Mainstream support | Crisp, Tidio, Intercom, Zendesk, Help Scout | About $19–95/mo | Email-based; Zendesk makes even "anonymous" requests verify an email |
| Forms | Google Forms, Jotform, Typeform | Free to $39/mo (Jotform HIPAA $99+) | **One way only, no reply channel** |
| Messaging | WhatsApp Business, Messenger | Free | Phone numbers, Meta; WhatsApp uploads the contact list (UK blog, not the ICO) [S] |
| Privacy-first | GhostChat, Hush Line, Signal, Proton | Free to $9/mo; Proton €6.99/user | GhostChat still stores chats; Hush Line needs the sender to have an account for ongoing chat |
| Anonymous two-way | Incogneato, FaceUp, WhistleblowerSoftware | About $49–349/mo, billed annually | Ticket or code model; sold for HR and compliance |
| Schools | Whisper, Tootoot | £350–500/yr | Anonymous reporting for pupils |

**[I] Pricing gap:**
- Free tools take data.
- Anonymous two-way tools cost about £40–100+/mo on annual contracts.
- A monthly plan under £10, with no annual contract, would undercut the whole anonymous two-way field.

## 6. Open questions for the user

1. **Which group first?** Vulnerable-client charities (strongest pain, little money), private sexual health and pharmacy (strongest anonymity, slower buyers), or therapists' first enquiries (can pay, narrower fit)?
2. **Pitch wording:** "less to hold, less to lose" and "safer than WhatsApp", not "no GDPR". Agree?
3. **Encryption only the host can read:** it strengthens our position as supplier, but if the host loses the key, the messages are lost. Do we want it?
4. **Price:** a monthly plan under £10? A free tier?
5. **A legal check** on the processor question before we make any claim on the website.

## Sources
- UKCP security and confidentiality guidelines: https://psychotherapy.org.uk/media/1ptnp1jt/ukcp-security-and-confidentiality-guidelines-2018-june2024.pdf
- BACP GPiA 065: https://www.bacp.co.uk/media/20401/bacp-confidentiality-and-record-keeping-crp-gpia065-jan24.pdf
- HIV Scotland fine: https://www.civilsociety.co.uk/news/hiv-charity-fined-10-000-over-data-breach.html
- Mermaids fine: https://www.wrigleys.co.uk/news/charity-social-economy/ico-fines-charity-f25000-for-failing-to-secure-personal-data
- ICO domestic abuse warning, 2023: https://ico.org.uk/about-the-ico/media-centre/news-and-blogs/2023/09/data-breaches-put-domestic-abuse-victims-lives-at-risk-uk-information-commissioner-warns/
- Charity breach types: https://www.civilsociety.co.uk/news/charities-report-125-data-breaches-to-the-ico-in-the-first-quarter-of-2019-20.html
- Northumberland CVA messaging guide: https://northumberlandcva.org.uk/wp-content/uploads/2024/12/A-‘How-to-Guide-on-using-Messaging-apps-template-agreement.pdf
- Addiction stigma: https://takingactiononaddiction.org.uk/news/almost-half-of-brits-impacted-by-addiction-yet-stigma-continues · https://www.researchsquare.com/article/rs-10538433
- Refuge tech abuse: https://refuge.org.uk/news/refuge-reveals-rise-in-tech-abuse-including-spycam-surveillance-ahead-of-uns-16-days-of-activism/
- Embarrassment surveys: https://healthcare-management.uk/half-brits-embarrassed-seek-medical-help-survey · https://www.djsresearch.co.uk/HealthMarketResearchInsightsAndFindings/article/Embarrassing-symptoms-cause-1-in-4-people-to-delay-visiting-GP-survey-finds-04286
- ChatHealth: https://www.wchc.nhs.uk/children-young-people/young-people/chathealth/
- Pharmacy count: https://www.nhsbsa.nhs.uk/index%2ephp/statistical-collections/general-pharmaceutical-services-england/general-pharmaceutical-services-england-201516-202425
- Charity count: https://www.gov.uk/government/publications/charity-commission-casework-and-registrations-data-2025-to-2026/casework-and-registrations-data-october-to-december-2025
- Culture Shift 2026: https://culture-shift.co.uk/resources/workplace/research-uk-employees-are-scared-to-speak-up-about-bad-behaviour-in-the-workplace
- ICO fee: https://www.lewissilkin.com/insights/2025/02/06/fee-payable-to-the-ico-to-increase-in-february-102jzb9 · https://ico.org.uk/for-organisations/data-protection-fee/
- DUAA commencement: https://www.legislation.gov.uk/uksi/2026/82/made
- ICO identifiers: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/personal-information-what-is-it/what-is-personal-data/what-are-identifiers-and-related-factors/
- ICO anonymisation guidance, 2025: https://www.twobirds.com/en/insights/2025/key-takeaways-from-the-ico’s-final-anonymisation-guidance
- ICO encryption scenarios: https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/security/encryption/encryption-scenarios
- EDPS v SRB: https://www.cliffordchance.com/insights/resources/blogs/talking-tech/en/articles/2025/09/pseudonymized-data-after-edps-v-srb.html
- Fines: DPP Law https://www.legalfutures.co.uk/latest-news/ico-fines-law-firm-60000-after-dark-web-publishes-client-data · Birthlink https://cy.ico.org.uk/action-weve-taken/enforcement/2025/07/birthlink · Staines https://www.pulsetoday.co.uk/news/regulation/gps-urged-to-review-data-protection-processes-as-practice-given-reprimand-by-ico/
- Breach survey: https://www.nibusinessinfo.co.uk/content/cyber-security-breaches-survey-202526
- Sector breach shares: https://beyondencryption.com/blog/data-security-an-analysis-of-the-latest-ico-findings
- Competitors: https://swgfl.org.uk/products/whisper · https://edtechimpact.com/products/tootoot/ · https://hushline.app · https://www.incognea.to/how-does-anonymous-reply-work/ · https://support.zendesk.com/hc/en-us/articles/10360295453082 · https://writeupp.com/pricing · https://www.consentstack.io/vendors/tawk-to
