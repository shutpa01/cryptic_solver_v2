# Beating Incogneato: research (2026-10-21)

**The aim:** be better than Incogneato at every tier, using an improved version of our Ask Us model.

- **Free tier:** matches Incogneato's lowest paid tier.
- **Top tier:** includes every feature, costs the same as their middle tier, and has no usage limits.
- **Both sides:** a far better experience for the person giving feedback, and a much better dashboard for the host (the organisation running the box).

**Sources:**
- The user's trial box `iD6do` (dashboard at whoanswered.me, public page at ansr.me), inspected in Chrome on 2026-10-21 without saving any changes.
- incognea.to pages.
- Review sites. These returned 403 errors when fetched, so their content comes from search summaries.

## 1. Their tiers (seen in the account page of the trial)

| Tier | Monthly | Annual | What it adds |
|---|---|---|---|
| Personal | ? | $16.58/mo (Capterra, unverified) | 1 box, unlimited feedback, website widget, real-time chat, 24/5 support |
| Professional | $49 | $41.58/mo ($499/yr) | 25 boxes, 4 questions per box, custom URL (path on ansr.me only), logo and button colour, conversation codes, Feedback Tone, Feedback Lens (AI), password or IP restriction, require email, unlimited box managers, 7 languages, labels and assignment |
| Premier | $99 | $83.25/mo ($999/yr) | unlimited boxes, 25 questions per box, all 33 templates, voting and discussion, box admins, SLA, priority support |
| Enterprise | $599+/mo | — | SAML SSO, SLAs, onboarding, invoice or PO, custom development |

**Notes:**
- **Personal is hidden.** The account page says "It looks like you're using a work email" and shows only Professional and Premier, even though the Professional card reads "Personal Plan +". The public pricing page starts at Professional.
- **No free plan.** There is a 14-day trial; this one expires 11/4/2026. Trial accounts are deleted 14 days after the trial ends (privacy page).
- **What the trial unlocks:** almost nothing. Everything under "Additional Box Settings" says "Professional Plan or above required": language, Tone, conversation code, password, IP restriction and require email. So do logo, button colours and confirmation text.

**DECIDED (user, 2026-10-21): Our free tier matches Professional ($49).**
- That covers 25 boxes, conversation codes (we use the device-held ticket instead), sentiment, the AI summary, custom URL and logo, password or IP restriction, unlimited managers and 7 languages.
- **Cost to work out:** the AI summary and sentiment cost us money for every free box.

## 2. The respondent's side (what the person giving feedback sees)

**How it works now:**
- One text box, an optional email field and a Send button.
- A loading screen reads "Anonymizing and securing your session…".
- **To get a reply:**
  - The respondent either gives an email address (Incogneato relays the reply), or keeps a conversation code (Professional and up), copies or downloads it, and types it back in later.
  - Lose the code and the conversation is gone.
  - Live chat works only while both sides are online.

**Privacy problems found:**
- **Google reCAPTCHA runs on the anonymous page.**
  - Checked in Chrome: `grecaptcha` is loaded, the badge is shown, and requests go to `www.google.com/recaptcha/api2`.
  - So Google receives the respondent's IP address and browser details, and their Google cookies if they're signed in.
  - The page also loads Google Fonts and jQuery from third-party servers.
- **The privacy policy contradicts the FAQ.**
  - The policy says they may collect "browser name, IP address", operating system and ISP, calling it "non-personal" information.
  - The FAQ says "No other respondent information is stored".
- **Encryption:** only the email address is said to be encrypted at rest. Nothing is said about responses.
- **No details on:**
  - Retention period.
  - Hosting region.
  - DPA (data processing agreement).
  - List of subprocessors.

**What reviewers complain about:**
- **Small teams lose anonymity:** "easy to guess who submitted each comment", and people "recognize writing styles".
- **No attachments.**

## 3. The host's side (the dashboard, as seen)

- **Layout:** a feedback list with a date filter, "Search Within" and a CSV export. Clicking an entry opens a reply pane and a Delete button.
- **AI and sentiment:**
  - Feedback Lens is in beta. It needs at least 10 responses, covers the latest 50 and refreshes once a day.
  - Feedback Tone scores words on 5 levels. Their own page admits it misreads mixed wording.
- **Alerts:** two switches only: "New Response Alerts (all boxes)" and "Weekly Summary Email (this box)".
- **Box setup:** one long page holding the nickname, colours, logo, title, intro, questions, tags, voting, time zone, the settings above, managers and admins, the share URL, the website widget code, a QR code and a printable flyer.
- **Team roles:** box managers can only view and reply. Box admins (Premier only) control everything. To reset a manager's password, you delete them and add them again.
- **Integrations:**
  - No API.
  - No mobile app.
  - No Teams integration.
  - Slack only by adding Slack's email-to-channel address as a box manager.
- **Missing workflow:**
  - No case status, so no way to show what was done about feedback (a reviewer asked for this).
  - No spam or abuse tooling documented.
- **Reviews:** "UI feels somewhat outdated", "UI is a little basic", and the overall dashboard is weak.
- Capterra 4.8 (26 reviews), G2 4.7 (23 reviews).

## 4. Where we can be better, tier by tier

### Free: matches their paid entry tier
- **Nothing for the respondent to keep:** no code, no email. The browser holds the ticket (the Ask Us model) and the reply waits there.
  - *Improvement needed:* a lost device means a lost ticket. Offer an optional "save to another device" link. It's a choice, not a requirement.
- **No Google on the anonymous page:** no reCAPTCHA, no third-party fonts or scripts. Fight spam with our own rate limiting or a proof-of-work check.
- **Truly unlimited:** responses, and conversations in both directions.
- **Includes the features they lock in the trial:** language, require email, password or IP restriction.

### Mid tier (their $49 to $99): everything, including Enterprise
- **Unlimited:** boxes, questions and team members.
- **Single sign-on** (SAML SSO) and an audit log of which team member did what.
- **Native integrations:** Slack, Teams, a webhook and an API.
- **Case status that respondents can see:** received → being looked at → done, plus "what we did" on a public board. This closes the loop reviewers asked for.
- **Writing-style protection:** an optional rewrite of the respondent's message before it reaches the host, so writing style doesn't give them away. This is the top anonymity complaint.
  - Pre-filled lists or questions where possible.
- **Attachments**, with metadata such as EXIF stripped.
- **AI summary from the first response:** no 10-response minimum and no daily wait.
- **Sentiment and topic tagging** done with a model rather than word lists.
- **A clear privacy statement:**
  - What is stored (nothing about the person, only the message and the ticket).
  - Retention period.
  - Hosting region.
  - DPA (data processing agreement).

### Host dashboard
- An inbox you can act on: unread, assigned, status, tags and keyboard shortcuts.
- Works on a phone.
- Replies from the dashboard or by email.
- Setup in steps instead of one long page.
- Resetting a team member's password doesn't mean deleting them.

## 5. Cost of the AI summary and sentiment on the free tier (2026-10-21)

**Prices** (Claude API, per million tokens, from the claude-api skill's model table dated 2026-10-06):

| Model | Input | Output |
|---|---|---|
| Haiku 5.5 | $0.10 | $0.50 |
| Sonnet 5.5 | $2 | $10 |

- The Batch API costs 50% less and returns results within 24 hours. That suits a daily summary.

**Assumptions** (mine; we have no real feedback data yet):
- A summary covers the latest 50 entries, as Incogneato's does. At about 150 tokens per entry plus about 500 tokens of instructions, that's about **8,000 tokens in**.
- About **400 tokens out**, doubled to 800 to allow for the model's thinking.
- A box's summary refreshes at most once a day, and only on a day that brings new feedback.
- Sentiment is one call per message: about 350 tokens in and about 50 out, including thinking.

**Cost per call:**

| Call | Haiku 5.5 | Haiku 5.5, batch | Sonnet 5.5 | Sonnet 5.5, batch |
|---|---|---|---|---|
| One summary | $0.0012 | $0.0006 | $0.024 | $0.012 |
| Sentiment, 1,000 messages | $0.06 | $0.03 | $1.20 | $0.60 |

**Cost per free account per month:**

| Case | Haiku, batch | Sonnet, batch |
|---|---|---|
| Typical: 1 active box, new feedback on 10 days | $0.006 | $0.12 |
| Worst case: all 25 boxes get feedback every day (750 summaries) | $0.45 | $9 |

**What this means:**
- **The summary cost is capped by design.** The window is fixed at 50 entries and refreshes at most once a day per box, so "unlimited responses" doesn't raise it. The worst case per free account is 25 boxes × 30 days.
- **Sentiment does grow with message volume**, so spam is the risk there. On Haiku, 100,000 spam messages cost about $3–6, so rate limiting matters more than the model bill.
- **On Haiku, the AI summary is close to free:** under half a dollar per account per month even in the worst case.
- **Sonnet is about 20 times the price.** Use it only if Haiku's summaries turn out not good enough.
- **Haiku's quality is untested on real feedback.** Test it on a sample before choosing.
- **Not covered here:** the writing-style rewrite (§4). It runs on every message before it reaches the host, so it can't use the batch discount. It needs its own costing.

## 6. Unknowns and risks

- **Prices:** the $99 Premier price and the Personal $16.58/mo figure are from their pages and Capterra. I haven't checked them against a checkout.
- **Our costs, not yet calculated:**
  - AI summaries and sentiment on a free tier.
  - SSO on our top tier.
  - The writing-style rewrite: Claude API calls cost money on every message.
- **Spam:** an anonymous box with no CAPTCHA and no email is a spam target. It's the hardest part (already noted in `research_2026-10-21.md` §2).
- **Liability:** whistleblowing and harassment reports may legally need a hosting region, retention policy and DPA. This needs real legal checking, not my reading.

## Sources
- https://www.incognea.to/ (pricing block)
- https://www.incognea.to/anonymous-suggestion-box/
- https://www.incognea.to/conversation-codes/
- https://www.incognea.to/how-does-anonymous-reply-work/
- https://www.incognea.to/anonymous-feedback-tone/
- https://www.incognea.to/how-to-view-your-ai-feedback-summary/
- https://www.incognea.to/privacy/
- https://www.incognea.to/faq/
- https://www.incognea.to/gdpr/
- https://www.incognea.to/compare/
- https://www.incognea.to/how-to-receive-feedback-notifications-in-slack/
- https://www.capterra.com/p/152578/Incogneato/pricing/
- https://www.capterra.com/p/152578/Incogneato/reviews/
- https://www.g2.com/products/incogneato/reviews
- https://www.softwareadvice.com/customer-satisfaction/incogneato-profile/reviews/
- Trial box screens: whoanswered.me/iD6do, /manage/iD6do, /account/iD6do; ansr.me/iD6do
