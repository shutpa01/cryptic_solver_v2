---
name: Always commit JS fixes immediately
description: JS fixes in puzzle.js and base.html have been lost multiple times - commit after every single change
type: feedback
---

JS fixes in puzzle.js, base.html, and helper templates have been lost or re-fixed multiple times across sessions. The pattern: a fix is made, tested, works — but then gets reverted by a git checkout, overwritten by another change, or simply not committed.

**Why:** User has experienced this repeatedly and it wastes significant time.

**How to apply:** After ANY change to a JS file or template, commit IMMEDIATELY before doing anything else. Do not batch JS fixes. Do not make multiple JS changes before committing. One fix, one commit, verify with `git log`.
