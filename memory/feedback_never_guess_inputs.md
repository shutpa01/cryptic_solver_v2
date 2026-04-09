---
name: Never guess when user input doesn't match
description: If user references a clue/file/entity that doesn't exist, say so immediately — never silently substitute a guess
type: feedback
---

When the user specifies something (clue number, file, column, etc.) and it doesn't exist, report that it doesn't exist and stop. Do NOT silently pick the closest match and carry on as if that's what they meant.

**Why:** User has repeatedly stressed no guessing. In this case, asked about "5a Guardian 29976" which doesn't exist — I silently used 5 down instead. This wastes the user's time and erodes trust.

**How to apply:** Any time a lookup returns no result for what was specified, say "X doesn't exist" and ask for clarification. Never assume intent.
