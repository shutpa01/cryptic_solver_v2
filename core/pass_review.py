"""Demote an engine PASS to PENDING — the only write this module can make.

Built 2026-09-10, after STET (Times 29644 1a) was served as a pass on a derivation
nobody could defend: `is -> PET`, from a synonym row harvested off a blog. It broke
no mechanical rule. Every clue word was accounted for, both indicator types were
bound, the container geometry was sound. What it could not survive was a reader.

No gate can make that judgement, and the reference DB cannot be cleaned into
safety: 1.35M synonym rows, 796k of them meshed from Merriam-Webster lists on the
assumption that every member of a sense list is a synonym of every other. It is
not — `bother -> meticulosity` and `dog -> wrench` are in there for the same
reason `is -> pet` was. Junk at that scale is a certainty, not a risk.

So the check happens at the point of USE, on the handful of claims an engine
actually made last night, by something that can read them.

THE SAFETY CASE IS THE ASYMMETRY. This module can move a verdict exactly one way:

    pass -> pending

Never fail -> pass. Never pending -> pass. Never pass -> fail. Its worst possible
failure is therefore extra review for the user — never a false claim on a page.
That is what makes it safe to run unattended.

FOUR REFUSALS, all enforced here rather than trusted to the caller:
  * not currently a `pass`               — nothing else may be moved
  * `manual` or `prefill`                — the user's own filing, and prefills are
                                           pending already; neither is engine output
  * not published TODAY                  — the nightly solves today's clues and the
                                           user publishes only after checking them,
                                           so today's date is the unpublished window.
                                           An older clue may be live; this must never
                                           reach one.
  * no reason given                      — a demotion the user cannot evaluate is
                                           just noise in their queue

The reason is appended to `wfw_notes`, which is what the /hs comment box reads, so
it appears where the user is already looking.
"""

import sqlite3
from datetime import date

from core import store

_NOT_ENGINE = ("manual", "prefill")


def demote_to_pending(conn, clue_id, reason, today=None):
    """Move one engine pass to pending, recording why. Returns a status string.

    Refuses — and changes nothing — unless every condition above holds. The caller
    gets a sentence saying which one failed, so a refusal is legible in the log.
    """
    reason = (reason or "").strip()
    if not reason:
        return "REFUSED %s: no reason given." % clue_id
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT s.status, s.solved_by, c.publication_date "
        "FROM wfw_solve s JOIN clues c ON c.id = s.clue_id WHERE s.clue_id = ?",
        (clue_id,)).fetchone()
    if row is None:
        return "REFUSED %s: no stored solve." % clue_id
    if row["status"] != "pass":
        return ("REFUSED %s: status is %r, not 'pass' — this may only demote a pass."
                % (clue_id, row["status"]))
    if (row["solved_by"] or "") in _NOT_ENGINE:
        return ("REFUSED %s: solved_by=%r is not engine output."
                % (clue_id, row["solved_by"]))
    pub = (row["publication_date"] or "")[:10]
    if pub != (today or date.today().isoformat()):
        return ("REFUSED %s: published %s, not today — only today's unpublished "
                "clues may be touched." % (clue_id, pub or "(unknown)"))

    store.set_status(conn, clue_id, "pending")
    prev = conn.execute("SELECT note FROM wfw_notes WHERE clue_id = ?",
                        (clue_id,)).fetchone()
    note = "Held for review (engine pass, automated check): %s" % reason
    if prev and (prev["note"] or "").strip():
        note = prev["note"].rstrip() + "\n" + note
        conn.execute("UPDATE wfw_notes SET note = ? WHERE clue_id = ?", (note, clue_id))
    else:
        conn.execute("INSERT INTO wfw_notes (clue_id, note) VALUES (?, ?)",
                     (clue_id, note))
    conn.commit()
    return "DEMOTED %s to pending: %s" % (clue_id, reason)
