"""Live enrichment-queue store for the Haiku definition fallback.

This is the ONE place the fallback touches a database, and it touches only the
existing human-review tables in clues_master.db:

  - pending_enrichments   — the queue you verify by your current process. A Haiku
                            definition lands here as type='definition', word=the
                            full (grammar-corrected) phrase, letters=answer.
  - rejected_enrichments  — definitions you have already rejected; never re-used.
  - definition_answers_augmented (cryptic_new.db) — NOT written here. A verified
                            definition reaches it only via the dashboard Accept
                            button (dashboard/pages/review.py:_add_definition).

So this store is the simulation boundary the redesign needs: the fallback queues,
you verify live, and an accepted definition becomes an ordinary DB hit on the
next run. The engines stay DB-decoupled — they receive an instance of this (or a
fake) injected, never importing it directly.
"""

import os
import sqlite3

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "clues_master.db")


class PendingStore:
    """Thin wrapper over the existing enrichment-queue tables. One short-lived
    connection per call — no open handle is held between solves."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DEFAULT_DB

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=30)

    def pending_definition(self, answer):
        """The phrase already queued as the definition for this answer, if any
        (so a re-run reuses it instead of paying Haiku again). Most recent wins."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT word FROM pending_enrichments "
                "WHERE type='definition' AND letters=? "
                "ORDER BY id DESC LIMIT 1", (answer,)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def is_rejected_definition(self, phrase, answer):
        """True if (phrase -> answer) was already rejected by a reviewer, so the
        fallback must not present or re-queue it."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM rejected_enrichments "
                "WHERE type='definition' AND word=? AND letters=?",
                (phrase, answer)).fetchone()
        finally:
            conn.close()
        return row is not None

    def queue_definition(self, phrase, answer, clue_text,
                         source=None, puzzle_number=None):
        """Queue a definition for human verification. INSERT OR IGNORE on the
        table's UNIQUE(type, word, letters) — a duplicate is a silent no-op.
        Returns True if a new row was added."""
        return self._queue("definition", phrase, answer, answer, clue_text,
                           source, puzzle_number)

    # --- indicators (the hidden-indicator fallback) ---------------------------
    # An indicator row is type='indicator', word=the phrase, letters=the
    # wordplay type (e.g. 'hidden'); Accept -> _add_indicator inserts it into the
    # indicators table. Reuse/reject are keyed the same way.

    def pending_indicator(self, answer, wp_type="hidden"):
        """The indicator phrase already queued for this answer+type, if any."""
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT word FROM pending_enrichments "
                "WHERE type='indicator' AND letters=? AND answer=? "
                "ORDER BY id DESC LIMIT 1", (wp_type, answer)).fetchone()
        finally:
            conn.close()
        return row[0] if row else None

    def is_rejected_indicator(self, phrase, wp_type="hidden"):
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM rejected_enrichments "
                "WHERE type='indicator' AND word=? AND letters=?",
                (phrase, wp_type)).fetchone()
        finally:
            conn.close()
        return row is not None

    def queue_indicator(self, phrase, answer, clue_text, wp_type="hidden",
                        source=None, puzzle_number=None):
        return self._queue("indicator", phrase, wp_type, answer, clue_text,
                           source, puzzle_number)

    def _queue(self, etype, word, letters, answer, clue_text,
               source, puzzle_number):
        pn = None
        if puzzle_number is not None:
            try:
                pn = int(puzzle_number)
            except (TypeError, ValueError):
                pn = None
        conn = self._conn()
        try:
            cur = conn.execute(
                "INSERT OR IGNORE INTO pending_enrichments "
                "(type, word, letters, answer, clue_text, source, puzzle_number) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (etype, word, letters, answer, clue_text, source, pn))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
