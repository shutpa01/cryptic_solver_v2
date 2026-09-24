"""The clue page's prose block — one definition of where drafts live and what a
record looks like.

    {"10094480": {"sentence": "Net means EARN, and picked up tells you it sounds
                               like ERNE.",
                  "gloss":    "A sea eagle, especially the white tailed variety.",
                  "answer":   "ERNE",
                  "approved": false}}

Written unapproved by `scripts/draft_prose.py`, ticked on /hs, and served only when
`approved` is true. It is a JSON file rather than a table because a draft is local
working state, not a record of the puzzle: `logs/*` is gitignored and the DBs are
not touched.

WHY THIS MODULE EXISTS AT ALL. The drafter and the /hs tick both need the path, the
key and the record shape, and this session has twice been bitten by the same fact
living in two places — the clue-type label in two renderers, the homophone middle
read in one place and not another. One definition, imported by both.

APPROVAL IS A SEPARATE ACT FROM DRAFTING. `save_draft` never sets `approved`, and
`set_approved` never rewrites the text unless the user supplies it. So a re-run of
the drafter cannot silently un-approve something the user has already ticked, and
ticking cannot silently alter what was drafted.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROSE = ROOT / "logs" / "prose.json"


def load():
    """Every draft, keyed by clue id as a string. Missing/!unreadable file = {}."""
    try:
        data = json.loads(PROSE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save(data):
    PROSE.parent.mkdir(parents=True, exist_ok=True)
    PROSE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def get(clue_id, data=None):
    """One clue's record, or None. `data` lets a caller read the file once."""
    rec = (load() if data is None else data).get(str(clue_id))
    return rec if isinstance(rec, dict) else None


def save_draft(clue_id, sentence, gloss, answer="", facts_hash=""):
    """File a NEW draft, unapproved. Refuses to touch a record already approved —
    an overnight re-run must never undo the user's tick.

    `facts_hash` fingerprints the record the prose was written from, so a later
    commit can tell whether the reading actually changed. See `facts_unchanged`.
    """
    data = load()
    key = str(clue_id)
    if (data.get(key) or {}).get("approved"):
        return False
    data[key] = {"sentence": sentence, "gloss": gloss,
                 "answer": (answer or "").upper(), "approved": False,
                 "facts_hash": facts_hash}
    save(data)
    return True


def facts_unchanged(clue_id, facts_hash, data=None):
    """True when a draft exists and was written from EXACTLY these facts.

    This is what makes drafting at the end of the nightly the cheap option. The
    user accepts the vast majority of prefill readings unchanged, so by the time
    they commit, the prose sitting in the box was written from the same record
    they are committing — and re-drafting it would spend 8-11 seconds and a model
    call to produce the same paragraph. Only a reading the user CHANGED has a
    different fingerprint, and only that one is drafted again.
    """
    rec = get(clue_id, data)
    return bool(rec and facts_hash and rec.get("facts_hash") == facts_hash)


def set_approved(clue_id, approved, sentence=None, gloss=None):
    """Tick or untick, optionally keeping edits the user made in the box.

    Returns False when there is nothing filed for this clue — approving something
    that does not exist would put a key in the file that no draft ever wrote.
    """
    data = load()
    key = str(clue_id)
    rec = data.get(key)
    if not isinstance(rec, dict):
        return False
    if sentence is not None:
        rec["sentence"] = sentence.strip()
    if gloss is not None:
        rec["gloss"] = gloss.strip()
    rec["approved"] = bool(approved)
    data[key] = rec
    save(data)
    return True


def approved_text(clue_id, data=None):
    """(sentence, gloss) when the user has ticked it, else None.

    The serving path calls THIS and nothing else, so an unapproved draft cannot
    reach a page by any route.
    """
    rec = get(clue_id, data)
    if not rec or not rec.get("approved"):
        return None
    return (rec.get("sentence") or "", rec.get("gloss") or "")
