"""Cordelia's recorded explanation of ONE clue, for the clue page's Listen button.

WHERE THE AUDIO COMES FROM. Nothing is synthesised for the website. The nightly
puzzle video already records her reading every clue in one take
(scripts/narrate_prose.build_track), and scripts/export_clue_audio.py cuts each
clue's stretch out of that take into data/clue_audio/<clue_id>.mp3. The Listen
button costs no ElevenLabs credits (user, 2026-09-29: "We need cordelia's voice").

NOT web/static. The deploy copies web/static whole with scp and a 60-second
budget on every code deploy (dashboard/pages/deploy.py); a few MB of audio per
puzzle, every day, would break that. This folder is synced on its own,
incrementally, like the grid JSONs.

STALE AUDIO IS NEVER PLAYED. Each mp3 has a <clue_id>.json beside it holding the
hash of the words she read. If the prose is edited after filming, the words on
the page and the words in the recording differ, and the button disappears
rather than read out something the page no longer says. body_hash() is the ONE
definition of "the words", used by the exporter and by the page alike.
"""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = ROOT / "data" / "clue_audio"


def body_text(approved, note):
    """The explanation she reads, as narrate_prose.clue_text_for builds it:
    the approved (sentence, gloss) when there is one, otherwise the author's
    comment (an INVALID or a clue type). Empty string when neither exists."""
    if approved:
        sentence, gloss = approved
        return "\n\n".join(p for p in ((sentence or "").strip(),
                                       (gloss or "").strip()) if p)
    return (note or "").strip()


def body_hash(approved, note):
    text = body_text(approved, note)
    return hashlib.sha1(text.encode("utf-8")).hexdigest() if text else ""


def mp3_path(clue_id):
    return AUDIO_DIR / ("%d.mp3" % int(clue_id))


def meta_path(clue_id):
    return AUDIO_DIR / ("%d.json" % int(clue_id))


def write_meta(clue_id, digest, **extra):
    meta_path(clue_id).write_text(
        json.dumps(dict(extra, clue_id=int(clue_id), body_hash=digest), indent=1),
        encoding="utf-8")


def is_current(clue_id, approved, note):
    """True only when a recording exists AND it says what the page says now."""
    try:
        if not mp3_path(clue_id).exists():
            return False
        meta = json.loads(meta_path(clue_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    want = body_hash(approved, note)
    return bool(want) and meta.get("body_hash") == want
