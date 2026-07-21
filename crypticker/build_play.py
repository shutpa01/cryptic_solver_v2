"""Inline a day's JSON into the play template -> a self-contained playable page.

    python -m crypticker.build_play            # uses days/sample_day.json
    python -m crypticker.build_play <day.json>

Opens with file:// (no server, no fetch) so the basics prototype just runs.
"""
import json, os, sys

HERE = os.path.dirname(__file__)


def build(day_path, out_path):
    with open(day_path, encoding="utf-8") as f:
        day = json.load(f)
    with open(os.path.join(HERE, "play_template.html"), encoding="utf-8") as f:
        tmpl = f.read()
    html = tmpl.replace("__PUZZLE_JSON__", json.dumps(day, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote", out_path)
    print("open it in a browser (file://) to play %d clues" % len(day["clues"]))


if __name__ == "__main__":
    day_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "days", "sample_day.json")
    build(day_path, os.path.join(HERE, "play.html"))
