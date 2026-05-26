"""Regression test for the clue-page WFW display contract.

This test protects the agreed page shape. It does not assert that a clue is
solved; it asserts that review-state WFW evidence is rendered through the fixed
WFW card, with every clue word visible and admin controls kept secondary.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web import create_app
from web.db import get_db
from web.routes.clue import generate_clue_slug


def _clue_slug(db, source, puzzle_number, clue_number, direction):
    row = db.execute(
        """SELECT id, clue_text
           FROM clues
           WHERE source = ?
             AND puzzle_number = ?
             AND clue_number = ?
             AND direction = ?
           LIMIT 1""",
        (source, str(puzzle_number), str(clue_number), direction),
    ).fetchone()
    assert row is not None
    return row["id"], generate_clue_slug(
        row["clue_text"] or "", clue_id=row["id"])


def _clue_slug_by_id(db, clue_id):
    row = db.execute(
        "SELECT id, clue_text FROM clues WHERE id = ?",
        (clue_id,),
    ).fetchone()
    assert row is not None
    return row["id"], generate_clue_slug(
        row["clue_text"] or "", clue_id=row["id"])


def _render(client, clue_id, slug):
    response = client.get("/clue/%s?admin=dev-admin-key" % slug)
    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "WFW needs review" in body or "How the clue works" in body
    assert '<p class="text-sm text-gray-500 mb-1">Answer</p>' not in body
    assert "Stage Three proof gate" not in body
    assert '<details class="mt-3" open>' not in body
    assert "Admin: word-role overrides" not in body
    assert "Admin: WFW word roles" in body
    assert "Admin: add WFW data" in body
    if "Admin: WFW evidence gaps" in body:
        assert '<details class="mt-4" open>' not in body
    assert "/admin/edit/%s" % clue_id in body
    assert "Re-verify" in body
    return body


def run_tests():
    app = create_app("development")
    with app.app_context():
        db = get_db()
        uncages_id, uncages_slug = _clue_slug(
            db, "dailymail", 17884, 1, "across")
        case_slugs = [
            _clue_slug_by_id(db, clue_id)
            for clue_id in (10068257, 10068258, 10068259, 10068260)
        ]

    with app.test_client() as client:
        uncages = _render(client, uncages_id, uncages_slug)
        for word in (
            "Releases", "new", "actor", "Nicolas", "in", "America",
        ):
            assert word in uncages
        for clue_id, slug in case_slugs:
            body = _render(client, clue_id, slug)
            assert "WFW needs review" in body or "How the clue works" in body

    print("Clue WFW render contract passed")
    return True


if __name__ == "__main__":
    ok = run_tests()
    sys.exit(0 if ok else 1)
