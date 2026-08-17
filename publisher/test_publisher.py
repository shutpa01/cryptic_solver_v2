"""Tests for the publisher widget.

Run: .venv\\Scripts\\python.exe -m unittest publisher.test_publisher -v

Stdlib unittest, so nothing new has to be installed to run them.

These cover the things that are expensive to get wrong: puzzles must not leak
into each other, solutions must not reach the browser, the match count must
never invent a red zero, and a full entry must never become a free Check.
"""

import json
import unittest

from publisher import create_app
from publisher.puzzles import build_model

SOLVED = "93933"        # Toughie 3740 — answers published
EMBARGOED = "84276"     # Prize Cryptic 31187 — answers not published
EXPLAINED = "90402"     # Cryptic 31268 — every clue has a WFW pass


class PublisherTests(unittest.TestCase):

    def setUp(self):
        app = create_app("development")
        app.config["RATE_LIMIT_ENABLED"] = False
        self.app = app
        self.client = app.test_client()

    # --- helpers ---------------------------------------------------------

    def embed(self, number):
        response = self.client.get(f"/embed/telegraph/{number}?k=demo")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        token = json.loads(html.split("token:", 1)[1].split(",\n", 1)[0].strip())
        model = json.loads(
            html.split('type="application/json">', 1)[1].split("</script>", 1)[0])
        return token, model, html

    def post(self, path, body, token):
        return self.client.post(
            path, json=body, headers={"Authorization": "Bearer " + token})

    @staticmethod
    def entry_of(model, entry_id):
        return next(e for e in model["entries"] if e["id"] == entry_id)

    # --- the model -------------------------------------------------------

    def test_puzzle_number_survives_the_grid_scan(self):
        """Regression: the square-label variable used to shadow the argument.

        Not cosmetic — an empty number collapsed every puzzle onto one browser
        storage key and one server-side solution cache.
        """
        model, _ = build_model("telegraph", SOLVED)
        self.assertEqual(model["number"], SOLVED)

    def test_linked_clue_reads_in_link_order_not_grid_order(self):
        """24 Across (5,5,3) links back to 1 Across; EDGAR is at 24, not at 1."""
        model, solutions = build_model("telegraph", "92493")
        entry = self.entry_of(model, "a24")
        self.assertEqual(entry["len"], 13)
        self.assertEqual(solutions["a24"], "EDGARALLANPOE")

    def test_answer_is_dropped_rather_than_forced_to_fit(self):
        """An embargoed puzzle yields no solutions at all, not truncated guesses."""
        model, solutions = build_model("telegraph", EMBARGOED)
        self.assertEqual(solutions, {})
        real = [e for e in model["entries"] if not e.get("stub_of")]
        self.assertTrue(all(e.get("unverified") for e in real))

    # --- what reaches the browser ----------------------------------------

    def test_no_solution_letters_in_the_embed_page(self):
        _, _, html = self.embed(SOLVED)
        _, solutions = build_model("telegraph", SOLVED)
        flat = html.upper().replace(" ", "")
        leaked = [a for a in solutions.values() if len(a) >= 6 and a in flat]
        self.assertEqual(leaked, [])

    def test_unknown_key_and_unknown_puzzle_are_refused(self):
        self.assertEqual(
            self.client.get(f"/embed/telegraph/{SOLVED}?k=nope").status_code, 403)
        self.assertEqual(
            self.client.get("/embed/telegraph/99999?k=demo").status_code, 404)

    # --- the match count --------------------------------------------------

    def test_empty_entry_gets_no_count_not_a_red_zero(self):
        """The red zero means "your letters are wrong". An entry with nothing
        in it must produce no count at all, or every empty entry shows zero."""
        token, model, _ = self.embed(SOLVED)
        entry = self.entry_of(model, "a1")
        response = self.post("/api/match-counts",
                             {"patterns": {"a1": "?" * entry["len"]}}, token)
        self.assertIsNone(response.json["counts"]["a1"])

    def test_full_entry_is_refused_so_the_count_is_never_a_free_check(self):
        token, _, _ = self.embed(SOLVED)
        _, solutions = build_model("telegraph", SOLVED)
        response = self.post("/api/match-counts",
                             {"patterns": {"a1": solutions["a1"]}}, token)
        self.assertIsNone(response.json["counts"]["a1"])

    def test_wrong_letters_do_produce_a_zero(self):
        token, _, _ = self.embed(SOLVED)
        response = self.post("/api/match-counts",
                             {"patterns": {"a1": "ZQXJ???"}}, token)
        self.assertEqual(response.json["counts"]["a1"], {"n": 0, "capped": False})

    def test_partial_entry_counts_and_caps(self):
        token, _, _ = self.embed(SOLVED)
        counts = self.post("/api/match-counts",
                           {"patterns": {"a1": "WAR????"}}, token).json["counts"]["a1"]
        self.assertTrue(0 < counts["n"] <= 100)
        capped = self.post("/api/match-counts",
                           {"patterns": {"a1": "W??????"}}, token).json["counts"]["a1"]
        self.assertTrue(capped["capped"])

    # --- check and reveal -------------------------------------------------

    def test_check_marks_only_the_wrong_letters(self):
        token, model, _ = self.embed(SOLVED)
        _, solutions = build_model("telegraph", SOLVED)
        entry = self.entry_of(model, "a1")
        answer = solutions["a1"]
        letters = {f"{r},{c}": answer[i] for i, (r, c) in enumerate(entry["cells"])}
        self.assertEqual(
            self.post("/api/check", {"letters": letters, "scope": "grid"}, token).json,
            {"wrong": [], "unknown": []})

        first = f"{entry['cells'][0][0]},{entry['cells'][0][1]}"
        letters[first] = "Z" if answer[0] != "Z" else "Q"
        self.assertEqual(
            self.post("/api/check", {"letters": letters, "scope": "grid"},
                      token).json["wrong"], [first])

    def test_embargoed_puzzle_reports_uncheckable_rather_than_passing(self):
        """Silence would read as "correct". It must say it cannot check."""
        token, model, _ = self.embed(EMBARGOED)
        entry = next(e for e in model["entries"] if e["len"])
        letters = {f"{r},{c}": "A" for r, c in entry["cells"]}
        body = self.post("/api/check", {"letters": letters, "scope": "grid"}, token).json
        self.assertEqual(body["wrong"], [])
        self.assertEqual(len(body["unknown"]), len(letters))

    def test_puzzles_do_not_share_solutions(self):
        """Regression for the shadowed number: loading one puzzle and then
        another must not answer the second with the first one's letters."""
        token_solved, model_solved, _ = self.embed(SOLVED)
        _, solutions = build_model("telegraph", SOLVED)
        entry = self.entry_of(model_solved, "a1")
        letters = {f"{r},{c}": solutions["a1"][i]
                   for i, (r, c) in enumerate(entry["cells"])}
        self.post("/api/check", {"letters": letters, "scope": "grid"}, token_solved)

        # Same squares, different puzzle: every one must be uncheckable.
        token_embargo, _, _ = self.embed(EMBARGOED)
        body = self.post("/api/check", {"letters": letters, "scope": "grid"},
                         token_embargo).json
        self.assertEqual(body["wrong"], [])
        self.assertEqual(len(body["unknown"]), len(letters))

    def test_reveal_returns_the_answer(self):
        token, model, _ = self.embed(SOLVED)
        _, solutions = build_model("telegraph", SOLVED)
        entry = self.entry_of(model, "a1")
        body = self.post("/api/reveal", {"scope": "entry", "entry": "a1"}, token).json
        got = "".join(body["letters"][f"{r},{c}"] for r, c in entry["cells"])
        self.assertEqual(got, solutions["a1"])

    def test_reveal_on_an_embargoed_puzzle_returns_nothing(self):
        token, model, _ = self.embed(EMBARGOED)
        entry = next(e for e in model["entries"] if e["len"])
        body = self.post("/api/reveal", {"scope": "entry", "entry": entry["id"]},
                         token).json
        self.assertEqual(body["letters"], {})
        self.assertTrue(body["unavailable"])

    # --- auth -------------------------------------------------------------

    def test_api_requires_a_token(self):
        self.assertEqual(
            self.client.post("/api/match-counts", json={"patterns": {}}).status_code,
            401)
        self.assertEqual(
            self.client.post("/api/match-counts", json={"patterns": {}},
                             headers={"Authorization": "Bearer rubbish"}).status_code,
            403)

    def test_token_carries_its_own_puzzle_scope(self):
        """There is no puzzle parameter on the API to tamper with — the puzzle
        is resolved from the token, so a token cannot reach another puzzle."""
        from publisher.auth import read_token
        token, _, _ = self.embed(SOLVED)
        with self.app.app_context():
            payload = read_token(token)
        self.assertEqual(payload["s"], "telegraph")
        self.assertEqual(payload["n"], SOLVED)

    def test_renewed_token_still_works(self):
        token, _, _ = self.embed(SOLVED)
        renewed = self.client.post("/api/token/renew", json={"token": token}).json["token"]
        response = self.post("/api/match-counts", {"patterns": {"a1": "WAR????"}}, renewed)
        self.assertEqual(response.status_code, 200)


    # --- the four tools ---------------------------------------------------

    def test_lookup_expands_only_the_length_that_fits_the_entry(self):
        """We know the entry length, so the group that matters is not trimmed
        to five with no way to see the rest."""
        token, model, _ = self.embed(EXPLAINED)
        entry = self.entry_of(model, "d1")
        data = self.post("/api/tools/lookup",
                         {"word": "Spirit", "entry": "d1"}, token).json
        fitting = [g for g in data["meanings"] if g.get("fits")]
        self.assertEqual(len(fitting), 1)
        self.assertEqual(fitting[0]["length"], entry["len"])
        self.assertGreater(len(fitting[0]["words"]), 5)
        others = [g for g in data["meanings"] if not g.get("fits")]
        self.assertTrue(all(len(g["words"]) <= 5 for g in others))

    def test_synonym_searches_both_directions(self):
        token, _, _ = self.embed(EXPLAINED)
        data = self.post("/api/tools/synonym", {"word": "spirit"}, token).json
        self.assertTrue(data["synonyms"])
        self.assertNotIn("SPIRIT", data["synonyms"])   # never echo the query

    def test_pattern_list_agrees_with_the_match_count(self):
        """A count of 34 above a list of 12 would read as broken. They run
        against the same corpus, so they must not disagree."""
        token, _, _ = self.embed(SOLVED)
        pattern = "WAR????"
        listed = self.post("/api/tools/pattern",
                           {"pattern": pattern, "entry": "a1"}, token).json
        counted = self.post("/api/match-counts",
                            {"patterns": {"a1": pattern}}, token).json["counts"]["a1"]
        self.assertEqual(listed["total"], counted["n"])

    def test_anagram_finds_the_answer_and_never_echoes_the_fodder(self):
        token, _, _ = self.embed(EXPLAINED)
        data = self.post("/api/tools/anagram",
                         {"letters": "CIANATOAL", "entry": "a24"}, token).json
        self.assertIn("CATALONIA", data["matches"])
        self.assertNotIn("CIANATOAL", data["matches"])

    def test_anagram_is_narrowed_by_the_letters_already_in_the_grid(self):
        """Fodder from the clue, narrowed by the grid — the composition no
        paper offers. A pattern that cannot fit must exclude the answer."""
        token, _, _ = self.embed(EXPLAINED)
        loose = self.post("/api/tools/anagram",
                          {"letters": "CIANATOAL", "entry": "a24"}, token).json
        self.assertIn("CATALONIA", loose["matches"])
        blocked = self.post("/api/tools/anagram",
                            {"letters": "CIANATOAL", "entry": "a24",
                             "pattern": "Z????????"}, token).json
        self.assertNotIn("CATALONIA", blocked["matches"])

    # --- the hint ladder --------------------------------------------------

    def test_hints_come_one_rung_at_a_time(self):
        token, _, _ = self.embed(EXPLAINED)
        for step in ("definition", "clue_type", "answer", "explanation"):
            body = self.post("/api/hints", {"entry": "d1", "step": step}, token).json
            self.assertEqual(body["step"], step)
            self.assertIsNotNone(body["value"])
            self.assertEqual(len(body), 2)   # only the step asked for

    def test_availability_says_which_rungs_exist_without_revealing_them(self):
        """The widget must learn whether to offer the ladder without being
        handed the answer to hide in the DOM."""
        token, _, _ = self.embed(EXPLAINED)
        body = self.post("/api/hints/available", {"entry": "d1"}, token).json
        self.assertIn("answer", body["steps"])
        answer = self.post("/api/hints", {"entry": "d1", "step": "answer"},
                           token).json["value"]
        self.assertNotIn(answer, json.dumps(body))

    def test_hints_never_serve_an_answer_the_feed_withholds(self):
        """An embargoed prize puzzle ships no solution, and Reveal says so.
        clues_master.db may still hold the answer, scraped once the embargo
        lifted — serving it through the hint ladder would contradict the same
        widget's own Reveal and leak an unpublished answer."""
        token, model, _ = self.embed(EMBARGOED)
        entry = next(e for e in model["entries"] if e["len"])

        available = self.post("/api/hints/available", {"entry": entry["id"]},
                              token).json["steps"]
        self.assertNotIn("answer", available)
        self.assertNotIn("explanation", available)

        for step in ("answer", "explanation"):
            body = self.post("/api/hints",
                             {"entry": entry["id"], "step": step}, token).json
            self.assertIsNone(body["value"], step + " leaked")
            self.assertIn("not been published", body["unavailable"])

    def test_hints_reject_an_unknown_step(self):
        token, _, _ = self.embed(EXPLAINED)
        self.assertEqual(
            self.post("/api/hints", {"entry": "d1", "step": "everything"},
                      token).status_code, 400)

    def test_full_explanation_is_the_site_wfw_breakdown(self):
        """Not a reinvented prose format — the same shape the site's overlay
        draws: a clue-type label, answer tiles coloured by the piece that
        placed each letter, the one-line assembly, and rows in clue order."""
        token, _, _ = self.embed(EXPLAINED)
        value = self.post("/api/hints", {"entry": "d1", "step": "explanation"},
                          token).json["value"]

        self.assertEqual(value["label"], "Container + acrostic")
        self.assertEqual("".join(t.get("char", "") for t in value["tiles"]), "MESCAL")
        self.assertIn("MESCAL", value["summary"])

        # Letters carry the colour of the piece that placed them, and the
        # container source differs from the acrostic ones.
        colours = [t.get("fg") for t in value["tiles"] if t.get("char")]
        self.assertEqual(len(set(colours)), 3)

        pills = [r["pill"] for r in value["rows"]]
        self.assertEqual(pills[0], "Definition")
        self.assertIn("Indicator", pills)
        self.assertIn("Synonym", pills)

    def test_clue_type_names_every_mechanism(self):
        """The detailed clue type is the part that is new to the world; a flat
        'Container' would throw away exactly what is being sold."""
        token, _, _ = self.embed(EXPLAINED)
        value = self.post("/api/hints", {"entry": "d1", "step": "clue_type"},
                          token).json["value"]
        self.assertEqual(value, "Container + acrostic")

    def test_tools_require_a_token(self):
        for path in ("/api/tools/lookup", "/api/tools/synonym",
                     "/api/tools/pattern", "/api/tools/anagram", "/api/hints"):
            self.assertEqual(self.client.post(path, json={}).status_code, 401,
                             path + " is unguarded")


if __name__ == "__main__":
    unittest.main()
