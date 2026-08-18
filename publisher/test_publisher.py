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

    def test_a_narrowed_entry_counts(self):
        token, _, _ = self.embed(SOLVED)
        counts = self.post("/api/match-counts",
                           {"patterns": {"a1": "WARSH??"}}, token).json["counts"]["a1"]
        self.assertTrue(0 < counts["n"] <= 9)

    def test_more_than_nine_shows_nothing_at_all(self):
        """A chip the solver cannot act on is noise, and a board covered in
        them cannot be scanned. Above the limit the entry stays silent."""
        token, _, _ = self.embed(SOLVED)
        counts = self.post("/api/match-counts",
                           {"patterns": {"a1": "W??????"}}, token).json["counts"]
        self.assertIsNone(counts["a1"])

    def test_an_entry_with_every_crossing_filled_is_offered_a_shortlist(self):
        """Such an entry can never narrow again from the grid, so staying
        silent would leave the solver stuck with no way forward."""
        token, _, _ = self.embed(SOLVED)
        counts = self.post("/api/match-counts",
                           {"patterns": {"a1": "W??????"},
                            "crossed": ["a1"]}, token).json["counts"]["a1"]
        self.assertEqual(counts["n"], 9)

    def test_the_number_is_the_length_of_the_list_it_opens(self):
        """The chip promises a list. A 9 above a list of 41, or of 3, would be
        the feature contradicting itself in one click."""
        token, _, _ = self.embed(SOLVED)
        for pattern, crossed in (("WARSH??", []), ("W??????", ["a1"])):
            counts = self.post("/api/match-counts",
                               {"patterns": {"a1": pattern},
                                "crossed": crossed}, token).json["counts"]["a1"]
            listed = self.post("/api/tools/pattern",
                               {"pattern": pattern, "entry": "a1"}, token).json
            self.assertEqual(counts["n"], len(listed["matches"]), pattern)

    def test_wrong_letters_get_no_shortlist_and_no_free_check(self):
        """A red zero says the letters are wrong. Listing the answer beside it
        would tell the solver which letters, for nothing."""
        token, _, _ = self.embed(SOLVED)
        counts = self.post("/api/match-counts",
                           {"patterns": {"a1": "ZQXJ???"},
                            "crossed": ["a1"]}, token).json["counts"]["a1"]
        self.assertEqual(counts["n"], 0)
        listed = self.post("/api/tools/pattern",
                           {"pattern": "ZQXJ???", "entry": "a1"}, token).json
        self.assertEqual(listed["matches"], [])

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

    def test_the_more_label_can_actually_fetch_the_rest(self):
        """The "+53 more" label is a control, as it is on the site. Asking for
        one length must return more than the five the summary showed."""
        token, _, _ = self.embed(EXPLAINED)
        summary = self.post("/api/tools/lookup", {"word": "Spirit"}, token).json
        trimmed = next(g for g in summary["meanings"] if g["more"] > 0)
        full = self.post("/api/tools/lookup",
                         {"word": "Spirit", "letters": trimmed["length"]}, token).json
        self.assertEqual(len(full["meanings"]), 1)
        self.assertEqual(full["meanings"][0]["length"], trimmed["length"])
        self.assertGreater(len(full["meanings"][0]["words"]), len(trimmed["words"]))

    def test_word_info_says_what_a_result_means(self):
        """Nine words that all fit are nine strings without this."""
        token, _, _ = self.embed(SOLVED)
        data = self.post("/api/tools/word-info", {"word": "WARSHIP"}, token).json
        self.assertEqual(data["word"], "WARSHIP")
        self.assertTrue(data["meanings"])
        self.assertTrue(all(len(m) <= len(data["meanings"][-1])
                            for m in data["meanings"]))   # shortest first

    def test_word_info_needs_a_token(self):
        self.assertEqual(
            self.client.post("/api/tools/word-info", json={"word": "WARSHIP"})
            .status_code, 401)

    def test_lookup_finds_a_phrase_the_solver_already_used(self):
        """"Jill's companion" -> JACK is in the reference DB, keyed as
        'jills companion'. Matching on LOWER(word) missed it and every other
        row whose key differs from its display form. Our own solver reads these
        tables by the normalised key (core/live_db.py:79); so must the widget,
        or it denies knowing what it used to solve the clue."""
        token, _, _ = self.embed(SOLVED)
        for spelling in ("Jill's companion", "jills companion", "JILL'S COMPANION"):
            data = self.post("/api/tools/lookup", {"word": spelling}, token).json
            words = [w for group in data["meanings"] for w in group["words"]]
            self.assertIn("JACK", words, spelling)

    def test_lookup_matches_a_hyphenated_phrase(self):
        token, _, _ = self.embed(SOLVED)
        data = self.post("/api/tools/lookup", {"word": "pen-pushers"}, token).json
        words = [w for group in data["meanings"] for w in group["words"]]
        self.assertIn("BORING WRITERS", words)

    def test_synonym_searches_both_directions(self):
        token, _, _ = self.embed(EXPLAINED)
        data = self.post("/api/tools/synonym", {"word": "spirit"}, token).json
        self.assertTrue(data["synonyms"])
        self.assertNotIn("SPIRIT", data["synonyms"])   # never echo the query

    def test_the_shortlist_is_capped_alphabetical_and_holds_the_answer(self):
        token, _, _ = self.embed(SOLVED)
        listed = self.post("/api/tools/pattern",
                           {"pattern": "W??????", "entry": "a1"}, token).json
        self.assertEqual(len(listed["matches"]), 9)
        self.assertIn("WARSHIP", listed["matches"])
        self.assertEqual(listed["matches"], sorted(listed["matches"]))
        self.assertTrue(listed["capped"])
        self.assertGreaterEqual(listed["total"], len(listed["matches"]))

    def test_must_include_narrows_the_pattern_search(self):
        """The site's second pattern field (puzzle.html:368). The pattern says
        where letters go; this says which must be in there somewhere."""
        token, _, _ = self.embed(SOLVED)
        listed = self.post("/api/tools/pattern",
                           {"pattern": "W??????", "entry": "a1",
                            "include": "SH"}, token).json
        self.assertTrue(listed["matches"])
        for word in listed["matches"]:
            letters = word.replace(" ", "").upper()
            self.assertIn("S", letters, word)
            self.assertIn("H", letters, word)

    def test_must_include_also_governs_the_answer(self):
        """The shortcut must not smuggle the answer past the solver's own
        filter — a list that answers a question nobody asked is worse than a
        short one."""
        token, _, _ = self.embed(SOLVED)
        listed = self.post("/api/tools/pattern",
                           {"pattern": "W??????", "entry": "a1",
                            "include": "ZQ"}, token).json
        self.assertNotIn("WARSHIP", listed["matches"])

    def test_the_shortlist_does_not_reshuffle_between_openings(self):
        """Two different nines for the same pattern would read as guessing."""
        token, _, _ = self.embed(SOLVED)
        first = self.post("/api/tools/pattern",
                          {"pattern": "W??????", "entry": "a1"}, token).json
        again = self.post("/api/tools/pattern",
                          {"pattern": "W??????", "entry": "a1"}, token).json
        self.assertEqual(first["matches"], again["matches"])

    def test_the_corpus_is_built_in_a_fixed_order(self):
        """Stability across restarts, not just within one. The buckets were
        built from sets, whose iteration order Python randomises per process,
        so the "unchanging" nine changed every time the server came up."""
        from publisher import corpus
        corpus.invalidate()
        first = corpus.load(self.app.config["CLUES_DB"], self.app.config["REF_DB"])
        snapshot = {n: list(words) for n, words in first.items()}
        corpus.invalidate()
        second = corpus.load(self.app.config["CLUES_DB"], self.app.config["REF_DB"])
        for length, words in snapshot.items():
            self.assertEqual(words, list(second[length]), f"length {length}")
        for length, words in second.items():
            self.assertEqual(words, sorted(words), f"length {length} unsorted")

    def test_an_answer_we_do_not_hold_is_never_added(self):
        """A prize puzzle under embargo has no solution here, and Check and
        Reveal both say so. The shortlist must be silent in the same breath."""
        token, model, _ = self.embed(EMBARGOED)
        entry = self.entry_of(model, "a1")
        pattern = "?" * entry["len"]
        pattern = "S" + pattern[1:]
        listed = self.post("/api/tools/pattern",
                           {"pattern": pattern, "entry": "a1"}, token).json
        with self.app.app_context():
            from publisher import corpus, reference
            plain = reference.pattern_matches(
                corpus, self.app.config["CLUES_DB"], self.app.config["REF_DB"],
                pattern, entry.get("enum"), "", answer=None, limit=9)
        self.assertEqual(listed["matches"], plain["matches"])

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
