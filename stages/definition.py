from resources import (
    clean_key,
    norm_letters,
    build_wordlist,
)

ARTICLES = ("a ", "an ", "the ")


def definition_candidates(clue_text, enumeration, graph):
    # wordlist = build_wordlist()

    definition_windows = generate_definition_windows(clue_text)

    answer_norm = None
    candidates = set()
    support = {}

    for dp in definition_windows:
        # Normalise smart apostrophes to ASCII before key lookup
        dp_norm = (
            dp
            .replace("’", "'")
            .replace("‘", "'")
        )
        key = clean_key(dp_norm)
        if key in graph:
            for cand in graph[key]:
                candidates.add(cand)
                support.setdefault(cand, set()).add(dp)

    # Article-variant matches
    for dp in definition_windows:
        dp_norm = (
            dp
            .replace("’", "'")
            .replace("‘", "'")
        )
        dp_clean = clean_key(dp_norm)
        for art in ARTICLES:
            key = clean_key(art + dp_clean)
            if key in graph:
                for cand in graph[key]:
                    candidates.add(cand)
                    support.setdefault(cand, set()).add(dp)

    return {
        "definition_windows": definition_windows,
        "candidates": list(candidates),
        "support": support,
    }


def generate_definition_windows(clue_text):
    # Same logic as always — scan start and end of clue
    # Normalise smart apostrophes to ASCII apostrophe
    clue_text = (
        clue_text
        .replace("’", "'")
        .replace("‘", "'")
    )

    words = clue_text.split()
    windows = set()

    for i in range(len(words)):
        window1 = " ".join(words[: i + 1])
        window2 = " ".join(words[-(i + 1):])
        if window1:
            windows.add(window1.strip())
        if window2:
            windows.add(window2.strip())

    # Apostrophe bifurcation (base-case): evaluate both possessive and de-possessed/plural forms
    # e.g. "Ray’s arrows" -> also try "rays arrows"
    expanded = set(windows)
    for w in windows:
        if "’s" in w or "'s" in w:
            expanded.add(w.replace("’s", "s").replace("'s", "s"))

    return list(expanded)
