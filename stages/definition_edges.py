import re
from stages.definition import definition_candidates as base_definition_candidates
from resources import clean_key, norm_letters

SEPARATOR_RE = re.compile(r"[,\;:\u2014]")  # comma, semicolon, colon, em dash
ARTICLES = ("a ", "an ", "the ")

def has_separator(clue_text: str) -> bool:
    if not clue_text:
        return False
    return bool(SEPARATOR_RE.search(clue_text))

def definition_candidates(clue_text, enumeration, graph):
    # Stage 1 — reuse base candidate logic
    result = base_definition_candidates(
        clue_text=clue_text,
        enumeration=enumeration,
        graph=graph,
    )

    result["has_separator"] = has_separator(clue_text)

    answer_norm = None
    if "answer" in result:
        answer_norm = norm_letters(result["answer"])

    # Pass-through if answer already found
    if answer_norm:
        for c in result["candidates"]:
            if norm_letters(c) == answer_norm:
                return result

    # Edge-based expansion using ARTICLE variants
    extra_candidates = set(result["candidates"])
    support = dict(result["support"])

    for dp in result["definition_windows"]:
        dp_clean = clean_key(dp)
        for art in ARTICLES:
            key = clean_key(art + dp_clean)
            if key in graph:
                for cand in graph[key]:
                    extra_candidates.add(cand)
                    support.setdefault(cand, set()).add(dp)

    return {
        "definition_windows": result["definition_windows"],
        "candidates": list(extra_candidates),
        "support": support,
        "has_separator": result["has_separator"],
    }
