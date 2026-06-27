"""Review-gate — make a high-risk engine's CLEAN PASS land as a 'pending' that DEMANDS human
review, instead of a silent green pass.

Some engines (e.g. container_deletion_selection, which builds BOTH container pieces) have a
larger, looser search space and so a higher chance of a coincidental exact reconstruction —
a false pass that looks clean. Rather than trust such an engine outright, we route its full
solves through human review: the answer is still claimed (a pending short-circuits the cascade
like a pass, so nothing else overrides it), but it shows amber with a REVIEW tag until a human
confirms it to a pass via the existing Set-status control.

Mechanism (deliberately tiny and reusable):
  * `gate(parse, engine_label)` downgrades a clean PASS to status 'pending' and prepends a
    warning beginning with REVIEW_PREFIX. It NEVER touches a genuine pending/fail.
  * The marker is just that warning string, which the store persists and reloads, so the
    REVIEW state survives a page reload with no extra schema. The renderer keys off
    `is_review(parse)` to show the REVIEW tag.

Applied at the CASCADE BOUNDARY (engine_registry.solve), not inside the engine — so the
engine's own 'return only on pass' contract is unaffected; the downgrade happens once the
cascade has accepted the solve.
"""

REVIEW_PREFIX = "REVIEW:"


def gate(parse, engine_label):
    """Downgrade `parse` from a clean PASS to a review-pending. No-op unless status == 'pass'."""
    if parse is None or getattr(parse, "status", None) != "pass":
        return parse
    parse.status = "pending"
    msg = ("%s fully solved by %s - a high-risk shape; confirm before trusting"
           % (REVIEW_PREFIX, engine_label))
    parse.warnings = [msg] + list(parse.warnings or [])
    return parse


def is_review(parse):
    """True if `parse` carries a REVIEW marker (set by gate()). Detected from the warnings so
    it survives store round-trips."""
    return any((w or "").startswith(REVIEW_PREFIX)
               for w in (getattr(parse, "warnings", None) or []))
