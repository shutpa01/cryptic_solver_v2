"""Hint reveal routes — progressive HTMX reveal + async explanation generation."""

from flask import Blueprint, request, render_template, abort, current_app, g
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from web.models import (
    get_clue_by_id, get_hint_steps, get_hint_content, compute_hint_tier,
)

bp = Blueprint("hints", __name__)

TOKEN_MAX_AGE = 3600  # 1 hour


def _get_serializer():
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"])


def generate_token(clue_id):
    """Generate a signed token encoding the clue ID."""
    s = _get_serializer()
    return s.dumps({"cid": clue_id})


def _validate_token(token):
    """Validate and decode a reveal token. Returns clue_id or None."""
    s = _get_serializer()
    try:
        data = s.loads(token, max_age=TOKEN_MAX_AGE)
        return data.get("cid")
    except (BadSignature, SignatureExpired):
        return None


@bp.route("/reveal", methods=["POST"])
def reveal():
    """Reveal a specific hint step (or all) for a clue.

    Expects form data: token (signed), step (int 1-based, or "all").
    Returns an HTML fragment for HTMX to swap in.
    """
    token = request.form.get("token", "")
    step_raw = request.form.get("step", "")

    if not token or not step_raw:
        abort(400)

    clue_id = _validate_token(token)
    if clue_id is None:
        return render_template("partials/hint_error.html",
                               message="Session expired — please reload the page."), 403

    clue = get_clue_by_id(clue_id)
    if clue is None:
        abort(404)

    tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue, tier=tier, is_admin=g.get("is_admin", False))
    if not steps:
        abort(400)

    show_all = step_raw == "all"

    if show_all:
        # Reveal everything
        revealed = []
        for s in steps:
            content = get_hint_content(clue, s["type"])
            revealed.append({
                "label": s["label"],
                "type": s["type"],
                "content": content,
            })
        return render_template(
            "partials/hint_step.html",
            revealed=revealed,
            token=token,
            has_next=False,
            next_step=None,
            next_label=None,
        )

    # Single step reveal
    try:
        step_num = int(step_raw)
    except ValueError:
        abort(400)

    if step_num < 1 or step_num > len(steps):
        abort(400)

    # Return just this one step
    s = steps[step_num - 1]
    content = get_hint_content(clue, s["type"])
    revealed = [{
        "label": s["label"],
        "type": s["type"],
        "content": content,
    }]

    return render_template(
        "partials/hint_step.html",
        revealed=revealed,
        token=token,
        has_next=False,
        next_step=None,
        next_label=None,
    )


@bp.route("/explain", methods=["POST"])
def explain():
    """Generate an explanation for a clue via the Sonnet API.

    Expects form data: token (signed).
    Calls the API, stores the result, returns updated hint buttons as HTML.
    """
    token = request.form.get("token", "")
    if not token:
        abort(400)

    clue_id = _validate_token(token)
    if clue_id is None:
        return render_template("partials/hint_error.html",
                               message="Session expired — please reload the page."), 403

    clue = get_clue_by_id(clue_id)
    if clue is None:
        abort(404)

    # Only generate if clue has an answer and isn't already HIGH tier
    if not clue["answer"]:
        return render_template("partials/hint_error.html",
                               message="No answer available for this clue.")

    tier, _ = compute_hint_tier(clue)
    if tier == "HIGH":
        return render_template("partials/hint_error.html",
                               message="This clue already has a full explanation.")

    # Call the API
    from web.explainer import generate_explanation
    try:
        success, message, result = generate_explanation(clue_id)
    except Exception as e:
        import logging
        logging.getLogger(__name__).exception("explain failed for clue %d", clue_id)
        return render_template("partials/hint_error.html",
                               message=f"Error generating explanation: {e}")

    if not success:
        return render_template("partials/hint_error.html", message=message)

    # Re-fetch the clue to get updated data, regenerate token for new hints
    clue = get_clue_by_id(clue_id)
    new_tier, _ = compute_hint_tier(clue)
    steps = get_hint_steps(clue, tier=new_tier, is_admin=g.get("is_admin", False))
    new_token = generate_token(clue_id)

    timing = {
        "total_ms": result.get("total_ms"),
        "api_ms": result.get("api_ms"),
        "score": result.get("score"),
    }

    return render_template(
        "partials/explain_result.html",
        clue=clue,
        tier=new_tier,
        steps=steps,
        token=new_token,
        timing=timing,
    )
