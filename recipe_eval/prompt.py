"""One prompt format shared by training data and evaluation, plus a parser for the model's reply.

render_prompt(request)  -> the user message (same text for training examples and eval examples)
target_json(recipe)     -> the assistant reply to train on (JSON the scorer can read)
parse_response(text)    -> dict for recipe_eval.scoring.score_response, or None if the reply is not valid
"""
import json
import re

INSTRUCTIONS = (
    "Write one complete recipe that uses at least one of these ingredients (salt, pepper, water and cooking oil are always available; "
    "you do not have to use every ingredient, and you may add other ingredients if the recipe needs them). "
    'Reply with JSON only, with keys: "title" (string), "servings" (number), '
    '"ingredients" (list of lines with amounts, like "2 cups flour"), "steps" (list of strings).'
)


REFUSAL_RULE = (
    ' If no recipe is possible without using an ingredient the user must avoid, reply instead with JSON '
    '{"refusal": "<short reason>"}.'
)


def render_prompt(request):
    """request = {"ingredients": [str], "tags": [str]} (as stored in the examples files).
    Optional keys: "craving" (free text) and "profile" = {"allergens": [...], "dislikes": [...]}. A request without them
    renders exactly as before, so older training data is unchanged."""
    ings = ", ".join(request["ingredients"])
    mood = ", ".join(t.replace("_", " ") for t in request.get("tags", [])) or "anything"
    lines = [f"I have these ingredients: {ings}.", f"I'm in the mood for: {mood}."]
    if request.get("craving"):
        lines.append(f"In my own words: {request['craving']}")
    profile = request.get("profile") or {}
    if profile.get("allergens"):
        lines.append("I am allergic to (must not appear anywhere in the recipe): " + ", ".join(profile["allergens"]) + ".")
    if profile.get("dislikes"):
        lines.append("I dislike: " + ", ".join(profile["dislikes"]) + ".")
    return "\n".join(lines) + "\n" + INSTRUCTIONS + (REFUSAL_RULE if profile.get("allergens") else "")


def target_json(recipe):
    """recipe = {"title", "servings" (or None), "ingredients", "steps"} -> JSON string used as the training target."""
    return json.dumps({
        "title": recipe["title"],
        "servings": recipe.get("servings"),
        "ingredients": list(recipe["ingredients"]),
        "steps": list(recipe["steps"]),
    }, ensure_ascii=False)


def parse_response(text):
    """Extract the recipe dict from a model reply. Returns None when no valid recipe JSON is found."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(d, dict) and d.get("refusal"):       # {"refusal": "reason"}: the model says no safe recipe exists
        return {"title": "", "servings": None, "ingredients": [], "steps": [], "refusal": str(d["refusal"])}
    if not isinstance(d, dict) or not isinstance(d.get("ingredients"), list) or not isinstance(d.get("steps"), list):
        return None
    return {"title": str(d.get("title", "")), "servings": d.get("servings"),
            "ingredients": [str(i) for i in d["ingredients"]], "steps": [str(s) for s in d["steps"]]}
