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


def render_prompt(request):
    """request = {"ingredients": [str], "tags": [str]} (as stored in the examples files)."""
    ings = ", ".join(request["ingredients"])
    mood = ", ".join(t.replace("_", " ") for t in request.get("tags", [])) or "anything"
    return f"I have these ingredients: {ings}.\nI'm in the mood for: {mood}.\n{INSTRUCTIONS}"


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
    if not isinstance(d, dict) or not isinstance(d.get("ingredients"), list) or not isinstance(d.get("steps"), list):
        return None
    return {"title": str(d.get("title", "")), "servings": d.get("servings"),
            "ingredients": [str(i) for i in d["ingredients"]], "steps": [str(s) for s in d["steps"]]}
