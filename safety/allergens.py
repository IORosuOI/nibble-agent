def check_allergens(ingredients: list, profile_allergens: list) -> tuple[str, list]:
    """
    Checks recipe ingredients against the user's allergen profile.

    Returns:
        ("PASS", [])              - No matching allergens.
        ("BLOCK", [...])          - Ingredients matching declared allergens.
        ("UNKNOWN", [...])        - Ingredients that cannot be verified.
    """

    if not ingredients:
        return "UNKNOWN", []

    blocked = []
    unknown = []

    allergens = {a.strip().casefold() for a in profile_allergens}

    for ingredient in ingredients:
        if not isinstance(ingredient, str) or not ingredient.strip():
            unknown.append(ingredient)
            continue

        if ingredient.strip().casefold() in allergens:
            blocked.append(ingredient)

    if blocked:
        return "BLOCK", blocked

    if unknown:
        return "UNKNOWN", unknown

    return "PASS", []
