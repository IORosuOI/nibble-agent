# Nibble – Milestone 2: Architecture and Plan

| Team member             | Sections                          |
| ----------------------- | --------------------------------- |
| Roșu Ioan-Luca          | 0. Overview, 4. Agent and tools, 7. Plan |
| Chelariu Răzvan-Dumitru | 1. Problem and data               |
| Crețu Vlad-Cristian     | 2. Golden dataset                 |
| Fazacaș Mihai-Tudor     | 3. RAG design                     |
| Ciocîia Marius Andrei   | 5. Model choices, 6. Safety       |

<!-- RULES: edit ONLY between your own BEGIN/END markers. Do not reformat the rest of the file. One PR per person, targeting main. -->

## 0. Architecture overview
<!-- BEGIN 0 (Luca) -->

_TODO: system diagram (Mermaid) and a short walkthrough of one request, from profile to recipe card._

<!-- END 0 -->

## 1. Problem and data
<!-- BEGIN 1 (Răzvan) -->
<!-- What task does the system handle? Which datasets (RecipeNLG, web recipes, USDA, Open Food Facts) and why? How are NEW datasets ingested (common recipe schema, accepted formats, pipeline steps)? -->

_TODO_

<!-- END 1 -->

## 2. Golden dataset
<!-- BEGIN 2 (Vlad) -->
<!-- How are representative test cases and expected answers created? Size, format, categories (normal, edge, adversarial), how expected answers are checked (constraints, not exact text), who reviews them. Which metrics run on it? -->

_TODO_

<!-- END 2 -->

## 3. RAG design
<!-- BEGIN 3 (Mihai) -->
<!-- Chunking strategy, embedding model, vector database, metadata filtering, RAG approaches to compare (e.g. dense vs hybrid, reranking, query rewriting). Explain WHY for each choice. -->

_TODO_

<!-- END 3 -->

## 4. Agent and tools
<!-- BEGIN 4 (Luca) -->
<!-- Which decisions the agent makes vs fixed pipeline steps, the list of tools (inputs/outputs), how they are exposed through the MCP server, agent framework. -->

_TODO_

<!-- END 4 -->

## 5. Model choices
<!-- BEGIN 5 (Marius) -->
<!-- SLMs considered and why (size, tool calling, license, hardware). Fine-tuning plan: method (LoRA), training data, task format, when it happens. -->

_TODO_

<!-- END 5 -->

## 6. Safety: toxicity and hallucinations
<!-- BEGIN 6 (Marius) -->
<!-- Allergen checks, poisonous/inedible blocklist, cooking temperatures and times, nutrition thresholds (50% / 150%), output moderation, grounding. -->

The general principle is the following: SLM proposes; specialized services verify; application decides whether a recipe can be served to the user.

We will have 4 validation outcomes:
 - PASS: The recipe satisfies the relevant checksa
 - WARN: The recipe is usable, but requires a visible warning
 - BLOCK: The recipe contains an identified safety violation
 - UNKNOWN: THere is insufficient information to establish safety

#### Alergen checking
Alergen categories defined in Annex II of Regulation (EU) No 1169/2011 (https://eur-lex.europa.eu/eli/reg/2011/1169/eng)

|Allergen category|	Examples|
|:---------------:|:-------:|
|Cereals containing gluten|	Wheat, rye, barley, oats, spelt|
|Crustaceans|	Crab, lobster, prawns|
|Eggs|	Chicken eggs, egg products|
|Fish|	Salmon, tuna, cod|
|Peanuts|	Peanuts and peanut products|
|Soybeans|	Soybeans, soy flour, soy protein|
|Milk|	Milk, cheese, butter, whey|
|Nuts|	Almonds, hazelnuts, walnuts, cashews, pistachios, etc.|
|Celery|	Celery root, celery leaves|
|Mustard|	Mustard seeds, mustard preparations|
|Sesame|	Sesame seeds, sesame products|
|Sulphur dioxide and sulphites|	Preservatives above 10 mg/kg or 10 mg/L|
|Lupin|	Lupin flour, lupin products|
|Molluscs|	Mussels, oysters, squid, snails|

Example of potential allergen clasification system:
```json
{
  "ingredient": "Whey protein",
  "canonical_id": "milk_whey",
  "allergens": [
    "milk"
  ],
  "derivatives": [
    "whey",
    "whey protein concentrate",
    "whey powder"
  ]
}
```

Example of potential user restriction system:
```json
{
  "profile": {
    "allergens": [
      "peanuts",
      "milk"
    ],
    "intolerances": [
      "lactose"
    ],
    "disliked_ingredients": [
      "mushrooms"
    ],
    "dietary_preferences": [
      "high_protein"
    ]
  }
}
```

We should also keep an eye to cross-contamination:
- Soy sauce can contain soy and wheat.
- Pesto can contain milk, nuts and sometimes other allergens.
- Stock cubes can contain celery, milk or other declared allergens.
- Chocolate can contain milk, soy and nuts.
- Bread crumbs can contain wheat.

#### Poisonous or inedible ingredient blocklist

|Category|Examples|Behaviour|
|:-:|:--:|:--:|
|Prohibited ingredients|	Poisonous mushrooms, toxic plant species, clearly inedible substances	|Always BLOCK|
|Conditionally unsafe ingredients|	Raw kidney beans, improperly processed cassava, unsafe plant parts|	BLOCK unless validated preparation requirements are satisfied|
|Food-safety hazards|	Spoiled ingredients, unsafe storage, contaminated food|	BLOCK or UNKNOWN depending on the evidence|

Example of potential hazard database structure:
```json
{
  "ingredient": {
    "canonical_id": "red_kidney_beans",
    "scientific_name": "Phaseolus vulgaris",
    "common_names": [
      "red beans",
      "kidney beans"
    ],
    "hazard": {
      "toxin": "phytohaemagglutinin",
      "severity": "high",
      "status": "conditional"
    },
    "preparation_requirements": {
      "soaking_required": true,
      "boiling_required": true,
      "slow_cooker_only": false
    },
    "source": "verified_food_safety_reference"
  }
}
```

Sources: 
- WHO: Natural toxins in food
- FDA: Foodborne pathogens and natural toxins
- EFSA: Chemical hazards and food cantaminants
- National food-safety authorities: Practical Preparation and storage requirements

Useful links:
- https://www.who.int/news-room/fact-sheets/detail/natural-toxins-in-food
- https://www.fda.gov/food/foodborne-pathogens/bad-bug-book-second-edition
- https://www.efsa.europa.eu/en/topics/topic/chemical-contaminants-food
  
**Recommendation**: Maintain a curated database rather than dynamically asking the SLM whether an ingredient is poisonous.


#### Cooking temperatures and times

The system should validate whether cooking instructions are sufficient to prepare food safely.

We should distinguish between:
- **Cooking time:** Estimated duration required to cook an ingredient.
- **Core temperature:** Actual temperature reached inside the food.
- **Holding time:** How long food must remain at a particular temperature to achieve the required safety treatment.

Example of temperature-time combinations from the UK Food Standards Agency (FSA):

| Core temperature | Required holding time |
|:----------------:|:---------------------:|
| 60°C | 45 minutes |
| 65°C | 10 minutes |
| 70°C | 2 minutes |
| 75°C | 30 seconds |
| 80°C | 6 seconds |

These are alternative temperature-time combinations, not interchangeable instant-temperature targets. The applicable rule must be selected according to the food category and preparation method.

We can also use category-specific recommendations from FoodSafety.gov:

| Food | Minimum internal temperature | Additional requirement |
|:----:|:----------------------------:|:---------------------:|
| Poultry | 74°C | — |
| Ground meat | 71°C | Depends on meat category |
| Fish | 63°C | — |
| Whole beef, pork, lamb and veal cuts | 63°C | 3-minute rest |
| Egg dishes | 71°C | — |
| Leftovers and casseroles | 74°C | — |

Sources:
- UK Food Standards Agency: Cooking your food (https://www.food.gov.uk/safety-hygiene/cooking-your-food)
- FoodSafety.gov: Safe Minimum Internal Temperatures (https://www.foodsafety.gov/food-safety-charts/safe-minimum-internal-temperatures)

**Recommendation**: Maintain a curated database of food-specific cooking requirements rather than letting the SLM invent cooking temperatures or assume that a cooking duration guarantees safety.

Example of potential cooking requirements database structure:

```json
{
  "food_category": "poultry",
  "temperature_rules": [
    {
      "temperature_c": 70,
      "holding_time_seconds": 120,
      "source": "FSA"
    },
    {
      "temperature_c": 75,
      "holding_time_seconds": 30,
      "source": "FSA"
    },
    {
      "temperature_c": 80,
      "holding_time_seconds": 6,
      "source": "FSA"
    }
  ],
  "measurement": "internal_core_temperature",
  "required": true
}
```

We should also include food handling, storage and reheating:
- Leaving perishable food at room temperature for too long.
- Improperly cooling cooked food.
- Reheating food inadequately.
- Incorrectly thawing frozen meat.
- Storing cooked rice under unsuitable conditions (risk of *Bacillus cereus* growth).

**Validation outcomes:**
- PASS: Valid temperature-time combination.
- WARN: Estimated cooking duration without a verifiable endpoint.
- BLOCK: Explicitly insufficient cooking conditions, until corrected.
- UNKNOWN: Insufficient information about the ingredient or its storage history.


#### Nutrition thresholds (50% / 150%)

Calculate the nutritional composition per serving and compare it against adult reference intakes.

We should use the adult Reference Intakes defined in Annex XIII of Regulation (EU) No 1169/2011 as the primary nutritional baseline.

| Nutrient | Adult reference intake |
|:--------:|:----------------------:|
| Energy | 2,000 kcal |
| Total fat | 70 g |
| Saturated fat | 20 g |
| Carbohydrates | 260 g |
| Sugars | 90 g |
| Protein | 50 g |
| Salt | 6 g |

Source: Regulation (EU) No 1169/2011 (https://eur-lex.europa.eu/eli/reg/2011/1169/eng)

For caffeine, there is no equivalent EU food-labelling reference intake. Instead, use EFSA's daily intake guidance:
- Healthy adults: up to 400 mg/day.
- Pregnant women: up to 200 mg/day.
- Children and adolescents: 3 mg/kg body weight/day as a level of no safety concern.

For sugar, distinguish between total sugars (EU reference intake of 90 g) and free sugars (WHO recommendation of less than 10% of daily energy intake, ideally less than 5%). For a 2,000 kcal diet, these correspond to approximately 50 g and 25 g, respectively.

Sources:
- EFSA: Caffeine (https://www.efsa.europa.eu/en/topics/topic/caffeine)
- WHO: Healthy diet (https://www.who.int/news-room/fact-sheets/detail/healthy-diet)

**Proposed threshold system:**

| Percentage of reference intake | Status | Application behaviour |
|:------------------------------:|:------:|:---------------------:|
| Below 50% | Normal | Display nutritional information |
| 50–100% | Warning | Highlight significant contribution |
| 100–150% | High contribution | Display a more prominent warning |
| Above 150% | Very high contribution | Strong warning and optional recipe modification |

Example warning: *One serving provides 123% of the EU adult reference intake for sugars.*

Calculation formula:

\[
\text{RI percentage} =
\frac{\text{Nutrient per serving}}
{\text{Adult reference intake}}\times100
\]

Example: If one serving contains 86 g of sugars:

\[
\frac{86}{90}\times100 \approx 95.6\%
\]

The system should calculate nutritional values using a dedicated nutrition service rather than relying on SLM estimations.

The service should:
- Retrieve nutritional values for each ingredient.
- Normalize quantities and measurement units.
- Calculate the complete recipe's nutritional composition.
- Divide the totals by the number of servings.
- Recalculate everything after ingredient substitutions.

**Important:** Exceeding 150% of an adult reference intake does not automatically mean a recipe is unsafe. It represents a very high nutritional contribution and should trigger a warning or optional recipe modification, not an automatic safety block.

Potential nutritional data sources:
- USDA FoodData Central (https://fdc.nal.usda.gov/)
- EFSA Food Composition (https://www.efsa.europa.eu/en/data-report/food-composition)
- Manufacturer nutritional declarations for packaged ingredients.

**Recommendation**: Use a dedicated nutrition database and calculation service. Nutritional values should be presented as estimates, since ingredient brands, serving sizes and cooking-related nutrient losses can introduce uncertainty.


#### Output moderation

Output moderation examines whether user requests, generated recipes and agent actions are safe and consistent with Nibble's intended functionality.

I would divide it into three components:

| Component | Responsibility |
|:---------:|:--------------|
| Input moderation | Identify harmful requests and dangerous intentions |
| Generated-output moderation | Validate generated recipes and responses |
| Agent and tool-use moderation | Prevent unauthorized or unsafe MCP tool execution |

**Input moderation examples:**

| User request | Expected behaviour |
|:------------|:------------------:|
| Make a lactose-free chicken recipe | ALLOW |
| Suggest a substitute for peanuts | ALLOW |
| How do I prepare raw kidney beans safely? | ALLOW with verified instructions |
| How can I make a poisonous mushroom edible? | BLOCK unsafe premise |
| How can I hide an allergen in someone's food? | BLOCK |
| Create an extremely restrictive starvation diet | BLOCK harmful instructions and redirect |

**Generated-output moderation** should inspect:
- Schema validity: Missing fields, incorrect types and malformed JSON.
- Dietary consistency: Conflicts with user preferences and restrictions.
- Allergen consistency: Prohibited allergens in generated ingredients.
- Food-safety consistency: Unsafe preparation and storage instructions.
- Nutritional consistency: Implausible values and incorrect calculations.
- Grounding consistency: Unsupported claims and fabricated citations.
- Content moderation: Dangerous instructions, misleading health claims and inappropriate responses.

**Agent and MCP moderation** should include:
- An explicit allowlist of MCP tools.
- JSON Schema validation for tool arguments.
- Restricted tool permissions.
- Maximum tool-call counts and execution timeouts.
- Validation of tool results before returning them to the model.
- No ability for the model to modify safety databases or validation thresholds.
- No ability for retrieved webpages to override system instructions.

We should also protect against prompt injection. Retrieved webpages must be treated as untrusted data, never as instructions to the agent.

Potential moderation models:
- Llama Guard (Meta).
- ShieldGemma (Google).
- Granite Guardian (IBM).

**Recommendation**: Combine rule-based validation with a dedicated moderation model where appropriate. Moderation models should classify potentially harmful requests and responses, but must not replace deterministic food-safety checks.

Useful reference: OWASP Top 10 for LLM Applications (https://genai.owasp.org/llm-top-10/).


#### Grounding

Grounding ensures that generated recipes and factual claims are supported by identifiable and reliable sources.

We should distinguish three types of grounding:

- **Recipe grounding:** Generated recipes should have identifiable real recipes as their basis.
- **Factual grounding:** Claims about ingredients, cooking methods and nutritional properties should have supporting evidence.
- **Safety grounding:** Safety-related claims should rely on authoritative food-safety sources, not arbitrary recipe websites.

**Recommended source hierarchy:**

| Information | Preferred sources |
|:-----------|:-----------------|
| Original recipes | Established recipe websites and licensed recipe datasets |
| Ingredient composition | USDA FoodData Central, EFSA and manufacturer data |
| Allergen classifications | EU Regulation 1169/2011, EFSA and product labels |
| Poisonous ingredients | WHO, FDA, EFSA and national food-safety authorities |
| Cooking temperatures | FSA, FoodSafety.gov and applicable national authorities |
| Nutrition thresholds | EU legislation, WHO and EFSA |
| Food-handling instructions | National food-safety authorities |


<!-- END 6 -->

## 7. Plan
<!-- BEGIN 7 (Luca) -->
<!-- Week-by-week timeline from week 6 to week 14, per member, including the experiments planned. This is a commitment, reviewed in week 12 via commit history. -->

_TODO_

<!-- END 7 -->
