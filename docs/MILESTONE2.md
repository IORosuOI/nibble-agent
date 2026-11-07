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

**Task.** Nibble generates a complete, structured recipe (title, servings, ingredient lines with amounts, steps) from the ingredients the user has plus a preference ("mood") such as high-protein, keto, vegan or spicy. The data work below produces (a) a recipe pool with nutrition information, (b) fine-tuning data for the SLM and (c) the golden test set (section 2).

**Datasets used and why**

| Dataset | Size | What it gives us | Why |
|---|---|---|---|
| Kaggle `recipes-dataset-64k-dishes` | 62,126 raw rows | title, category, ingredient lines, step lists | Large recipe text source. No nutrition and no servings. |
| Hugging Face `xiaoxl/recipe_data` (labelled MIT) | 107,064 rows; 24,367 have servings and calories/protein/carbs/fat | recipes with per-serving nutrition | The only source we found with full recipe text and per-serving macros. Rows with nutrition link to allrecipes.com. |
| USDA FoodData Central, SR Legacy (public domain) | 7,793 foods | per-100 g macros and household-measure gram weights | Lets us compute macros from any ingredient list and check the labels. |

Looked at and not used: RecipeNLG (about 2.2M recipes, but no nutrition and a non-commercial research licence) and MM-Food-100K (food photos with no steps, quantity-less ingredient lists, macros estimated from photos, non-commercial licence). Open Food Facts, FlavorDB and food2vec from the RFC are not used in the dataset work so far.

**How a new dataset is ingested.** Every source is converted to one common schema and goes through the same steps. The steps are scripts in `scripts/` and cells in `recipes.ipynb`. A new source needs a small adapter that maps its columns to the schema (CSV, JSON and Parquet have been read so far); there is no generic upload tool yet.

Common schema: `recipe_title`, `category`, `subcategory`, `description`, `ingredients` (list of lines with amounts), `directions` (list of steps), optional `servings` and per-serving `kcal`, `protein_g`, `carb_g`, `fat_g`, `fiber_g`, `sodium_mg`. The pipeline adds `recipe_id`, `source`, `nutrition_basis` and `single_step`.

1. **Acquire.** Kaggle through `kagglehub`; Hugging Face Parquet shards downloaded one at a time, text and nutrition columns kept, shard deleted (`scripts/pull_xiaoxl.py`); USDA SR Legacy as a CSV zip.
2. **Parse** ingredient and direction text into lists.
3. **Clean.** Drop recipes with fewer than 4 ingredients (many are truncated lists such as only `cooking spray`), under 60 characters of directions, or non-food categories (pet food, DIY). Single-step recipes are flagged, not dropped. Later (step 5) recipes with more than 4,000 kcal in total are also dropped, because big-batch recipes do not suit a home-cooking app.
4. **Deduplicate.** Key = normalized title + sorted normalized ingredients. About 60% of Kaggle rows were the same recipe listed under several categories, so one row is kept (most specific category) and all categories are stored. A new source is also checked against the existing pool (title, directions or ingredient match) so no recipe enters twice.
5. **Nutrition check.** Macros are recomputed from the ingredient lines with USDA data (`recipe_eval/nutrition.py`: hand-written overrides, then cached ingredient-to-USDA matches, then word overlap). A source label is kept only if the recomputed values agree: calories within 25%, other macros within 40%, at least 80% of ingredients matched. Recipes without servings keep computed whole-recipe totals (so calorie shares). Recipes with a total above 4,000 kcal are dropped (4,859 recipes), and for the Kaggle recipes that remain, `servings` is set to `round(total / 750 kcal)` (1 to 5) so a serving is about 750 kcal, flagged `servings_estimated`. This is a normalising choice, not a recovered fact.
6. **Merge and label.** One file, `data/recipes_merged.jsonl`. Tags are derived by rule (section 2). Training and test files are built from it (`scripts/build_train.py`, `scripts/build_examples.py`).

**Resulting pool** (`data/recipes_merged.jsonl`, 23,395 recipes)

| Source | Nutrition | Recipes |
|---|---|---|
| Kaggle | computed from ingredients, whole-recipe totals; servings estimated at about 750 kcal each | 15,251 |
| Kaggle | none (under 80% of ingredients matched), no servings | 4,798 |
| xiaoxl | per-serving label, agrees with the USDA recomputation | 3,346 |

A record has the common schema plus `calc_kcal_total`, `calc_protein_total`, `calc_carb_total`, `calc_fat_total` and `calc_coverage` (computed totals and the share of ingredients matched) and `servings_estimated`. No recipe in the pool is above 4,000 kcal in total. Computed macros are about 15% off at the median against labels, so they are estimates, not ground truth.

**How the data is used.** The pool feeds (1) the golden test set (section 2), (2) fine-tuning data for the SLM (section 5): 19,692 training and 1,036 validation examples, built from recipes that are not in the test set, and (3) the nutrition calculator, which is also used by the scorer and could be reused by the agent's nutrition tool (section 4). Check the licences before training or releasing a model: the recipe text comes from recipe sites.

<!-- END 1 -->

## 2. Golden dataset
<!-- BEGIN 2 (Vlad) -->
<!-- How are representative test cases and expected answers created? Size, format, categories (normal, edge, adversarial), how expected answers are checked (constraints, not exact text), who reviews them. Which metrics run on it? -->

**Purpose.** A test set for the recipe generator: given ingredients and a mood, does the model return a usable recipe that stays inside the ingredients and fits the mood? Expected answers are constraints, not exact text.

**How test cases are created** (`scripts/build_examples.py`). Each case is built from one recipe in the pool:
- **Request:** the recipe's ingredient names (staples removed) plus 0-2 random *distractor* ingredients taken from other recipes, shuffled, plus 1-2 mood tags the recipe really has.
- **Reference answer:** the source recipe (title, servings, ingredients, steps) and its nutrition: per serving for xiaoxl rows, whole-recipe computed totals (with servings estimated at about 750 kcal each) for Kaggle rows.
- **Tags** are derived by rule (`recipe_eval/labels.py`, thresholds in one place). Macro tags use shares of calories: `high_protein` (protein at least 25%), `high_carb` (carbs at least 55%), `low_carb` (carbs at most 20%), `keto` (carbs at most 10% and fat at least 60%). Computed shares must clear a threshold by 3 points. `low_calorie` (at most 400 kcal per serving) needs real servings, so it exists only for xiaoxl rows. Keyword tags are read from the ingredient list: `vegetarian`, `vegan`, `dairy_free`, `gluten_free`, `spicy` (approximate).

**Size and format.** 1,200 cases, 120 per tag, JSON Lines in `data/eval_examples.jsonl`. Fields: `example_id`, `recipe_id` (link to the pool), `source`, `nutrition_basis`, `request` (`ingredients`, `tags`), `distractors`, `golden`. 768 cases come from Kaggle recipes and 432 from xiaoxl; 753 request two tags; 818 contain distractors.

**Categories**

| Category | What is in the set |
|---|---|
| Normal | Ingredients of a real recipe plus 1-2 matching mood tags (all 1,200 cases) |
| Edge | Distractor ingredients the model should not be forced to use (818); single-step recipes (flagged in the pool); cases whose nutrition is computed rather than labelled (768 cases) |
| Adversarial | **Not built yet.** Planned: allergen conflicts, inedible or unsafe ingredient combinations, contradictory moods, requests that cannot be satisfied from the given ingredients |

**How answers are checked.** Not by matching the reference text. `recipe_eval/scoring.py: score_response(request, response)` returns three scores in [0, 1]:
1. **Compliance (hard gate).** Every non-staple ingredient in the answer must come from the user's list (salt, pepper, water and cooking oil are always allowed). Any extra ingredient makes the overall score 0.
2. **Preference.** How well the answer fits each requested tag. Keyword tags are pass/fail. Macro tags compute the answer's own nutrition from its ingredient lines (USDA) and score calorie shares with a 15% tolerance band, because the calculator is about 15% off at the median. Calorie shares do not depend on servings, so the model cannot game this score by claiming a different serving count.
3. **Feasibility.** Rule checks: at least 2 steps, ingredients used in the steps, sane temperatures (100-600 F / 40-320 C) and times, an oven temperature when baking, amounts given. Optionally blended 50/50 with a 1-5 LLM-judge rating.

`overall = 0.3 x compliance + 0.4 x preference + 0.3 x feasibility`.

**Metrics reported.** Mean overall score; mean of each sub-score; compliance-failure rate (share of answers with an invented ingredient); all broken down by tag and by nutrition source (labelled vs computed). Sanity checks done: the reference recipes scored as if they were model answers average 0.992 (minimum 0.93); an answer with an invented ingredient scores 0.

**Who reviews.** Right now verification is automatic only (label-versus-USDA agreement and the sanity checks above); no person has reviewed the cases yet. Proposed: a hand review of 50-100 sampled cases by the evaluation owner and a second team member, recorded in a `reviewed_by` field.

**Link to fine-tuning.** Training and validation files use the same prompt and answer format (`recipe_eval/prompt.py`) and exclude every test recipe, so a fine-tuned model can be scored with no conversion.

**Known limits.** No adversarial cases and no human review yet. User allergen and dislike profiles, such as the RFC example "lactose intolerant, hates mushrooms", are only partly covered (`dairy_free`, `gluten_free`; dislikes are not modelled). Nutrition is approximate (about 15% at the median), 4,798 pool recipes have none, and Kaggle servings are estimated (about 750 kcal each), not recorded. Feasibility rules catch obvious nonsense, not subtle cooking errors.

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

_TODO_

<!-- END 6 -->

## 7. Plan
<!-- BEGIN 7 (Luca) -->
<!-- Week-by-week timeline from week 6 to week 14, per member, including the experiments planned. This is a commitment, reviewed in week 12 via commit history. -->

_TODO_

<!-- END 7 -->
