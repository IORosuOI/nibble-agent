# Recipe golden dataset: what was built and how to score a model on it

## 1. Goal
Evaluate a model that takes (ingredients the user has + a mood such as "spicy" or "high-protein") and returns a full recipe.
The evaluation checks that the recipe is built around the user's ingredients, nutritional fit, and culinary feasibility.

## 2. Source datasets
| Dataset | Used for | Notes |
|---|---|---|
| Kaggle `prashantsingh001/recipes-dataset-64k-dishes` | Base recipes (62,126 rows) | Title, category, ingredients, directions. **No nutrition, no servings.** |
| Hugging Face `xiaoxl/recipe_data` (MIT) | Recipes with nutrition (107,064 rows, 20 parquet shards) | Only 24,367 rows have servings plus calories/protein/carbs/fat; 76% of rows have none. Rows with nutrition link to allrecipes.com; the card does not name its source. |
| USDA FoodData Central, SR Legacy (public domain) | Computing macros from ingredient lists | 7,793 foods, per-100 g macros, household-measure gram weights. |

Looked at and rejected: `Humanbased-AI/MM-Food-100K` (food photos; no cooking steps, ingredient lists have no quantities, macros are model estimates from photos, non-commercial license) and RecipeNLG (no nutrition, non-commercial research license).

## 3. How the dataset was curated
**Kaggle set (62,126 rows -> 23,785):**
1. Parsed ingredient and direction lists.
2. Dropped recipes with fewer than 4 ingredients (many 1-2 ingredient rows were truncated lists, e.g. only `cooking spray`, which breaks ingredient-compliance scoring).
3. Dropped recipes with under 60 characters of directions, and Pet Food / DIY categories (78 rows).
4. Deduplicated on normalized title + sorted normalized ingredients. About 60% of rows were the same recipe listed under several categories. Kept the row with the most specific subcategory and stored every category in `all_categories`.
5. Single-step recipes were flagged (`single_step`), not removed; sampled ones were genuinely usable.

**xiaoxl set (24,367 rows with nutrition -> 4,469 in the merged set):**
1. Kept rows with servings and all four macros.
2. Removed any row matching the Kaggle set by title, directions, or ingredient list (12,980 of 24,367 matched; 4,310 title-only matches were excluded to be safe). 11,346 remained.
3. Applied the same quality rules (4 or more ingredients, 60 or more direction characters).
4. **Macro verification:** recomputed per-serving macros from the ingredient lines using USDA data and compared with the label. A row is `macro_verified` only if ingredient coverage is 80% or more, calories are within 25%, protein/carbs/fat are within 40%, and the label's own calories fit its macros (4/4/9). Of 11,346 rows: 4,597 verified, 910 `label_suspect` (calories off by over 60% despite good coverage), 2,189 not checkable (low coverage), the rest partial disagreements. Only verified rows were merged.

**Merged set:** `data/recipes_merged.jsonl`, 23,395 recipes (Kaggle 20,049; xiaoxl 3,346) after the calorie cap below. Column `source` records origin. Before the cap it had 28,254 (Kaggle 23,785; xiaoxl 4,469).

**Calorie cap (`scripts/add_computed_nutrition.py`, `MAX_TOTAL_KCAL = 4000`):** big-batch recipes (popcorn for a crowd, whole turkeys, fruitcakes, 40,000 kcal of cookie dough) are useless for a home-cooking app, so every recipe whose whole-recipe total is above 4,000 kcal is dropped. For Kaggle rows the total is the computed USDA total; for xiaoxl rows it is per-serving kcal x servings. This removed 4,859 recipes (Kaggle 3,736; xiaoxl 1,123). The cap is one constant; 3,000 would remove about 2,500 more Kaggle recipes.

**Computed nutrition and servings for Kaggle rows (`scripts/add_computed_nutrition.py`):** Kaggle has no servings. For the 15,251 Kaggle recipes with at least 80% ingredient coverage and a total of at most 4,000 kcal, USDA-computed totals are stored (`calc_*_total`) and `nutrition_basis = "computed, no servings"` (the name is kept for compatibility). Their `servings` is now set by rule to `round(total kcal / 750)`, minimum 1 (so 1 to 5), so that one serving is about 750 kcal; these rows carry `servings_estimated = true` and must never be treated as labels. xiaoxl rows are `"label, agrees with USDA recompute"` (3,346) and keep their real servings and per-serving label values; the remaining 4,798 Kaggle rows have `"none"` and no servings. Computed values are estimates, not verified.

**Servings estimation experiment (a model-based estimate was not applied; the 750 kcal rule above is used instead):** could per-serving values be recovered for Kaggle rows? Tested on 200 xiaoxl recipes with known servings. Servings stated in text: found for only 1% of Kaggle rows, and right 29% of the time on xiaoxl. Guessing from total calories: within 25% of the true servings 20-27% of the time. A Haiku model estimating from ingredients and steps: exact 45%, within 25% 52%, within 50% 74%, within 2x 86% (the model may partly recall these well-known recipes, so this is optimistic for new recipes). Using those estimates, per-serving calories were within 25% of the label 56% of the time, and the low_calorie tag agreed 90% (vs 79% for always answering "low"). Because only about half are right and running it on thousands of rows is costly, the model estimate was NOT used. Servings for Kaggle rows are instead a normalising choice (about 750 kcal per serving), not a recovered fact; guessing servings from total calories was right only about a quarter of the time, so treat them as approximate. `low_calorie` still exists only for xiaoxl rows, because every estimated Kaggle serving is about 750 kcal by construction.

## 4. How the ingredient-to-nutrition calculator works (`recipe_eval/nutrition.py`)
Parse each line (amount, unit, name), match the name to a USDA food, convert to grams, sum macros, divide by servings. Matching order: hand overrides (about 270 common ingredients), cached model-assisted matches (`data/ingredient_map.json`, 1,487 names, produced by Haiku subagents choosing among USDA candidates, with ingredient lines that said "to taste" or "for frying" skipped), then a word-overlap fallback.
Measured accuracy against the labels on 1,500 sampled recipes: 81% reached 80% or more ingredient coverage; among those, calories were within 25% of the label for 65% (median error 15%; protein 16%, carbs 14%, fat 15%). Treat results as roughly +/-15-20%, and use tolerance bands.

## 5. Labels (`recipe_eval/labels.py`, thresholds in `THRESHOLDS`)
- Macro tags are defined on **shares of calories**, so they need no servings and work for labelled and computed rows alike: high_protein (protein 25% or more of calories), high_carb (carbs 55% or more), low_carb (carbs 20% or less), keto (carbs 10% or less and fat 60% or more). Computed (Kaggle) shares must clear a threshold by 3 points (`COMPUTED_MARGIN`), because computed shares are about 1-3 points off.
- `low_calorie` (400 kcal or less per serving) needs real servings, so it exists only for the xiaoxl rows.
- Keyword tags (from the ingredient list, approximate): vegetarian, vegan, dairy_free, gluten_free, spicy.
- Tag counts in the merged set (after the calorie cap): vegetarian 14,043; gluten_free 11,877; dairy_free 10,166; high_carb 5,870; spicy 4,142; vegan 4,079; low_carb 3,543; low_calorie 2,662; high_protein 2,574; keto 1,002. (Before computed nutrition: high_protein 688, keto 487.)
- Thresholds are my choices, picked so no tag is tiny and so the share medians of both sources match (protein/carb/fat about 10/45/44%); change them in `labels.THRESHOLDS`.

## 6. Evaluation examples (`data/eval_examples.jsonl`, built by `scripts/build_examples.py`)
1,200 examples, 120 per tag. Each has the available ingredients (the recipe's non-staple ingredients plus 0-2 distractors the model need not use), 1-2 mood tags, the golden recipe (title, servings, ingredients, steps), its `nutrition_basis`, and either per-serving label nutrition (xiaoxl) or computed totals (Kaggle). For macro tags, up to half of the examples come from labelled xiaoxl rows and the rest from computed Kaggle rows; `low_calorie` is xiaoxl only.

## 7. How a model is scored on one ground-truth entry
Input: `request = {ingredients, tags}` from the example. The model returns `{servings, ingredients (with amounts), steps}`. `score_response(request, response, calc)` computes:
1. **Compliance (hard gate).** The recipe must use at least one of the ingredients the user listed (staples are always allowed and do not count: salt, pepper, water, oil, ice, cooking spray). Extra ingredients the user did not list are allowed. A recipe that uses none of the user's ingredients gives overall 0. Also reports `usage` (share of the available ingredients used), `extras` (the added ingredients) and `pantry_share` (share of the recipe's non-staple ingredients that came from the user). `scoring.EXTRA_PENALTY` (default 0, so extras are free) can be raised to lower the compliance score as the share of extras grows.
2. **Preference.** For each requested tag: keyword tags are pass/fail on the answer's ingredients; macro tags use the calculator on the answer's own ingredients and are scored on calorie shares (so the servings the model claims cannot change the score), 1 when inside the threshold band (15% tolerance) and falling linearly to 0 outside it. Only `low_calorie` uses per-serving calories; there the model's claimed servings are clamped to 0.67x-1.5x the golden recipe's servings (`reference_servings`). Averaged over tags.
3. **Feasibility.** Rule checks (at least 2 steps, ingredients actually used in the steps, sane temperatures and times, oven temperature given when baking or roasting, amounts given), optionally blended 50/50 with a 1-5 LLM-judge rating (prompt provided in `scoring.LLM_JUDGE_PROMPT`).
4. **Overall** = 0 if compliance fails, else 0.3 x compliance + 0.4 x preference + 0.3 x feasibility. Report the three sub-scores too.
Model performance = the mean of these per-entry scores, broken down by tag, plus the compliance-fail rate (share of answers that use none of the user's ingredients) and the average share of extras.
The golden recipe is the reference: it supplies the request (ingredients, tags). The answer is judged against the request and rules, not by text match to the golden recipe. Sanity check: golden recipes scored as answers average 0.992 (minimum 0.93); a recipe with no ingredient from the user's list scores 0; adding extra ingredients does not trigger the gate.

## 8. Limitations
- Not human-verified. "Verified" means two imperfect methods (label and USDA recompute) agree, and the quality rules passed. No person has reviewed a sample yet.
- Nutrition for 15,251 Kaggle rows is computed (totals/shares only), not verified; their servings are estimated as total / 750 kcal, not recorded; 4,798 Kaggle rows have no nutrition and no servings. Macro tags for computed rows are consistent with the same calculator that scores model answers, so they are partly self-consistent by construction; the labelled xiaoxl rows are the independent check.
- Calculator error is about 15% at the median; keyword tags are approximate; compliance matching uses word overlap (with plural handling), so near-miss ingredient names can count as a match (or a miss). Because extras are allowed, the model can add anything it likes, so only the 'uses at least one' gate is hard; check `extras` and `pantry_share` to see how far answers drift from the pantry. Training targets still use only the listed ingredients (1:1 with the source recipe), so a fine-tuned model will mostly stay inside the list.
- Feasibility is rule-based and lenient (an absurd bake time still scored 0.85 overall); an LLM judge or human check is recommended before trusting it.
- The scorer has only been tested on golden recipes and hand-made bad answers, not on real model output.
- xiaoxl license is labeled MIT, but its recipe text appears scraped from a recipe site; check before redistributing.

## 9. Files and how they link
All keys are `recipe_id` (`r00000`...), from `data/recipes_merged.jsonl`.

| File | What it is | Rows | Use |
|---|---|---|---|
| `data/recipes_merged.jsonl` | The recipe pool: every cleaned recipe with source, tags inputs, nutrition | 23,395 | Source of truth; do not train on it directly |
| `data/eval_examples.jsonl` | Test set: request (ingredients + mood + distractors) + golden recipe + nutrition | 1,200 | Evaluation only |
| `data/train_examples.jsonl` | Training set: same request format, `messages` (user prompt -> assistant JSON recipe) + `target` | 19,692 | Send to training |
| `data/val_examples.jsonl` | Validation split of the same | 1,036 | Training-time validation |
| `data/ingredient_map.json` | Ingredient name -> USDA food id (used by the nutrition calculator) | 1,487 | Scoring |
| `data/xiaoxl_nutrition_checked.csv` | xiaoxl rows with label vs USDA-recompute comparison (intermediate) | 11,346 | Audit trail |

Train and eval are disjoint: `build_train.py` excludes every `recipe_id` in `eval_examples.jsonl` (checked with an assert). Train, val and eval share one prompt format (`recipe_eval/prompt.py: render_prompt`) and one answer format (JSON with `title`, `servings`, `ingredients`, `steps`), so a model trained on `train_examples` is scored with `parse_response` + `score_response` on `eval_examples` without any conversion.
Code: `recipes.ipynb` (cleaning, merging, labels, examples), `recipe_eval/` (core, nutrition, overrides, labels, scoring, prompt), `scripts/` (`pull_xiaoxl.py`, `verify_xiaoxl.py`, `add_computed_nutrition.py`, `build_examples.py`, `build_train.py`).
Training caveat: the recipe text comes from recipe sites (Kaggle scrape; xiaoxl rows link to allrecipes.com). Check the licenses before training or releasing a model commercially. Kaggle training targets carry estimated servings (about 750 kcal per serving, see section 3); 3,973 training targets (Kaggle rows with no nutrition) still have `servings: null`.

## 10. What is in the repo and why
### Library: `recipe_eval/` (scores answers and computes nutrition)
| File | What it does | Why it is needed |
|---|---|---|
| `__init__.py` | Empty. | Makes the folder an importable package. |
| `core.py` | Parses ingredient lines (amount, unit, name), loads USDA, converts to grams. | Everything else needs ingredient text turned into grams. |
| `overrides.py` | About 270 hand-written ingredient-to-USDA matches. | Word matching gets common items wrong (flour, butter, eggs); hand matches are applied first. |
| `nutrition.py` | The calculator: ingredient lines to kcal, protein, carbs, fat. | Checks xiaoxl labels, fills nutrition for Kaggle rows, measures a model's answer. |
| `labels.py` | Rules that turn nutrition and ingredients into mood tags (keto, high_protein, vegan, spicy, ...). | The tags are the "mood" in a request and the scorer checks them; thresholds live in one place. |
| `scoring.py` | The scorer: compliance, preference, feasibility, overall. | This is how a model's answer is graded. |
| `prompt.py` | One prompt format and one answer format (JSON). | Training and testing use the same format, so a trained model needs no conversion. |

### Scripts: `scripts/` (build the data; re-runnable)
Run order: `pull_xiaoxl.py`, `verify_xiaoxl.py`, the cleaning and merge cells of `recipes.ipynb`, `add_computed_nutrition.py`, `build_examples.py`, `build_train.py`. `build_train.py` must run after `build_examples.py` because it excludes the test recipes.
| File | Why it is needed |
|---|---|
| `pull_xiaoxl.py` | Downloads xiaoxl in pieces and keeps only rows with nutrition. |
| `verify_xiaoxl.py` | Recomputes xiaoxl macros and compares them with the labels, so only agreeing rows are kept. |
| `add_computed_nutrition.py` | Adds computed totals for Kaggle rows, drops recipes over 4,000 kcal, estimates servings at about 750 kcal each. |
| `build_examples.py` | Builds the 1,200 golden test cases (ingredients, distractors, mood tags, reference recipe). |
| `build_train.py` | Builds training and validation files, leaving out every test recipe. |

### Notebook and docs
- `recipes.ipynb`: runs the whole flow in order (download, clean, dedup, merge, nutrition, labels, examples, scoring demo). It is the readable record of how the dataset was made.
- `DATASET_SUMMARY.md`: this file.
- `docs/MILESTONE2.md`: the team file; only sections 1 and 2 are filled in.

### Committed data
- `data/eval_examples.jsonl`: the golden test set (1.7 MB); it is the deliverable and small enough to commit.
- `data/ingredient_map.json`: the 1,487 ingredient-to-USDA matches made by Haiku subagents; slow to recreate and the calculator needs it.

### Not committed, on purpose
- Big generated files: `data/recipes_merged.jsonl`, `data/train_examples.jsonl`, `data/val_examples.jsonl` (47, 74 and 4 MB); the scripts regenerate them. `data/xiaoxl_nutrition_checked.csv` (22 MB audit trail) is also left out.
- Raw downloads: Kaggle, xiaoxl and the USDA folders.
- `usda_sr/` (38 MB, public domain) is needed by the calculator. Anyone who clones the repo must download USDA SR Legacy into `usda_sr/` before running the scripts or the scorer; a download step is not automated yet.
