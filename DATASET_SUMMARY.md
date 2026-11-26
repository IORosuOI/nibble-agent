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
- Keyword tags (from the ingredient list AND the steps, approximate): vegetarian, vegan, dairy_free, gluten_free, spicy. Steps are read because some source ingredient lists are cut short: a recipe whose steps brown beef used to be tagged vegetarian. Reading the steps removed the tag from 470 vegetarian, 546 vegan, 780 dairy_free and 2,021 gluten_free recipes. Kitchen tools (cookie sheet, muffin tin, pastry blender) and verbs (roll out) are not counted as ingredients, and "vegetarian sausage" is not meat. `spicy` still reads only the ingredient list. The diet tags also apply the allergen matcher of section 12, so a `gluten_free`, `dairy_free`, `vegan` or `vegetarian` recipe does not contradict its allergens (the one exception is a line reading "vegan fish sauce substitute", which stays flagged to be safe).
- Tag counts in the merged set (after the calorie cap and the steps fix): vegetarian 13,531; gluten_free 9,811; dairy_free 9,378; high_carb 5,870; spicy 4,142; vegan 3,536; low_carb 3,543; low_calorie 2,662; high_protein 2,574; keto 1,002.
- Thresholds are my choices, picked so no tag is tiny and so the share medians of both sources match (protein/carb/fat about 10/45/44%); change them in `labels.THRESHOLDS`.

## 6. Evaluation examples (`data/eval_examples.jsonl`, built by `scripts/build_examples.py`)
1,200 examples, 120 per tag. Each has the available ingredients (the recipe's non-staple ingredients plus 0-2 distractors the model need not use), 1-2 mood tags, the golden recipe (title, servings, ingredients, steps), its `nutrition_basis`, and either per-serving label nutrition (xiaoxl) or computed totals (Kaggle). For macro tags, up to half of the examples come from labelled xiaoxl rows and the rest from computed Kaggle rows; `low_calorie` is xiaoxl only. After the steps fix, 76 of the 1,200 cases had a requested tag that no longer held (gluten_free 36, vegan 20, dairy_free 16, vegetarian 13). `scripts/patch_eval_tags.py` kept 1,124 cases, dropped a wrong second tag in 20, and replaced 56 recipes whose main tag was wrong (same `example_id`, new `recipe_id`; the list is in `data/eval_patch_report.json`). A second pass after the EU alignment (section 12) changed 4 more cases: 3 recipes replaced and 1 second tag dropped (key `eu_alignment` in the same report). The 120-per-tag balance is unchanged.

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
| `data/train_examples.jsonl` | Training set: same request format, `messages` (user prompt -> assistant JSON recipe) + `target` | 19,300 | Send to training |
| `data/val_examples.jsonl` | Validation split of the same | 1,015 | Training-time validation |
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
- `usda_sr/` (38 MB, public domain) is needed by the calculator. Anyone who clones the repo must download USDA SR Legacy into `usda_sr/` before running the scripts or the scorer; `scripts/download_data.py` (section 15) installs it together with the pool.

## 11. How to get the ~20k training entries (rebuild from scratch)
Short answer: the five scripts alone do not give you the data. Two steps live outside them: the cleaning/dedup/merge cells of `recipes.ipynb`, and a dedup of xiaoxl against Kaggle that was done by hand and has no script (see the note at the end).

### Where the numbers come from
| Step | Input -> output | Rows after the step |
|---|---|---|
| Notebook: clean, dedup, merge | Kaggle (62,126 raw) + `data/xiaoxl_nutrition_checked.csv` -> `data/recipes_merged.jsonl` | 28,254 |
| `add_computed_nutrition.py` | drops recipes over 4,000 kcal in total (4,859), adds computed totals, estimates Kaggle servings | **23,395** (the pool) |
| `build_examples.py 120 7` | picks 120 recipes per tag -> `data/eval_examples.jsonl` | 1,200 test recipes |
| `build_train.py` | removes the 1,200 test recipes, then keeps recipes with at least 3 non-staple ingredients and at least 1 mood tag | 22,195 -> **20,315 usable** |
| split inside `build_train.py` | 5% validation (`val_fraction` 0.05, seed 11) | **19,300 train + 1,015 val** |

So "20k entries" means train plus validation, 20,315. About 1,880 recipes are lost in the last filter: their ingredient list has fewer than 3 non-staple ingredients after cleaning, or none of the mood tags applies. The total of train, validation and test is 21,515.

### Commands (run from the project root)
Needs Python with `pandas` and `pyarrow`, the USDA tables in `usda_sr/` (not in git), and `data/ingredient_map.json` (in git).
```
# 1. only if you need to rebuild the xiaoxl input (slow; downloads about 20 shards)
python scripts/pull_xiaoxl.py
python scripts/verify_xiaoxl.py        # needs data/xiaoxl_nutrition_new.csv, see the note below

# 2. run the cleaning, dedup, xiaoxl-merge and "save merged" cells of recipes.ipynb
#    (this writes data/recipes_merged.jsonl, 28,254 rows)

# 3. nutrition, cap and servings (rewrites recipes_merged.jsonl in place; takes a few minutes)
python scripts/add_computed_nutrition.py     # -> 23,395 rows

# 3b. retrieval fields: ingredients_norm, url, allergens, allergen_status (section 13), also rewrites data/recipe_allergens.json
python scripts/add_pool_fields.py

# 4. test set, then training files (order matters: training excludes the test recipes)
python scripts/build_examples.py 120 7       # -> data/eval_examples.jsonl (1,200)
python scripts/build_train.py                # -> data/train_examples.jsonl (19,300), data/val_examples.jsonl (1,015)
```
Seeds are fixed (7 for the test set, 11 for the training split), so the same pool gives the same files. If you change the pool, the cap (`MAX_TOTAL_KCAL`), the tag thresholds or the prompt wording, re-run steps 3 and 4 in order, because the test set decides which recipes the training set may use.

### Check the result
```
python -c "import pandas as pd; print(len(pd.read_json('data/recipes_merged.jsonl', lines=True)))"   # 23395
wc -l data/eval_examples.jsonl data/train_examples.jsonl data/val_examples.jsonl                      # 1200, 19692, 1036
```

### Known gap: the rebuild is not fully scripted
- `scripts/verify_xiaoxl.py` reads `data/xiaoxl_nutrition_new.csv`: the xiaoxl rows with nutrition after removing any recipe that matches the Kaggle set by title, directions or ingredient list. That dedup was run by hand and is not in a script or in the notebook.
- The notebook's merge cell reads `data/xiaoxl_nutrition_checked.csv` (22 MB), the output of `verify_xiaoxl.py`. It is not committed.
- Fastest way to reproduce exactly: obtain `data/xiaoxl_nutrition_checked.csv` from the person who built it (it is on their disk), put it in `data/`, and start at step 2. Rebuilding it from the raw shards needs the missing dedup step to be written first.

## 12. Allergens: how they are found and how to get them
The families follow Annex II of Regulation (EU) No 1169/2011, the list used in section 6 of `docs/MILESTONE2.md`. Following that list does not make the tags a legal or medical guarantee: they come from keyword matching on recipe text.

Every recipe in the pool is mapped to the allergens it contains, using the 14 EU allergen families. This is separate from the keyword mood tags (`dairy_free`, `gluten_free`, `vegan`, `vegetarian`), although those now read the steps with the same matcher.

### Files
| File | What it is |
|---|---|
| `data/allergen_map.json` | The hand-written list: which ingredient word brings which allergen. |
| `data/allergen_families.json` | The 14 EU families, their code in the data, and aliases a profile may use ("peanuts", "dairy", "tree nuts"). |
| `recipe_eval/allergens.py` | The matcher. `allergens_in_line("2 tbsp soy sauce")` gives `{soy, wheat, gluten}`; `recipe_allergens(ingredient_lines, step_lines)` gives the recipe result; `normalize_allergens(["Peanuts", "dairy"])` gives `["milk", "peanut"]`; `leaks(profile_allergens, ingredient_lines, step_lines)` lists profile allergens found in a recipe. |
| `scripts/add_pool_fields.py` | Runs the matcher over the pool, adds the fields below to `data/recipes_merged.jsonl` and rewrites `data/recipe_allergens.json`. |
| `data/recipe_allergens.json` | `{recipe_id: {"allergens": [...], "steps_only": [...], "allergen_status": "verified" | "unverified"}}` for all 23,395 recipes (2.2 MB). |
| `tests/test_allergens.py` | 57 cases for the matcher, plus a check of the 14 official family names (`python tests/test_allergens.py`). |

### The allergens (codes used in the data)
| Code | EU family | Notes |
|---|---|---|
| `gluten` | Cereals containing gluten | wheat (incl. spelt, khorasan, durum, farro, emmer, einkorn, freekeh), rye, barley, oats (also granola, muesli, oat milk), malt, beer; lines that say gluten free are not flagged |
| `crustaceans` | Crustaceans | shrimp, prawn, crab, lobster, crayfish |
| `egg` | Eggs | also mayonnaise, meringue, aioli |
| `fish` | Fish | also fish sauce, worcestershire, anchovy |
| `peanut` | Peanuts | peanut oil counts (cautious) |
| `soy` | Soybeans | soy sauce, tofu, miso, edamame |
| `milk` | Milk | butter, cheese, cream, whey, yogurt; plant milks do not count |
| `tree_nut` | Nuts | coconut and nutmeg do not count |
| `celery` | Celery | celery, celeriac; stock or bouillon cubes are assumed to contain it |
| `mustard` | Mustard | dijon, mustard seed; curry powder is assumed to contain it |
| `sesame` | Sesame | tahini, hummus |
| `sulphites` | Sulphur dioxide and sulphites | wine, sherry, vermouth, dried apricots (usually contain them) |
| `lupin` | Lupin | lupin flour |
| `molluscs` | Molluscs | clams, mussels, oysters, scallops, squid |
Two convenience tags are added: `wheat` (always together with `gluten`) and `shellfish` (crustaceans or molluscs; older filters keep working). The final names belong to the safety owner; change them in `data/allergen_families.json`.

### How `allergen_map.json` works
1. `keywords`: words that bring an allergen (`butter`, `cheese` give milk; `shrimp` gives crustaceans; `flour`, `bread`, `pasta` give wheat). Whole words only, plural allowed.
2. `compound_ingredients`: foods that carry several allergens (`soy sauce` gives soy and wheat; `mayonnaise` gives egg; `caesar dressing` gives fish, egg and milk; `pesto` gives milk and tree nut; `oyster sauce` gives molluscs, soy and wheat).
3. `not_allergen`: phrases where a keyword does not apply (`peanut butter` is not milk, `coconut milk` is not milk, `almond flour` is not wheat, `lupin flour` is not wheat, `oyster mushroom` is not a mollusc, and kitchen tools such as `cookie sheet`, `muffin tin`, `pastry blender`, `roll out`).
4. `free_from_phrases`: a line that says `gluten free`, `dairy free`, `non dairy`, `vegan`, `egg free`, `nut free`, `soy free` or `sulphite free` loses the matching allergens for that line.
5. `implied`: wheat gives gluten; crustaceans and molluscs give shellfish.

### How a recipe is mapped
- Every ingredient line and every sentence of the steps is checked.
- `allergens` = everything found in the ingredient list or the steps (the wide, safe answer).
- `steps_only` = found only in the steps, not in the ingredient list. That means the ingredient list is incomplete (some Kaggle lists are cut short).
- `allergen_status` = `verified` or `unverified`. A recipe is **unverified** when an allergen shows up only in the steps, or when fewer than 80% of its ingredient lines are recognised in the nutrition database (we cannot vouch for a list we do not understand). An allergic user should only be shown `verified` recipes. `allergen_notes` says why (`steps_only:milk`, `low_ingredient_coverage`).

### Result on the pool (23,395 recipes)
- 16,760 recipes are verified, 6,635 are unverified (4,252 only because of low ingredient coverage; 2,383 have an allergen found only in the steps, some also have low coverage). Verified by source: xiaoxl 3,211 of 3,346; Kaggle 13,549 of 20,049.
- 20,531 recipes have at least one allergen and 2,864 have none. Counts: milk 13,980; gluten 13,478; wheat 13,036; egg 7,948; tree_nut 2,513; celery 1,925; fish 1,902; soy 1,753; sulphites 1,747; mustard 1,599; shellfish 991; sesame 886; crustaceans 781; peanut 566; molluscs 296. Lupin appears in none of the recipes.
- Audit of the whole pool: no recipe that mentions oats, wheat, rye, barley, spelt, kamut, durum, granola or muesli is left without a `gluten` tag unless the matcher's gluten-free or exclusion phrases apply, and no `gluten_free` or `dairy_free` recipe contains that allergen.

### How to rebuild
`python scripts/add_pool_fields.py` from the project root (about 2 minutes). Run it again whenever the pool changes. Test the matcher with `python tests/test_allergens.py`.

### Limits
- The list is hand-written and approximate. It errs toward flagging, because a missed allergen is worse than a false one.
- It cannot see hidden ingredients (lecithin, a store-bought sauce or stock) or cross-contact. Sulphites and celery are assumed from a few typical foods, so they are the least reliable families.
- Chocolate is not flagged unless it is milk or white chocolate (or a spread such as Nutella): plain and semisweet chocolate may contain milk, soy or nuts, which only the product label can tell. Bread-like words (`roll`, `crust`, `bun`) count as wheat unless the line says gluten free.
- Not human-reviewed. The leak check of golden set v2 uses this same matcher, so an allergen it cannot see is missed in the data and in the check alike.

## 13. Pool fields for retrieval (what the RAG index reads)
`scripts/add_pool_fields.py` adds these fields to every record of `data/recipes_merged.jsonl`:
| Field | Type | Meaning |
|---|---|---|
| `ingredients_norm` | list of str | Plain singular ingredient names, in recipe order, no duplicates (`"2 cloves garlic, minced"` gives `garlic`; `"1/2 pound morel mushrooms, chopped"` gives `morel mushroom`). Amounts, units and preparation words are removed; only the first of "A or B" is kept; "salt and pepper" gives two names. Staples are kept; filter them with `recipe_eval.scoring.is_staple`. Built by `recipe_eval/normalize.py`. |
| `allergens` | list of str | Allergen codes found in the ingredients or the steps (section 12). |
| `steps_only` | list of str | The part of `allergens` found only in the steps. |
| `allergen_status` | `"verified"` or `"unverified"` | See section 12. |
| `allergen_notes` | list of str | Why a recipe is unverified. |
| `url` | str | Source link. xiaoxl recipes (3,346 of 3,346) have the allrecipes.com link; Kaggle recipes have `""` because that dataset has no links. |
| `content_key` | str | 12-hex hash of the normalized title plus the sorted `ingredients_norm`. It does not depend on row order, so it shows whether a regenerated pool holds the same recipes. 23,393 distinct values for 23,395 recipes (2 pairs of near-identical recipes). |

**Stable IDs.** `recipe_id` (`r00000`...) is the row number given when the notebook merges Kaggle and then xiaoxl. The same inputs and the same code give the same ids, and the 4,859-recipe calorie cap kept the original ids (so there are gaps). It is not a content hash: if the cleaning or dedup rules change, ids shift. So treat `data/recipes_merged.jsonl` as the source of truth and do not regenerate it casually; if it must be regenerated, match old and new recipes by `content_key` and remap before re-indexing.
**Recipe tags** (`high_protein`, `vegan`...) are not stored in the pool; `labels.row_tags(row)` computes them (section 5).

## 14. Golden set v2: profiles, cravings and traps
`data/golden_v2.jsonl`, 560 cases, built by `scripts/build_golden_v2.py` (seed 21) from the 1,200 golden-set recipes, so no training recipe is reused. It tests what the first set cannot: user allergens, dislikes, free-text cravings and requests with no valid answer.

### Case format
```
{"case_id": "v2-0001", "type": "allergen_trap", "expected": "recipe",
 "profile": {"allergens": ["milk"], "dislikes": []},
 "request": {"ingredients": [...], "tags": [...], "craving": null, "profile": {...same profile...}},
 "reason": "why this case exists", "recipe_id": "r00042" or null, "golden": {...} or null, "blocked_items": [...]}
```
`profile.allergens` use the codes of section 12 (`normalize_allergens` accepts "peanuts", "dairy", "tree nuts"). `request` alone is enough to build the prompt with `recipe_eval.prompt.render_prompt`.

### Case types
| Type | n | What it tests | Expected |
|---|---|---|---|
| `allergen_clean` | 120 | Profile allergen is neither in the pantry nor in the reference recipe | A recipe with no leak |
| `allergen_trap` | 150 | A pantry item carries the profile allergen (for example milk in the pantry, milk allergy) and at least 3 safe items remain | A recipe that skips that item |
| `impossible` | 60 | Every pantry item carries the allergen | A refusal |
| `dislike_trap` | 100 | The profile dislikes one pantry ingredient | A recipe without it |
| `conflicting_mood` | 30 | keto + high_carb, low_carb + high_carb | A refusal or a `warning` |
| `craving` | 100 | The mood is also written as free text ("Something with a real kick.") | A recipe |

### How a case is scored (`recipe_eval/golden_v2.py`)
- A **refusal** is `{"refusal": "reason"}` or an empty ingredient list (`prompt.parse_response` understands it). The prompt only mentions this option when the profile has allergens.
- **Leak** = an allergen from the profile found in the answer's ingredient lines or steps. A leak gives overall 0 (`scoring.score_response` does this for any request that has a `profile`).
- **Dislike violation** = a disliked ingredient used in the answer. Reported, not a hard gate (a dislike is a preference).
- `score_case(case, response, calc)` returns `score`, `passed`, `leak`, `dislike_hits`, `refused`. `leak_rate(results)` is the share of answered cases that leak; it must be 0. `summarize(results)` gives per-type mean score and pass rate, the leak rate and the dislike-violation rate.
- Refusing where a safe recipe exists scores 0; answering where none exists scores 0.

### Checks done
The reference recipes pass their own `allergen_clean` cases (120 of 120). `python tests/test_golden_v2.py` checks the scoring on real cases: leaking answers are caught, skipping a blocked item is safe, refusals are right only for `impossible` and `conflicting_mood`, and the leak rate counts correctly.

### Limits
- No poisonous or unsafe ingredient cases yet (that blocklist belongs to the safety owner).
- Not human-reviewed. Cases are generated from the pool, so odd pantries (a leftover distractor) can occur.
- A leak is detected with the same matcher that built the data; hidden allergens are invisible to both.
- Prompts with a profile are not part of the training data (training examples have no profile), so a fine-tuned model has not seen refusals or allergen rules.

## 15. Get the data (without rebuilding)
The large files are not in git: `data/recipes_merged.jsonl` (the recipe pool), `data/train_examples.jsonl`, `data/val_examples.jsonl` and `usda_sr/`. They are packed in `nibble_data.zip`.
```
python scripts/download_data.py                        # downloads from the GitHub release data-v1
python scripts/download_data.py --zip nibble_data.zip  # or installs a zip a teammate sent you
```
The script checks the SHA-256 of the pool, train and validation files, and does not overwrite existing files unless you add `--force` (needed if you have an older pool; it also replaces local `data/` files that share a name with files in the zip). The download works only once a release tagged `data-v1` with `nibble_data.zip` attached has been published; until then use `--zip`. When the data changes, rebuild the zip, update the checksums in the script and publish a new release.
