
# RFC - Nibble, Recipe Companion

- **Deliverable:** Week 4 RFC

| Team member             | Responsibility                                     |
| ----------------------- | -------------------------------------------------- |
| Roșu Ioan-Luca          | Team leader, agent tools (MCP server), fine-tuning |
| Chelariu Răzvan-Dumitru | Data ingestion                                     |
| Fazacaș Mihai-Tudor     | RAG                                                |
| Ciocîia Marius Andrei   | Toxicity and hallucination handling, fine-tuning   |
| Crețu Vlad-Cristian     | Evaluation                                         |

---
### Overview

**Nibble** is an **agentic LLM-based app** that helps users generate recipes tailored to their dietary preferences (likes, dislikes, allergens), with creative ingredient combinations and substitutions. It runs on a small language model (**SLM**), grounds its recipes in real ones retrieved from the web through **RAG**, and uses a set of tools exposed through an **MCP server**. Every recipe passes a safety layer that double-checks allergens and flags inedible, poisonous or clearly unhealthy combinations.

Example request:
`I'm lactose intolerant, hate mushrooms, and have chicken, rice and spinach; make me dinner.`

Nibble answers with a recipe that respects the user's requests, estimates nutritional information and wraps it all nicely in minimalistic easy-on-the-eyes chat, with descriptive icons that mark ingredients, steps and nutritional value of the ingredients.

---
### Motivation

Recipe generation is one of the most common everyday uses of LLM chatbots. A new personally tailored recipe is a prompt away. But...

General purpose LLMs often meet setbacks in this case. They may confuse subtle but important things, like order of steps or ingredients, combine things that don't make sense together or hyperfocus on a certain ingredient if the user mentions it briefly and, most often, advise wrong quantities, and substitute ingredients poorly. An AI-generated recipe is a sandbox for hallucinations affecting something as essential as the food you eat.

Those shortcomings can be meaningfully reduced, using persistent personalization and allergen profiles, grounding via RAG, safety checks, or even fine-tuning.

---
### Goals

1) **Personalized, creative recipes:** ingest user profiles (likes, dislikes, allergens) and generate recipes for the active profile, optionally from the ingredients the user has at hand, including creative ingredient combinations and substitutions, each with estimated nutritional information. Users can swap any ingredient with one tap if it’s unavailable or disliked; Nibble suggests substitutes and updates quantities, nutrition and allergen checks accordingly.

2) **Grounded answers:** base generated recipes on real recipes retrieved from online sources (RAG), and cite the sources used.

3) **Domain-adapted SLM:** fine-tune an SLM on recipe corpora so it reads and writes recipes in a consistent, structured format (ingredients, quantities, steps).

4) **Agentic reasoning:** let the agent decide which tools to call (search, allergen check, substitution, nutrition) through an MCP server, and reason until a "correct" and satisfactory recipe is provided.

5) **Safety:** adapt or block a recipe containing a profile allergen (checked against the 14 major EU allergens); flag inedible or toxic combinations and unsafe cooking temperatures and times, and compare each serving against adult daily reference values (**WHO**/**EFSA**) for calories, sugar, fat, salt and caffeine: from 50% to 150% of a daily value the recipe shows a warning with the exact share (e.g. "*one serving has 123% of the daily recommended sugar for an adult*"); above 150% it is flagged as unsafe and the agent lowers quantities, substitutes ingredients or switches the recipe.

6) **Measurable quality:** evaluate retrieval relevance, flavor pairing of central ingredients, substitutions and safety violations.

---
### Stack summary

- **Model plan:** a base SLM during development, fine-tuned (***LoRA***) in the final stages.

- **Data:** ***RecipeNLG*** and web recipes, ***USDA FoodData Central*** for nutrition, ***Open Food Facts*** for allergens, ***FlavorDB*** and ***food2vec*** for pairing.

- **"Creativity" metric:** high ***FlavorDB*** overlap with low ***food2vec*** co-occurrence. Novel, but plausible.
