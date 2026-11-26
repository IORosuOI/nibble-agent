# Nibble

An agentic recipe assistant on a fine-tuned SLM, with RAG over real recipes, MCP tools, and allergen and nutrition safety checks.

## Documents
- [Milestone 2: architecture and plan](docs/MILESTONE2.md)
- [Milestone 1 feedback](milestone1-feedback.md)

## Repository layout
Each folder has a README with its purpose and owner.

The recipe pool is in `data/recipes_merged.jsonl`, created by the cleaning and merge steps in `recipes.ipynb` and processed by `scripts/add_computed_nutrition.py`.
