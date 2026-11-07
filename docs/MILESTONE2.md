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

#### **Model Choices, License and Hardware**

| **Criteria** | **Qwen3.5-4B** | **IBM Granite 4.2-3B** | **Ministral 3-3B** |
|:---|:---:|:---:|:---:|
| Parameters | 4B | 3B | 3B |
| Tool calling | Native agentic/tool support | Native reasoning and tool calling | Function calling and structured outputs |
| License | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Main Consideration | Best starting candidate | Particularly interesting for agent orchestration | Compact edge deployment |
| Recomended Hardware and deployment | GPU, 8–12 GB VRAM with more comfortable inference and small-context QLoRA experiments, *maybe more VRAM if we do LoRA tuning* | GPU, 8–12 GB VRAM with more comfortable inference and small-context QLoRA experiments, *maybe more VRAM if we do LoRA tuning* | CPU, 8–16 GB RAM with 4-bit quantization and slower inference **OR** GPU 6–8 VRAM with 4-bit inference |
| What hardware we could actually use | Powerful Gaming PC or a complex distributed architecture if we're all poor | Powerful Gaming PC or a complex distributed architecture if we're all poor | My old crappy laptop with 8GB RAM we could ruin |

---

#### **Fine tuning**

LoRA or QLoRA with Hugging Face's PEFT or TRL libraries:

- https://huggingface.co/docs/peft/index
- https://huggingface.co/docs/trl/sft_trainer

**Suggested initial configuration**

| **Parameter** | **Suggested starting value** |
|:---|:---:|
| Method | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 16 or 32 |
| LoRA alpha | 32 or 64 |
| LoRA dropout | 0.05 |
| Learning rate | 1e-4 to 2e-4 |
| Epochs | 1–3 initially |
| Optimizer | Paged AdamW |
| Gradient checkpointing | Enabled |
| Training objective | Causal language modelling / SFT |

---

#### **Training Data**

| **Dataset component** | **Contents** | **Purpose** |
|:---|:---|:---|
| Recipe corpus | Ingredients, quantities, preparation steps, servings, cooking times | Learn recipe structure and terminology |
| User instructions | Dietary profiles, available ingredients, preferences and dislikes | Learn personalization |
| Agentic trajectories | User request, tool selection, tool results and final answer | Learn tool calling |
| Safety examples | Allergens, toxic ingredients, unsafe cooking conditions and conflicting constraints | Learn when to reject or request verification |

**Recipe Corpus suggestion:** *RecipeNLG https://recipenlg.cs.put.poznan.pl/dataset*

**Recommended dataset splitting strategy:** 80/10/10 split (training, validation, testing)

**Suggested Distribution of training tasks**

| **Task** | **Training proportion** |
|:---|:---:|
| Structured recipe generation | 35% |
| Tool selection and calling | 25% |
| Ingredient substitutions | 15% |
| Dietary restrictions and conflicting preferences | 15% |
| Safety-related responses and refusals | 10% |

---

#### **Task format and training examples**

Conversational instruction-tuning data with structured JSON outputs and explicit tool calls.

Hugging Face TRL supports this type of training through its SFTTrainer, which supports conversational messages, tool-call records, tool responses and a separate tools column containing JSON schemas: 

- *https://github.com/huggingface/trl/blob/main/docs/source/dataset_formats.md*
- *https://huggingface.co/docs/trl/sft_trainer*

**Example of JSON task format:**

```json
{
  "recipe": {
    "title": "Lemon Garlic Chicken with Spinach Rice",
    "servings": 2,
    "preparation_time": 10,
    "cooking_time": 25,
    "ingredients": [
      {
        "name": "Chicken breast",
        "quantity": 300,
        "unit": "g"
      },
      {
        "name": "Rice",
        "quantity": 150,
        "unit": "g"
      },
      {
        "name": "Spinach",
        "quantity": 100,
        "unit": "g"
      }
    ],
    "steps": [
      {
        "step": 1,
        "instruction": "Cook the rice according to its instructions."
      },
      {
        "step": 2,
        "instruction": "Cook the chicken thoroughly."
      },
      {
        "step": 3,
        "instruction": "Add spinach, garlic and lemon."
      }
    ]
  },
  "nutrition": {
    "calories_kcal": 420,
    "fat_g": 12,
    "sugar_g": 3,
    "salt_g": 1.2,
    "caffeine_mg": 0
  },
  "safety": {
    "status": "approved",
    "allergen_checks": [],
    "warnings": []
  },
  "sources": []
}
```

---

#### **Summary and experimental plan**

| **Component** | **Choice** |
|:---|:---|
| Primary SLM | Qwen3.5-4B |
| Comparative SLM | Granite 4.2-3B |
| Additional baseline | Ministral 3-3B |
| Fine-tuning method | QLoRA |
| Fine-tuning framework | Hugging Face TRL + PEFT |
| Training objective | Supervised Fine-Tuning |
| Training data | Recipe corpus + synthetic conversations + MCP trajectories |
| Data split | 80% / 10% / 10%, grouped by recipe family |
| Output format | Structured JSON |
| Tool format | Conversational messages with JSON-schema function calls |
| Retrieval | Separate RAG pipeline |
| Safety | Deterministic validation services |
| Evaluation | Held-out recipe families and adversarial safety test cases |

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
