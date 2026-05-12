# Data

This directory should contain the training data for TEAR. Due to sensitivity of the content, we do not provide the actual data. Below are the expected formats.

## SFT Training Data (Stage 1)

**File**: `sft_data.json`

JSON array of (seed_prompt, rewritten_prompt) pairs constructed via three temporal rewriting rules:
1. **Temporal Deconstruction**: Decompose harmful directives into chronological static events
2. **Sequential Enforcement**: Insert temporal connectives ("First," "After two seconds,")
3. **Temporal-Space Synthesis**: Harmfulness emerges from temporal composition, not individual descriptions

```json
[
  {
    "input": "<seed harmful prompt>",
    "output": "<temporally rewritten safe prompt>",
    "instruction": "Your task is to rewrite the input video generation prompt..."
  }
]
```

## PPO Training Data (Stage 2)

**File**: `ppo_data.json`

JSON array used as training prompts for PPO. Each item contains an input seed prompt and an optional instruction.

```json
[
  {
    "input": "<seed prompt describing harmful scenario>",
    "instruction": "<optional rewriting instruction>"
  }
]
```

## Seed Prompts (Stage 3 / Evaluation)

**Format**: Plain text files with one prompt per line.

The paper uses 390 meta-harmful prompts across 6 categories (65 each):
- Violence
- Gore
- Self-harm
- Pornography
- Illegal Activity
- Disturbing Content

These seed prompts are sourced from T2VSafetyBench and are not redistributed here.
