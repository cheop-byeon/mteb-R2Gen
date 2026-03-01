# RFC8205 Rationale Generation

This directory contains scripts for generating and evaluating rationales for RFC8205 design decisions using Retrieval-Augmented Generation (RAG).

## Overview

The generation pipeline consists of two stages:
1. **Generation**: Use retrieved contexts from Stage 1 (retrieval) to generate rationales via LLM
2. **Evaluation**: Assess the quality of generated rationales using established metrics

## Prerequisites

- Completed Stage 1 retrieval evaluation (results in `../results/stage1/test/c2i/{model}/`)
- OpenRouter API key set: `export OPENROUTER_API_KEY="your-api-key"`
- Required Python packages installed (see main repo requirements.txt)

## Usage

### Generate Rationales

```bash
# Submit generation job
sbatch generation.sh

# Or run directly
python rfc_rationale_generator.py --model "jiebi/RFC-DRAlign-LN"
```

**Output:** `qa_data_{model}.jsonl` containing query IDs and generated rationales

**Note:** The `--model` parameter refers to the **retrieval model** used in Stage 1, not the generation model (configured as `MODEL` in `rfc_rationale_generator.py`).

### Evaluate Quality

```bash
# Submit evaluation job
sbatch eval.sh
```

This evaluates generated rationales using two metrics:
- **Context recall**: Relevance to design decisions based on retrieved context
- **Faithfulness**: Grounding of generated rationales in retrieved evidence

## Directory Structure

```
c2i/test/
  ├── queries.jsonl       # Design decisions to explain
  ├── corpus.386.jsonl    # Retrieved discussion contexts
  └── qrels/              # Ground truth relevance judgments

eval_outputs/             # Evaluation results
```

## Customization

To use a different retrieval model's predictions:
1. Edit `generation.sh` and change the `--model` parameter
2. Ensure the corresponding predictions exist in `../results/stage1/test/c2i/{model}/`

To change the generation model:
1. Edit `MODEL` constant in `rfc_rationale_generator.py`
2. Update prompt templates if needed for different model behavior