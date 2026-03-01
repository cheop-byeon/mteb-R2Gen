# Custom Model Wrappers for Fine-Tuned Models

## Overview

This directory contains custom wrapper implementations for the MTEB benchmark framework, specifically designed to integrate fine-tuned dense retrieval models trained on RFC standards and internet architecture data.

## Available Model Wrappers

### 1. **BM25 Search** (`bm25.py`)
- **Description**: Traditional lexical-based retrieval using BM25 algorithm
- **Use Case**: Baseline for comparison with neural models
- **Dependencies**: `bm25s`, `PyStemmer`

### 2. **PromptRetriever** (`promptriever_models.py`)
- **Description**: Wrapper for Samaya AI's PromptRetriever model
- **Model**: `samaya-ai/promptriever-mistral-v0.1-7b-v1`
- **Use Case**: Out-of-the-box instruction-tuned retriever

### 3. **Retrieval Models on GitHub Data** (`retrieval_mistral_models.py`)
- **Description**: Custom fine-tuned models trained on real-world data (GitHub)
- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Training Data**: GitHub CodeConvo
- **Models Available**:
  - `jiebi/IDs-C2I-Dec`: Code-to-issue retrieval on I-Ds (out of CodeConvo)
  - `jiebi/IDs-I2C-Dec`: Issue-to-code retrieval on I-Ds (out of CodeConvo)

### 4. **Non-Verbose Synthetic Models** (`non_verbose_retrieval_mistral_models.py`)
- **Description**: Models trained on concise synthetic RFC-aligned data
- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Training Data**: RFCAlign synthetic dataset (non-verbose variant)
- **Models Available**:
  - `RFC-DRAlign-QN`:  RFCAlign-QN (training data)
  - `RFC-DRAlign-LN`:  RFCAlign-LN (training data)

### 5. **Verbose Synthetic Models** (`verbose_retrieval_mistral_models.py`)
- **Description**: Models trained on detailed synthetic RFC-aligned data
- **Base Model**: `mistralai/Mistral-7B-v0.1`
- **Training Data**: RFCAlign synthetic dataset (verbose variant)
- **Models Available**:
  - `RFC-DRAlign-QV`: RFCAlign-QV (training data)
  - `RFC-DRAlign-LV`: RFCAlign-QN (training data)

## Base Classes and Infrastructure

### Wrapper Base Class (`wrapper.py`)
Provides utility functions for:
- Prompt template management
- Task-specific prompt selection
- Instruction handling for models
- Cross-model compatibility layer

### Key Features
- Automatic prompt selection based on task type and prompt type
- Support for multiple prompt strategies
- Flexible instruction template support
- MTEB framework integration

## Model Loading and Configuration

### Path Structure
```
base_models/
  └── mistralai/
      └── Mistral-7B-v0.1/        # Base model weights

peft_models/
  └── jiebi/
      ├── IDs-C2I-Dec/
      ├── IDs-I2C-Dec/
      ├── RFC-DRAlign-QV/             # Verbose query LoRA adapters
      ├── RFC-DRAlign-QN/             # Non-verbose query LoRA adapters
      ├── RFC-DRAlign-LV/             # Verbose passage LoRA adapters
      ├── RFC-DRAlign-LN/             # Non-verbose passage LoRA adapters
      └── [other LoRA adapters]/
```

### Model Download Instructions

Before downloading models, create the required directory structure:
```bash
mkdir -p base_models peft_models
```

**Step 1: Download base model**
```bash
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./base_models/mistralai/Mistral-7B-v0.1
```

**Step 2: Download fine-tuned models (PEFT adapters)**

Download all required PEFT models from Hugging Face:
```bash
# GitHub CodeConvo-based models
huggingface-cli download jiebi/IDs-C2I-Dec --local-dir ./peft_models/jiebi/IDs-C2I-Dec
huggingface-cli download jiebi/IDs-I2C-Dec --local-dir ./peft_models/jiebi/IDs-I2C-Dec

# RFC-DRAlign synthetic models (non-verbose)
huggingface-cli download jiebi/RFC-DRAlign-QN --local-dir ./peft_models/jiebi/RFC-DRAlign-QN
huggingface-cli download jiebi/RFC-DRAlign-LN --local-dir ./peft_models/jiebi/RFC-DRAlign-LN

# RFC-DRAlign synthetic models (verbose)
huggingface-cli download jiebi/RFC-DRAlign-QV --local-dir ./peft_models/jiebi/RFC-DRAlign-QV
huggingface-cli download jiebi/RFC-DRAlign-LV --local-dir ./peft_models/jiebi/RFC-DRAlign-LV
```

**Alternative: Download all models with a bash script**
```bash
#!/bin/bash
mkdir -p base_models peft_models

# Base model
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir ./base_models/mistralai/Mistral-7B-v0.1

# PEFT models
for model in IDs-C2I-Dec IDs-I2C-Dec RFC-DRAlign-QN RFC-DRAlign-LN RFC-DRAlign-QV RFC-DRAlign-LV; do
    huggingface-cli download jiebi/$model --local-dir ./peft_models/jiebi/$model
done
```

## Implementation Details

### CustomRepLLaMAWrapper
The core wrapper class for our fine-tuned models:

```python
class CustomRepLLaMAWrapper(Wrapper):
    def __init__(
        self,
        base_model_name_or_path: str,           # Path to base model
        peft_model_name_or_path: str,          # Path to LoRA adapters
        torch_dtype: torch.dtype,              # Model precision (e.g., torch.float16)
        device_map: str,                       # Device allocation strategy
        model_prompts: dict[str, str] | None = None,  # Task-specific prompts
        **kwargs
    ):
        # Initializes tokenizer, loads base model, and applies PEFT adapters
```

## Integration with MTEB Pipeline

These wrappers seamlessly integrate with the MTEB evaluation framework for evaluating retrieval tasks.

## Training and Fine-Tuning

For details on model training and synthetic data generation:
- **Fine-tuning Framework**: https://github.com/cheop-byeon/FlagEmbedding
- **Synthetic Data Generation**: https://github.com/cheop-byeon/synthetic-data-kit
- **Dataset**: RFCAlign (available on Hugging Face)

## Evaluation Metrics

Models are evaluated on:
- **NDCG@10**: Ranking quality of top-10 results
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **HIT**: Hit rate at different cutoff values
- **Recall@k**: Coverage of relevant documents in top-k results

## Additional Resources

- **MTEB Documentation**: https://github.com/embeddings-benchmark/mteb
- **Model Hub**: https://huggingface.co/jiebi
- **Dataset Repository**: https://huggingface.co/datasets/jiebi/RFCAlign

## Contributing

When adding new model wrappers:
1. Inherit from the `Wrapper` base class
2. Implement required `encode()` method compatible with MTEB interface
3. Add model registration in `__init__.py`
4. Document model configuration in this file
5. Update the models dictionary with new model variants
