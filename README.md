# Repository Overview

This repository accompanies our paper **“Beyond the Rules: Understanding the Design Logic of Internet Standards.”**  
It includes scripts, models, and resources used throughout our studies.

The implementation is adapted from the original MTEB repository (<https://github.com/embeddings-benchmark/mteb>), but has been streamlined into lightweight utilities focused solely on **retrieval** and **reranking** tasks.  
Selected MTEB models are included, and our evaluation scripts are compatible with any dataset that follows the same format. Additionally, we introduce a generator that uses retrieved documents to produce a coherent, context‑aware response.

```
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022}
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```

---

## Installation

```
conda create -p path/to/conda_env --python 3.11
conda activate path/to/conda_env
pip install -e .
```

Other packages need to be installed for RAG. Please check the package list in requirements.txt.

```
pip install langchain-text-splitters==1.1.0
pip install llama-index-core==0.14.8
pip install ragas==0.4.2
```

## Dataset
The synthetic training data is released:
- **RFC‑Align**  
  https://huggingface.co/datasets/jiebi/RFCAlign

## Models
The models trained on synthetic data are (V: verbose; N: non-verbose; D: decision; R: rationale):

- **RFC‑DRAlign‑QV**  
  https://huggingface.co/jiebi/RFC-DRAlign-QV

- **RFC‑DRAlign‑QL**  
  https://huggingface.co/jiebi/RFC-DRAlign-QL

- **RFC‑DRAlign‑LV**  
  https://huggingface.co/jiebi/RFC-DRAlign-LV

- **RFC‑DRAlign‑LN**  
  https://huggingface.co/jiebi/RFC-DRAlign-LN

We strongly recommend that you download the base model, such as mistralai/Mistral-7B-v0.1, and place it in the base_models folder; similarly, download the peft model and place it in the peft_models folder.
