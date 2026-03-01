# Repository Overview

This repository accompanies our paper 
**“Linking Rationale to Decision on Internet Standards: A Retrieval-Based Approach Using Synthetic Data.”**
**“Beyond the Rules: Understanding the Design Logic of Internet Standards.”**  
It includes scripts, models, and resources used throughout our studies.

## Repository Structure

The repository is organized into two main components:

### Information Retrieval (`ir/`)
see **“Linking Rationale to Decision on Internet Standards: A Retrieval-Based Approach Using Synthetic Data.”**
Implements retrieval-based approaches to connect rationales with technical decisions:
- **i2c (issue-to-comments)**: Maps discussion threads (rationales/explanations) to decisions in standards

These tasks leverage the IETF mail archives as a discussion base and RFC/Internet-Draft repositories as decision sources.

### Generation (`gen/`)
see **“Beyond the Rules: Understanding the Design Logic of Internet Standards.”**  
Implements retrieval-based approaches to connect rationales with technical decisions:
- **c2i (code-to-issues)**: Retrieves relevant discussions for given technical decisions
Extends the retrieval pipeline with a RAG (Retrieval-Augmented Generation) component that uses retrieved documents to generate coherent, context-aware explanations for design decisions.

## Technical Implementation

This implementation adapts the MTEB benchmark framework (<https://github.com/embeddings-benchmark/mteb>), streamlined for **retrieval** and **reranking** tasks. Our evaluation scripts are compatible with any dataset following MTEB format, and we provide a generator that produces context-aware responses from retrieved documents.


---

## Installation

```
module load Miniconda3/22.11.1-1
export PS1=\$
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINCONDA3: ${EBROOTMINICONDA3}"

conda create -p path/to/conda_env python=3.11
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

  python download_RFCAlign.py can help you download the dataset
  
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

## Models Fine-tuning
https://github.com/cheop-byeon/FlagEmbedding

## Synthetic Data Generation
https://github.com/cheop-byeon/synthetic-data-kit


## Acknowledgements

We acknowledge the MTEB benchmark framework developed by Muennighoff et al. (2022):

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