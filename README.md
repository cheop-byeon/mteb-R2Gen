# Repository Overview

This repository accompanies two related papers:

- LREC paper: [Linking Rationale to Decision on Internet Standards: A Retrieval-Based Approach Using Synthetic Data](https://doi.org/10.63317/3szh4omfcsxb)
- SemTech paper: [Beyond the Rules: Understanding the Design Logic of Internet Standards](https://doi.org/10.1145/3774905.3795082)

## Citation

### Linking Rationale to Decision on Internet Standards

```bibtex
@inproceedings{bian-etal-2026-linking,
  title = {Linking Rationale to Decision on Internet Standards: A Retrieval-Based Approach Using Synthetic Data},
  author = {Bian, Jie and Welzl, Michael},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  year = {2026},
  pages = {7149--7162},
  doi = {10.63317/3szh4omfcsxb}
}
```

### Beyond the Rules

```bibtex
@inproceedings{bian2026beyond,
  title={Beyond the Rules: Understanding the Design Logic of Internet Standards},
  author={Bian, Jie and Welzl, Michael and Arefyev, Nikolay},
  booktitle={Companion Proceedings of the ACM Web Conference 2026},
  year={2026},
  pages={1025--1032},
  doi={10.1145/3774905.3795082}
}
```

This repository includes the scripts, models, and resources used throughout our studies.

## Repository Structure

The repository is organized into two main components:

### Information Retrieval (`ir/`)
See **“Linking Rationale to Decision on Internet Standards: A Retrieval-Based Approach Using Synthetic Data.”**
Implements retrieval-based approaches to connect rationales with technical decisions:
- **i2c (issue/email comments to code/textual edit)**: Maps discussion threads (rationales/explanations) to decisions in standards

These tasks leverage the IETF mail archives as a discussion base and RFC/Internet-Draft repositories as decision sources.

### Generation (`gen/`)
See **“Beyond the Rules: Understanding the Design Logic of Internet Standards.”**  
Implements retrieval-based approaches to connect rationales with technical decisions:
- **c2i (code/textual edit to issue/email comments)**: Retrieves relevant discussions for given technical decisions
Extends the retrieval pipeline with a RAG (Retrieval-Augmented Generation) component that uses retrieved documents to generate coherent, context-aware explanations for design decisions.

## Technical Implementation

This implementation adapts the MTEB benchmark framework (<https://github.com/embeddings-benchmark/mteb>), streamlined for **retrieval** and **reranking** tasks. Our evaluation scripts are compatible with any dataset following MTEB format, and we provide a generator that produces context-aware responses from retrieved documents.


---

## Installation

For most users (local setup):

```bash
conda create -n mteb-r2gen python=3.11 -y
conda activate mteb-r2gen
pip install -e .
```

If you are using an HPC environment with EasyBuild modules:

```bash
module load Miniconda3/22.11.1-1
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
echo "Conda environments: $(conda info --envs)"
echo "EBROOTMINICONDA3: ${EBROOTMINICONDA3}"

conda create -p path/to/conda_env python=3.11
conda activate path/to/conda_env
pip install -e .
```

Other packages need to be installed for RAG. Please check the package list in requirements.txt.

```bash
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

## Quick Start

Run a minimal end-to-end workflow:

```bash
# 1) Install
conda create -n mteb-r2gen python=3.11 -y
conda activate mteb-r2gen
pip install -e .

# 2) Download datasets
python download_CodeConvo.py
python download_RFCAlign.py

# 3) Run evaluation (HPC)
sbatch evaluation.sh
```

If you are not on an HPC cluster, open `evaluation.sh` and run the corresponding Python command directly in your shell.

## Reproducibility

- Python version: `3.11` (used in the installation examples).
- Evaluation entry point: `RFCAlign_IR_mteb.py`.
- Dataset split/path convention: `ir/<name>/<direction>/<split>` (as used in `evaluation.sh` and `bm25.sh`).
- Dataset revision should be checked from the source dataset page before running (for RFC-Align: https://huggingface.co/datasets/jiebi/RFCAlign).
- Example dense-eval HPC resources (from `evaluation.sh`): `1x A100 GPU`, `32G mem-per-cpu`, `1 CPU task`, `1 hour` wall time.
- BM25 baseline (`bm25.sh`) runs without GPU directives.
- Result location: `results/stage1/<split>/<direction>/` with per-model prediction folders (`save_predictions=True`).
- Current behavior overwrites prior result files (`overwrite_results=True`).
- No explicit random seed is set in `RFCAlign_IR_mteb.py`, `evaluation.sh`, or `bm25.sh`.


## Evaluation

To run evaluations on retrieval tasks:

### Dense Embedding Models
```bash
# See evaluation.sh for comprehensive evaluation setup with all models and datasets
# The script includes all available fine-tuned models (RFC-DRAlign, CodeConvo-based)
sbatch evaluation.sh
```

### BM25 Baseline
```bash
# See bm25.sh for BM25-based retrieval evaluation
sbatch bm25.sh
```

**Note**: Before running evaluations, download the datasets using:
```bash
python download_CodeConvo.py      # Download CodeConvo dataset
python download_RFCAlign.py       # Download RFC-Align dataset
```

See [DATASET_PATH_USAGE.md](DATASET_PATH_USAGE.md) for detailed dataset download and path resolution instructions.

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

Note: Some datasets, base models, and third-party assets referenced by this repository may have their own licenses and usage terms. Please check the corresponding source pages before reuse.

## Contributing

Contributions are welcome.

- Open an issue first to discuss bug reports, feature requests, or larger changes.
- Submit focused pull requests with a clear description of what changed and why.
- For evaluation-related updates, include the command you ran and the relevant output path.

## Contact

For questions about code, datasets, or models, please open a GitHub issue in this repository.

## Acknowledgements

We acknowledge the MTEB benchmark framework developed by Muennighoff et al. (2022):

```
@article{muennighoff2022mteb,
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},
  year = {2022},
  url = {https://arxiv.org/abs/2210.07316},
  doi = {10.48550/ARXIV.2210.07316},
}
```