from __future__ import annotations

import logging
from functools import partial

from mteb.model_meta import ModelMeta, sentence_transformers_loader

logger = logging.getLogger(__name__)

DEFAULT_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
DEFAULT_RELEASE_DATE = "2026-01-15"

model_prompts = {
    "query": "query:  ",
    "passage": "passage:  ",
}


def get_hf_repo_revision(repo_id: str) -> str:
    """Return the target commit hash for the repository's main branch if available.

    Falls back to DEFAULT_REVISION on any error.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        refs = api.list_repo_refs(repo_id)
        if refs.branches:
            for branch in refs.branches:
                if branch.name == "main":
                    return branch.target_commit
            return refs.branches[0].target_commit
        if refs.tags:
            return refs.tags[0].target_commit
    except Exception as error:
        logger.warning("Could not fetch HF revision for %s: %s", repo_id, error)

    return DEFAULT_REVISION

bge_small_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-small-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=24_000_000,
    memory_usage=None,
    embed_dim=512,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)

bge_base_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-base-en-v1.5",
        revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-base-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
    release_date="2023-09-11",  # initial commit of hf model.
    n_parameters=438_000_000,
    memory_usage=None,
    embed_dim=768,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)

bge_large_en_v1_5 = ModelMeta(
    loader=partial(  # type: ignore
        sentence_transformers_loader,
        model_name="BAAI/bge-large-en-v1.5",
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        model_prompts=model_prompts,
    ),
    name="BAAI/bge-large-en-v1.5",
    languages=["eng_Latn"],
    open_weights=True,
    revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
    release_date="2023-09-12",  # initial commit of hf model.
    n_parameters=1_340_000_000,
    memory_usage=None,
    embed_dim=1024,
    license="mit",
    max_tokens=512,
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    similarity_fn_name="cosine",
    framework=["Sentence Transformers", "PyTorch"],
    use_instructions=True,
)


# Model configurations
MODELS_CONFIG = {
    # Use the two Hugging Face repositories as PEFT checkpoints. The
    # revision will be resolved dynamically from the repo refs.
    "jiebi/IDs-C2I-Enc": "jiebi/IDs-C2I-Enc",
    "jiebi/IDs-I2C-Enc": "jiebi/IDs-I2C-Enc",
    "jiebi/Kubernetes-C2I-Enc": "jiebi/Kubernetes-C2I-Enc",
    "jiebi/Kubernetes-I2C-Enc": "jiebi/Kubernetes-I2C-Enc",
    "jiebi/SIGIR-C2I-Enc": "jiebi/SIGIR-C2I-Enc",
    "jiebi/SIGIR-I2C-Enc": "jiebi/SIGIR-I2C-Enc",
}


def _create_model_meta(name: str, model_path: str) -> ModelMeta:
    """Factory function to create ModelMeta instances with common defaults."""
    revision = (
        get_hf_repo_revision(model_path)
        if "/" in model_path
        else DEFAULT_REVISION
    )

    return ModelMeta(
        loader=partial(  # type: ignore
            sentence_transformers_loader,
            model_name=model_path,
            revision=revision,
            model_prompts=model_prompts,
        ),
        name=name,
        languages=["eng_Latn"],
        open_weights=True,
        revision=revision,
        release_date=DEFAULT_RELEASE_DATE,
        n_parameters=1_340_000_000,
        memory_usage=None,
        embed_dim=1024,
        license="mit",
        max_tokens=512,
        similarity_fn_name="cosine",
        framework=["Sentence Transformers", "PyTorch"],
        use_instructions=True,
    )


# Dynamically create model instances
for model_name, model_path in MODELS_CONFIG.items():
    globals()[model_name] = _create_model_meta(model_name, model_path)