from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from transformers import AutoModel, AutoTokenizer
from mteb.encoder_interface import Encoder, PromptType
from mteb.model_meta import ModelMeta

from .wrapper import Wrapper

logger = logging.getLogger(__name__)

# Constants
BASE_PATH = "./base_models"
PEFT_PATH = "./peft_models"
BASE_MODEL_PATH = f"{BASE_PATH}/mistralai/Mistral-7B-v0.1"
DEFAULT_BATCH_SIZE = 16
MAX_LENGTH = 4096
PADDING_MULTIPLE = 8
DEFAULT_REVISION = "7231864981174d9bee8c7687c24c8344414eae6b"
DEFAULT_RELEASE_DATE = "2026-01-15"
MODEL_PROMPTS = {
    PromptType.query.value: "query:  ",
    PromptType.passage.value: "",
}

model_prompts = {
    PromptType.query.value: "query:  ",
    PromptType.passage.value: "",
}


def get_hf_repo_revision(repo_id: str) -> str:
    """Return the target commit hash for the repository's main branch if available.

    Falls back to BASE_REVISION on any error.
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        refs = api.list_repo_refs(repo_id)
        # Prefer branch named 'main', otherwise take first branch, then tags
        if refs.branches:
            for b in refs.branches:
                if b.name == "main":
                    return b.target_commit
            return refs.branches[0].target_commit
        if refs.tags:
            return refs.tags[0].target_commit
    except Exception as e:
        logger.warning("Could not fetch HF revision for %s: %s", repo_id, e)

    return DEFAULT_REVISION


class CustomRepLLaMAWrapper(Wrapper):
    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        torch_dtype: torch.dtype,
        device_map: str,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ):
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "To use the RepLLaMA based models `peft` is required. Please install it with `pip install 'mteb[peft]'`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.base_model = AutoModel.from_pretrained(base_model_name_or_path, torch_dtype=torch_dtype, device_map=device_map)

        self.model = PeftModel.from_pretrained(self.base_model, peft_model_name_or_path)
        self.model = self.model.merge_and_unload()
        # set the max_length for the evals as they did, although the model can handle longer
        self.model.config.max_length = MAX_LENGTH
        self.tokenizer.model_max_length = MAX_LENGTH

        self.model_prompts = (
            self.validate_task_to_prompt_name(model_prompts) if model_prompts else None
        )

    def create_batch_dict(self, tokenizer, input_texts):
        max_length = self.model.config.max_length
        batch_dict = tokenizer(
            input_texts,
            max_length=max_length - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=PADDING_MULTIPLE,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,  # noqa
    ) -> np.ndarray:
        batch_size = DEFAULT_BATCH_SIZE if "batch_size" not in kwargs else kwargs.pop("batch_size")
        all_embeddings = []
        prompt = self.get_prompt_name(self.model_prompts, task_name, prompt_type)
        if prompt:
            sentences = [f"{prompt}{sentence}".strip() for sentence in sentences]
        for i in tqdm.tqdm(range(0, len(sentences), batch_size)):
            batch_texts = sentences[i : i + batch_size]

            batch_dict = self.create_batch_dict(self.tokenizer, batch_texts)
            batch_dict = {
                key: value.to(self.model.device) for key, value in batch_dict.items()
            }

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.model(**batch_dict)
                    last_hidden_state = outputs.last_hidden_state
                    sequence_lengths = batch_dict["attention_mask"].sum(dim=1) - 1
                    batch_embeddings_size = last_hidden_state.shape[0]
                    reps = last_hidden_state[
                        torch.arange(batch_embeddings_size, device=last_hidden_state.device),
                        sequence_lengths,
                    ]
                    embeddings = F.normalize(reps, p=2, dim=-1)
                    all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def _create_model_meta(model_name: str, peft_model_path: str) -> ModelMeta:
    """Factory function to create ModelMeta instances for verbose retrieval models."""
    # Resolve revision dynamically from the peft repo (if it's a HF repo id)
    revision = (
        get_hf_repo_revision(peft_model_path)
        if "/" in peft_model_path
        else DEFAULT_REVISION
    )

    return ModelMeta(
        loader=lambda: CustomRepLLaMAWrapper(
            base_model_name_or_path=BASE_MODEL_PATH,
            peft_model_name_or_path=f"{PEFT_PATH}/{peft_model_path}",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            model_prompts=model_prompts,
        ),
        name=model_name,
        languages=["eng_Latn"],
        open_weights=True,
        revision=revision,
        release_date=DEFAULT_RELEASE_DATE,
    )


# Configuration dictionary for all verbose models
MODELS_CONFIG = {
    # Use the two Hugging Face repositories as PEFT checkpoints. The
    # revision will be resolved dynamically from the repo refs.
    "jiebi/RFC-DRAlign-LV": "jiebi/RFC-DRAlign-LV",
    "jiebi/RFC-DRAlign-QV": "jiebi/RFC-DRAlign-QV",
}

# Dynamically create models
for model_name, peft_path in MODELS_CONFIG.items():
    globals()[model_name] = _create_model_meta(model_name, peft_path)
