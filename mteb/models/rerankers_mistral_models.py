from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder
from mteb.evaluation.evaluators.RetrievalEvaluator import DenseRetrievalExactSearch

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
BASE_REVISION = "7231864981174d9bee8c7687c24c8344414eae6b"
RELEASE_DATE = "2024-09-15"
DEVICE_MAP = "auto"
FP_OPTIONS = torch.bfloat16
MAX_LENGTH = 2048
DEFAULT_BATCH_SIZE = 4

RERANKER_TEMPLATE = """<s>[INST] You are a developer, whose job is to determine if the following document is relevant to the query (true/false).
Query: {query}
Document: {text}
Relevant (either "true" or "false"): [/INST]"""

QUERY_INSTRUCT_TEMPLATE = "{query} {instruction}"


class CustomMistralReranker(DenseRetrievalExactSearch):
    name = "mistral"

    def __init__(
        self,
        base_model_name_or_path: str,
        peft_model_name_or_path: str,
        batch_size: int = DEFAULT_BATCH_SIZE,
        fp_options: bool = None,
        silent: bool = False,
        is_classification: bool = False,
        **kwargs,
    ):
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "To use the RepLLaMA based models `peft` is required. Please install it with `pip install 'mteb[peft]'`."
            )

        self.batch_size = batch_size
        self.fp_options = fp_options if fp_options is not None else torch.float32
        if self.fp_options == "auto":
            self.fp_options = torch.float32
        elif self.fp_options == "float16":
            self.fp_options = torch.float16
        elif self.fp_options == "float32":
            self.fp_options = torch.float32
        elif self.fp_options == "bfloat16":
            self.fp_options = torch.bfloat16
        logger.info(f"Using fp_options of {self.fp_options}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.silent = silent
        self.first_print = True  # for debugging

        if "torch_compile" in kwargs:
            del kwargs["torch_compile"]

        self.template = RERANKER_TEMPLATE
        self.query_instruct_template = QUERY_INSTRUCT_TEMPLATE
        logger.info(f"Using query_instruct_template of {self.query_instruct_template}")
        self.is_classification = is_classification

        logger.info(self.template)
        logger.info(base_model_name_or_path)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, padding_side="left")
        # MODIFIED
        special_tokens_dict = {'additional_special_tokens': ['<ins>', '</ins>', '<del>', '</del>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=self.fp_options)
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(self.base_model, peft_model_name_or_path)
        self.model = self.model.merge_and_unload()
        # set the max_length for the evals as they did, although the model can handle longer
        self.model.config.max_length = MAX_LENGTH
        self.tokenizer.model_max_length = MAX_LENGTH
        self.model.to(self.device)

        self.token_false_id = self.tokenizer.get_vocab()["false"]
        self.token_true_id = self.tokenizer.get_vocab()["true"]
        self.max_length = min(MAX_LENGTH, self.tokenizer.model_max_length)
        logger.info(f"Using max_length of {self.max_length}")
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            logger.info(f"Using {self.gpu_count} GPUs")
            self.model = torch.nn.DataParallel(self.model)
        self.model.eval()

    @torch.inference_mode()
    def encode(
        self,
        sentences: list[str],
        *,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        batch_size = self.batch_size
        embeddings = []
        for batch_start in tqdm.tqdm(range(0, len(sentences), batch_size)):
            batch_end = batch_start + batch_size
            batch = sentences[batch_start:batch_end]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            embeddings.append(model_output.logits.cpu())

        return torch.cat(embeddings, dim=0).numpy()


def _create_model_meta(
    model_name: str,
    base_model_path: str,
    peft_checkpoint: str,
) -> ModelMeta:
    """Factory function to create ModelMeta instances for reranker models."""
    return ModelMeta(
        loader=lambda: CustomMistralReranker(
            base_model_name_or_path=base_model_path,
            peft_model_name_or_path=peft_checkpoint,
            device_map=DEVICE_MAP,
            fp_options=FP_OPTIONS,
        ),
        name=model_name,
        languages=["eng_Latn"],
        open_weights=True,
        revision=BASE_REVISION,
        release_date=RELEASE_DATE,
    )


# Configuration dictionary for all reranker models
MODELS_CONFIG = {
    ("reranker_mistral_i2c", "mistralai/Mistral-7B-Instruct-v0.2", "/cluster/projects/nn9851k/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_mistralai_Mistral-7B-instruction-v0.2/i2c/checkpoint-788/"),
    ("reranker_mistral_c2i", "mistralai/Mistral-7B-Instruct-v0.2", "/cluster/projects/nn9851k/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_mistralai_Mistral-7B-instruction-v0.2/c2i/checkpoint-2208"),
    ("reranker_mistral7b_i2c_ids", "/cluster/work/users/jiebi/mistralai/Mistral-7B-Instruct-v0.2", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_mistralai_Mistral-7B-Instruct-v0.2/i2c/checkpoint-131"),
    ("reranker_llama8b_i2c_ids", "/cluster/work/users/jiebi/meta-llama/Llama-3.1-8B-Instruct", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_meta-llama_llama_8b/i2c/checkpoint-131"),
    ("reranker_rank17b_i2c_ids", "/cluster/work/users/jiebi/jhu-clsp/rank1-7b", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_jhu-clsp_rank1_7b/i2c/checkpoint-131"),
    ("reranker_qwen7b_i2c_ids", "/cluster/work/users/jiebi/Qwen/Qwen2.5-7B-Instruct", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_Qwen-Qwen2.5-7B-Instruct/i2c/checkpoint-131"),
    ("reranker_qwen14b_i2c_ids", "/cluster/work/users/jiebi/Qwen/Qwen2.5-14B-Instruct", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_Qwen-Qwen2.5-14B-Instruct/i2c/checkpoint-65"),
    ("reranker_qwen32b_i2c_ids", "/cluster/work/users/jiebi/Qwen/Qwen2.5-32B-Instruct", "/cluster/work/users/jiebi/FlagEmbedding/examples/finetune/reranker/decoder_only/decoder_only_base_Qwen-Qwen2.5-32B-Instruct/i2c/checkpoint-131"),
}

# Dynamically create models
for model_name, base_model_path, peft_checkpoint in MODELS_CONFIG:
    globals()[model_name] = _create_model_meta(model_name, base_model_path, peft_checkpoint)
