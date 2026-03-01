from __future__ import annotations

import logging

import datasets

from mteb.abstasks.AbsTaskRetrieval import AbsTaskRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata

# Constants
COMMON_DATASET_PATH = "./sample/" # this path could be passed dynamically
COMMON_DATASET_REVISION = "d3c5e1fc0b855ab6097bf1cda04dd73947d7caab"
COMMON_DATE_RANGE = ("2022-01-01", "2024-10-30")
COMMON_EVAL_SPLIT = ["test"]
COMMON_MAIN_SCORE = "mrr_at_10"
COMMON_CATEGORY = "p2p"
COMMON_MODALITIES = ["text"]
COMMON_REFERENCE = None
COMMON_TYPE = "Retrieval"
COMMON_DOMAINS = ["Programming"]
COMMON_SUBTYPES = ["Code retrieval", "Reasoning as Retrieval"]
COMMON_LICENSE = "cc-by-4.0"
COMMON_ANNOTATIONS = "derived"
COMMON_DESCRIPTION = "Retrieval"

# Task configurations: (class_name, task_name, eval_langs, has_trailing_comma)
TASKS_CONFIG = [
    ("KubernetesRetrieval", "kubernetes", ["eng-Latn", "go-Code"]),
    ("NixpkgsRetrieval", "nixpkgs", ["eng-Latn", "c++-Code"]),
    ("PytorchRetrieval", "pytorch", ["eng-Latn", "python-Code"]),
    ("ReactRetrieval", "react", ["eng-Latn", "javascript-Code"]),
    ("VSCodeRetrieval", "vscode", ["eng-Latn", "javascript-Code", "typescript-Code"]),
    ("FreeCodeCampRetrieval", "freeCodeCamp", ["eng-Latn", "javascript-Code"]),
    ("RustRetrieval", "rust", ["eng-Latn", "rust-Code"]),
    ("IDsRetrieval", "ids", ["eng-Latn"]),
    ("IDsSuppRetrieval", "ids-supp", ["eng-Latn"]),
    ("RFCRetrieval", "rfcs", ["eng-Latn"]),
    ("RFC8335Retrieval", "rfc8335", ["eng-Latn"]),
    ("RFC7657Retrieval", "rfc7657", ["eng-Latn"]),
    ("RFC8205Retrieval", "rfc8205", ["eng-Latn"]),
    ("RFCAlignRetrieval", "rfcAlign", ["eng-Latn"]),
    ("WebsiteRetrieval", "website", ["eng-Latn"]),
    ("SweRetrieval", "swe", ["eng-Latn"]),
]


def _create_retrieval_task(class_name: str, task_name: str, eval_langs: list[str]) -> type:
    """Factory function to create Retrieval task classes."""
    return type(
        class_name,
        (AbsTaskRetrieval,),
        {
            "metadata": TaskMetadata(
                name=task_name,
                description=COMMON_DESCRIPTION,
                reference=COMMON_REFERENCE,
                type=COMMON_TYPE,
                category=COMMON_CATEGORY,
                modalities=COMMON_MODALITIES,
                eval_splits=COMMON_EVAL_SPLIT,
                eval_langs=eval_langs,
                main_score=COMMON_MAIN_SCORE,
                dataset={
                    "path": COMMON_DATASET_PATH,
                    "revision": COMMON_DATASET_REVISION,
                },
                date=COMMON_DATE_RANGE,
                domains=COMMON_DOMAINS,
                task_subtypes=COMMON_SUBTYPES,
                license=COMMON_LICENSE,
                annotations_creators=COMMON_ANNOTATIONS,
            )
        },
    )


# Dynamically create all task classes
for class_name, task_name, eval_langs in TASKS_CONFIG:
    globals()[class_name] = _create_retrieval_task(class_name, task_name, eval_langs)
