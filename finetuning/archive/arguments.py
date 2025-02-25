import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from transformers import GlueDataTrainingArguments as DataTrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    wandb_project: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases project name"}
    )
    wandb_run_name: Optional[str] = field(
        default=None, metadata={"help": "Weights and Biases project name"}
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "use flash attn"}
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "checkpoint path"}
    )
    save_file: Optional[str] = field(
        default=None,
        metadata={"help": "checkpoint path"}
    )

@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."},
    )
    prompt_key: Optional[str] = field(
        default=None,
        metadata={"help": "The name of text field in the dataset"},
    )