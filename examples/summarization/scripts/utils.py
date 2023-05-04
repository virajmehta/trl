from tqdm import tqdm
import pyarrow as pa
import pyarrow.dataset as ds
import numpy as np
import pandas as pd
from datasets import Dataset
import random
import evaluate

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
import torch

accuracy = evaluate.load("accuracy")

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch

class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], attention_mask=inputs["attention_mask_j"])
        rewards_k = model(input_ids=inputs["input_ids_k"], attention_mask=inputs["attention_mask_k"])
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

def get_word_list(ds):
    words = set()
    for i in ds:
        words.update(i['info']['post'].split())
    return words

def create_random_dataset(vocab_ds, n, tokenizer_bos_token):
    vocab = get_word_list(vocab_ds)
    examples = {"text_j": [], "text_k": []}
    for i in tqdm(range(n)):
        post_length = random.randint(100, 500)
        summary1_length = random.randint(10, 50)
        summary2_length = random.randint(10, 50)
        post = ' '.join(random.sample(vocab, post_length))
        summary1 = ' '.join(random.sample(vocab, summary1_length))
        summary2 = ' '.join(random.sample(vocab, summary2_length))
        examples["text_j"].append(f"{summary1} {tokenizer_bos_token} {post}")
        examples["text_k"].append(f"{summary2} {tokenizer_bos_token} {post}")

    return Dataset.from_dict(examples)