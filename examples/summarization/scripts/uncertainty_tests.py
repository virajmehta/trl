from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import evaluate
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel
)
from transformers.utils import PaddingStrategy
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from models import DropoutLLMForRewardModeling
from utils import *

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    resume_from_checkpoint: Optional[bool] = field(
        default=True, metadata={"help": "If you want to resume training where it left off."}
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model path of pretrained model"
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}_summarization_reward_model",
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=5,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

if script_args.model_name == 'gpt2':
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
elif script_args.model_name == 'llama':
    tokenizer = LlamaTokenizer.from_pretrained("llama_hf_7B")

# CHANGE THIS, DON'T HAVE CHECKPOINT YET
model = DropoutLLMForRewardModeling(num_labels=100, dropout=0.1, model_name=script_args.model_name, tokenizer=tokenizer)
model.load_state_dict(torch.load('models/gpt2_reward_model.pt'))

ds = load_dataset("openai/summarize_from_feedback", name="comparisons")
num_proc = 8  # Can adjust to be higher if you have more processors. Should work even if you don't have 8 CPUs, though.
original_columns = ds["train"].column_names
def turn_into_text_classification_format(examples):
    new_examples = {"text_j": [], "text_k": [], "sites": []}
    for info, summaries, choice in zip(examples["info"], examples["summaries"], examples["choice"]):
        if len(summaries) != 2 or choice not in (0, 1):
            raise ValueError(
                f"There should be two summaries with a choice that's either 0 or 1. Received {len(summaries)} summaries and choice={choice}."
            )
        original_text_field = "post" if info["post"] is not None else "article"
        new_examples["text_j"].append(
            summaries[choice]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field]
        )
        new_examples["text_k"].append(
            summaries[0 if choice == 1 else 1]["text"] + " " + tokenizer.bos_token + " " + info[original_text_field]
        )
        new_examples["sites"].append("news" if info["site"] is not None else "reddit")

    return new_examples

random_ds = create_random_dataset(ds['train'], 10000, tokenizer.bos_token)
ds = ds.map(turn_into_text_classification_format, batched=True, num_proc=num_proc, remove_columns=original_columns)

# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_j = tokenizer(examples["text_j"], truncation=True)
    tokenized_k = tokenizer(examples["text_k"], truncation=True)
    return {
        "input_ids_j": tokenized_j["input_ids"],
        "attention_mask_j": tokenized_j["attention_mask"],
        "input_ids_k": tokenized_k["input_ids"],
        "attention_mask_k": tokenized_k["attention_mask"],
    }


tokenized_ds = ds.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text_j", "text_k"])
tokenized_random_ds = random_ds.map(preprocess_function, batched=True, num_proc=num_proc, remove_columns=["text_j", "text_k"])

trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

rewards_articles_good = []
rewards_articles_bad = []
rewards_random_good = []
rewards_random_bad = []
for _ in range(50):
    preds_articles = trainer.predict(tokenized_ds['validation'])
    preds_random = trainer.predict(tokenized_random_ds)
    rewards_articles_good.append(preds_articles.predictions[0])
    rewards_articles_bad.append(preds_articles.predictions[1])
    rewards_random_good.append(preds_random.predictions[0])
    rewards_random_bad.append(preds_random.predictions[1])
# print("rewards_articles_good", rewards_articles_good)
rewards_articles_good = np.array(rewards_articles_good)
rewards_articles_bad = np.array(rewards_articles_bad)
rewards_random_good = np.array(rewards_random_good)
rewards_random_bad = np.array(rewards_random_bad)
std_articles_good = np.std(rewards_articles_good, axis=0)
std_articles_bad = np.std(rewards_articles_bad, axis=0)
std_random_good = np.std(rewards_random_good, axis=0)
std_random_bad = np.std(rewards_random_bad, axis=0)

# print("std_articles_good", std_articles_good)
# print("std_articles_bad", std_articles_bad.shape)
# print("std_random_good", std_random_good.shape)
# print("std_random_bad", std_random_bad.shape)
# print("rewards_articles_good", rewards_articles_good.shape)
# print("rewards_articles_bad", rewards_articles_bad.shape)
# print("rewards_random_good", rewards_random_good.shape)
# print("rewards_random_bad", rewards_random_bad.shape)
# print("ds", len(ds['validation']))

assert len(std_articles_good) == len(std_articles_bad)
assert len(std_random_good) == len(std_random_bad)
assert len(std_articles_good) == len(ds['validation'])

std_reddit_good = []
std_reddit_bad = []
std_news_good = []
std_news_bad = []
for idx, article in enumerate(ds['validation']):
    if article['sites'] == 'reddit':
        std_reddit_good.append(std_articles_good[idx])
        std_reddit_bad.append(std_articles_bad[idx])
    else:
        std_news_good.append(std_articles_good[idx])
        std_news_bad.append(std_articles_bad[idx])
std_reddit_good = np.array(std_reddit_good)
std_reddit_bad = np.array(std_reddit_bad)
std_news_good = np.array(std_news_good)
std_news_bad = np.array(std_news_bad)

# Plot and save distplots without histograms and only curves of rewards.
sns.kdeplot(std_reddit_good.flatten(), label='Reddit preferred')
sns.kdeplot(std_reddit_bad.flatten(), label='Reddit not preferred')
sns.kdeplot(std_news_good.flatten(), label='News preferred')
sns.kdeplot(std_news_bad.flatten(), label='News not preferred')
sns.kdeplot(std_random_good.flatten(), label='Random preferred')
sns.kdeplot(std_random_bad.flatten(), label='Random not preferred')
plt.legend()
plt.savefig('output.png')

