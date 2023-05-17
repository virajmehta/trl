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
)
from transformers.utils import PaddingStrategy
import torch

from models import UncertaintyEstimationLLM
from utils import *

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=0, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False
        , metadata={"help": "If you want to resume training where it left off."}
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=16)
    per_device_eval_batch_size: Optional[int] = field(default=16)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[int] = field(default=2e-5)
    weight_decay: Optional[int] = field(default=0.001)
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    bf16: Optional[bool] = field(
        default=False,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    epochs: Optional[int] = field(
        default="10", metadata={"help": "The number of training epochs for the reward model. OpenAI used 5."}
    )
    dropout: Optional[float] = field(
        default=0.1, metadata={"help": "The dropout rate for the reward model."}
    )
    ensemble: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to ensemble the reward model"}
    )
    n_ensembles: Optional[int] = field(
        default=50, metadata={"help": "The number of ensemble members to use for ensembling or number of repeats for dropout"}
    )
    num_labels: Optional[int] = field(
        default=100,
        metadata={"help": "The dimension of the linear head for ensembles"}
    )
    ensemble_dropout: Optional[float] = field(
        default=1.0,
        metadata={"help": "Fraction of heads to use for ensemble"}
    )



parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Load the human comparisons dataset for tuning the reward model.
ds = load_dataset("openai/summarize_from_feedback", name="comparisons")

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}_summarization_reward_model",
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.epochs,
    weight_decay=script_args.weight_decay,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
)

if script_args.model_name == 'gpt2':
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    # Need to do this for gpt2, because it doesn't have an official pad token.
    tokenizer.pad_token = tokenizer.eos_token
elif script_args.model_name == 'llama':
    tokenizer = LlamaTokenizer.from_pretrained("llama_hf_7B")
model = UncertaintyEstimationLLM(num_labels=script_args.num_labels, dropout=script_args.dropout, model_name=script_args.model_name,
                                    tokenizer=tokenizer, ensemble=script_args.ensemble, n_ensembles=script_args.n_ensembles)
# model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1)

# Turn the dataset into pairs of post + summaries, where text_j is the preferred post + summary and text_k is the other.
def turn_into_text_classification_format(examples):
    new_examples = {"text_j": [], "text_k": []}
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

    return new_examples


num_proc = 8  # Can adjust to be higher if you have more processors. Should work even if you don't have 8 CPUs, though.
original_columns = ds["train"].column_names
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


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)
print(model)
trainer.train(script_args.resume_from_checkpoint)
# trainer.save_model(f"models/gpt2_reward_model")

torch.save(model.state_dict(), f'models/{script_args.model_name}_reward_model_{f"ensemble{script_args.n_ensembles}_{script_args.num_labels}_{script_args.ensemble_dropout}" if script_args.ensemble else f"dropout_{script_args.num_labels}"}.pt')

# Push to the hub so you can share it with people :D
# model.push_to_hub('gpt2_reward_model')
