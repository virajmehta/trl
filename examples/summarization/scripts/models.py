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
    LlamaTokenizer
)
from transformers.utils import PaddingStrategy
import torch

class DropoutLLMForRewardModeling(nn.Module):
    def __init__(self, num_labels, dropout, model_name, tokenizer, model_path=None):
        super(DropoutLLMForRewardModeling, self).__init__()
        if model_name == "gpt2":
            if model_path is not None:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
            self.model.config.pad_token_id = tokenizer.eos_token_id # Needed only for GPT2
        elif model_name == "llama":
            self.model = LlamaForCausalLM.from_pretrained('llama_hf_7B', num_labels=num_labels)
        self.value_head = nn.Linear(num_labels, 1)
        self.dropout = dropout

    def forward(self, input_ids=None, attention_mask=None):
        # Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.value_head(torch.nn.functional.dropout(nn.ReLU()((outputs[0])), p=self.dropout, training=True))
