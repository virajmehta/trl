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

class UncertaintyEstimationLLM(nn.Module):
    def __init__(self, num_labels, model_name, tokenizer, model_path=None, dropout=0.1, ensemble=False, n_ensembles=50):
        super(UncertaintyEstimationLLM, self).__init__()
        if model_name == "gpt2":
            if model_path is not None:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=num_labels)
            self.model.config.pad_token_id = tokenizer.eos_token_id # Needed only for GPT2
        elif model_name == "llama":
            self.model = LlamaForCausalLM.from_pretrained('llama_hf_7B', num_labels=num_labels)
        self.dropout = dropout
        self.ensemble = ensemble
        self.n_ensembles = n_ensembles
        if self.ensemble:
            self.value_heads = nn.ModuleList([nn.Linear(num_labels, 1) for _ in range(n_ensembles)])
        else:
            self.value_head = nn.Linear(num_labels, 1)
        self.uncertainty_mode = False

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if not self.uncertainty_mode:
            if not self.ensemble:
                return self.value_head(nn.functional.dropout(nn.ReLU()((outputs[0])), p=self.dropout, training=True))
            else:
                head_outputs = []
                for head in self.value_heads:
                    head_outputs.append(head(nn.ReLU()((outputs[0]))))
                return torch.mean(torch.stack(head_outputs), dim=0)
        else:
            if not self.ensemble:
                # Calculate the output n_ensembles times and return the std
                return torch.std(torch.stack(
                    [self.value_head(nn.functional.dropout(nn.ReLU()((outputs[0])), p=self.dropout, training=True))
                     for _ in range(self.n_ensembles)]), dim=0)
            else:
                head_outputs = []
                for head in self.value_heads:
                    head_outputs.append(head(nn.ReLU()((outputs[0]))))
                return torch.std(torch.stack(head_outputs), dim=0)

    def set_uncertainty_mode(self, uncertainty_mode):
        self.uncertainty_mode = uncertainty_mode