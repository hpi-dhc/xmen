import torch
import numpy as np

from dataclasses import dataclass
from transformers.tokenization_utils_base import (
    PreTrainedTokenizerBase,
    PaddingStrategy,
)
from transformers import EvalPrediction

from typing import Optional, Union, Dict


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def top_1_metrics(preds, labels):
    tp_nil = tp = tp_no_nil = fp = fn_nil = fn = fn_no_nil = fp_no_nil = 0
    for p, l in zip(preds, labels):
        if p[0] == 0 and l == 0:
            tp_nil += 1
        elif p[0] == l:
            tp += 1
        elif p[0] == 0:
            if p[1] == l and l != 0:
                tp_no_nil += 1
            else:
                fn_no_nil += 1
                fp_no_nil += 1
            fn_nil += 1
        elif p[0] != 0:
            fn += 1
            fp += 1
    f1 = lambda x, y: 2 * x * y / (x + y) if x + y > 0 else 0

    pr_nil = tp / (tp + fp) if tp > 0 else 0
    re_nil = tp / (tp + fn + fn_nil + tp_nil) if tp > 0 else 0

    pr = (tp + tp_no_nil) / (tp + tp_no_nil + fp + fp_no_nil + tp_nil)
    re = (tp + tp_no_nil) / (tp + tp_no_nil + fn + fn_no_nil + tp_nil)

    return {
        "pr_nil": pr_nil,
        "re_nil": re_nil,
        "f1_nil": f1(pr_nil, re_nil),
        "pr": pr,
        "re": re,
        "f1": f1(pr, re),
    }


def compute_entity_linking_metrics(p: EvalPrediction) -> Dict:
    preds = np.argsort(p.predictions, axis=1)[:, -1::-1]

    metrics = top_1_metrics(preds, p.label_ids)
    metrics["acc"] = simple_accuracy(preds[:, 0], p.label_ids)
    return metrics


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        input_ids = batch['input_ids']
        
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
