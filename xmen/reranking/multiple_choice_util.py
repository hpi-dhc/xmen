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
    """
    Computes the simple accuracy of predictions.

    Args:
    - preds (Union[List, np.ndarray]): List or array of predicted labels.
    - labels (Union[List, np.ndarray]): List or array of true labels.

    Returns:
    - float: Simple accuracy of predictions.
    """
    return (preds == labels).mean()


def top_1_metrics(preds, labels):
    """
    Computes top-1 metrics for a given set of predictions.

    Args:
    - preds (np.ndarray): Array of predicted labels.
    - labels (np.ndarray): Array of true labels.

    Returns:
    - Dict[str, float]: A dictionary containing the following metrics:
        - pr_nil: Precision for predicting NIL entities.
        - re_nil: Recall for predicting NIL entities.
        - f1_nil: F1 score for predicting NIL entities.
        - pr: Precision for predicting all entities.
        - re: Recall for predicting all entities.
        - f1: F1 score for predicting all entities.

    """
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
    """
    Computes entity linking metrics for a given set of predictions.

    Args:
    - p (EvalPrediction): Object containing predictions and labels.

    Returns:
    - Dict[str, float]: A dictionary containing the following metrics:
        - pr_nil: Precision for predicting NIL entities.
        - re_nil: Recall for predicting NIL entities.
        - f1_nil: F1 score for predicting NIL entities.
        - pr: Precision for predicting all entities.
        - re: Recall for predicting all entities.
        - f1: F1 score for predicting all entities.
        - acc: Simple accuracy of predictions.
    """
    preds = np.argsort(p.predictions, axis=1)[:, -1::-1]

    metrics = top_1_metrics(preds, p.label_ids)
    metrics["acc"] = simple_accuracy(preds[:, 0], p.label_ids)
    return metrics


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
    - tokenizer (:obj:`PreTrainedTokenizerBase`): The tokenizer used for encoding the data.
    - padding (:obj:`Union[bool, str, PaddingStrategy]`, optional, defaults to True):
         Select a padding strategy. Default strategy is to pad to the longest sample in the batch.
    - max_length (:obj:`int`, optional):
         The maximum length of the tokenized input sequences. Will truncate all samples longer than this.
    - pad_to_multiple_of (:obj:`int`, optional):
         If set, the input sequences will be padded to a length that is a multiple of the given value.

    Returns:
    - A dictionary with the following key-value pairs:
        - input_ids (:obj:`torch.Tensor`): The input token ids.
        - attention_mask (:obj:`torch.Tensor`): The input attention mask.
        - token_type_ids (:obj:`torch.Tensor`, optional): The token type
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
        input_ids = batch["input_ids"]

        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
