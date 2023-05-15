from typing import Union
from pathlib import Path
import numpy as np
import torch
from torch import nn

from tqdm.autonotebook import tqdm

from xmen.reranking import Reranker
from xmen.reranking.scored_cross_encoder import ScoredInputExample, ScoredCrossEncoder
from xmen.reranking.ranking_util import get_flat_candidate_ds

from sentence_transformers.readers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

from sentence_transformers.cross_encoder.evaluation import (
    CEBinaryClassificationEvaluator,
)

from datasets import DatasetDict
from xmen.data import IndexedDatasetDict, IndexedDataset

import logging
from sentence_transformers import LoggingHandler

logger = logging.getLogger(__name__)

from typing import Union, List


def flat_ds_to_cross_enc_dataset(flat_candidate_ds, doc_index, context_length, mask_mention, encode_sem_type):
    """
    Convert a flat candidate dataset to a cross-encoding dataset.

    Args:
    - flat_candidate_ds (list): A list of dictionaries where each dictionary represents a single training example. The keys
                                of each dictionary are "context_left", "mention", "context_right", "candidates", "scores",
                                "synonyms", and "label".
    - doc_index (list): A list of indices for each document in the dataset.
    - context_length (int): The number of tokens to include in the context before and after the mention. If set to 0, the
                            context will not be truncated.
    - mask_mention (bool): Whether or not to mask the mention in the generated mention-candidate pairs.

    Returns:
    - res (list): A list of batches, where each batch is a list of ScoredInputExamples. Each ScoredInputExample represents a
                  single mention-candidate pair with a label and a score.
    - res_index (list): A list of indices corresponding to the batches in res.
    """
    res = []
    res_index = []
    for doc, idx in zip(tqdm(flat_candidate_ds), doc_index):
        l_ctx, m, r_ctx = doc["context_left"], doc["mention"], doc["context_right"]
        mention = f"{l_ctx[-context_length:] if context_length else ''} [START] {m if not mask_mention else '[MASK]'} [END] {r_ctx[:context_length] if context_length else ''}"
        batch = []
        for c, score, syns, semtype in zip(doc["candidates"], doc["scores"], doc["synonyms"], doc["types"]):
            candidate = syns[0] + " [TITLE] " + " [SEP] ".join(syns[1:])
            if encode_sem_type:
                candidate = ",".join(semtype) + " [TYPE] " + candidate
            label = 1 if c in doc["label"] else 0
            batch.append(ScoredInputExample(texts=[mention, candidate], label=label, score=score))
        if batch:
            res.append(batch)
            res_index.append(idx)
    return res, res_index


def create_cross_enc_dataset(
    candidate_ds,
    ground_truth,
    kb,
    context_length: int,
    expand_abbreviations: bool,
    encode_sem_type: bool,
    masking: bool,
):
    """
    Create a cross-encoding dataset from a candidate dataset.

    Args:
    - candidate_ds (list): A list of dictionaries where each dictionary represents a single candidate. The keys of each
                           dictionary are "entity", "context", and "mention".
    - ground_truth (dict): A dictionary where the keys are mentions and the values are lists of entities that are valid
                           candidates for that mention.
    - kb (KnowledgeBase): A KnowledgeBase object that contains information about valid entity types and relations between
                          entities.
    - context_length (int): The number of tokens to include in the context before and after the mention. If set to 0, the
                            context will not be truncated.
    - expand_abbreviations (bool): Whether or not to expand abbreviations in the context before and after the mention.
    - masking (bool): Whether or not to mask the mention in the generated mention-candidate pairs.

    Returns:
    - res (list): A list of batches, where each batch is a list of ScoredInputExamples. Each ScoredInputExample represents a
                  single mention-candidate pair with a label and a score.
    - res_index (list): A list of indices corresponding to the batches in res.
    """
    flat_candidate_ds, doc_index = get_flat_candidate_ds(
        candidate_ds, ground_truth, expand_abbreviations=expand_abbreviations, kb=kb
    )
    return flat_ds_to_cross_enc_dataset(
        flat_candidate_ds, doc_index, context_length, mask_mention=masking, encode_sem_type=encode_sem_type
    )


def _cross_encoder_predict(cross_encoder, cross_enc_dataset, show_progress_bar, convert_to_numpy):
    """
    Generate cross-encoder predictions for a cross-encoder and a cross-encoding dataset.

    Args:
    - cross_encoder (CrossEncoder): A CrossEncoder object that will be used to generate predictions.
    - cross_enc_dataset (list): A list of batches, where each batch is a list of ScoredInputExamples. Each ScoredInputExample
                                represents a single mention-candidate pair with a label and a score.
    - show_progress_bar (bool): Whether or not to display a progress bar while generating predictions. If None, the default
                                behavior is to show the progress bar if the logger's effective level is set to INFO or
                                DEBUG.
    - convert_to_numpy (bool): Whether or not to convert the generated predictions to numpy arrays.

    Returns:
    - pred_scores (list): A list of tensors, where each tensor represents the predicted scores for a single mention-candidate
                          pair.
    """
    inp_dataloader = torch.utils.data.DataLoader(
        BatchedCrossEncoderDataset(cross_enc_dataset), num_workers=0, batch_size=None
    )
    inp_dataloader.collate_fn = cross_encoder.smart_batching_collate

    if show_progress_bar is None:
        show_progress_bar = logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG

    iterator = inp_dataloader
    if show_progress_bar:
        iterator = tqdm(inp_dataloader, desc="Batches")

    pred_scores = []
    cross_encoder.model.eval()
    # model.to(self._target_device)
    with torch.no_grad():
        for features, _ in iterator:
            model_predictions = cross_encoder.model(**features, return_dict=True)
            logits = model_predictions.logits
            logits = torch.nn.functional.softmax(logits.reshape(-1), dim=0)
            pred_scores.append(logits)

    if convert_to_numpy:
        pred_scores = [score.cpu().detach().numpy() for score in pred_scores]

    return pred_scores


def rerank(doc, index, doc_idx, ranking, reject_nil=False):
    """
    Re-ranks entities in a given document based on their normalized scores.

    Args:
    - doc (dict): A dictionary containing the document to be re-ranked.
    - index (int): The index of the document.
    - doc_idx (list): A list of indices indicating the position of entities in the document.
    - ranking (list): A list of scores to rank the entities.
    - reject_nil (bool): A boolean value indicating whether or not to reject entities with empty normalized scores.

    Raises:
    - AssertionError: If an entity has an empty normalized score and `reject_nil` is set to `True`.
    - AssertionError: If an entity's normalized score has a different length than its corresponding ranking score.

    Returns:
    - A dictionary containing the re-ranked entities in the given document.
    """
    entities = []
    for ei, e in enumerate(doc["entities"]):
        mask = (np.array(doc_idx) == [index, ei]).all(axis=1)
        ranking_idx = mask.argmax()
        if mask[ranking_idx] == False:
            assert len(e["normalized"]) == 0
        else:
            rank = ranking[ranking_idx]
            assert len(e["normalized"]) == len(rank), (len(e["normalized"]), len(rank))
            for n, r in zip(e["normalized"], rank):
                n["score"] = r
            e["normalized"].sort(key=lambda k: k["score"], reverse=True)
        entities.append(e)
    return {"entities": entities}


class CrossEncoderTrainingArgs:
    """
    A class to store arguments for training a CrossEncoder model.

    Args:
    - args (dict): A dictionary containing the arguments for training.

    Attributes:
    - args (dict): A dictionary containing the arguments for training, which can also be set individually.
    """

    def __init__(self, args: dict):
        self.args = args

    def __init__(
        self,
        model_name: str,
        num_train_epochs: int,
        fp16: bool = True,
        label_smoothing: bool = False,
        score_regularization: bool = False,
        train_layers: list = None,
        softmax_loss: bool = True,
    ):
        self.args = {}
        self.args["model_name"] = model_name
        self.args["num_train_epochs"] = num_train_epochs
        self.args["fp16"] = fp16
        self.args["label_smoothing"] = label_smoothing
        self.args["score_regularization"] = score_regularization
        self.args["train_layers"] = train_layers
        self.args["softmax_loss"] = softmax_loss

    def __getitem__(self, key):
        return self.args[key]


class CrossEncoderReranker(Reranker):
    """
    Reranker that uses a cross-encoder to score a set of candidate passages against a query.
    Inherits from the abstract class Reranker.
    """

    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def load(checkpoint, device):
        """
        Loads a pre-trained model from a checkpoint and returns a new instance of CrossEncoderReranker.

        Args:
        - checkpoint: The path to the checkpoint file to load.
        - device: The device to load the model onto.

        Returns:
        - new instance of CrossEncoderReranker.
        """
        model = CrossEncoder(checkpoint)
        model.model.to(torch.device(device))
        return CrossEncoderReranker(model)

    @staticmethod
    def prepare_data(
        candidates,
        ground_truth,
        kb,
        context_length: int = 128,
        expand_abbreviations: bool = False,
        encode_sem_type: bool = False,
        masking: bool = False,
        **kwargs,
    ):
        """
        Prepares the data for training or evaluation.

        Args:
        - candidates: A Dataset or DatasetDict containing the candidate passages to score.
        - ground_truth: A Dataset or DatasetDict containing the ground-truth passages.
        - kb: The knowledge base to use for context enrichment.
        - context_length: The maximum character length of the left / right context to use for scoring (default 128 chars).
        - expand_abbreviations: Whether to expand abbreviations in the passages.
        - encode_sem_type: Whether to include the semantic type of the concept in its representation
        - masking: Whether to mask entities in the passages.

        Returns:
        - IndexedDataset or IndexedDatasetDict containing the encoded passages.
        """
        print("Context length:", context_length)

        if type(candidates) == DatasetDict:
            assert type(ground_truth) == DatasetDict
            res = IndexedDatasetDict()
            for split, cand in candidates.items():
                gt = ground_truth[split]
                ds, doc_index = create_cross_enc_dataset(
                    cand, gt, kb, context_length, expand_abbreviations, encode_sem_type, masking
                )
                res[split] = IndexedDataset(ds, doc_index)
            return res
        else:
            ds, doc_index = create_cross_enc_dataset(
                candidates,
                ground_truth,
                kb,
                context_length,
                expand_abbreviations,
                encode_sem_type,
                masking,
            )
            return IndexedDataset(ds, doc_index)

    def fit(
        self,
        train_dataset,
        val_dataset,
        output_dir: Union[str, Path],
        training_args: CrossEncoderTrainingArgs,
        train_continue=False,
        loss_fct=None,
        callback=None,
        add_special_tokens=True,
        max_length=512,
        eval_callback=None,
    ):
        """
        Fits the model using the given training and validation datasets.

        Args:
        - train_dataset (List[InputExample]): The list of InputExample objects representing the training dataset.
        - val_dataset (List[InputExample]): The list of InputExample objects representing the validation dataset.
        - output_dir (Union[str, Path]): The directory where the trained model will be saved.
        - train_continue (bool, optional): If True, the training will be continued from the current state of the model. Defaults to False.
        - softmax_loss (bool, optional): If True, uses CrossEntropyLoss as the loss function. Otherwise, no loss function is used. Defaults to True.
        - loss_fct (optional): The loss function to be used. If None, the function will be automatically set based on the value of softmax_loss. Defaults to None.
        - callback (optional): A callback function to be called at the end of each epoch. Defaults to None.
        - add_special_tokens (bool, optional): If True, additional special tokens are added to the tokenizer. Defaults to True.
        - max_length (int, optional): The maximum length of the input sequence. Defaults to 512.
        - fp16 (bool, optional): If True, uses mixed-precision training. Defaults to False.
        - eval_callback (optional): A callback function to be called during evaluation. Defaults to None.
        - **training_args: A dictionary containing the training arguments to be passed to the ScoredCrossEncoder model.

        Raises:
        - AssertionError: If train_continue is False and the model already exists.
        """
        for k, v in training_args.args.items():
            print(k, ":=", v)
        if not self.model:
            self.model = ScoredCrossEncoder(training_args["model_name"], num_labels=1, max_length=max_length)
            if add_special_tokens:
                self.model.tokenizer.add_special_tokens({"additional_special_tokens": ["[START]", "[END]", "[TITLE]"]})
                self.model.model.resize_token_embeddings(len(self.model.tokenizer))
        else:
            assert train_continue, "Training must be continued if model is set"
            print("Continue training")

        if not loss_fct:
            if training_args["softmax_loss"]:
                loss_fct = nn.CrossEntropyLoss()
            else:
                loss_fct = None

        # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
        train_dataloader = torch.utils.data.DataLoader(
            BatchedCrossEncoderDataset(train_dataset), num_workers=0, batch_size=None, shuffle=True
        )
        evaluator = EntityLinkingEvaluator(val_dataset, name="eval", eval_callback=eval_callback)

        #### Just some code to print debug information to stdout
        logging.basicConfig(
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=[LoggingHandler()],
        )
        #### /print debug information to stdout

        # Train the model
        self.model.fit_with_scores(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=training_args["num_train_epochs"],
            loss_fct=loss_fct,
            warmup_steps=100,
            output_path=output_dir,
            callback=callback,
            use_amp=training_args["fp16"],
            label_smoothing=training_args["label_smoothing"],
            score_regularization=training_args["score_regularization"],
            train_layers=training_args["train_layers"],
        )

    def rerank_batch(self, candidates, cross_enc_dataset, show_progress_bar=True, convert_to_numpy=True):
        """
        Re-ranks a batch of candidates using a cross encoder and returns the re-ranked candidates.

        Args:
        - candidates: a dataset of candidate inputs to be re-ranked.
        - cross_enc_dataset: a dataset of cross-encoder inputs for scoring the candidates.
        - show_progress_bar: a boolean indicating whether to display a progress bar during prediction.
        - convert_to_numpy: a boolean indicating whether to convert predictions to numpy arrays.

        Returns:
        - A dataset of re-ranked candidates.
        """
        predictions = _cross_encoder_predict(self.model, cross_enc_dataset.dataset, show_progress_bar, convert_to_numpy)
        return candidates.map(
            lambda d, i: rerank(d, i, cross_enc_dataset.index, predictions),
            with_indices=True,
            load_from_cache_file=False,
        )


class EntityLinkingEvaluator:
    """
    Evaluates a model on an entity linking dataset and returns the accuracy.

    Args:
    - el_dataset (Dataset): The entity linking dataset to evaluate on.
    - name (str): The name of the dataset being evaluated. Defaults to "-".
    - show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
    - eval_callback (function): A callback function to call during evaluation. Defaults to None.
    - k_max (int): The maximum value of k to calculate accuracy at. Defaults to 64.

    Returns:
    - float: The accuracy of the model on the entity linking dataset.
    """

    def __init__(self, el_dataset, name="-", show_progress_bar: bool = False, eval_callback=None, k_max: int = 64):
        self.name = name
        self.el_dataset = el_dataset
        self.show_progress_bar = show_progress_bar
        self.eval_callback = eval_callback
        self.k_max = k_max

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EntityLinkingEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        model_pred_scores = _cross_encoder_predict(
            model,
            self.el_dataset,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
        )

        def get_accuracy(pred_scores, k=1):
            tp = fp = 0

            for scores, batch in zip(pred_scores, self.el_dataset):
                labels = np.array([b.label for b in batch])
                pred_index = scores.argmax()
                pred_indices = scores.argsort()[-1::-1]
                top_k = pred_indices[0:k]
                ground_truth = labels.argmax() if (labels > 0).any() else -1

                if any(top_k == ground_truth):
                    tp += 1
                else:
                    fp += 1
            return tp / (tp + fp)

        acc = get_accuracy(model_pred_scores, 1)
        acc5 = get_accuracy(model_pred_scores, 5)
        k_max = self.k_max
        accmax = get_accuracy(model_pred_scores, k_max)

        logger.info(f"Accuracy: {acc}")
        logger.info(f"Accuracy @ 5: {acc5}")
        logger.info(f"Accuracy @ {k_max}: {accmax}")

        if self.eval_callback:
            self.eval_callback(
                {
                    "train/epoch": epoch,
                    f"{self.name}/accuracy_1": acc,
                    f"{self.name}/accuracy_5": acc5,
                    f"{self.name}/accuracy_{k_max}": accmax,
                }
            )

        baseline_pred_scores = np.array([[0]] * len(self.el_dataset))
        baseline_acc = get_accuracy(baseline_pred_scores, 1)
        logger.info(f"Baseline Accuracy: {baseline_acc}")

        return acc


class BatchedCrossEncoderDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset class for batched cross-encoder data.

    Args:
    - cross_enc_data (list): The batched cross-encoder data.
    """

    def __init__(self, cross_enc_data):
        self.cross_enc_data = cross_enc_data

    def __len__(self):
        return len(self.cross_enc_data)

    def __getitem__(self, idx):
        return self.cross_enc_data[idx]
