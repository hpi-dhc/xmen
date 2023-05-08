from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List, Union
import transformers
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers import SentenceTransformer

import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm.autonotebook import tqdm, trange

import re

logger = logging.getLogger(__name__)


class ScoredInputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0, score: float = None):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        :score score
            prior score for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label
        self.score = score

    def __repr__(self):
        return "<ScoredInputExample> label: {}, score: {}, texts: {}".format(
            str(self.label), str(self.score), "; ".join(self.texts)
        )


class ScoredCrossEncoder(CrossEncoder):
    def __init__(
        self,
        model_name: str,
        num_labels: int = None,
        max_length: int = None,
        device: str = None,
        tokenizer_args: Dict = {},
        automodel_args: Dict = {},
        default_activation_function=None,
    ):
        super().__init__(
            model_name, num_labels, max_length, device, tokenizer_args, automodel_args, default_activation_function
        )

    def smart_batching_collate_with_scores(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []
        scores = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())
            labels.append(example.label)
            scores.append(example.score)

        tokenized = self.tokenizer(
            *texts, padding=True, truncation="longest_first", return_tensors="pt", max_length=self.max_length
        )
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device
        )
        scores = torch.tensor(scores, dtype=torch.float).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels, scores

    def score_regularizer(self, logits, scores):  # , mean = False):
        pdist = torch.nn.PairwiseDistance(p=2)
        dist = pdist(logits, scores)
        # print(dist)
        return dist
        # sm = torch.nn.Softmax(dim=0)
        # if mean:
        #    return torch.mean(scores - sm(logits))
        # else:
        #    return torch.sum(scores - sm(logits))

    def fit_with_scores(
        self,
        train_dataloader: DataLoader,
        evaluator: SentenceEvaluator = None,
        epochs: int = 1,
        loss_fct=None,
        activation_fct=torch.nn.Identity(),
        scheduler: str = "WarmupLinear",
        warmup_steps: int = 10000,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params: Dict[str, object] = {"lr": 2e-5},
        weight_decay: float = 0.01,
        score_regularization: bool = True,
        train_layers: list = [],
        label_smoothing: float = 0.0,
        evaluation_steps: int = 0,
        output_path: str = None,
        save_best_model: bool = True,
        max_grad_norm: float = 1,
        use_amp: bool = False,
        callback: Callable[[float, int, int], None] = None,
        show_progress_bar: bool = True,
    ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        :param show_progress_bar: If True, output a tqdm progress bar
        """
        print("Using score regularization:", score_regularization)
        print("Using label smoothing factor:", label_smoothing)

        train_dataloader.collate_fn = self.smart_batching_collate_with_scores

        if use_amp:
            from torch.cuda.amp import autocast

            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

        if train_layers:
            print("Training only layers: ", train_layers)
            for n, param in param_optimizer:
                if not any([n.startswith(f) or re.match(f, n) for f in train_layers]):
                    print("Freezing", n)
                    param.requires_grad = False

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(
                optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps
            )

        if loss_fct is None:
            loss_fct = torch.nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else torch.nn.CrossEntropyLoss()

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch", disable=not show_progress_bar):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels, scores in tqdm(
                train_dataloader, desc="Iteration", smoothing=0.05, disable=not show_progress_bar
            ):
                if label_smoothing > 0:
                    labels = (1 - label_smoothing) * labels + (label_smoothing / len(labels))
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)
                        if score_regularization:
                            loss_value += self.score_regularizer(logits, scores)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    if score_regularization:
                        # loss_value = self.score_regularizer(logits, scores)
                        # Try next: only logits, no softmax
                        loss_value += self.score_regularizer(logits, scores)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def predict(
        self,
        sentences: List[List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = None,
        num_workers: int = 0,
        activation_fct=None,
        apply_softmax=False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
    ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.

        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(
            sentences,
            batch_size=batch_size,
            collate_fn=self.smart_batching_collate_text_only,
            num_workers=num_workers,
            shuffle=False,
        )

        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)
