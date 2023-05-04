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


def flat_ds_to_cross_enc_dataset(flat_candidate_ds, doc_index, context_length, mask_mention):
    print('Masking:', mask_mention)
    res = []
    res_index = []
    for doc, idx in zip(tqdm(flat_candidate_ds), doc_index):
        l_ctx, m, r_ctx = doc["context_left"], doc["mention"], doc["context_right"]
        mention = f"{l_ctx[-context_length:] if context_length else ''} [START] {m if not mask_mention else '[MASK]'} [END] {r_ctx[:context_length] if context_length else ''}"
        batch = []
        for c, score, syns in zip(doc["candidates"], doc["scores"], doc["synonyms"]):
            candidate = syns[0] + " [TITLE] " + " [SEP] ".join(syns[1:])
            label = 1 if c in doc["label"] else 0
            batch.append(ScoredInputExample(texts=[mention, candidate], label=label, score=score))
        if batch:
            res.append(batch)
            res_index.append(idx)
    return res, res_index


def create_cross_enc_dataset(candidate_ds, ground_truth, kb, context_length: int, expand_abbreviations: bool, masking: bool):
    flat_candidate_ds, doc_index = get_flat_candidate_ds(
        candidate_ds, ground_truth, expand_abbreviations=expand_abbreviations, kb=kb
    )
    return flat_ds_to_cross_enc_dataset(flat_candidate_ds, doc_index, context_length, mask_mention=masking)


def cross_encoder_predict(cross_encoder, cross_enc_dataset, show_progress_bar=True, convert_to_numpy=True):
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

class CrossEncoderTrainingArgs:
    
    def __init__(self, args : dict):
        self.args = args
    
    def __init__(self,
           model_name : str,
           num_train_epochs : int,
           fp16 : bool = True,
           label_smoothing : bool = False,
           score_regularization : bool = False,
           train_layers : list  = None,
           softmax_loss : bool = True,
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
    def __init__(self, model=None):
        self.model = model

    @staticmethod
    def load(checkpoint, device):
        model = CrossEncoder(checkpoint)
        model.model.to(torch.device(device))
        return CrossEncoderReranker(model)

    @staticmethod
    def prepare_data(
        candidates,
        ground_truth,
        kb,
        context_length: int,
        expand_abbreviations: bool = False,
        masking: bool = False,
        **kwargs,
    ):
        print("Context length:", context_length)

        if type(candidates) == DatasetDict:
            assert type(ground_truth) == DatasetDict
            res = IndexedDatasetDict()
            for split, cand in candidates.items():
                gt = ground_truth[split]
                ds, doc_index = create_cross_enc_dataset(
                    cand,
                    gt,
                    kb,
                    context_length,
                    expand_abbreviations,
                    masking
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
        eval_callback=None
    ):
        for k, v in training_args.args.items():
            print(k, ':=', v)
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
            train_layers=training_args["train_layers"]
        )

    def rerank_batch(self):
        pass


class EntityLinkingEvaluator:
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
        model_pred_scores = cross_encoder_predict(
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
            self.eval_callback({
                'train/epoch' : epoch,
                f'{self.name}/accuracy_1' : acc,
                f'{self.name}/accuracy_5' : acc5,
                f'{self.name}/accuracy_{k_max}' : accmax,
            })

        baseline_pred_scores = np.array([[0]] * len(self.el_dataset))
        baseline_acc = get_accuracy(baseline_pred_scores, 1)
        logger.info(f"Baseline Accuracy: {baseline_acc}")

        return acc


class BatchedCrossEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, cross_enc_data):
        self.cross_enc_data = cross_enc_data

    def __len__(self):
        return len(self.cross_enc_data)

    def __getitem__(self, idx):
        return self.cross_enc_data[idx]
