from typing import Union
from pathlib import Path
from tqdm import tqdm

from datasets import DatasetDict

from xmen.reranking import Reranker, multiple_choice_util, ranking_util
from xmen.data import IndexedDataset, IndexedDatasetDict

from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForMultipleChoice,
)


def index_of(val, in_list):
    try:
        return in_list.index(val) + 1
    except ValueError:
        return 0


def preprocess_el_dataset(examples, tokenizer, context_length: int, n_candidates: int, encode_positions : bool = False):
    actual_n_candidates = n_candidates + 1  # if candidate not included
    mentions = [
        [f"{l_ctx[-context_length:] if context_length > 0 else ''} [START] {m} [END] {r_ctx[:context_length] if context_length > 0 else ''}"] * actual_n_candidates
        for l_ctx, m, r_ctx in zip(examples["context_left"], examples["mention"], examples["context_right"])
    ]
    mentions = sum(mentions, [])
    candidates = []
    for e in examples["synonyms"]:
        if len(e) < n_candidates:
            e.extend([""] * (n_candidates - len(e)))
        e.insert(0, ["[UNK]"])

    candidates = [(f"[POSSTART] {i} [POSEND] " if encode_positions else "") + (" [SEP] ".join(s)) for e in examples["synonyms"] for i, s in enumerate(e)]

    tokenized_examples = tokenizer(mentions, candidates, truncation=True, max_length=512)

    return {
        k: [v[i : i + actual_n_candidates] for i in range(0, len(v), actual_n_candidates)]
        for k, v in tokenized_examples.items()
    }


def create_multiple_choice_dataset(
    candidate_ds,
    ground_truth,
    kb,
    tokenizer,
    context_length: int,
    k: int,
    expand_abbreviations: bool,
):
    flat_candidate_ds, doc_index = ranking_util.get_flat_candidate_ds(
        candidate_ds, ground_truth, expand_abbreviations, kb
    )

    # get index of the correct answer or 0 if not in the dataset
    flat_candidate_ds = flat_candidate_ds.map(
        lambda e: {"label": index_of(e["label"][0], e["candidates"])},
        load_from_cache_file=False,
    )

    ds_tokenized = flat_candidate_ds.map(
        lambda e: preprocess_el_dataset(e, tokenizer, context_length, k),
        batched=True,
        remove_columns=[c for c in flat_candidate_ds.column_names if c != "label"],
        load_from_cache_file=False,
    )
    return IndexedDataset(ds_tokenized, doc_index)


class ListWiseReranker(Reranker):
    def __init__(self):
        pass

    @staticmethod
    def prepare_data(
        candidates,
        ground_truth,
        kb,
        tokenizer,
        context_length: int,
        k: int,
        expand_abbreviations: bool,
        **kwargs,
    ):
        print("Context length:", context_length)
        print("n candidates:", k)
        if type(candidates) == DatasetDict:
            assert type(ground_truth) == DatasetDict
            res = IndexedDatasetDict()
            for split, cand in candidates.items():
                gt = ground_truth[split]
                ds = create_multiple_choice_dataset(
                    cand,
                    gt,
                    kb,
                    tokenizer,
                    context_length,
                    k,
                    expand_abbreviations,
                )
                res[split] = ds
            return res
        else:
            return create_multiple_choice_dataset(
                candidates,
                ground_truth,
                kb,
                tokenizer,
                context_length,
                k,
                expand_abbreviations,
            )

    @staticmethod
    def get_trainer(
        train_dataset,
        val_dataset,
        tokenizer,
        output_dir: Union[str, Path],
        **training_args,
    ):
        model = AutoModelForMultipleChoice.from_pretrained(training_args["model_name"])

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            # save_strategy="no",
            save_strategy="epoch",
            gradient_checkpointing=True,
            load_best_model_at_end=True,
            report_to="wandb",
            save_total_limit=2,
            run_name=training_args.get("run_name", None),
            metric_for_best_model="f1",
            **{k: v for k, v in training_args.items() if k != "model_name"},
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=multiple_choice_util.DataCollatorForMultipleChoice(tokenizer=tokenizer),
            compute_metrics=multiple_choice_util.compute_entity_linking_metrics,
        )

        return trainer

    def rerank_batch(self):
        pass
