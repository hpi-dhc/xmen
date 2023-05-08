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
    """
    Returns the index of the first occurrence of the specified value in the given list.
    If the value is not found, returns 0.

    Args:
    - val: the value to search for
    - in_list: the list to search in

    Returns:
    - The index of the first occurrence of the specified value in the given list.
      If the value is not found, returns 0.
    """
    try:
        return in_list.index(val) + 1
    except ValueError:
        return 0


def preprocess_el_dataset(examples, tokenizer, context_length: int, n_candidates: int, encode_positions: bool = False):
    """
    Preprocesses a dataset of examples for entity linking, converting them to tokenized input
    suitable for use with a neural network model. Each example consists of a mention and a list of
    candidate entities, represented as their synonyms.

    Args:
    - examples: a dictionary containing the "mention", "synonyms", "context_left", and "context_right"
               fields for each example.
    - tokenizer: the tokenizer to use for tokenizing the input
    - context_length: the maximum length of context to include on either side of the mention, in tokens
    - n_candidates: the number of candidates to include for each mention
    - encode_positions: if True, includes position encoding in the output tokens

    Returns:
    - A dictionary containing the tokenized examples, split into inputs for each mention and its
    corresponding candidate entities. The keys are the names of the inputs ("input_ids",
    "attention_mask", "token_type_ids"), and the values are lists of lists, where each inner list
    corresponds to the inputs for a single mention and its candidate entities.
    """
    actual_n_candidates = n_candidates + 1  # if candidate not included
    mentions = [
        [
            f"{l_ctx[-context_length:] if context_length > 0 else ''} [START] {m} [END] {r_ctx[:context_length] if context_length > 0 else ''}"
        ]
        * actual_n_candidates
        for l_ctx, m, r_ctx in zip(examples["context_left"], examples["mention"], examples["context_right"])
    ]
    mentions = sum(mentions, [])
    candidates = []
    for e in examples["synonyms"]:
        if len(e) < n_candidates:
            e.extend([""] * (n_candidates - len(e)))
        e.insert(0, ["[UNK]"])

    candidates = [
        (f"[POSSTART] {i} [POSEND] " if encode_positions else "") + (" [SEP] ".join(s))
        for e in examples["synonyms"]
        for i, s in enumerate(e)
    ]

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
    """
    Creates a multiple choice dataset suitable for training a neural network model for entity linking.
    The dataset consists of a list of mentions and their corresponding candidate entities, where one of
    the candidates is the correct entity for the mention. The dataset is tokenized and indexed, ready for
    use with a neural network.

    Args:
    - candidate_ds: the dataset of candidate entities for each mention
    - ground_truth: the correct entity for each mention
    - kb: the knowledge base to use for entity linking
    - tokenizer: the tokenizer to use for tokenizing the input
    - context_length: the maximum length of context to include on either side of the mention, in tokens
    - k: the number of candidates to include for each mention
    - expand_abbreviations: if True, expands abbreviations in the dataset using the knowledge base

    Returns:
    - An IndexedDataset object containing the tokenized and indexed multiple choice dataset.
    """
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
        """
        Given a list of candidates, their corresponding ground truth labels, a knowledge base (kb), a tokenizer, and other optional arguments, creates a dataset that can be used to train a listwise reranker. Returns an IndexedDatasetDict containing train, validation, and/or test datasets depending on the input types.

        Args:
        - candidates: Union[DatasetDict, List[Dict[str, Union[str, List[str]]]]] - A dataset or list of dictionaries containing the candidates, their labels, and optionally other metadata.
        - ground_truth: Optional[DatasetDict] - A dataset containing the ground truth labels for the candidates. Only necessary if candidates is a DatasetDict.
        - kb: Union[Path, str] - A path to a file or directory containing the knowledge base for the task.
        - tokenizer: PreTrainedTokenizerBase - A tokenizer object to use for tokenizing the candidates and ground truth labels.
        - context_length: int - The maximum number of tokens to include in the context of each candidate during tokenization.
        - k: int - The number of candidates to consider in each listwise comparison.
        - expand_abbreviations: bool - Whether to expand abbreviations in the candidates and ground truth labels during tokenization.
        - **kwargs: Additional optional keyword arguments to pass to create_multiple_choice_dataset().

        Returns:
        - res: IndexedDatasetDict - An IndexedDatasetDict containing the train, validation, and/or test datasets.

        Raises:
        - AssertionError if `candidates` is not a DatasetDict
        """
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
        """
        Given train and validation datasets, a tokenizer, an output directory, and other optional training arguments, creates a Trainer object that can be used to train a listwise reranker. Returns the Trainer object.

        Args:
        - train_dataset: Dataset - A dataset containing the training examples.
        - val_dataset: Dataset - A dataset containing the validation examples.
        - tokenizer: PreTrainedTokenizerBase - A tokenizer object to use for tokenizing the examples.
        - output_dir: Union[Path, str] - The path to the output directory where the trained model will be saved.
        - **training_args: Additional optional keyword arguments to pass to the Trainer object.

        Returns:
        - trainer: Trainer - A Trainer object that can be used to train a listwise reranker.
        """
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
