import hydra
import os
from pathlib import Path
import logging

from xmen import load_kb
from xmen.data import get_cuis, CUIReplacer, EmptyNormalizationFilter, ConceptMerger, AbbreviationExpander
from xmen.linkers import SapBERTLinker, TFIDFNGramLinker, EnsembleLinker
from xmen.evaluation import evaluate

from dataloaders import load_dataset

log = logging.getLogger(__name__)


def train_val_test_split(dataset):
    """Split a dataset into train, validation, and test sets."""
    # TODO: implement for datasets without a train/val/test split
    return dataset


def log_cuis_stats(dataset, kb):
    for split, ds in dataset.items():
        cuis = get_cuis(ds)
        log.info(f"Split: {split}")
        log.info(f"CUIs (unique): {len(cuis)} ({len(set(cuis))})")
        missing_cuis = [cui for cui in cuis if cui not in kb.cui_to_entity]
        if len(missing_cuis) > 0:
            log.warn(f"Missing CUIs (unique): {len(missing_cuis)} ({len(set(missing_cuis))})")
        else:
            log.info("No missing CUIs")


class EvalLogger:
    """A class for logging evaluation results."""

    def __init__(self, split, ground_truth, ks=[1, 2, 4, 8, 16, 32, 64]) -> None:
        self.ground_truth = ground_truth
        self.ks = ks
        self.split = split
        self.file_name = f"{split}_results.csv"
        with open(self.file_name, "w") as f:
            f.write("key,")
            f.write(",".join(["r@_{k}" for k in ks]))
            f.write("\n")

    def eval_and_log_at_k(self, key: str, prediction):
        """Evaluate a prediction and log the results at the given k values."""
        line = key
        for k in self.ks:
            recall = evaluate(self.ground_truth, prediction, top_k_predictions=k)["strict"]["recall"]
            log.info("Recall@k", recall)
            line += f",{recall}"
        with open(self.file_name, "a") as f:
            f.write(line)
            f.write("\n")


def prepare_data(config, kb):
    log.info("Loading dataset")
    dataset = load_dataset(config.benchmark.dataset)
    log.info(dataset)

    log.info("Filtering empty concept IDs")
    dataset = EmptyNormalizationFilter().transform_batch(ConceptMerger().transform_batch(dataset))

    log_cuis_stats(dataset, kb)

    log.info("Replace Retired CUIs")
    dataset = CUIReplacer(config.benchmark.dict.umls.meta_path).transform_batch(dataset)
    log_cuis_stats(dataset, kb)

    if config.data.expand_abbreviations:
        dataset = AbbreviationExpander().transform_batch(dataset)

    return dataset


def generate_candidates(index_base_path, dataset, config):
    ngram_linker = TFIDFNGramLinker(index_base_path / "ngram", **config.linker.ngram)
    candidates_ngram = ngram_linker.predict_batch(dataset)
    return candidates_ngram


@hydra.main(version_base=None, config_path=".", config_name="benchmark.yaml")
def main(config) -> None:
    """Run a benchmark with the given config file."""

    log.info(f"Running in {os.getcwd()}")

    base_path = Path(config.hydra_work_dir)

    dict_name = base_path / f"{config.benchmark.name}.jsonl"

    if not dict_name.exists():
        log.error(f"{dict_name} does not exist, please run: xmen dict <config name>")
        return

    index_base_path = base_path / "index"

    if not index_base_path.exists():
        log.error(f"{index_base_path} does not exist, please run: xmen index <config name> --all")
        return

    log.info("Loading KB")
    kb = load_kb(dict_name)
    log.info(f"Loaded {dict_name} with {len(kb.cui_to_entity)} concepts and {len(kb.alias_to_cuis)} aliases")

    dataset = prepare_data(config, kb)

    val_logger = EvalLogger(ground_truth=dataset["val"], split="val")
    test_logger = EvalLogger(ground_truth=dataset["test"], split="test")

    candidates = generate_candidates(dataset, kb)
    test_logger.eval_and_log_at_k("ngram", candidates["test"])
    val_logger.eval_and_log_at_k("ngram", candidates["val"])


if __name__ == "__main__":
    main()
