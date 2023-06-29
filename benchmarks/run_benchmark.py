import hydra
import os
from pathlib import Path
import logging

from xmen import load_kb
from xmen.data import get_cuis, CUIReplacer, EmptyNormalizationFilter, ConceptMerger, AbbreviationExpander
from xmen.linkers import SapBERTLinker, TFIDFNGramLinker, EnsembleLinker
from xmen.linkers.util import filter_and_apply_threshold
from xmen.reranking.cross_encoder import CrossEncoderTrainingArgs, CrossEncoderReranker
from xmen.evaluation import evaluate

import dataloaders

log = logging.getLogger(__name__)


def log_cuis_stats(dataset, kb):
    """Log the number of CUIs in the dataset and the number of missing CUIs compared to the KB."""
    for split, ds in dataset.items():
        cuis = get_cuis(ds)
        log.info(f"Split: {split}")
        log.info(f"CUIs (unique): {len(cuis)} ({len(set(cuis))})")
        missing_cuis = [cui for cui in cuis if cui not in kb.cui_to_entity]
        if len(missing_cuis) > 0:
            log.warning(f"Missing CUIs (unique): {len(missing_cuis)} ({len(set(missing_cuis))})")
        else:
            log.info("No missing CUIs")


class EvalLogger:
    """A helper class for logging evaluation results."""

    def __init__(self, ground_truth, file_prefix, ks=[1, 2, 4, 8, 16, 32, 64]) -> None:
        self.ground_truth = ground_truth
        self.ks = ks
        self.file_prefix = file_prefix
        self.file_name = f"{file_prefix}_results.csv"
        with open(self.file_name, "w") as f:
            f.write("key,")
            f.write(",".join([f"recall_{k}" for k in ks]))
            f.write("\n")

    def eval_and_log_at_k(self, key: str, prediction):
        """Evaluate a prediction and log the results at the given k values."""
        line = key
        for k in self.ks:
            recall = evaluate(self.ground_truth, prediction, top_k_predictions=k)["strict"]["recall"]
            log.info(f"{key} - {self.file_prefix} - Recall@{k}:{recall}")
            line += f",{recall}"
        with open(self.file_name, "a") as f:
            f.write(line)
            f.write("\n")


def prepare_data(dataset, config, kb):
    """Prepare the dataset for the benchmark."""

    log.info("Filtering empty concept IDs")
    dataset = EmptyNormalizationFilter().transform_batch(ConceptMerger().transform_batch(dataset))

    log_cuis_stats(dataset, kb)

    if umls := config.benchmark.dict.get("umls", None):
        log.info("Replace Retired CUIs")
        dataset = CUIReplacer(umls.meta_path).transform_batch(dataset)
        log_cuis_stats(dataset, kb)

    if config.data.expand_abbreviations:
        dataset = AbbreviationExpander().transform_batch(dataset)

    return dataset


def generate_candidates(dataset, config):
    """Generate candidates with n-gram, SapBERT and Ensemble."""
    batch_size = config.linker.batch_size
    k_ngram = config.linker.candidate_generation.ngram.k
    k_sapbert = config.linker.candidate_generation.sapbert.k
    k_ensemble = k_sapbert + k_ngram

    log.info("Generating n-gram candidates")
    ngram_linker = TFIDFNGramLinker(**config.linker.candidate_generation.ngram)
    candidates_ngram = ngram_linker.predict_batch(dataset)

    test_logger.eval_and_log_at_k("ngram", candidates_ngram["test"])
    val_logger.eval_and_log_at_k("ngram", candidates_ngram["validation"])

    log.info("Generating SapBERT candidates")
    sapbert_linker = SapBERTLinker.instance if SapBERTLinker.instance else SapBERTLinker(**config.linker.candidate_generation.sapbert)
    candidates_sapbert = sapbert_linker.predict_batch(dataset, batch_size=batch_size)

    test_logger.eval_and_log_at_k("sapbert", candidates_sapbert["test"])
    val_logger.eval_and_log_at_k("sapbert", candidates_sapbert["validation"])

    log.info("Generating ensemble candidates")
    ensemble_linker = EnsembleLinker()
    ensemble_linker.add_linker("sapbert", sapbert_linker, k=k_sapbert)
    ensemble_linker.add_linker("ngram", ngram_linker, k=k_ngram)

    # Re-use predictions for efficiency
    candidates_ensemble = ensemble_linker.predict_batch(
        dataset,
        batch_size,
        k_ensemble,
        reuse_preds={"sapbert": candidates_sapbert, "ngram": candidates_ngram},
    )

    test_logger.eval_and_log_at_k("ensemble", candidates_ensemble["test"])
    val_logger.eval_and_log_at_k("ensemble", candidates_ensemble["validation"])

    return candidates_ensemble


@hydra.main(version_base=None, config_path=".", config_name="benchmark.yaml")
def main(config) -> None:
    """Run a benchmark with the given config file."""

    log.info(f"Running in {os.getcwd()}")

    base_path = Path(config.hydra_work_dir)
    output_base_dir = Path(config.output)

    dict_name = base_path / f"{config.benchmark.name}.jsonl"
    if not dict_name.exists():
        log.error(f"{dict_name} does not exist, please run: xmen dict benchmarks/benchmark/<config name>")
        return

    index_base_path = base_path / "index"
    if not index_base_path.exists():
        log.error(f"{index_base_path} does not exist, please run: xmen index benchmarks/benchmark/<config name> --all")
        return

    log.info("Loading KB")
    kb = load_kb(dict_name)
    log.info(f"Loaded {dict_name} with {len(kb.cui_to_entity)} concepts and {len(kb.alias_to_cuis)} aliases")

    log.info("Loading dataset")
    splits = dataloaders.load_dataset(config.benchmark.dataset, data_dir=config.benchmark.get("data_dir", None))
    log.info(f"Running on {len(splits)} splits")
    for fold, dataset in enumerate(splits):
        fold_prefix = f"{fold}-{config.benchmark.name}"

        dataset = prepare_data(dataset, config, kb)

        global val_logger
        val_logger = EvalLogger(ground_truth=dataset["validation"], file_prefix=f"{fold_prefix}_validation")

        global test_logger
        test_logger = EvalLogger(ground_truth=dataset["test"], file_prefix=f"{fold_prefix}_test")

        candidates = generate_candidates(dataset, config)

        # Prepare Dataset
        candidates = filter_and_apply_threshold(candidates, config.linker.reranking.k, 0.0)

        cross_enc_ds = CrossEncoderReranker.prepare_data(candidates, dataset, kb)

        train_args = CrossEncoderTrainingArgs(
            num_train_epochs=2,
        )

        rr = CrossEncoderReranker()
        output_dir = output_base_dir / fold_prefix / "cross_encoder_training"

        rr.fit(
            train_dataset=cross_enc_ds["train"].dataset,
            val_dataset=cross_enc_ds["validation"].dataset,
            output_dir=output_dir,
            training_args=train_args,
            show_progress_bar=False,
        )

        rr = CrossEncoderReranker.load(output_dir, device=0)

        cross_enc_pred_val = rr.rerank_batch(candidates["validation"], cross_enc_ds["validation"])
        val_logger.eval_and_log_at_k("cross_encoder", cross_enc_pred_val)

        cross_enc_pred_test = rr.rerank_batch(candidates["test"], cross_enc_ds["test"])
        test_logger.eval_and_log_at_k("cross_encoder", cross_enc_pred_test)


if __name__ == "__main__":
    main()
