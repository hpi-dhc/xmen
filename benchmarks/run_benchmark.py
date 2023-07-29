import hydra
from hydra.core.hydra_config import HydraConfig
import os
import sys
from pathlib import Path
import logging
import dataloaders
from omegaconf import OmegaConf, SCMode
import wandb
import submitit
import torch
from transformers.trainer_utils import enable_full_determinism
import traceback
import uuid

from xmen import load_kb
from xmen.data import (
    get_cuis,
    filter_and_apply_threshold,
    CUIReplacer,
    EmptyNormalizationFilter,
    MissingCUIFilter,
    AbbreviationExpander,
    SemanticGroupFilter,
    Sampler,
)
from xmen.linkers import SapBERTLinker, TFIDFNGramLinker, EnsembleLinker
from xmen.reranking.cross_encoder import CrossEncoderTrainingArgs, CrossEncoderReranker
from xmen.evaluation import evaluate

log = logging.getLogger(__name__)

from hydra.core.config_store import ConfigStore
from hydra.core.plugins import Plugins
from hydra_plugins.hydra_process_launcher import InProcessLauncher, InProcessQueueConf
from hydra.utils import to_absolute_path

Plugins.instance().register(InProcessLauncher)

ConfigStore.instance().store(
    group="hydra/launcher",
    name="submitit_inprocess",
    node=InProcessQueueConf(),
    provider="submitit_launcher",
)


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

    def __init__(
        self, ground_truth, split, file_prefix, ks=[1, 2, 4, 8, 16, 32, 64], callback=None, subsets=[]
    ) -> None:
        self.ground_truth = ground_truth
        self.ks = ks
        self.split = split
        self.subsets = subsets
        self.file_prefix = file_prefix + "_" + split
        self.file_name = f"{self.file_prefix}_results.csv"
        self.callback = callback
        self.metrics = ["recall", "precision", "fscore"]
        with open(self.file_name, "w") as f:
            f.write("key,subset")
            for k in ks:
                for m in self.metrics:
                    f.write(f",{m}_{k}")
            f.write("\n")

    def eval_and_log_at_k(self, key: str, prediction):
        """Evaluate a prediction and log the results at the given k values."""
        for subset in [None] + self.subsets:
            subset_key = key + ("_" + subset if subset else "")
            line = key + "," + (subset if subset else "")
            gt = self.ground_truth
            p = prediction
            if subset:
                gt = gt.filter(lambda x: x["corpus_id"] == subset)
                p = p.filter(lambda x: x["corpus_id"] == subset)
            for k in self.ks:
                for metric in self.metrics:
                    res = evaluate(gt, p, top_k_predictions=k)["strict"][metric]
                    log_key = f"{subset_key}-{self.file_prefix}-{metric}@{k}"
                    log.info(f"{log_key}: {res}")
                    if self.callback:
                        self.callback({f"{self.split}/{subset_key}-{metric}@{k}": res})
                    line += f",{res}"
            with open(self.file_name, "a") as f:
                f.write(line)
                f.write("\n")


def prepare_data(dataset, config, kb):
    """Prepare the dataset for the benchmark."""

    log.info("Filtering empty concept IDs")
    dataset = EmptyNormalizationFilter().transform_batch(dataset)

    log_cuis_stats(dataset, kb)

    if umls := config.dict.get("umls", None):
        log.info("Replace Retired CUIs")
        dataset = CUIReplacer(umls.meta_path).transform_batch(dataset)
        log_cuis_stats(dataset, kb)

    if config.data.get("ignore_missing_cuis", False):
        log.info("Ignoring entities missing CUIs")
        dataset = MissingCUIFilter(kb).transform_batch(dataset)
        log_cuis_stats(dataset, kb)

    if config.data.expand_abbreviations:
        log.info("Expanding Abbreviations")
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
    sapbert_linker = (
        SapBERTLinker.instance
        if SapBERTLinker.instance
        else SapBERTLinker(**config.linker.candidate_generation.sapbert)
    )
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
    try:
        """Run a benchmark with the given config file."""
        log.info(f"Running in {os.getcwd()}")
        log.info(f"# CUDA Devices: {torch.cuda.device_count()}")

        enable_full_determinism(config.random_seed)

        if HydraConfig.get().runtime.choices["hydra/launcher"] == "local":
            log.info("Running locally")
            job_id = "local"
            hostname = "local"
        else:
            env = submitit.JobEnvironment()
            job_id = env.job_id
            hostname = env.hostname

        base_path = Path(config.work_dir)

        dict_name = base_path / f"{config.name}.jsonl"
        if not dict_name.exists():
            log.error(f"{dict_name} does not exist, please run: xmen dict benchmarks/benchmark/<config name>")
            return

        index_base_path = base_path / "index"
        if not index_base_path.exists():
            log.error(
                f"{index_base_path} does not exist, please run: xmen index benchmarks/benchmark/<config name> --all"
            )
            return

        log.info("Loading KB")
        kb = load_kb(dict_name)
        log.info(f"Loaded {dict_name} with {len(kb.cui_to_entity)} concepts and {len(kb.alias_to_cuis)} aliases")

        subsets = config.get("subsets", [])
        if subsets:
            log.info(f"Considering subsets: {subsets}")

        log.info("Loading dataset")
        if local_dataset := config.get("local_dataset", None):
            splits = dataloaders.load_dataset(to_absolute_path(config.local_dataset))
        else:
            if data_dir := config.get("data_dir", None):
                splits = dataloaders.load_dataset(config.dataset, data_dir=data_dir, subsets=subsets)
            else:
                splits = dataloaders.load_dataset(config.dataset, subsets=subsets)
        log.info(f"Running on {len(splits)} splits")
        for fold, dataset in enumerate(splits):
            if sample := config.get("sample", None):
                log.info(f"Sampling {sample} examples")
                dataset = Sampler(config.random_seed, n=sample).transform_batch(dataset)

            fold_prefix = f"{fold}-{config.name}"
            log.info(f"Fold: {fold_prefix}")
            run = None
            try:
                output_base_dir = Path(config.output) / str(uuid.uuid4())
                log.info(f"Writing to output dir {output_base_dir}")
                if not config.disable_wandb:
                    log.info("Initializing Weights & Biases run")
                    run = wandb.init(name=fold_prefix, project=config.wandb_project, tags=config.get("tags", None))
                    eval_callback = run.log
                    dict_config = OmegaConf.to_container(config, structured_config_mode=SCMode.DICT_CONFIG)
                    log.info(dict_config)
                    run.log(dict_config)
                    run.log({"cp_dir": str(output_base_dir)})
                    run.log({"hydra_dir": os.getcwd()})
                    run.log({"job_id": job_id, "hostname": hostname})
                else:
                    eval_callback = None

                dataset = prepare_data(dataset, config, kb)
                log.info(dataset)

                global val_logger
                val_logger = EvalLogger(
                    ground_truth=dataset["validation"],
                    file_prefix=f"{fold_prefix}",
                    split="validation",
                    callback=eval_callback,
                    subsets=subsets,
                )

                global test_logger
                test_logger = EvalLogger(
                    ground_truth=dataset["test"],
                    file_prefix=f"{fold_prefix}",
                    split="test",
                    callback=eval_callback,
                    subsets=subsets,
                )

                dataset.save_to_disk(fold_prefix + "_dataset")

                log.info("Generating candidates")
                candidates = generate_candidates(dataset, config)

                if config.data.get("filter_semantic_groups", False):
                    log.info("Filtering semantic groups")
                    group_filter = SemanticGroupFilter(kb, "v03")
                    candidates = group_filter.transform_batch(candidates)

                candidates = filter_and_apply_threshold(candidates, config.linker.reranking.k, 0.0)

                test_logger.eval_and_log_at_k("candidates", candidates["test"])
                val_logger.eval_and_log_at_k("candidates", candidates["validation"])

                candidates.save_to_disk(fold_prefix + "_candidates")

                # Prepare Dataset for Cross Encoder
                log.info("Preparing data for cross encoder training")
                cross_enc_ds = CrossEncoderReranker.prepare_data(
                    candidates, dataset, kb, **config.linker.reranking.data
                )

                log.info("Training cross encoder (this might take a while...)")
                train_args = CrossEncoderTrainingArgs(
                    random_seed=config.random_seed, **config.linker.reranking.training
                )

                rr = CrossEncoderReranker()
                output_dir = output_base_dir / fold_prefix / "cross_encoder_training"

                rr.fit(
                    train_dataset=cross_enc_ds["train"].dataset,
                    val_dataset=cross_enc_ds["validation"].dataset,
                    output_dir=output_dir,
                    training_args=train_args,
                    show_progress_bar=True,
                    eval_callback=eval_callback,
                )

                log.info("Running prediction")

                rr = CrossEncoderReranker.load(output_dir, device=0)

                for allow_nil in [True, False]:
                    suffix = "_no_nil" if not allow_nil else ""
                    cross_enc_pred_val = rr.rerank_batch(
                        candidates["validation"], cross_enc_ds["validation"], allow_nil=allow_nil
                    )
                    val_logger.eval_and_log_at_k(f"cross_encoder{suffix}", cross_enc_pred_val)
                    cross_enc_pred_val.save_to_disk(fold_prefix + f"_cross_enc_pred_val{suffix}")

                    cross_enc_pred_test = rr.rerank_batch(candidates["test"], cross_enc_ds["test"], allow_nil=allow_nil)
                    test_logger.eval_and_log_at_k(f"cross_encoder{suffix}", cross_enc_pred_test)
                    cross_enc_pred_test.save_to_disk(fold_prefix + f"_cross_enc_pred_test{suffix}")

                    if thresholds := config.get("thresholds", None):
                        for t in thresholds:
                            log.info(f"Thresholding at {t}")
                            cross_enc_pred_val_t = filter_and_apply_threshold(cross_enc_pred_val, k=1, threshold=t)
                            val_logger.eval_and_log_at_k(f"cross_encoder_t_{t}{suffix}", cross_enc_pred_val_t)

                            cross_enc_pred_test_t = filter_and_apply_threshold(cross_enc_pred_test, k=1, threshold=t)
                            test_logger.eval_and_log_at_k(f"cross_encoder_t_{t}{suffix}", cross_enc_pred_test_t)

            finally:
                if run:
                    run.finish()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
