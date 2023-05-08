from typing import Dict
from xmen.linkers import EntityLinker
from .util import filter_and_apply_threshold
from datasets import Dataset, utils
import numpy as np
import itertools


class EnsembleLinker(EntityLinker):
    """
    A class for combining multiple entity linking models together into an ensemble.

    Attributes:
    - linkers_fn (Dict[str, Callable]): a dictionary of linker functions, each associated with a unique name
    - linkers_k (Dict[str, int]): a dictionary of integers representing the number of top entities to keep for each linker
    - linker_thresholds (Dict[str, float]): a dictionary of floats representing the score threshold to use for each linker
    - linker_weigths (Dict[str, float]): a dictionary of floats representing the weight to give to each linker during combination
    """

    # Ignore caching when using dataset.map
    def __getstate__(self):
        return {}

    def __init__(self):
        self.linkers_fn = {}
        self.linkers_k = {}
        self.linker_thresholds = {}
        self.linker_weigths = {}

    def add_linker(self, name, linker, k=1, threshold=0.0, weight=1.0):
        """
        Adds a new linker to the ensemble.

        Args:
        - name: a unique name to associate with the linker
        - linker: an instance of the EntityLinker class
        - k: an integer representing the number of top entities to keep for this linker (default is 1)
        - threshold: a float representing the score threshold to use for this linker (default is 0.0)
        - weight: a float representing the weight to give to this linker during combination (default is 1.0)
        """
        self.add_linker_fn(name, lambda: linker, k, threshold, weight)

    def add_linker_fn(self, name, linker_fn, k=1, threshold=0.0, weight=1.0):
        """
        Adds a new linker function to the ensemble.

        Args:
        - name: a unique name to associate with the linker
        - linker_fn: a function that returns an instance of the EntityLinker class
        - k: an integer representing the number of top entities to keep for this linker (default is 1)
        - threshold: a float representing the score threshold to use for this linker (default is 0.0)
        - weight: a float representing the weight to give to this linker during combination (default is 1.0)
        """
        self.linkers_fn[name] = linker_fn
        self.linkers_k[name] = k
        self.linker_thresholds[name] = threshold
        self.linker_weigths[name] = weight

    def predict_batch(self, dataset, batch_size, top_k=None, reuse_preds=None):
        """
        Runs the ensemble on a dataset in batches.

        Args:
        - dataset: a dataset of documents to run entity linking on
        - batch_size: an integer representing the batch size to use
        - top_k: an optional integer representing the number of top entities to keep per document
        - reuse_preds: an optional dictionary of precomputed predictions for each linker

        Returns:
        - a dictionary containing the predicted entities for each document in the dataset
        """

        def merge_linkers(batch, index):
            progress = utils.logging.is_progress_bar_enabled()
            try:
                mapped = {}
                if progress:
                    utils.logging.disable_progress_bar()

                for linker_name, linker_fn in self.linkers_fn.items():
                    if reuse_preds:
                        linked = reuse_preds[linker_name].select(index)
                    else:
                        linker = linker_fn()
                        self.get_logger().info(f"Running{linker_name}")
                        linked = linker.predict_batch(Dataset.from_dict(batch), batch_size)
                    mapped[linker_name] = filter_and_apply_threshold(
                        linked,
                        self.linkers_k[linker_name],
                        self.linker_thresholds[linker_name],
                    )["entities"]

                entities = []

                for i, doc in enumerate(batch["entities"]):
                    for j, e in enumerate(doc):
                        merged_scores = []
                        e["normalized"] = []
                        for linker_name in mapped.keys():
                            for n in mapped[linker_name][i][j]["normalized"]:
                                n["predicted_by"] = linker_name
                                if "score" in n:
                                    n["score"] *= self.linker_weigths[linker_name]
                                merged_scores.append(n)
                        # Merge duplicate predictions
                        key_fn = lambda cand: cand["db_id"]
                        for key, grp in itertools.groupby(sorted(merged_scores, key=key_fn), key=key_fn):
                            preds = list(grp)
                            db_name = {p["db_name"] for p in preds}
                            score = max([p["score"] for p in preds])
                            pred_by = [p["predicted_by"] for p in preds]
                            assert len(db_name) == 1
                            e["normalized"].append(
                                {
                                    "db_id": key,
                                    "db_name": db_name.pop(),
                                    "score": score,
                                    "predicted_by": pred_by,
                                }
                            )
                        e["normalized"] = sorted(e["normalized"], key=lambda n: n["score"])[-1::-1]
                        if top_k:
                            e["normalized"] = e["normalized"][:top_k]

                    entities.append(doc)
            finally:
                if progress:
                    utils.logging.enable_progress_bar()
            return {"entities": entities}

        return dataset.map(
            merge_linkers,
            with_indices=True,
            batched=True,
            batch_size=batch_size,
            load_from_cache_file=False,
        )

    def predict(self, unit: str, entities: dict) -> dict:
        raise NotImplementedError()
