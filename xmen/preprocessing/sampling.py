from datasets import DatasetDict
import numpy as np


class Sampler:
    def __init__(self, random_seed: int, n: int = None, frac: float = None):
        self.random_seed = random_seed
        self.n = n
        self.frac = frac
        assert n or frac, "n or frac need to be provided"
        assert not (n and frac), "either n or frac can be provided"

    def transform_batch(self, dataset, aligned_dataset=None):
        def _sample(ds, aligned_ds=None):
            n_samples = self.n if self.n else int(self.frac * len(ds))
            rng = np.random.default_rng(seed=52)
            doc_indices = rng.choice(np.arange(0, len(ds)), size=n_samples, replace=False)

            sampled = ds.select(doc_indices)
            if not aligned_ds:
                return sampled
            return sampled, aligned_ds.select(doc_indices)

        if type(dataset) == DatasetDict:
            res = DatasetDict()
            if aligned_dataset:
                aligned_res = DatasetDict()
            for k, v in dataset.items():
                if aligned_dataset:
                    res[k], aligned_res[k] = _sample(v, aligned_dataset[k])
                else:
                    res[k] = _sample(v)
            return (res, aligned_res) if aligned_dataset else res
        else:
            return _sample(dataset, aligned_dataset or None)
