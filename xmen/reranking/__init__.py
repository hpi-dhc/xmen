from abc import ABC, abstractmethod
from datasets.arrow_dataset import Dataset
import numpy as np


class Reranker(ABC):
    @abstractmethod
    def rerank_batch(self, dataset: Dataset) -> Dataset:
        pass

    @staticmethod
    def sort_concepts(concepts):
        idx = np.argsort(np.array(concepts["score"]))[-1::-1]
        result = {}
        for k, v in concepts.items():
            result[k] = list(np.array(v)[idx])
        return result
