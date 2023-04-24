# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

from abc import ABC, abstractmethod
from datasets.arrow_dataset import Dataset
import logging

from xmen.reranking import Reranker

logger = logging.getLogger("transformers")


class EntityLinker(ABC):
    def predict_batch(self, dataset: Dataset, batch_size: int = None) -> Dataset:
        """Naive default implementation of batch prediction.
        Should be overridden if the particular model provides an efficient way to predict in batch (e.g., on a GPU)

        Args:
            dataset (Dataset): Input (arrow) dataset with entities but without concepts

        Returns:
            Dataset: (arrow) dataset with linked concepts
        """
        return dataset.map(
            lambda unit: {"entities": self.predict(unit["passages"], unit["entities"])},
            load_from_cache_file=False,
        )

    def get_logger(self):
        return logger

    @abstractmethod
    def predict(self, passages: list, entities: list) -> list:
        pass


class RerankedLinker(EntityLinker):
    def __init__(self, linker: EntityLinker, ranker: Reranker):
        self.linker = linker
        self.ranker = ranker

    def predict_batch(self, dataset: Dataset, **linker_kwargs) -> Dataset:
        result = self.linker.predict_batch(dataset, linker_kwargs)
        return self.ranker.rerank_batch(result)

    def predict(self, passages: list, entities: list) -> list:
        raise NotImplementedError()


from .tf_idf_ngram_linker import TFIDFNGramLinker
from .sap_bert_linker import SapBERTLinker
from .ensemble import EnsembleLinker
