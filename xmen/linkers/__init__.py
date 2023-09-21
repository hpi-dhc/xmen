# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

from abc import ABC, abstractmethod
from datasets.arrow_dataset import Dataset
from pathlib import Path
from xmen.log import logger

from xmen.reranking import Reranker


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


def default_ensemble(index_base_path, k_ngram=100, k_sapbert=1000, cuda=True):
    """
    Creates the default ensemble candidate generator consisting of equally weighted SapBERT and TF-IDF N-Gram Linker
    """
    index_base_path = Path(index_base_path)
    ngram_linker = TFIDFNGramLinker(index_base_path=index_base_path / "ngrams", k=k_ngram)

    SapBERTLinker.clear()
    sapbert_linker = SapBERTLinker(index_base_path=index_base_path / "sapbert", k=k_sapbert, cuda=cuda)

    ensemble = EnsembleLinker()
    ensemble.add_linker("ngram", ngram_linker, k=k_ngram)
    ensemble.add_linker("sapbert", sapbert_linker, k=k_sapbert)

    return ensemble
