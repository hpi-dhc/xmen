# pylint: disable=g-import-not-at-top,g-bad-import-order,wrong-import-position

from abc import ABC, abstractmethod
from datasets.arrow_dataset import Dataset
from pathlib import Path
from xmen.log import logger

from xmen.reranking import Reranker
from xmen.data import from_spans
from typing import Union


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

    def predict_no_context(
        self, entities: Union[str, list[str]], label: Union[str, list[str]] = None, batch_size: int = None
    ) -> list:
        """
        Generates candidate concepts for the given entities (one or more) without any context.

        Args:
        - entities (str | list[str]): The entity or entities for which to generate candidates.
        - label (str | list[str]): The label or labels for the entities. If a single label is provided, it will be used for all entities.
        - batch_size (int): The batch size to use for prediction. If None, the default batch size of the model will be used.
        """
        is_str = False
        if isinstance(entities, str):
            is_str = True
            entities = [entities]
            assert label is None or isinstance(label, str)
            label = [label]
        elif label is None or isinstance(label, str):
            label = [label] * len(entities)
        assert len(entities) == len(label)

        spans = []
        sentences = []
        indices = []
        for e, l in zip(entities, label):
            indices.append(len(sentences))
            spans.append([{"char_start_index": 0, "char_end_index": len(e), "label": l, "span": e}])
            sentences.append(e)
        ds = from_spans(entities=spans, sentences=sentences)
        result = self.predict_batch(ds, batch_size)
        if is_str:
            assert len(result["entities"]) == 1 and len(result["entities"][0]) == 1
            return result["entities"][0][0]
        else:
            _result = []
            for r in result["entities"]:
                assert len(r) == 1
                _result.append(r[0])
            return _result


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
