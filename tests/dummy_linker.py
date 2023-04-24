from xmen.linkers import EntityLinker
from datasets.arrow_dataset import Dataset


class CopyLinker(EntityLinker):
    def predict_batch(self, dataset: Dataset) -> Dataset:
        return dataset

    def predict(self, sentence: str, entities: list) -> list:
        return entities


class NullLinker(EntityLinker):
    def predict_batch(self, dataset: Dataset) -> Dataset:
        return dataset.map(lambda e: {"entities": []})

    def predict(self, sentence: str, entities: list) -> list:
        return []
