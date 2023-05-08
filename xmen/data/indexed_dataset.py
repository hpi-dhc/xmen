import pickle
from pathlib import Path
import datasets


class IndexedDatasetDict(dict):
    """
    A dictionary of IndexedDataset objects that can be saved to and loaded from disk.

    Inherits from dict.

    Methods:
    - save_to_disk(folder: str) -> None: Saves the IndexedDataset objects to the specified folder.
    - load_from_disk(folder: str) -> IndexedDatasetDict: Loads the IndexedDataset objects from the specified folder.

    """

    def save_to_disk(self, folder):
        for k, v in self.items():
            v.save_to_disk(Path(folder) / k)

    @staticmethod
    def load_from_disk(folder):
        res = IndexedDatasetDict()
        for f in Path(folder).glob("*"):
            if f.is_dir():
                res[f.name] = IndexedDataset.load_from_disk(f)
        return res


class IndexedDataset:
    """
    A class that wraps a dataset and an index and provides methods to save and load them from disk.

    Attributes:
    - dataset: The dataset to wrap.
    - index: The index of the dataset.

    Methods:
    - save_to_disk(folder: str) -> None: Saves the dataset and the index to the specified folder.
    - load_from_disk(folder: str) -> IndexedDataset: Loads the dataset and the index from the specified folder.
    - __repr__() -> str: Returns a string representation of the IndexedDataset object.
    """

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index

    def save_to_disk(self, folder):
        """
        Saves the dataset and the index to the specified folder.

        Args:
        - folder (str or Path): The path to the folder where the dataset and index will be saved.
        """
        folder = Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        with open(folder / "dataset", "wb") as fh:
            pickle.dump(self.dataset, fh)
        with open(folder / "index", "wb") as fh:
            pickle.dump(self.index, fh)

    @staticmethod
    def load_from_disk(folder):
        """
        Loads the dataset and the index from the specified folder.

        Args:
        - folder (str or Path): The path to the folder where the dataset and index will be saved.
        """
        with open(Path(folder) / "dataset", "rb") as fh:
            dataset = pickle.load(fh)
        with open(Path(folder) / "index", "rb") as fh:
            index = pickle.load(fh)
        return IndexedDataset(dataset, index)

    def __repr__(self):
        if type(self.dataset) == datasets.Dataset:
            return self.dataset.__repr__()
        else:
            return f"[{len(self.dataset)} items]"
