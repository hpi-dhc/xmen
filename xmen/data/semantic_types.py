from xmen.umls import get_sem_type_tree, expand_tuis
from typing import Dict


class SemanticTypeFilter:
    """
    A class to filter out examples based on semantic types.

    Args:
    - type_to_tui (Dict): A dictionary that maps a semantic type to a set of TUIs.
    - kb: A knowledge base object.
    - expand_types: Whether to expand the given TUIs based on the semantic type tree

    Attributes:
    - type_to_tui (Dict): A dictionary that maps a semantic type to a set of TUIs.
    - kb: A knowledge base object.
    - tree: A semantic type tree object.
    - expand_types: flag indicating whether to expand semantic types
    """

    def __init__(self, type_to_tui: Dict, kb, expand_types=False):
        self.type_to_tui = type_to_tui
        self.kb = kb
        self.tree = get_sem_type_tree()
        self.expand_types = expand_types

    def __getstate__(self):
        return {}

    def get_tuis(self, cui):
        """
        Returns the set of TUIs associated with a given CUI.

        Args:
        - cui (str): A CUI string.

        Returns:
        - tuis (set): A set of TUIs associated with the given CUI.
        """
        return self.kb.cui_to_entity[cui].types

    def filter_semantic_types(self, example):
        """
        Filters out normalized entities from the given example that are not associated with any of the valid TUIs.

        Args:
        - example (dict): A dictionary representing a single example, with keys "text" and "entities".

        Returns:
        - filtered_example (dict): A dictionary representing the filtered example, with key "entities".
        """
        entities = example["entities"]
        for e in entities:
            valid_tuis = self.type_to_tui[e["type"]]
            if self.expand_types:
                valid_tuis = expand_tuis(valid_tuis, self.tree)
            filtered = []
            for n in e["normalized"]:
                concept_tuis = self.get_tuis(n["db_id"])
                if any([t for t in concept_tuis if t in valid_tuis]):
                    filtered.append(n)
            e["normalized"] = filtered
        return {"entities": entities}

    def transform_batch(self, ds):
        """
        Transforms the given dataset by applying the filter_semantic_groups method to each example in the dataset.

        Args:
        - ds (tf.data.Dataset): A dataset of examples.

        Returns:
        - transformed_ds (tf.data.Dataset): A transformed dataset of examples.
        """
        return ds.map(lambda e: self.filter_semantic_types(e), load_from_cache_file=False)
