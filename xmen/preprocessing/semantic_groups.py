from xmen.umls import get_semantic_groups


class SemanticGroupFilter:
    """
    A class for filtering entities in a dataset based on their semantic group.

    Args:
    - kb (EntityKnowledgeBase): The knowledge base containing the entities and their types.
    - version (str, optional): The version of the semantic group mapping to use.
    """

    def __init__(self, kb, version=None):
        self._kb = kb
        self._sem_groups = get_semantic_groups(version).set_index("TUI")

    def __getstate__(self):
        return {}

    def get_sem_groups(self, cui):
        """
        Returns the semantic group(s) of an entity.

        Args:
        - cui (str): The concept unique identifier of the entity.

        Returns:
        - A list of the semantic group(s) of the entity, or an empty list if the entity is not found in the knowledge base.
        """
        entry = self._kb.cui_to_entity.get(cui, None)
        if not entry:
            return []
        return [self._sem_groups.loc[t].GRP for t in entry.types]

    def filter_semantic_groups(self, example):
        """
        Filters the entities in an example based on their semantic group.

        Args:
        - example (dict): A dictionary containing the example entities.

        Returns:
        - A dictionary containing the filtered entities.
        """
        entities = example["entities"]
        for e in entities:
            sem_groups = e["type"]
            filtered = []
            for n in e["normalized"]:
                concept_groups = self.get_sem_groups(n["db_id"])
                if any([g for g in sem_groups if g in concept_groups]):
                    filtered.append(n)
            e["normalized"] = filtered
        return {"entities": entities}

    def transform_batch(self, ds):
        """
        Applies the semantic group filter to a dataset.

        Args:
        - ds (Dataset): The dataset to apply the filter to.

        Returns:
        - The filtered dataset.
        """
        return ds.map(lambda e: self.filter_semantic_groups(e), load_from_cache_file=False)
