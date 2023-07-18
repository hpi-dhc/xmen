class EmptyNormalizationFilter:
    """
    A class to filter out entities with empty normalized values from text data.
    """

    def transform_batch(self, dataset):
        """
        Filters entities with empty normalized values from a given dataset.

        Args:
        - dataset (Dataset): A dataset object from the Hugging Face datasets library.

        Returns:
        - Dataset: A transformed dataset object with the same number of examples, but with empty-normalized entities removed.
        """

        def filter_empty(d):
            ents = [e for e in d["entities"] if len(e["normalized"]) > 0]
            return {"entities": ents}

        return dataset.map(filter_empty)


class MissingCUIFilter:
    """
    A class to filter out entities with missing CUIs according to a given knowledge base.
    """

    def __init__(self, kb):
        self.kb = kb

    def __getstate__(self):
        return {}

    def transform_batch(self, dataset):
        def filter_missing_cuis(entities):
            result = []
            for e in entities:
                filtered = []
                for n in e["normalized"]:
                    if n["db_id"] in self.kb.cui_to_entity:
                        filtered.append(n)
                if len(filtered) > 0:
                    e["normalized"] = filtered
                    result.append(e)
            return result

        return dataset.map(lambda i: {"entities": filter_missing_cuis(i["entities"])})
