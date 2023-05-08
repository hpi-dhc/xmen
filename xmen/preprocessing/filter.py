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
