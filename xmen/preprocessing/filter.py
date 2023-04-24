class EmptyNormalizationFilter:
    def transform_batch(self, dataset):
        def filter_empty(d):
            ents = [e for e in d["entities"] if len(e["normalized"]) > 0]
            return {"entities": ents}

        return dataset.map(filter_empty)
