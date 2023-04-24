from xmen.umls import get_semantic_groups


class SemanticGroupFilter:
    def __init__(self, kb, version=None):
        self._kb = kb
        self._sem_groups = get_semantic_groups(version).set_index("TUI")

    def __getstate__(self):
        return {}

    def get_sem_groups(self, cui):
        entry = self._kb.cui_to_entity.get(cui, None)
        if not entry:
            return []
        return [self._sem_groups.loc[t].GRP for t in entry.types]

    def filter_semantic_groups(self, example):
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
        return ds.map(lambda e: self.filter_semantic_groups(e), load_from_cache_file=False)
