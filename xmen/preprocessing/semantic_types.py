from xmen.umls import get_sem_type_tree, expand_tuis
from typing import Dict

class SemanticTypeFilter:
    def __init__(self, type_to_tui : Dict, kb):
        self.type_to_tui = type_to_tui
        self.kb = kb
        self.tree = get_sem_type_tree()

    def __getstate__(self):
        return {}

    def get_tuis(self, cui):
        return self.kb.cui_to_entity[cui].types
    
    def filter_semantic_groups(self, example):
        entities = example["entities"]
        for e in entities:
            valid_tuis = self.type_to_tui[e["type"]]
            valid_tuis = expand_tuis(valid_tuis, get_sem_type_tree)
            filtered = []
            for n in e["normalized"]:
                concept_tuis = self.get_tuis(n["db_id"])
                if any([t for t in concept_tuis if t in valid_tuis]):
                    filtered.append(n)
            e["normalized"] = filtered
        return {"entities": entities}

    def transform_batch(self, ds):
        return ds.map(lambda e: self.filter_semantic_groups(e), load_from_cache_file=False)
