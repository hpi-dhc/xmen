from collections.abc import Mapping
from typing import List
from xmen.reranking import Reranker


class RuleBasedReranker(Reranker):
    """
    Reranks a dataset using a list of rules.

    Args:
    - rules (List[Mapping]): A list of mappings representing the rules to be applied.

    Attributes:
    - rules (List[Mapping]): A list of mappings representing the rules to be applied.

    Returns:
    - The reranked dataset.
    """

    def __init__(self, rules: List[Mapping]) -> None:
        self.rules = rules

    def rerank_batch(self, dataset):
        def apply_rules(concepts, context):
            for r in self.rules:
                concepts = r(concepts, context)
            return concepts

        def apply_rules_unit(entry):
            mapped_entities = []
            for e in entry["entities"]:
                e = e.copy()
                concepts = e["normalized"]
                concepts = apply_rules(concepts, e)
                e.update({"normalized": concepts})
                mapped_entities.append(e)
            return {"entities": mapped_entities}

        return dataset.map(apply_rules_unit, load_from_cache_file=False)
