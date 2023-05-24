from collections import defaultdict
import json
from pathlib import Path
from typing import List, Union
import pandas as pd
from scispacy.linking_utils import KnowledgeBase, Entity
from collections.abc import Mapping


def load_kb(file_path: Union[str, Path]):
    return CompositeKnowledgebase([file_path])


class CompositeKnowledgebase(KnowledgeBase):
    """
    Initializes a CompositeKnowledgebase object with the given list of file paths and optional list of mappers.

    Args:
    - file_paths (List[Union[str, Path]]): A list of file paths for JSONL files that contain the knowledge to be added to the CompositeKnowledgebase.
    - mappers (List[Mapping], optional): A list of mappers to transform each entry in the JSONL files before adding them to the CompositeKnowledgebase. Defaults to None.

    Raises:
    - AssertionError: If the length of the mappers list does not match the length of the file_paths list, or if any of the file paths does not have a ".jsonl" extension.

    Attributes:
    - alias_to_cuis (defaultdict): A defaultdict containing the mapping from each entity alias to the set of CUIs it is associated with.
    - cui_to_entity (dict): A dictionary mapping each CUI to its corresponding Entity object in the CompositeKnowledgebase.
    """

    def __init__(
        self,
        file_paths: List[Union[str, Path]],
        mappers: List[Mapping] = None,
    ):
        if not mappers:
            mappers = [lambda x: x] * len(file_paths)
        assert len(mappers) == len(file_paths)
        alias_to_cuis = defaultdict(set)
        self.cui_to_entity = {}

        for file_path, mapper in zip(file_paths, mappers):
            file_path = Path(file_path)
            assert file_path.suffix == ".jsonl"
            raw = [json.loads(line) for line in open(file_path)]

            for entry in raw:
                if mapper:
                    entry = mapper(entry)
                if not entry:
                    continue
                if type(entry) != list:
                    entry = [entry]
                for concept in entry:
                    if type(concept["concept_id"]) == int:
                        concept["concept_id"] = str(concept["concept_id"])
                    unique_aliases = set(concept["aliases"])
                    if "canonical_name" in concept:
                        unique_aliases.add(concept["canonical_name"])
                    for alias in unique_aliases:
                        alias_to_cuis[alias].add(concept["concept_id"])
                    if not concept["concept_id"] in self.cui_to_entity:
                        self.cui_to_entity[concept["concept_id"]] = Entity(**concept)
                    else:
                        self.cui_to_entity[concept["concept_id"]] = _merge_entities(
                            Entity(**concept), self.cui_to_entity[concept["concept_id"]]
                        )

            self.alias_to_cuis = {**alias_to_cuis}


def _merge_entities(e1: Entity, e2: Entity):
    """
    Merges two entities and returns the resulting entity. The two entities must have the same concept ID.

    Args:
    - e1 (Entity): The first entity to merge.
    - e2 (Entity): The second entity to merge.

    Returns:
    - Entity: The resulting entity after merging the two input entities.

    Raises:
    - AssertionError: If the two input entities have different concept IDs.
    """
    assert e1.concept_id == e2.concept_id

    canonical_name = e1.canonical_name
    if not canonical_name:
        canonical_name = e2.canonical_name
    definition = e1.definition
    if not definition:
        definition = e2.definition

    aliases = list(set(e1.aliases).union(set(e2.aliases)))
    types = list(set(e1.types).union(set(e2.types)))

    return Entity(e1.concept_id, canonical_name, aliases, types, definition)


def create_flat_term_dict(concept_names_jsonl: List[Union[str, Path]], mappers: List[Mapping] = None):
    """
    Creates a flattened pandas dataframe of UMLS terms and their metadata from a list of UMLS concept names JSON files.

    Args:
    - concept_names_jsonl (List[Union[str, Path]]): A list of file paths to UMLS concept names JSON files.
    - mappers (List[Mapping], optional): A list of mappers to apply to each JSON entry, one per JSON file.
    If not provided, defaults to None.

    Returns:
    - pd.DataFrame: A pandas dataframe containing UMLS term data. The columns are 'cui' (str), 'term' (str),
    'canonical' (str), and 'tuis' (list of str), representing the UMLS concept unique identifier, the term or alias,
    the canonical name, and the associated semantic types, respectively.

    Raises:
    - AssertionError: If the number of mappers provided is not equal to the number of JSON files provided, or if
    an entry does not have a canonical name.
    """
    term_dict = []
    if not mappers:
        mappers = [lambda x: x] * len(concept_names_jsonl)
    assert len(mappers) == len(concept_names_jsonl)
    for jsonl_file, mapper in zip(concept_names_jsonl, mappers):
        with open(jsonl_file) as f:
            for entry in f:
                entry = json.loads(entry)
                if mapper != None:
                    entry = mapper(entry)
                    if not entry:
                        continue
                if type(entry) != list:
                    entry = [entry]
                for e in entry:
                    assert e["canonical_name"]
                    cui = e["concept_id"]
                    tuis = e["types"]
                    term_dict.append(
                        {
                            "cui": str(cui),
                            "term": e["canonical_name"],
                            "canonical": e["canonical_name"],
                            "tuis": tuis,
                        }
                    )
                    for alias in e["aliases"]:
                        term_dict.append(
                            {
                                "cui": str(cui),
                                "term": alias,
                                "canonical": e["canonical_name"],
                                "tuis": tuis,
                            }
                        )
    term_dict = pd.DataFrame(term_dict)
    return term_dict.drop_duplicates(subset=["cui", "term"])
