from typing import Union, List


def init_schema(dataset):
    """
    Initializes the non-BigBIO elements of a dataset
    """
    n_docs = len(dataset["document_id"])
    if not "corpus_id" in dataset:
        dataset["corpus_id"] = [None] * n_docs
    if not "lang" in dataset:
        dataset["lang"] = [None] * n_docs
    for es in dataset["entities"]:
        for e in es:
            for n in e["normalized"]:
                if not "score" in n:
                    n["score"] = 0.0
                if not "predicted_by" in n:
                    n["predicted_by"] = []
            if not "long_form" in e:
                e["long_form"] = None
    return dataset


class Concept:
    """
    A class representing a concept.

    Args:
    - db_id (str): The id of the concept.
    - score (float): The score of the concept.
    - db_name (str): The name of the knowledge base containing the concept.
    - type (str): The type of the concept.

    Attributes:
    - _dict (dict): A dictionary containing the 'db_id', 'target_kb', 'type', and 'score' keys.
    """

    def __init__(
        self,
        db_id: str = None,
        score: float = None,
        db_name: str = None,
        type: str = None,
    ):
        self._dict = {
            "db_id": db_id,
            "target_kb": db_name,
            "type": type,
            "score": score,
        }


class Entity:
    """
    A class representing an entity.

    Args:
    - offsets: A tuple or list of tuples representing the start and end character offsets of the entity.
    - text (Union[str, List[str]]): A string or list of strings representing the text of the entity.
    - id (str): The id of the entity.
    - entity_type (str): The type of the entity.
    - concepts (List[Concept]): A list of concepts associated with the entity.

    Attributes:
    - _dict (dict): A dictionary containing the 'id', 'text', 'offsets', 'type', and 'normalized' keys.
    """

    def __init__(
        self,
        offsets,
        text: Union[str, List[str]],
        id: str = "1",
        entity_type: str = None,
        concepts: List[Concept] = None,
    ):
        self._dict = {
            "id": id,
            "text": [text] if type(text) == str else text,
            "offsets": offsets,
            "type": entity_type,
            "normalized": [c._dict for c in concepts] if concepts else [],
        }


def make_document(entities: List[Entity], document_id: str = "1", corpus_id: str = "x") -> dict:
    """
    Creates a dictionary representing a document.

    Args:
    - entities (List[Entity]): A list of entities in the document.
    - document_id (str): The id of the document.
    - corpus_id (str): The id of the corpus containing the document.

    Returns:
    - A dictionary with the 'corpus_id', 'document_id', and 'entities' keys.
    """
    return {
        "corpus_id": corpus_id,
        "document_id": document_id,
        "entities": [e._dict for e in entities],
    }


def get_cuis(dataset):
    """
    Extracts the unique CUIs (Concept Unique Identifiers) from a dataset.

    Args:
    - dataset: A dictionary containing the 'entities' key, which is a list of Entity objects.

    Returns:
    - A list of unique CUIs.
    """
    return [c["db_id"] for doc in dataset["entities"] for e in doc for c in e["normalized"]]


def filter_and_apply_threshold(input_pred, k: int, threshold: float):
    """
    Filters the predicted entities based on a score threshold and returns the top k entities for each entity.

    Args:
    - input_pred (list): A list of predicted entities to be filtered and thresholded.
    - k (int): The number of top entities to keep after filtering.
    - threshold (float): The score threshold below which entities will be removed.

    Returns:
    - list: A list of filtered and thresholded predicted entities.
    """
    assert k >= 0

    def apply(entry):
        entities = entry["entities"]
        for e in entities:
            filtered = [n for n in e["normalized"] if "score" in n and n["score"] >= threshold]
            e["normalized"] = sorted(filtered, key=lambda n: n["score"], reverse=True)[:k]
        return {"entities": entities}

    return input_pred.map(apply, load_from_cache_file=False)
