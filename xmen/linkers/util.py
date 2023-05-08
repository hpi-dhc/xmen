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
