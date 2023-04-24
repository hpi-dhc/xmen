def filter_and_apply_threshold(input_pred, k: int, threshold: float):
    assert k >= 0

    def apply(entry):
        entities = entry["entities"]
        for e in entities:
            filtered = [n for n in e["normalized"] if "score" in n and n["score"] >= threshold]
            e["normalized"] = sorted(filtered, key=lambda n: n["score"], reverse=True)[:k]
        return {"entities": entities}

    return input_pred.map(apply, load_from_cache_file=False)
