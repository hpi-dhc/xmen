import pandas as pd


def get_dataframe(predictions, kb):
    ents = []
    for d in predictions:
        for e in d["entities"]:
            span = " ".join(e["text"])
            label = e["type"]
            top_concept = e["normalized"][0] if len(e["normalized"]) > 0 else None
            if top_concept:
                cui = top_concept["db_id"]
                ents.append(
                    {
                        "mention": span,
                        "class": label,
                        "cui": cui,
                        "canonical name": kb.cui_to_entity[cui].canonical_name,
                        "linked by": top_concept["predicted_by"],
                        "score": top_concept["score"],
                    }
                )
            else:
                ents.append({"mention": span, "class": label, "cui": "Not linkable"})
    return pd.DataFrame(ents)
