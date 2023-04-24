import pandas as pd


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.distemist_path
    distemist_dict = pd.read_csv(path, sep="\t")
    distemist_dict.sort_values("code", inplace=True)

    concept_details = {}

    for _, entry in distemist_dict.iterrows():
        sid = entry.code
        if not sid in concept_details:
            concept_details[sid] = {"concept_id": sid, "canonical_name": None, "types": [], "aliases": []}
        if entry.mainterm:
            assert not concept_details[sid]["canonical_name"]
            concept_details[sid]["canonical_name"] = entry.term
        else:
            concept_details[sid]["aliases"].append(entry.term)
        if entry.semantic_tag not in concept_details[sid]["types"]:
            concept_details[sid]["types"].append(entry.semantic_tag)

    for v in concept_details.values():
        if not v["canonical_name"]:
            v["canonical_name"] = v["aliases"].pop()

    return concept_details
