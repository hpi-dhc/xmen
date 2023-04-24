import pandas as pd


def get_concept_details(cfg) -> dict:
    # load all concepts in a df and initialize empty the output dict
    path = cfg.dict.custom.dic_path
    concepts = pd.read_csv(path)
    concept_details = {}

    for _, row in concepts.iterrows():
        # consider term only if the language and sem groups are contemplated in the config.yaml
        if row["lang"] in cfg.dict.custom.lang and row["sem"] in cfg.dict.custom.sem:
            cui = row["cui"]
            # make a new key for new cuis, the first alias will be canonical term and all terms for same CUI share type
            if cui not in concept_details:
                concept_details[cui] = {"canonical_name": row["alias"], "types": [row["sem"]], "aliases": []}
            elif cui in concept_details:
                concept_details[cui]["aliases"].append(row["alias"])

    return concept_details
