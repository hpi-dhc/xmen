import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.ops_path
    cols = list(np.arange(0, 15))
    cols[3], cols[6], cols[8] = "type", "code", "text"
    icd_dict = pd.read_csv(path, sep=";", names=cols)

    concept_details = {}
    for _, entry in icd_dict.iterrows():
        sid = entry.code
        if not sid in concept_details:
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)
        if entry.type not in concept_details[sid]["types"]:
            concept_details[sid]["types"].append(entry.type)

    return concept_details
