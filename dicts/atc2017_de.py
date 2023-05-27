import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
import openpyxl


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.atc_path
    wb = openpyxl.load_workbook(path)
    sheet = wb["WIdO-Index alphabetisch_2017"]
    atc_dict = pd.DataFrame(sheet.values)
    atc_dict = atc_dict.rename(columns={0: "code", 2: "text", 4: "DDD_Info"})
    atc_dict.drop(0, axis=0, inplace=True)
    atc_dict.dropna(axis=0, how="all", inplace=True)
    atc_dict.drop(atc_dict[atc_dict.text.isnull()].index, inplace=True)

    concept_details = {}
    for _, entry in atc_dict.iterrows():
        sid = entry.code
        if not sid in concept_details:
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)

    return concept_details
