import pandas as pd
import numpy as np
from omegaconf import OmegaConf, DictConfig
import openpyxl


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.atc_path
    wb = openpyxl.load_workbook(f"{path}/ATC GKV AI 2017.xlsx")
    sheet = wb["WIdO-Index sortiert_2017"]
    sort = pd.DataFrame(sheet.values)
    sort = sort.rename(columns={0: "code", 2: "text", 4: "DDD_Info"})
    sort.drop(0, axis=0, inplace=True)
    sort.dropna(axis=0, how="all", inplace=True)
    sort.drop(sort[sort.text.isnull()].index, inplace=True)
    sort.drop(sort[sort.code.isnull()].index, inplace=True)

    sort["code"] = sort["code"].str.rstrip()
    sort = sort[sort["code"].str.len() > 3]

    concept_details = {}
    for _, entry in sort.iterrows():
        sid = entry.code
        if not sid in concept_details:
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)

    return concept_details
