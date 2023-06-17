import pandas as pd
from omegaconf import OmegaConf, DictConfig


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.ops_path

    # load alphabetic codes
    alph_cols = ["ArtDerKodierung", "DIMDI-Nummer", "code", "code2", "text"]
    alph = pd.read_csv(
        f"{path}/p2set2017/ops2017alpha_edvtxt_20161028.txt",
        sep="|",
        names=alph_cols,
        index_col=False,
        encoding="latin1"
    )

    # load systematic codes
    syst_cols = list(range(30))
    syst_cols[3] = "type"
    syst_cols[6] = "code"
    syst_cols[8] = "text"
    syst_cols[4] = "code2"
    syst_cols[5] = "code3"
    syst = pd.read_csv(f"{path}/p1smt2017/Klassifikationsdateien/ops2017syst_kodes.txt", sep=";", names=syst_cols)

    # get the codes that are only in the systematic list
    syst_codes = list(syst["code"].unique())
    alph_codes = list(alph["code"].unique())
    only_syst = [code for code in syst_codes if code not in alph_codes]
    syst = syst[syst["code"].isin(only_syst)]

    # get all concepts from alphabetic (they have aliases and "s.a.", etc.)
    concept_details = {}
    for _, entry in alph.iterrows():
        sid = entry.code
        if not sid in concept_details and not pd.isna(sid):
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)

    # get from systematic the concepts whose codes are only there and not in alphabetic
    for _, entry in syst.iterrows():
        sid = entry.code
        if not sid in concept_details and not pd.isna(sid):
            concept_details[sid] = {"concept_id": sid, "canonical_name": entry.text, "types": [], "aliases": []}
        elif sid in concept_details:
            concept_details[sid]["aliases"].append(entry.text)
        if entry.type not in concept_details[sid]["types"]:
            concept_details[sid]["types"].append(entry.type)

    # remove all resulting duplicates
    for k, v in concept_details.items():
        concept_details[k]["aliases"] = list(set(v["aliases"]))

    return concept_details
