import pandas as pd
from omegaconf import OmegaConf, DictConfig
import re


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.icd10gm_path

    # load alphabetic codes
    alph_cols = ["ArtDerKodierung", "DIMDI-Nummer", "Druckkennzeichen", "code", "code2", "code3", "text"]
    alph = pd.read_csv(
        f"{path}/x3get2017/icd10gm2017alpha_edvtxt_20161005.txt", sep="|", names=alph_cols, index_col=False
    )

    # load systematic codes
    syst_cols = list(range(30))
    syst_cols[3], syst_cols[6], syst_cols[8] = "type", "code", "text"
    syst = pd.read_csv(f"{path}/x1gmt2017/Klassifikationsdateien/icd10gm2017syst_kodes.txt", sep=";", names=syst_cols)

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

    # deal with "sehe auch"
    for k, v in concept_details.items():
        # if there is " - s. a." in canonical, leave the first term there and move the rest to aliases
        chunks = v["canonical_name"].split(" - s.a. ")
        if len(chunks) == 2:
            concept_details[k]["canonical_name"] = chunks[0]
            concept_details[k]["aliases"].append(chunks[1])

        # deal with first instance for aliases
        for alias in v["aliases"]:
            chunks = alias.split(" - s.a. ")
            if len(chunks) == 2:
                concept_details[k]["aliases"].append(chunks[0])
                concept_details[k]["aliases"].append(chunks[1])
                concept_details[k]["aliases"].remove(alias)

        # deal with instances with multiple "s.a."s
        for alias in v["aliases"]:
            chunks = alias.split(" s.a. ")
            if len(chunks) > 1:
                for a in chunks:
                    concept_details[k]["aliases"].append(a)
                concept_details[k]["aliases"].remove(alias)

    # deal with abbreviations
    for k, v in concept_details.items():
        brackets = r"\[(.*?)\]"

        text = v["canonical_name"]
        if "[" in text and "]" in text:
            abbs = re.findall(brackets, text)
            main_text = re.sub(brackets, "", text)
            concept_details[k]["canonical_name"] = main_text
            for abb in abbs:
                concept_details[k]["aliases"].append(abb)

        for alias in v["aliases"]:
            text = alias
            if "[" in text and "]" in text:
                abbs = re.findall(brackets, text)
                main_text = re.sub(brackets, "", text)

                concept_details[k]["aliases"].append(main_text)
                for abb in abbs:
                    concept_details[k]["aliases"].append(abb)

                concept_details[k]["aliases"].remove(alias)

    # remove all resulting duplicates
    for k, v in concept_details.items():
        concept_details[k]["aliases"] = list(set(v["aliases"]))

    return concept_details
