import pandas as pd
import re
from xmen.umls import get_semantic_groups
import os
from pathlib import Path
from xmen.evaluation import evaluate

def get_error_types(row, kb, sem_group_version):  
    if sem_group_version:
        semgroups = get_semantic_groups(sem_group_version)
        tui2grp = semgroups.reset_index().set_index("TUI").GRP

    def get_semantic_types(scui):
        tuis = set(kb.cui_to_entity[str(scui)].types)
        return tuis

    def sem_group_for_cui(scui):
        groups = {
            tui2grp.loc[t]
            for t in kb.cui_to_entity[str(scui)].types
            if t in tui2grp.index
        }
        return groups
    
    def get_synset(scui):
        kb_ix = kb.cui_to_entity[str(scui)]
        return set([kb_ix.canonical_name] + kb_ix.aliases)
    
    gold_id = row["gold_concept"]["db_id"]
    pred_id = row["pred_top"]

    if gold_id == pred_id:
        return "TP"
    
    if not gold_id in kb.cui_to_entity:
        return "MISSING_CUI_GOLD"
    
    if not pred_id in kb.cui_to_entity:
        return "MISSING_CUI_PRED"
    
    if get_word_len(row) >= 3:
        return "COMPLEX_ENTITY"
    
    if (len(row.gt_text) == 1) and bool(
        re.match("[A-Z]{2,3}", row.gt_text[0])
    ):
        return "ABBREV"
    
    if (
        len(get_semantic_types(gold_id).intersection(get_semantic_types(pred_id)))
            == 0
        ):
        return "WRONG_SEMANTIC_TYPE"
    
    
    if len(get_synset(gold_id).intersection(get_synset(pred_id))) > 0:
        return "SAME_SYNONYMS"
    
    if sem_group_version:
        if len(sem_group_for_cui(gold_id).intersection(set(row.gold_type))) == 0:
            return "INVALID_SEMANTIC_GROUP"

    if sem_group_version:
        if (
            len(sem_group_for_cui(gold_id).intersection(sem_group_for_cui(pred_id)))
            == 0
        ):
            return "WRONG_SEMANTIC_GROUP"



    return "UNKNOWN_ERROR"

def get_word_len(row):
    return len(" ".join(row.gt_text).split(" "))

def get_entity_category(row, kb):
    gold_id = row["gold_concept"]["db_id"]
    res = {}
    res["word_len"] = get_word_len(row)
    res["ABREV"] = (len(row.gt_text) == 1) and bool(
        re.match("[A-Z]{2,3}", row.gt_text[0])
    )
    res["exists"] = gold_id in kb.cui_to_entity

    return res

def get_category(r):
    if r.error_type == "TP":
        return "TP"
    if not r.exists:
        assert r.pred_index == -1
        return "MISSING_CUI"
    if r.ABREV and r.error_type != "TP":
        return "ABREV"
    if r.pred_index != -1 and r.pred_index < 10:
        return "TOP10_" + r.error_type
    if r.word_len < 3 and r.error_type == "NOT_FOUND":
        return "SIMPLE_NOT_FOUND"
    return r.error_type


def analyze(ea_df, kb, name, sem_group_version='v03'):
    entity_types = pd.DataFrame(
        list(ea_df.apply(lambda r: get_entity_category(r, kb), axis=1))
    )
    error_types = ea_df.apply(lambda r: get_error_types(r, kb, sem_group_version), axis=1)
    error_types.name = "error_type"
    edf = pd.concat([error_types, entity_types, ea_df], axis=1)
    cats = edf.apply(get_category, axis=1)
    cats.name = "error_cat"
    edf = pd.concat([cats, edf], axis=1)
    cat_errors = cats.value_counts() / len(cats)
    cat_errors.name = name

    return edf, cat_errors