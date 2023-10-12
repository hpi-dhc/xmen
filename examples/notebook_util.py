import pandas as pd
from xmen.umls import get_semantic_groups
from xmen.evaluation import get_synset


def get_error_types(row, kb, sem_group_version, return_fn):
    if sem_group_version:
        semgroups = get_semantic_groups(sem_group_version)
        tui2grp = semgroups.reset_index().set_index("TUI").GRP

    def get_semantic_types(scui):
        tuis = set(kb.cui_to_entity[str(scui)].types)
        return tuis

    def sem_group_for_cui(scui):
        groups = {tui2grp.loc[t] for t in kb.cui_to_entity[str(scui)].types if t in tui2grp.index}
        return groups

    gold_id = row["gold_concept"]["db_id"]
    pred_id = row["pred_top"]
    pred_index = row["pred_index"]

    if gold_id == pred_id:
        return "TP"

    if return_fn and pred_index == -1:
        return "FN"

    if not gold_id in kb.cui_to_entity:
        return "MISSING_CUI_GOLD"
    if not pred_id in kb.cui_to_entity:
        return "MISSING_CUI_PRED"

    if row["_word_len"] >= 3:
        return "COMPLEX_ENTITY"

    if row["_abbrev"]:
        return "ABBREV"

    if len(get_synset(kb, gold_id).intersection(get_synset(kb, pred_id))) > 0:
        return "SAME_ALIAS"

    sem_types = get_semantic_types(gold_id)
    if sem_types and len(sem_types.intersection(get_semantic_types(pred_id))) == 0:
        return "WRONG_SEMANTIC_TYPE"

    if sem_group_version:
        if len(sem_group_for_cui(gold_id).intersection({row.gold_type})) == 0:
            return "INVALID_SEMANTIC_GROUP"

    if sem_group_version:
        if len(sem_group_for_cui(gold_id).intersection(sem_group_for_cui(pred_id))) == 0:
            return "WRONG_SEMANTIC_GROUP"

    if pred_index > 0 and pred_index < 5:
        return "OTHER_TOP5"

    if pred_index > 0:
        return "OTHER_RANKING_ERROR"

    if pred_index == -1:  # No explanation left
        return "OTHER_FN"

    return "OTHER_ERROR"


def analyze(ea_df, kb, sem_group_version="v03", return_fn=True):
    error_types = ea_df.apply(lambda r: get_error_types(r, kb, sem_group_version, return_fn), axis=1)
    error_types.name = "error_type"
    edf = pd.concat([error_types, ea_df], axis=1)
    return edf
