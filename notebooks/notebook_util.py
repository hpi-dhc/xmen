import pandas as pd
import re
from xmen.umls import get_semantic_groups
import os
from pathlib import Path
from xmen.evaluation import evaluate

def evaluate_at_k(ground_truth, pred, eval_k=[1,5,8,20,64]):
    for ki in eval_k:
        print(f'Perf@{ki}', evaluate(ground_truth, pred, top_k_predictions=ki)["strict"]['recall'])

def get_temp_path(conf):
    return Path(os.path.expanduser("~")) / 'runs'/ f'{conf.name}'

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
    
    gold_id = row["gold_concept"]["db_id"]
    pred_id = row["pred_top"]

    if gold_id == pred_id:
        return "TP"
    
    if not gold_id in kb.cui_to_entity:
        return "MISSING_CUI_GOLD"
    
    if not pred_id in kb.cui_to_entity:
        return "MISSING_CUI_PRED"

    if sem_group_version:
        if len(sem_group_for_cui(gold_id).intersection(set(row.gold_type))) == 0:
            return "INVALID_SEMANTIC_GROUP"
    
    if row.pred_index == -1:
        return "NOT_FOUND"

    def get_synset(scui):
        kb_ix = kb.cui_to_entity[str(scui)]
        return set([kb_ix.canonical_name] + kb_ix.aliases)

    if len(get_synset(gold_id).intersection(get_synset(pred_id))) > 0:
        return "SAME_SYNONYMS"

    if sem_group_version:
        if (
            len(sem_group_for_cui(gold_id).intersection(sem_group_for_cui(pred_id)))
            == 0
        ):
            return "WRONG_SEMANTIC_GROUP"

    if row.pred_top_score > 0.8 and row.pred_top_score - row.pred_index_score > 0.1:
        return "SUSPECTED_ANNOTATION_ERROR"
    
    if get_word_len(row) >= 3:
        return "COMPLEX_ENTITY"

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

import subprocess, getpass

def get_gpu_usage():
    """
    Returns a dict which contains information about memory usage for each GPU.
    In the following output, the GPU with id "0" uses 5774 MB of 16280 MB.
    253 MB are used by other users, which means that we are using 5774 - 253 MB.
    {
        "0": {
            "used": 5774,
            "used_by_others": 253,
            "total": 16280
        },
        "1": {
            "used": 5648,
            "used_by_others": 253,
            "total": 16280
        }
    }
    """

    # Name of current user, e.g. "root"
    current_user = getpass.getuser()

    # Find mapping from process ids to user names
    command = ["ps", "axo", "pid,user"]
    output = subprocess.check_output(command).decode("utf-8")
    pid_user = dict(row.strip().split()
        for row in output.strip().split("\n")[1:])

    # Find all GPUs and their total memory
    command = ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv"]
    output = subprocess.check_output(command).decode("utf-8")
    total_memory = dict(row.replace(",", " ").split()[:2]
        for row in output.strip().split("\n")[1:])

    # Store GPU usage information for each GPU
    gpu_usage = {gpu_id: {"used": 0, "used_by_others": 0, "total": int(total)}
        for gpu_id, total in total_memory.items()}

    # Use nvidia-smi to get GPU memory usage of each process
    command = ["nvidia-smi", "pmon", "-s", "m", "-c", "1"]
    output = subprocess.check_output(command).decode("utf-8")
    for row in output.strip().split("\n"):
        if row.startswith("#"): continue

        gpu_id, pid, type, mb, command = row.split()

        # Special case to skip weird output when no process is running on GPU
        if pid == "-": continue

        gpu_usage[gpu_id]["used"] += int(mb)

        # If the GPU user is different from us
        if pid_user[pid] != current_user:
            gpu_usage[gpu_id]["used_by_others"] += int(mb)

    return gpu_usage

def get_free_gpus():
    """
    Returns the ids of GPUs which are occupied to less than 1 GB by other users.
    """

    return [gpu_id for gpu_id, usage in get_gpu_usage().items()
        if usage["used"] == 0]

if __name__ == "__main__":
    import json

    print("GPU memory usage information:")
    print(json.dumps(get_gpu_usage(), indent=4))
    print()
    print("GPU ids of free GPUs:", get_free_gpus())
