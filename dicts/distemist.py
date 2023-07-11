import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

from xmen.log import logger
from xmen.umls import read_umls_file_headers, get_umls_concepts


def read_snomed2cui_mapping(meta_path):
    mrconso = "MRCONSO.RRF"
    snomed2cui = defaultdict(list)
    headers = read_umls_file_headers(meta_path, mrconso)
    with open(f"{meta_path}/{mrconso}") as fin:
        for line in tqdm(fin.readlines()):
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            concept = dict(zip(headers, splits))
            if concept["SAB"] in ["SNOMEDCT_US", "SCTSPA"]:
                snomed2cui[str(concept["SCUI"])].append(concept["CUI"])
    return snomed2cui


def get_concept_details(cfg) -> dict:
    path = cfg.dict.custom.distemist_path
    distemist_dict = pd.read_csv(path, sep="\t")
    distemist_dict.sort_values("code", inplace=True)

    concept_details = {}

    for _, entry in distemist_dict.iterrows():
        sid = str(entry.code)
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

    if meta_path := cfg.dict.custom.get("umls_meta_path"):
        langs = cfg.dict.custom.lang
        logger.info(f"Extending Distemist by UMLS synonyms from {meta_path} in languages {langs}")
        umls_concepts = get_umls_concepts(
            meta_path, langs, sabs=None, sources=None, semantic_groups=None, semantic_types=None
        )
        snomed2cui = read_snomed2cui_mapping(meta_path)

        for sid, concept in concept_details.items():
            for c in snomed2cui[sid]:
                if c in umls_concepts:
                    for alias in umls_concepts[c]["aliases"] + [umls_concepts[c]["canonical_name"]]:
                        if not alias in concept["aliases"]:
                            concept["aliases"].append(alias)

    return concept_details
