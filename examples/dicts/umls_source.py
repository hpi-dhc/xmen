from tqdm.auto import tqdm
from xmen.umls import read_umls_file_headers, get_umls_concepts
from collections import defaultdict
from xmen.log import logger


def get_concept_details(cfg):
    mrconso = "MRCONSO.RRF"
    concept_details = {}
    meta_path = cfg.dict.custom.umls_meta_path
    sabs = cfg.dict.custom.sabs
    id_key = cfg.dict.custom.get("id_key")

    headers = read_umls_file_headers(meta_path, mrconso)

    scui2cui = defaultdict(list)

    with open(f"{meta_path}/{mrconso}") as fin:
        for line in tqdm(fin.readlines()):
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            concept = dict(zip(headers, splits))
            if concept["SAB"] in sabs:
                cui = concept["CUI"]
                if id_key:
                    sid = concept[id_key]
                else:
                    sid = concept["SDUI"]
                    if not sid:
                        sid = concept["SCUI"]
                if not sid:
                    logger.warn(
                        f"Skipping concept with CUI {cui} because we could not find a valid source vocabulary ID"
                    )

                if not sid in concept_details:
                    concept_details[sid] = {"concept_id": sid, "types": [], "aliases": []}
                name = concept["STR"]

                is_canonical = concept["TTY"] == "PT" and concept["TS"] == "P"

                if not is_canonical or "canonical_name" in concept_details[sid]:
                    # not a canonical name or a canonical name already found
                    concept_details[sid]["aliases"].append(name)  # add it as an alias
                else:
                    concept_details[sid]["canonical_name"] = name  # set as canonical name

                scui2cui[sid].append(cui)

    for concept in concept_details.values():
        if not "canonical_name" in concept:
            concept["canonical_name"] = concept["aliases"].pop()

    if umls_extend := cfg.dict.custom.get("umls_extend"):
        # Optionally extend with UMLS synonyms
        other_umls_concepts = get_umls_concepts(
            meta_path,
            umls_extend.get("lang"),
            sabs=umls_extend.get("sabs"),
            sources=umls_extend.get("sources"),
            semantic_groups=umls_extend.get("semantic_groups"),
            semantic_types=umls_extend.get("semantic_types"),
        )

        for scui, concept in tqdm(concept_details.items()):
            for cui in scui2cui[scui]:
                if cui in other_umls_concepts:
                    for t in other_umls_concepts[cui]["types"]:
                        if not t in concept["types"]:
                            concept["types"].append(t)
                    for new_alias in other_umls_concepts[cui]["aliases"] + [other_umls_concepts[cui]["canonical_name"]]:
                        if new_alias not in concept["aliases"] and new_alias != concept["canonical_name"]:
                            concept["aliases"].append(new_alias)

    return concept_details
