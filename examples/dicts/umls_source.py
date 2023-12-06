from tqdm.auto import tqdm
from xmen.umls import read_umls_file_headers


def get_concept_details(cfg):
    mrconso = "MRCONSO.RRF"
    concept_details = {}
    meta_path = cfg.dict.custom.umls_meta_path
    sabs = cfg.dict.custom.sabs

    headers = read_umls_file_headers(meta_path, mrconso)

    with open(f"{meta_path}/{mrconso}") as fin:
        for line in tqdm(fin.readlines()):
            splits = line.strip().split("|")
            assert len(headers) == len(splits)
            concept = dict(zip(headers, splits))
            if concept["SAB"] in sabs:
                sid = concept["SDUI"]
                name = concept["STR"]
                if sid in concept_details:
                    concept_details[sid]["aliases"].append(name)
                else:
                    concept_details[sid] = {"concept_id": sid, "canonical_name": name, "types": [], "aliases": []}
    return concept_details
